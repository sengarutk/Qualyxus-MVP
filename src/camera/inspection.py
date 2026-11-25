"""From a part in view to one view's result: queueing, batching, judging, recording."""
import logging
import time
from typing import Any

import cv2
import numpy as np

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

from src.process.models import NOT_INSPECTED
from src.camera.presence import PresenceDetector
from src.camera.part_inspection import CompletedPart, PartPresenceTracker, part_pose
from src.inference.anomaly.image_quality import image_fault
from src.inference.inspection_stack import PartJudgement
from src.inference.postprocess import severity_from_conf
from src.logging.event_logger import EventLogger
from src.system.station_clock import station_time
from src.telemetry.plc_adapter import PLC_FLAG_NOT_INSPECTED, PLC_PASS

from src.camera.worker import CameraWorkerState, BACKLOG_BYTES, INSPECT_BATCH, EVIDENCE_PENDING, UNKNOWN_RECIPE

logger = logging.getLogger("src.camera.camera_manager")


class InspectionMixin:
    @property
    def anomaly_detector(self):
        return self.stack_provider.get()[0].anomaly

    @property
    def segmenter(self):
        return self.stack_provider.get()[0].segmenter

    @property
    def type_classifier(self):
        return self.stack_provider.get()[0].typer

    def set_recipe(self, camera_id: str, recipe: str | None) -> None:
        """Switch the product/recipe a camera inspects (models load on its next part)."""
        if recipe:
            self.camera_recipes[camera_id] = recipe
        else:
            self.camera_recipes.pop(camera_id, None)
        logger.info("[Inspection] %s now inspects recipe '%s'", camera_id, recipe or self.stack_provider.default_recipe)

    def _recipe_for(self, camera_id: str, cycle_id: str | None) -> str:
        """The camera's recipe or the default; with recipe_from_scalar, the producing cycle's value.

        A machine that should name the product and does not gives UNKNOWN_RECIPE, which no model
        judges: the wrong product's model must never pass a part.
        """
        if self.recipe_from_scalar and cycle_id and self.process_service is not None:
            try:
                cycle = self.process_service.store.get_cycle(cycle_id)
                value = (cycle.scalars or {}).get(self.recipe_from_scalar) if cycle else None
                if value:
                    return str(value)
            except Exception as e:
                logger.warning("[Inspection] recipe lookup for %s failed: %s", cycle_id, e)
        if self.recipe_from_scalar:
            return UNKNOWN_RECIPE
        return self.camera_recipes.get(camera_id) or self.stack_provider.default_recipe

    def _model_recipe(self, camera_id: str, product: str) -> str:
        """Each face of a part looks different, so a camera with ``model_view`` has its own model."""
        view = getattr(self, "_model_views", {}).get(camera_id)
        return f"{product}-{view}" if view else product

    def _good_pool(self, recipe: str):
        if not self._pool_cfg.get("enabled", True):
            return None
        pool = self._good_pools.get(recipe)
        if pool is None:
            from src.inference.good_pool import GoodPartPool

            pool = self._good_pools.setdefault(recipe, GoodPartPool(
                self._pool_cfg.get("dir", "data/good_pool"), recipe,
                every_n=int(self._pool_cfg.get("every_n", 5)), max_items=int(self._pool_cfg.get("max_items", 3000)),
                max_score=self._pool_cfg.get("max_score", 0.8),
                calibration_every_n=int(self._pool_cfg.get("calibration_every_n", 20)),
                calibration_max_items=int(self._pool_cfg.get("calibration_max_items", 2000)),
                max_gb=float(self._pool_cfg.get("max_gb", 5.0))))
        return pool

    def _track_part(self, state: CameraWorkerState, frame: np.ndarray, frame_id: int,
                    event_ids: list[str], defect_types: list[str]) -> None:
        if not self.inspection_enabled:
            return
        tracker = self.part_trackers.get(state.camera_id)
        if tracker is None:
            tracker = self.part_trackers.setdefault(state.camera_id, PartPresenceTracker(
                state.camera_id, on_frames=self._presence_on_frames, off_frames=self._presence_off_frames,
            ))
        if self.trigger is not None:
            completed = self.trigger.frame(state.camera_id, frame, frame_id, station_time(), event_ids, defect_types)
        else:
            present = self._presence_of(state.camera_id)(frame)
            completed = tracker.update(frame, present, station_time(), frame_id, event_ids, defect_types)
        state.last_infer_ts = time.perf_counter()   # this frame made it through the part pipeline
        if completed is not None:
            self._queue_inspection(state, completed)

    def _presence_of(self, camera_id: str) -> PresenceDetector:
        detector = self.presence.get(camera_id)
        return detector if detector is not None else self.presence.setdefault(camera_id, PresenceDetector())

    def _queue_inspection(self, state: CameraWorkerState, part: CompletedPart) -> None:
        size = part.best_frame.nbytes
        with self._backlog_lock:
            fits = not self._backlog_bytes or self._backlog_bytes + size <= BACKLOG_BYTES
        if not fits or not self._inspection_slots.acquire(blocking=False):
            logger.error("[%s] Inspection backlog full: part recorded as not inspected.", state.camera_id)
            state.parts_not_inspected += 1
            self._view_faults[state.camera_id] = True
            self._report_view(state, part.trigger_key, PLC_FLAG_NOT_INSPECTED)
            return
        with self._backlog_lock:
            self._backlog_bytes += size
            self._pending_parts.append((state, part))
        try:
            self.inspection_executor.submit(self._inspect_pending)
        except RuntimeError:  # shutting down
            with self._backlog_lock:
                queued = (state, part) in self._pending_parts
                if queued:
                    self._pending_parts.remove((state, part))
                    self._backlog_bytes -= size
            if queued:
                self._inspection_slots.release()
                state.parts_not_inspected += 1
                self._report_view(state, self._part_key(state.camera_id, part.ts, part.trigger_key)[2],
                                  PLC_FLAG_NOT_INSPECTED)

    def _inspect_pending(self) -> None:
        """Inspect what is waiting: views already queued go through the model together.

        Nothing is held back to fill a batch, and a camera appears at most once in one, because
        linking its next part needs this one's record.
        """
        batch: list[tuple[CameraWorkerState, CompletedPart]] = []
        with self._backlog_lock:
            cameras: set[str] = set()
            while self._pending_parts and len(batch) < INSPECT_BATCH:
                if self._pending_parts[0][0].camera_id in cameras:
                    break
                batch.append(self._pending_parts.popleft())
                cameras.add(batch[-1][0].camera_id)
        if not batch:
            return
        prepared = self._judge_together(batch) if len(batch) > 1 else {}
        for i, (state, part) in enumerate(batch):
            try:
                self._inspect_queued(state, part, *([prepared[i]] if i in prepared else []))
            except Exception as e:      # one view must not cost the others in the batch
                logger.error("[%s] Part inspection failed: %s", state.camera_id, e, exc_info=True)
            finally:
                with self._backlog_lock:
                    self._backlog_bytes -= part.best_frame.nbytes
        with self._backlog_lock:    # a camera that came up twice in this batch is still waiting
            waiting = bool(self._pending_parts)
        if waiting:
            try:
                self.inspection_executor.submit(self._inspect_pending)
            except RuntimeError:
                self._inspect_pending()

    def _judge_together(self, batch) -> dict[int, Any]:
        """Anomaly results of several views in one pass; any problem leaves them to be judged singly."""
        from src.inference.anomaly.base import predict_many

        self._inspecting_since = time.monotonic()
        try:
            jobs, index = [], []
            for i, (state, part) in enumerate(batch):
                if self.recipe_from_scalar:      # the product is known only once the part is linked
                    return {}
                recipe = self._model_recipe(state.camera_id, self._recipe_for(state.camera_id, None))
                stack, _ = self.stack_provider.get(recipe)
                detector = None if stack.empty else stack.anomaly
                if detector is None:
                    continue
                if image_fault(part.best_frame, (detector.calibration or {}).get("image_quality")):
                    continue
                jobs.append((detector, part.best_frame))
                index.append((i, detector))
            if len(jobs) < 2:
                return {}
            return {i: (detector, result) for (i, detector), result in zip(index, predict_many(jobs), strict=True)}
        except Exception as e:
            logger.error("[Inspection] Views could not be judged together: %s", e, exc_info=True)
            return {}

    def _part_key(self, camera_id: str, ts: float, trigger_key: str | None = None):
        """(process service if this camera is linked, alignment, key of the physical part).

        A trigger proves which views are the same part, so it stands in where no cycle is known.
        """
        service, alignment, key = self._cycle_key(camera_id, ts)
        if trigger_key is not None and key is not None and "#" not in key:
            self._trigger_numbers[key] = self._trigger_numbers.get(trigger_key, 0)
        return service, alignment, key or trigger_key

    def _cycle_key(self, camera_id: str, ts: float):
        service = self.process_service
        link = service.aligner.link_for(camera_id) if service is not None else None
        if link is None:
            return None, None, None   # camera not linked to a machine: no process label to record
        alignment = service.aligner.align_view(camera_id, ts)
        key = alignment.cycle_id if alignment else None
        if key is not None and link.parts_per_cycle > 1:
            key = f"{key}#{alignment.cavity}"   # each cavity is its own part
        return service, alignment, key

    def _inspect_queued(self, state: CameraWorkerState, part: CompletedPart, *judged) -> None:
        self._inspecting_since = time.monotonic()
        try:
            self._inspect_part(state, part, *judged)
        finally:
            self._inspecting_since = None
            self._inspection_slots.release()

    def _inspect_part(self, state: CameraWorkerState, part: CompletedPart, batched=None) -> None:
        """One view's result for one physical part, recorded against its producing cycle.

        Decision stack (per recipe): good-parts-only anomaly model (catches anything unusual)
        + supervised segmenter (names and delineates known defect types) + frame-level
        detections seen while the part was in view. Any of them failing the part fails it.
        A shadow stack, when registered, judges the same part without affecting the verdict.
        """
        part_key = None
        reported = False
        try:
            frame = part.best_frame
            service, alignment, part_key = self._part_key(state.camera_id, part.ts, part.trigger_key)
            cycle_id = alignment.cycle_id if alignment else None
            recipe = self._model_recipe(state.camera_id, self._recipe_for(state.camera_id, cycle_id))
            stack, shadow = self.stack_provider.get(recipe)
            judged, fault = self._judge_view(stack, part, batched)
            failed = judged.failed or bool(part.defect_event_ids)
            event_id = part.defect_event_ids[0] if part.defect_event_ids else None
            if failed and event_id is None:
                event_id = self._log_part_defect(state, part, stack, judged, cycle_id)

            # No loaded model means nobody looked at this part, which is not a pass.
            not_inspected = (stack.empty or bool(fault) or self.frame_detector_missing) and not failed
            verdict = "not_inspected" if not_inspected else ("fail" if failed else "pass")
            self._view_faults[state.camera_id] = not_inspected
            flag = (self._part_severity(judged.anomaly, judged.top, judged.confidence) if failed
                    else PLC_FLAG_NOT_INSPECTED if not_inspected else PLC_PASS)
            reported = True    # anything failing from here on must not change this view's result
            self._report_view(state, part_key, flag)
            self._save_view(state.camera_id, part_key, frame, stack, judged.anomaly)
            self._note_not_inspected(state, verdict, fault, recipe)
            image_sha = self._image_sha(frame, shadow)
            shadow_record = self._run_shadow(state, part, stack, shadow, judged, verdict, cycle_id, image_sha, recipe)
            self._update_model_health(state.camera_id, recipe, stack, judged.normalized_score, failed)
            self._in_background(lambda: self._keep_images(
                state, part, frame, recipe, stack, judged, verdict, not_inspected, cycle_id, event_id, image_sha))
            self._audit_view(state, part, stack, judged, verdict, fault, recipe, cycle_id, event_id, image_sha,
                             shadow_record)
            self._record_view(state, part, service, alignment, judged, verdict, event_id)
        except Exception as e:
            logger.error("[%s] Part inspection failed: %s", state.camera_id, e, exc_info=True)
            if not reported:
                state.parts_not_inspected += 1
                self._view_faults[state.camera_id] = True
                self._report_view(state, part_key, PLC_FLAG_NOT_INSPECTED)

    @staticmethod
    def _judge_view(stack, part: CompletedPart, batched) -> tuple[PartJudgement, str | None]:
        baseline = (getattr(stack.anomaly, "calibration", None) or {}).get("image_quality")
        fault = image_fault(part.best_frame, baseline)
        if fault:
            return PartJudgement(False, None, None, None), fault
        # A batched result counts only for the very model that judges this part now.
        ready = {"anomaly": batched[1]} if batched is not None and batched[0] is stack.anomaly else {}
        return stack.judge(part.best_frame, frame_defect_types=list(part.defect_types or []), **ready), None

    def _log_part_defect(self, state: CameraWorkerState, part: CompletedPart, stack, judged: PartJudgement,
                         cycle_id: str | None) -> str | None:
        """No frame-level event exists for this part: log one with part-level evidence."""
        frame, anomaly, top = part.best_frame, judged.anomaly, judged.top
        event_id = EventLogger.new_event_id()
        overlay = None
        if top is not None and top.mask is not None:
            overlay = frame.copy()
            overlay[top.mask] = (0.5 * overlay[top.mask] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.rectangle(overlay, top.bbox[:2], top.bbox[2:], (0, 0, 255), 2)
        elif anomaly is not None and anomaly.is_anomalous:
            overlay = stack.anomaly.heatmap_overlay(frame, anomaly, stack.anomaly.threshold)
        h, w = frame.shape[:2]
        logged = self._record_new_defect(
            state, frame, part.best_frame_id, event_id, float(judged.confidence or 0.5), judged.bbox or (0, 0, w, h),
            self._telemetry_context(state.camera_id, part.ts, cycle_id), defect_type=judged.defect_type or "anomaly",
            severity=self._part_severity(anomaly, top, judged.confidence), overlay=overlay)
        return event_id if logged else None

    def _note_not_inspected(self, state: CameraWorkerState, verdict: str, fault: str | None, recipe: str) -> None:
        if verdict == "not_inspected" and not state.not_inspected_logged:
            state.not_inspected_logged = True
            logger.error("[%s] Parts are recorded as not_inspected: %s", state.camera_id,
                         fault or f"no inspection model is loaded for recipe '{recipe}'")
        elif verdict == "pass":
            state.not_inspected_logged = False

    def _image_sha(self, frame: np.ndarray, shadow) -> str | None:
        if (self.audit_log is None and shadow is None and self.review_queue is None
                and not self._pool_cfg.get("enabled", True)):
            return None
        from src.inference.audit_log import image_sha256

        return image_sha256(frame)

    def _run_shadow(self, state, part, stack, shadow, judged, verdict, cycle_id, image_sha, recipe
                    ) -> dict[str, Any] | None:
        if shadow is None or shadow.empty:
            return None
        try:
            other = shadow.judge(part.best_frame, frame_defect_types=list(part.defect_types or []))
            record = {
                "ts": part.ts, "camera_id": state.camera_id, "cycle_id": cycle_id, "image_sha": image_sha,
                "active_version": stack.versions.get("anomaly", {}).get("version"),
                "shadow_version": shadow.versions.get("anomaly", {}).get("version"),
                "active_verdict": verdict, "shadow_verdict": "fail" if (other.failed or part.defect_event_ids) else "pass",
                "active_score": judged.normalized_score, "shadow_score": other.normalized_score,
                "active_type": judged.defect_type, "shadow_type": other.defect_type,
            }
            if self.stack_provider.registry is not None:
                self.stack_provider.registry.log_shadow(recipe, "anomaly", record)
            return record
        except Exception as e:
            logger.warning("[%s] Shadow model failed on a part: %s", state.camera_id, e)
            return None

    def _audit_view(self, state, part, stack, judged, verdict, fault, recipe, cycle_id, event_id, image_sha,
                    shadow_record) -> None:
        if self.audit_log is None:
            return
        self.audit_log.append({
            "ts": part.ts, "camera_id": state.camera_id, "recipe": recipe, "cycle_id": cycle_id,
            "image_fault": fault, "verdict": verdict, "defect_type": judged.defect_type if verdict == "fail" else None,
            "normalized_score": judged.normalized_score,
            "raw_score": judged.anomaly.score if judged.anomaly is not None else None,
            "threshold": stack.anomaly.threshold if stack.anomaly is not None else None,
            "models": stack.versions, "image_sha256": image_sha, "event_id": event_id,
            "frame_event_ids": list(part.defect_event_ids or []),
            "shadow": None if shadow_record is None else {
                "version": shadow_record["shadow_version"], "verdict": shadow_record["shadow_verdict"],
                "score": shadow_record["shadow_score"]},
        })

    def _record_view(self, state, part, service, alignment, judged, verdict, event_id) -> None:
        if verdict == "not_inspected":
            state.parts_not_inspected += 1
            if service is not None:
                service.record_inspection(state.camera_id, part.ts, NOT_INSPECTED, alignment=alignment)
            return
        if service is not None:
            service.record_inspection(
                state.camera_id, part.ts, verdict, defect_type=judged.defect_type, anomaly_score=judged.normalized_score,
                event_id=event_id, alignment=alignment, position=part_pose(part.best_frame))
        state.parts_inspected += 1
        if verdict == "fail":
            state.parts_failed += 1

    def _in_background(self, job) -> None:
        """Write images off the inspection thread; when the disk falls behind, they are skipped."""
        executor = getattr(self, "evidence_executor", None)
        if executor is None:
            try:
                job()
            except Exception as e:
                logger.error("[Inspection] Images not kept: %s", e, exc_info=True)
            return
        with self._backlog_lock:
            if self._evidence_pending >= EVIDENCE_PENDING:
                logger.warning("[Inspection] Review and training images skipped: the disk is too slow.")
                return
            self._evidence_pending += 1

        def run():
            try:
                job()
            except Exception as e:
                logger.error("[Inspection] Images not kept: %s", e, exc_info=True)
            finally:
                with self._backlog_lock:
                    self._evidence_pending -= 1
        try:
            executor.submit(run)
        except RuntimeError:    # shutting down
            run()

    def _keep_images(self, state, part, frame, recipe, stack, judged, verdict, not_inspected, cycle_id, event_id,
                     image_sha) -> None:
        """Training, calibration and review images of one judged part, while the disk has room."""
        monitor = getattr(self, "disk_monitor", None)
        if monitor is not None and not monitor.allow_defect_save():
            return
        anomaly, normalized, failed = judged.anomaly, judged.normalized_score, verdict == "fail"
        pool = self._good_pool(recipe)
        # An unbiased sample of parts, whatever they scored, so a retrained model's threshold is
        # not calibrated on easy parts only. One the pool cannot vouch for needs a person.
        sampled = False
        if pool is not None and not not_inspected and not part.defect_event_ids:
            try:
                sampled = pool.sample(frame, verdict=verdict, normalized_score=normalized,
                                      camera_id=state.camera_id, image_sha=image_sha) == "needs_review"
            except OSError as e:
                logger.error("[%s] Calibration sample not stored: %s", state.camera_id, e)
        if self.review_queue is not None and (sampled or self.review_queue.should_review(normalized)):
            self.review_queue.add(
                state.camera_id, frame, normalized_score=normalized, verdict=verdict,
                cycle_id=cycle_id, event_id=event_id,
                overlay=(stack.anomaly.heatmap_overlay(frame, anomaly, stack.anomaly.threshold)
                         if anomaly is not None else None),
                bbox=judged.bbox if failed else (anomaly.bbox if anomaly is not None else None),
                predicted_type=judged.defect_type, image_sha=image_sha, recipe=recipe,
            )
        elif verdict == "pass" and pool is not None:
            try:
                pool.offer(frame, verdict=verdict, normalized_score=normalized, camera_id=state.camera_id,
                           image_sha=image_sha)
            except OSError as e:
                logger.error("[%s] Good-pool image not stored: %s", state.camera_id, e)

    def _save_view(self, camera_id: str, part_key: str | None, frame: np.ndarray, stack, anomaly) -> None:
        store = getattr(self, "view_store", None)
        if store is None or part_key is None:
            return
        try:
            heat = (stack.anomaly.heatmap_overlay(frame, anomaly, stack.anomaly.threshold)
                    if anomaly is not None and anomaly.is_anomalous else None)
            store.save(part_key, camera_id, frame, heat)
        except Exception as e:
            logger.warning("[%s] View of part %s not stored: %s", camera_id, part_key, e)

    @staticmethod
    def _part_severity(anomaly, top, confidence: float | None) -> int:
        if anomaly is not None and anomaly.is_anomalous and top is None:
            return 3 if anomaly.normalized_score >= 2.0 else (2 if anomaly.normalized_score >= 1.3 else 1)
        return severity_from_conf(confidence or 0.5)

    def _update_model_health(self, camera_id: str, recipe: str, stack, normalized: float | None, failed: bool) -> None:
        det = stack.anomaly
        version = (stack.versions.get("anomaly") or {}).get("digest")
        key = (camera_id, recipe, version)   # a new model version starts a fresh health baseline
        monitor = self.model_health.get(key)
        if monitor is None:
            from src.inference.anomaly.monitoring import ModelHealthMonitor

            monitor = self.model_health.setdefault(key, ModelHealthMonitor(
                calibration=getattr(det, "calibration", None), threshold=getattr(det, "threshold", None),
                window=int(self._health_cfg.get("window", 300)),
                drift_ratio=float(self._health_cfg.get("drift_ratio", 1.5)),
                expected_reject_rate=float(self._health_cfg.get("expected_reject_rate", 0.05)),
            ))
        status = monitor.update(normalized, failed)
        status["recipe"] = recipe
        status["models"] = {r: v.get("version") for r, v in stack.versions.items()}
        self._health_status = getattr(self, "_health_status", {})
        self._health_status[camera_id] = status
        registry = self.stack_provider.registry
        if (self.retrain_on_drift and registry is not None and det is not None
                and status.get("score_drift") and status["score_drift"] > monitor.drift_ratio):
            registry.request_retrain(recipe, camera_id=camera_id, since_ts=time.time(),
                                     reason=f"score drift {status['score_drift']:.2f}x on {camera_id}")

    def _drop_visit(self, state: CameraWorkerState) -> None:
        """A part cut off by a restart was not inspected, and must not be joined to the next one."""
        tracker = self.part_trackers.pop(state.camera_id, None)
        if tracker is not None and tracker.part_in_view:
            state.parts_not_inspected += 1
            self._report_view(state, self._part_key(state.camera_id, station_time())[2], PLC_FLAG_NOT_INSPECTED)

