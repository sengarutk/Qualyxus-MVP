"""Defect events, their evidence images, and the optional geometry, vibration and language-model checks."""
import asyncio
import copy
import dataclasses
import logging
import queue
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

from src.camera.presence import is_part_present
from src.core.factory_manager import FactoryManager
from src.logging.event_logger import EventLogger
from src.logging.schema import DefectEvent

from src.camera.worker import CameraWorkerState

logger = logging.getLogger("src.camera.camera_manager")


class DefectMixin:
    def enqueue_disk_write(self, target_path: Path | str, image: np.ndarray) -> bool:
        """Offloads cv2.imwrite calls to a background queue. Returns False when the write was shed."""
        try:
            self.disk_write_queue.put_nowait((Path(target_path), image.copy()))
            return True
        except queue.Full:
            return False
        except Exception as e:
            logger.error("[DiskWriter] Failed to enqueue %s: %s", target_path, e)
            return False

    def _disk_writer_loop(self):
        """Dedicated background thread for non-blocking disk persistence."""
        while not self.shutdown_event.is_set() or not self.disk_write_queue.empty():
            try:
                target_path, image = self.disk_write_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target_path), image):
                    logger.error("[DiskWriter] cv2.imwrite returned False for %s", target_path)
            except Exception as e:
                logger.error("[DiskWriter] Non-blocking imwrite failed for %s: %s", target_path, e)

    def _maybe_save_raw_frame(self, camera_id: str, frame: np.ndarray, frame_id: int, save_every: int) -> None:
        """Persist every Nth non-blank raw frame while the disk policy allows raw saves."""
        if save_every <= 0 or frame_id % save_every != 0:
            return
        try:
            if self.disk_monitor.allow_raw_save() and is_part_present(frame):
                self.enqueue_disk_write(self.raw_dir / f"{camera_id}_frame_{frame_id:04d}.png", frame)
        except Exception as disk_err:
            logger.error("[%s] Failed to persist raw frame: %s", camera_id, disk_err)

    def _handle_inference_result(
        self,
        state: CameraWorkerState,
        frame: np.ndarray,
        frame_id: int,
        boxes: np.ndarray,
        xyxy: np.ndarray,
        conf: np.ndarray,
        *,
        defect_type: str,
        severity_fn,
    ) -> tuple[list[str], list[str]]:
        """Track detections, log new defects; returns (new event ids, their defect types)."""
        state.frames_inferred += 1
        state.last_infer_ts = time.perf_counter()

        num_boxes = len(boxes)
        detections = np.empty((num_boxes, 6), dtype=np.float32)
        if num_boxes > 0:
            detections[:, :4] = xyxy
            detections[:, 4] = conf
            detections[:, 5] = boxes[:, 5] if boxes.shape[1] > 5 else 0.0

        tracks = state.tracker.update(detections) if state.tracker is not None else []
        new_ids: list[str] = []
        new_types: list[str] = []

        if num_boxes > 0:
            telemetry_ctx = self._telemetry_context(state.camera_id)
            for idx in range(num_boxes):
                c_conf = float(conf[idx]) if idx < len(conf) else 0.99
                if idx < len(tracks):
                    _track_id, event_id, tx1, ty1, tx2, ty2, is_new = tracks[idx]
                else:
                    is_new = True
                    event_id = EventLogger.new_event_id()
                    tx1, ty1, tx2, ty2 = (float(v) for v in xyxy[idx])
                if is_new:
                    logged = self._record_new_defect(
                        state, frame, frame_id, event_id, c_conf, (tx1, ty1, tx2, ty2),
                        telemetry_ctx, defect_type=defect_type, severity=int(severity_fn(c_conf)),
                    )
                    if logged:
                        new_ids.append(event_id)
                        new_types.append(defect_type)
            self._process_vibration_flow(state, xyxy)

        self._submit_geometry_check(state, frame)
        return new_ids, new_types

    def _record_new_defect(
        self,
        state: CameraWorkerState,
        frame: np.ndarray,
        frame_id: int,
        event_id: str,
        confidence: float,
        bbox: tuple[float, float, float, float],
        telemetry_ctx,
        *,
        defect_type: str,
        severity: int,
        overlay: np.ndarray | None = None,
    ) -> bool:
        # The defect snapshot is the evidence image referenced by the event record.
        evidence_path, overlay_path, crop_path = "", None, None
        h, w = frame.shape[:2]
        x1_c = int(np.clip(bbox[0], 0, w - 1))
        y1_c = int(np.clip(bbox[1], 0, h - 1))
        x2_c = int(np.clip(bbox[2], 0, w))
        y2_c = int(np.clip(bbox[3], 0, h))
        crop_matrix = frame[y1_c:y2_c, x1_c:x2_c].copy()
        if self.disk_monitor.allow_defect_save():
            base = self.defect_dir / state.camera_id / f"{state.camera_id}_defect_{event_id}_frame_{frame_id:04d}"
            if self.enqueue_disk_write(base.with_suffix(".png"), frame):
                evidence_path = str(base.with_suffix(".png").resolve())
            if overlay is not None and self.enqueue_disk_write(Path(f"{base}_overlay.png"), overlay):
                overlay_path = str(Path(f"{base}_overlay.png").resolve())
            if crop_matrix.size and self.enqueue_disk_write(Path(f"{base}_crop.png"), crop_matrix):
                crop_path = str(Path(f"{base}_crop.png").resolve())

        event = DefectEvent(
            event_id=event_id,
            timestamp=EventLogger.now(),
            source=state.camera_id,
            defect_type=defect_type,
            confidence=confidence,
            severity=severity,
            batch_id=None,
            operator_id=None,
            machine_id=telemetry_ctx.machine_id,
            shift_id=telemetry_ctx.shift_id,
            line_id=telemetry_ctx.line_id,
            station_id=telemetry_ctx.station_id,
            temperature=telemetry_ctx.temperature,
            vibration=telemetry_ctx.vibration,
            speed=telemetry_ctx.speed,
            lot_id=telemetry_ctx.lot_id,
            full_image_path=evidence_path,
            overlay_path=overlay_path,
            crop_path=crop_path,
        )

        event_logger = FactoryManager().get_instance("EventLogger")
        if not self._dispatch_event(event_logger, event):
            return False
        state.events_logged += 1
        state.last_log_ts = time.time()
        self._submit_vlm_reasoning(crop_matrix, event_id, event_logger)
        return True

    def _dispatch_event(self, event_logger, event: DefectEvent) -> bool:
        """Hand a DefectEvent to the EventLogger (async batch path when an event loop is running)."""
        if event_logger is None:
            now = time.time()
            if now - self._last_missing_logger_warning > 5.0:
                logger.warning("[CameraManager] EventLogger not registered; defect events are not being persisted.")
                self._last_missing_logger_warning = now
            return False
        try:
            loop = self.main_loop
            if loop is not None and loop.is_running():
                payload = {k: v for k, v in dataclasses.asdict(event).items() if k != "frame_data"}
                asyncio.run_coroutine_threadsafe(event_logger.log_batch_async([payload]), loop)
            else:
                event_logger.log_event(event)
            return True
        except Exception as e:
            logger.error("[%s] Failed to persist defect event %s: %s", event.source, event.event_id, e)
            return False

    def _submit_vlm_reasoning(self, crop_matrix: np.ndarray, event_id: str, event_logger) -> None:
        """Queue a bounded, non-blocking VLM root-cause pass for a freshly logged defect."""
        if not self.vlm_enabled or crop_matrix.size == 0:
            return
        with self._vlm_lock:
            if self._vlm_pending >= self.vlm_max_pending:
                return  # shed: the VLM backend is slower than the defect rate
            self._vlm_pending += 1

        from src.inference.analysis.vlm_gateway import analyze_defect_crop_sync
        from src.utils.profile_manager import ProfileManager

        def _run():
            try:
                expert_ctx = ProfileManager().get_vlm_expert_context()
                analysis_text = analyze_defect_crop_sync(crop_matrix, expert_ctx, self.config)
                if analysis_text:
                    event_logger.append_vlm_narrative(event_id, analysis_text)
            except Exception as e:
                logger.error("[VLM] Reasoning pass failed for %s: %s", event_id, e)
            finally:
                with self._vlm_lock:
                    self._vlm_pending -= 1

        try:
            self.vlm_executor.submit(_run)
        except RuntimeError:
            with self._vlm_lock:
                self._vlm_pending -= 1

    def _process_vibration_flow(self, state: CameraWorkerState, xyxy: np.ndarray) -> None:
        """Feed the centroid displacement of the dominant detection into the vibration DSP."""
        analyzer = self.vibration_analyzers.get(state.camera_id)
        if analyzer is None or len(xyxy) == 0:
            return
        try:
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            best = int(np.argmax(areas))
            cx = float((xyxy[best, 0] + xyxy[best, 2]) * 0.5)
            cy = float((xyxy[best, 1] + xyxy[best, 3]) * 0.5)
            prev_cx, prev_cy = state.prev_centroid if state.prev_centroid is not None else (cx, cy)
            state.prev_centroid = (cx, cy)
            analyzer.ingest(cx - prev_cx, cy - prev_cy)
        except Exception as e:
            logger.error("[%s] Vibration flow update failed: %s", state.camera_id, e)

    def _submit_geometry_check(self, state: CameraWorkerState, frame: np.ndarray) -> None:
        geom_engine = self.geometry_engines.get(state.camera_id)
        # Without a golden template every check is an ALIGNMENT_FAULT; skip the wasted work.
        if geom_engine is None or geom_engine.template is None:
            return
        # Bounded in-flight work: never let geometry checks queue up behind a slow CPU.
        with self._geometry_lock:
            if state.geometry_inflight >= 2:
                return
            state.geometry_inflight += 1

        def _geom_callback(fut):
            with self._geometry_lock:
                state.geometry_inflight -= 1
            try:
                res = fut.result()
                if not res:
                    return
                state.last_geometric_diagnostic = res
                violation = res.get("status") == "GEOMETRIC_VIOLATION"

                now = time.time()
                if violation or (now - state.last_geometry_log_ts) >= self.geometry_ok_log_interval:
                    state.last_geometry_log_ts = now
                    res_payload = copy.deepcopy(res)
                    res_payload["camera_id"] = state.camera_id
                    res_payload["timestamp"] = now
                    ev_logger = FactoryManager().get_instance("EventLogger")
                    if ev_logger:
                        loop = self.main_loop
                        if loop is not None and loop.is_running():
                            asyncio.run_coroutine_threadsafe(ev_logger.log_batch_async([res_payload]), loop)
                        else:
                            ev_logger._sync_execute_transaction([res_payload])

                if violation:
                    try:
                        self.alert_queue.put_nowait({
                            "type": "GEOMETRIC_CRITICAL",
                            "camera_id": state.camera_id,
                            "timestamp": now,
                            "diagnostic": res,
                        })
                    except queue.Full:
                        pass
            except Exception as e:
                logger.error("[%s] Background geometry check crashed: %s", state.camera_id, e)

        try:
            fut = self.executor.submit(geom_engine.inspect_part_geometry, frame.copy(), False)
            fut.add_done_callback(_geom_callback)
        except RuntimeError:
            with self._geometry_lock:
                state.geometry_inflight -= 1  # executor already shut down

    def _telemetry_context(self, camera_id: str, ts: float | None = None, cycle_id: str | None = None):
        """Telemetry of the cycle that produced the part in view (falls back to live values)."""
        ctx = self.telemetry.get_telemetry(camera_id)
        service = self.process_service
        if service is None:
            return ctx
        try:
            if cycle_id is None:
                cycle_id = service.aligner.align(camera_id, ts if ts is not None else time.time()).cycle_id
            overrides = {k: v for k, v in service.telemetry_for(cycle_id).items() if v is not None}
            known = {f.name for f in dataclasses.fields(ctx)}
            return dataclasses.replace(ctx, **{k: v for k, v in overrides.items() if k in known})
        except Exception as e:
            logger.debug("[%s] Cycle telemetry unavailable: %s", camera_id, e)
            return ctx

