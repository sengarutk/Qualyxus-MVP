"""From the views' results to one verdict per part at the PLC, and the heartbeat that says it is trusted."""
import asyncio
import logging
import queue
import time
from typing import Any


try:
    import psutil as _psutil
except ImportError:
    _psutil = None

from src.process.models import NOT_INSPECTED, InspectionRecord
from src.core.factory_manager import FactoryManager
from src.system.station_clock import station_time
from src.telemetry.plc_adapter import PLC_FLAG_NOT_INSPECTED

from src.camera.worker import CameraWorkerState, VERDICT_LOST_HOLD_S, UNSEEN_CHECK_S, UNSEEN_MARGIN_S

logger = logging.getLogger("src.camera.camera_manager")


class VerdictMixin:
    def _report_view(self, state: CameraWorkerState, part_key: str | None, flag: int) -> None:
        verdict = self.part_fuser.add(part_key, state.camera_id, int(flag), time.monotonic())
        if verdict is not None:
            self._publish_part_verdict(verdict)

    @staticmethod
    def _part_number(cycle_id: str | None) -> tuple[int, int | None]:
        """(machine cycle counter, cavity) of a part; 0 when no machine cycle is known.

        An invented number could be another real part's, so an unmatched part gets none.
        """
        cycle, _, cavity = str(cycle_id or "").partition("#")
        tail = cycle.rsplit("-", 1)[-1]
        return (int(tail) & 0xFFFF if tail.isdigit() else 0), (int(cavity) if cavity.isdigit() else None)

    def _publish_part_verdict(self, verdict: dict[str, Any]) -> None:
        """One physical part's final verdict: queued for the PLC and kept in the audit trail."""
        cycle_id = verdict["cycle_id"]
        # A view that stopped reporting is unhealthy too, even if another view failed the part.
        # An update is about an older part and says nothing about the station now.
        if not verdict.get("update"):
            self._inspection_faulted = verdict["verdict"] == "not_inspected" or bool(verdict["missing"])
        part, cavity = self._part_number(cycle_id)
        if cycle_id in getattr(self, "_trigger_numbers", {}):
            part = self._trigger_numbers[cycle_id] & 0xFFFF     # the PLC's own count of the part it triggered on
        signal = {**verdict, "type": "PART_VERDICT", "part": part, "cavity": cavity, "timestamp": time.time(),
                  "published_mono": time.monotonic()}
        if verdict["verdict"] != "pass" and verdict["missing"]:
            logger.warning("[Inspection] Part %s %s: no result from %s", part, verdict["verdict"], verdict["missing"])
        self._last_verdict = {k: signal[k] for k in ("part", "cycle_id", "verdict", "flag", "timestamp")}
        # The PLC comes first: a broken log or a full disk must never cost a verdict.
        try:
            self.fast_queue.put_nowait(signal)
        except queue.Full:
            self._verdict_lost(signal)
        if self.audit_log is not None:
            try:
                self.audit_log.append({"ts": signal["timestamp"], "record": "part_verdict", "part": signal["part"],
                                       "cycle_id": cycle_id, "verdict": verdict["verdict"], "flag": verdict["flag"],
                                       "views": verdict["views"], "missing_views": verdict["missing"],
                                       "update": bool(verdict.get("update")),
                                       "shutdown": self.shutdown_event.is_set()})
            except Exception as e:
                logger.error("[Inspection] Audit record for part %s not written: %s", part, e)

    def _deliver_verdict(self, adapter, signal: dict[str, Any]) -> None:
        """Write one verdict; a verdict that never arrived must show up as a stopped heartbeat."""
        # By then the part has left the station: a late verdict would land on another part.
        deadline = signal.get("published_mono", time.monotonic()) + self.max_verdict_age_s
        try:
            delivered = time.monotonic() < deadline and adapter.write_part_verdict(signal, deadline)
        except Exception as e:
            logger.error("[Inspection] Verdict for part %s not written: %s", signal.get("part"), e)
            delivered = False
        if delivered:
            return
        self._verdict_lost(signal)

    def _verdict_lost(self, signal: dict[str, Any]) -> None:
        logger.error("[Inspection] Verdict for part %s did not reach the PLC.", signal.get("part"))
        self._delivery_fault_until = time.monotonic() + VERDICT_LOST_HOLD_S
        if self.audit_log is not None:
            try:
                self.audit_log.append({"ts": time.time(), "record": "verdict_not_delivered", "part": signal.get("part"),
                                       "cycle_id": signal.get("cycle_id"), "verdict": signal.get("verdict"),
                                       "flag": signal.get("flag")})
            except Exception as e:
                logger.error("[Inspection] Audit record for an undelivered verdict not written: %s", e)

    def _silent_views(self) -> list[str]:
        """Views that stopped delivering frames. A station with no cameras running counts too."""
        with self._lock:
            workers = dict(self.workers)
        expected = set(self.part_fuser.required) or set(workers)
        if not expected:
            return ["no cameras running"]
        now = time.perf_counter()
        silent = []
        for camera_id in expected:
            state = workers.get(camera_id)
            newest = min(state.last_capture_ts, state.last_infer_ts) if state is not None else 0.0
            if state is None or state.failed or not newest or now - newest > self.max_frame_age_s:
                silent.append(camera_id)
        return sorted(silent)

    def _heartbeat_hold(self, now: float) -> str | None:
        """Why inspection cannot be called healthy right now, or None."""
        if self.shutdown_event.is_set():
            return "inspection is shutting down"
        if self._inspection_faulted:
            return "the last part was not fully inspected"
        required = self.part_fuser.required
        faulted = sorted(c for c, bad in self._view_faults.items() if bad and (not required or c in required))
        if faulted:
            return f"cannot inspect on {faulted}"
        started = self._inspecting_since
        if started is not None and now - started > self.max_inspect_s:
            return f"a part has been in inspection for over {self.max_inspect_s:.0f} s"
        if now < self._delivery_fault_until:
            return "a part verdict did not reach the PLC"
        source = getattr(self, "trigger_source", None)
        if source is not None and not source.alive():
            return "the PLC trigger cannot be read"
        silent = self._silent_views()
        return f"no frames from {silent}" if silent else None

    def _unseen_parts(self, wall_now: float) -> list[dict[str, Any]]:
        """Parts of finished cycles that should have reached the cameras by now and did not."""
        service = self.process_service
        if service is None or not self.expect_part_per_cycle or wall_now < self._next_unseen_check:
            return []
        self._next_unseen_check = wall_now + UNSEEN_CHECK_S
        machines: dict[str, Any] = {}
        for link in service.aligner.links.values():
            if link.mode == "time":
                machines.setdefault(link.machine_id, link)
        verdicts = []
        for machine_id, link in machines.items():
            until = wall_now - (link.transport_delay_s + link.tolerance_s + self.part_fuser.timeout_s + UNSEEN_MARGIN_S)
            since = self._cycles_checked_to.setdefault(machine_id, wall_now)
            if until <= since:
                continue
            self._cycles_checked_to[machine_id] = until
            keys = []
            for cycle_id, _end_ts in service.store.cycles_between(machine_id, since, until):
                keys += ([f"{cycle_id}#{i}" for i in range(link.parts_per_cycle)]
                         if link.parts_per_cycle > 1 else [cycle_id])
            unseen = self.part_fuser.unseen(keys)
            for verdict in unseen:   # so the dashboard counts them, not only the PLC and audit trail
                cycle_id, _, cavity = verdict["cycle_id"].partition("#")
                try:
                    service.store.save_inspection(InspectionRecord(
                        inspection_id="", camera_id="unseen", ts=wall_now, verdict=NOT_INSPECTED,
                        cycle_id=cycle_id, source="unseen", cavity=int(cavity or 0)))
                except Exception as e:
                    logger.error("[Inspection] Could not record unseen part %s: %s", verdict["cycle_id"], e)
            verdicts += unseen
        return verdicts

    def _on_trigger(self, number: int, lost: int) -> None:
        """The PLC says a part is in position (``lost``: triggers that came faster than they were read)."""
        now = station_time()
        for _ in range(lost):
            self._publish_part_verdict({"cycle_id": None, "verdict": "not_inspected", "flag": PLC_FLAG_NOT_INSPECTED,
                                        "views": {}, "missing": sorted(self.trigger.cameras)})
        key = f"trigger-{number}-{int(now)}"    # the PLC's counter wraps; the key must not repeat
        self._trigger_numbers[key] = number
        while len(self._trigger_numbers) > 1024:
            self._trigger_numbers.popitem(last=False)
        self._report_missed(self.trigger.fire(key, now))

    def _report_missed(self, missed) -> None:
        for camera_id, key, ts in missed:
            with self._lock:
                state = self.workers.get(camera_id)
            logger.error("[%s] No image for triggered part %s.", camera_id, key)
            if state is not None:
                state.parts_not_inspected += 1
                self._report_view(state, self._part_key(camera_id, ts, key)[2], PLC_FLAG_NOT_INSPECTED)

    def _reflex_tick(self, adapter) -> None:
        """Close parts nobody finished, and keep the PLC heartbeat moving."""
        if getattr(self, "trigger", None) is not None:
            self._report_missed(self.trigger.expire(station_time()))
        now = time.monotonic()   # a clock correction must not stall the heartbeat or a part deadline
        for verdict in self.part_fuser.expired(now) + self._unseen_parts(station_time()):
            self._publish_part_verdict(verdict)
        if adapter is None:
            return
        hold = self._heartbeat_hold(now)
        if hold != self._hold_reason:
            self._hold_reason = hold
            if hold:
                logger.warning("[Inspection] PLC heartbeat stopped: %s", hold)
            else:
                logger.info("[Inspection] PLC heartbeat running.")
        busy = self._heartbeat_write is not None and not self._heartbeat_write.done()
        if hold is None and not busy and now - self._last_heartbeat >= self.heartbeat_s:
            self._last_heartbeat = now
            try:
                self._heartbeat_write = self.plc_executor.submit(adapter.write_heartbeat)
            except RuntimeError:
                pass  # shutting down

    def _next_reflex_signal(self) -> dict[str, Any] | None:
        """Verdicts first; a machine alert waits at most one short poll."""
        try:
            return self.fast_queue.get(timeout=0.05)
        except queue.Empty:
            pass
        try:
            return self.alert_queue.get_nowait()
        except queue.Empty:
            return None

    def _reflex_worker(self):
        adapter = getattr(self.telemetry, "adapter", None)
        # Runs until inspection has finished, and then until every verdict is handed over.
        while not (self._reflex_stop.is_set() and self.fast_queue.empty()):
            try:
                self._reflex_tick(adapter)
            except Exception as e:   # this thread carries every verdict: it must outlive any one error
                logger.error("[Reflex] Tick failed: %s", e, exc_info=True)
            signal_data = self._next_reflex_signal()
            if signal_data is None:
                continue
            try:
                if adapter is not None:
                    try:
                        if signal_data.get("type") == "PART_VERDICT":
                            self.plc_executor.submit(self._deliver_verdict, adapter, signal_data)
                        elif self._alert_write is None or self._alert_write.done():
                            self._alert_write = self.plc_executor.submit(adapter.write_advisory_flag, signal_data)
                        else:   # an unreachable PLC must not build a queue in front of the verdicts
                            logger.warning("[Reflex] Machine alert skipped: the PLC has not taken the previous one.")
                    except RuntimeError:
                        pass

                hub_connector = FactoryManager().get_instance("HubConnector")
                loop = self.main_loop
                if hub_connector is not None and hasattr(hub_connector, "bridge") and loop is not None and loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        hub_connector.bridge.publish_telemetry("qualyxus/reflex/anomaly", signal_data),
                        loop,
                    )
            except Exception as e:
                logger.error("[Reflex] Failed to dispatch reflex signal: %s", e)

    def _finish_inspections(self, states: list[CameraWorkerState]) -> None:
        trigger = getattr(self, "trigger", None)
        if trigger is not None:
            self.trigger_source.stop()
            by_camera = {s.camera_id: s for s in states}
            for camera_id, key, ts in trigger.expire(float("inf")):
                if camera_id in by_camera:
                    by_camera[camera_id].parts_not_inspected += 1
                    self._report_view(by_camera[camera_id], self._part_key(camera_id, ts, key)[2], PLC_FLAG_NOT_INSPECTED)
        self.inspection_executor.shutdown(wait=True)
        while getattr(self, "_pending_parts", None):
            self._inspect_pending()
        for state in states:
            tracker = self.part_trackers.get(state.camera_id)
            if trigger is None and tracker is not None and tracker.part_in_view:   # stopped in the middle of a part
                state.parts_not_inspected += 1
                self._report_view(state, self._part_key(state.camera_id, station_time())[2], PLC_FLAG_NOT_INSPECTED)
        for verdict in self.part_fuser.expired(float("inf")):  # their other views will never come
            self._publish_part_verdict(verdict)

    def health_problems(self) -> list[str]:
        """Why this station cannot be called healthy now; empty when it can."""
        with self._lock:
            workers = dict(self.workers)
        problems = [f"camera {c.get('id')} is not running" for c in self._camera_entries() if str(c.get("id")) not in workers]
        hold = self._heartbeat_hold(time.monotonic())
        if hold:
            problems.append(hold)
        if self.disk_monitor.get_policy().level == "critical":
            problems.append("the disk is full")
        return problems

