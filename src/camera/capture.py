"""Reading frames from cameras, and handing them on."""
import logging
import queue
import time
from typing import Any

import cv2
import numpy as np

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

from src.core.factory_manager import FactoryManager
from src.inference.postprocess import severity_from_conf
from src.logging.schema import SystemState
from src.system.station_clock import station_time

from src.camera.worker import FramePacket, FrameSlot, pool_slots, CameraWorkerState, _SHEDDING_STATES

logger = logging.getLogger("src.camera.camera_manager")


class CaptureMixin:
    def _open_camera(self, source: Any, width: int = 640, height: int = 480, fps: float = 30.0):
        network = isinstance(source, str) and "://" in source
        if network:
            # Without these a stream that stops answering holds the thread for ever.
            timeout_ms = int(1000 * self.stream_timeout_s)
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                                                           cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])
        else:
            cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            return (cap, "OpenCV-HW" if str(source).startswith("rtsp://") else "OpenCV-SW")
        cap.release()
        return (None, "OpenCV-SW")

    def _interruptible_sleep(self, seconds: float, state: CameraWorkerState) -> bool:
        """Interruptible zero-CPU sleep that prevents framework pacing distortions."""
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            if self.shutdown_event.is_set() or state.stop_event.is_set():
                return False
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                break
            state.stop_event.wait(min(0.1, remaining))
        return not (self.shutdown_event.is_set() or state.stop_event.is_set())

    def _current_policy_state(self) -> SystemState:
        with self._policy_lock:
            sampled_at, policy = self._policy_cache
            now = time.monotonic()
            if now - sampled_at >= self._policy_cache_ttl:
                try:
                    policy = self.governor.get_policy_state()
                except Exception as e:
                    logger.debug("[CameraManager] Governor evaluation failed: %s", e)
                self._policy_cache = (now, policy)
            return policy

    def _should_shed_load(self, state: CameraWorkerState) -> bool:
        """True while the resource governor reports CRITICAL/EMERGENCY pressure."""
        if not self.governor:
            return False
        policy = self._current_policy_state()
        if policy not in _SHEDDING_STATES:
            return False
        current_ts = time.time()
        if current_ts - state.last_throttle_log_ts > 1.0:
            logger.warning("[%s] Governor %s: resources saturated. Aggressive shedding active.", state.camera_id, policy.value)
            state.last_throttle_log_ts = current_ts
        return True

    def _capture_loop(self, state: CameraWorkerState, camera_cfg: dict):
        width = int(self._camera_setting(camera_cfg, "width", 640))
        height = int(self._camera_setting(camera_cfg, "height", 480))
        fps = float(self._camera_setting(camera_cfg, "fps", 20))
        reconnect_delay = float(self._camera_setting(camera_cfg, "reconnect_delay", 5.0))
        save_every = int(self._camera_setting(camera_cfg, "save_every", 0))

        is_synthetic = isinstance(state.source, str) and state.source.startswith("synthetic://")
        mock_inference_cfg = bool(camera_cfg.get("mock_inference", False))

        # Pre-allocate the mutable ring buffer managed within CameraWorkerState
        state.pool = [FrameSlot(height, width) for _ in range(pool_slots(height, width))]
        state.pool_write_index = 0

        cap = None
        if not is_synthetic:
            cap, state.ingest_mode = self._open_camera(state.source, width=width, height=height, fps=fps)
        else:
            state.ingest_mode = "Synthetic"

        frame_interval = 1.0 / max(1.0, fps)
        frame_id = 0

        try:
            while not self.shutdown_event.is_set() and not state.stop_event.is_set():

                if self._should_shed_load(state):
                    if cap is not None and not is_synthetic:
                        cap.grab()  # keep the RTSP buffer drained while shedding
                    state.last_heartbeat_ts = time.perf_counter()
                    self._interruptible_sleep(0.1, state)
                    continue

                loop_start = time.perf_counter()

                if is_synthetic:
                    synthetic_fail_after = camera_cfg.get("synthetic_fail_after_frames")
                    if synthetic_fail_after is not None and state.restarts == 0 and frame_id >= int(synthetic_fail_after):
                        raise RuntimeError("Synthetic capture failure injected")
                    slot = state.pool[state.pool_write_index % len(state.pool)]
                    part_feed = self._part_feeds.get(state.camera_id)
                    if part_feed is not None:
                        rendered, _ = part_feed.render(station_time(), width, height)
                        np.copyto(slot.frame, rendered)
                        frame = slot.frame
                        state.pool_write_index += 1
                        frame_id += 1
                        state.frames_captured = frame_id
                        state.last_capture_ts = time.perf_counter()
                        state.last_heartbeat_ts = state.last_capture_ts
                        self._dispatch_frame(state, camera_cfg, frame, frame_id, save_every, is_synthetic=True,
                                             mock=mock_inference_cfg or state.force_mock_inference)
                        elapsed = time.perf_counter() - loop_start
                        if elapsed < frame_interval and not self._interruptible_sleep(frame_interval - elapsed, state):
                            break
                        continue
                    slot.frame[:] = (40, 40, 40)

                    # Render dynamic synthetic part geometry with high-contrast borders and text overlay
                    cx = int((320 + (frame_id * 8)) % width)
                    cy = int((240 + (frame_id * 5)) % height)
                    top_left = (max(0, cx - 60), max(0, cy - 40))
                    bottom_right = (min(width - 1, cx + 60), min(height - 1, cy + 40))
                    cv2.rectangle(slot.frame, top_left, bottom_right, (180, 180, 180), -1)
                    cv2.rectangle(slot.frame, top_left, bottom_right, (255, 255, 255), 2)
                    cv2.putText(slot.frame, f"{state.camera_id} - FRAME {frame_id:04d}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    frame = slot.frame
                    state.pool_write_index += 1
                else:
                    if cap is None:
                        if not self._interruptible_sleep(reconnect_delay, state):
                            break
                        cap, state.ingest_mode = self._open_camera(state.source, width=width, height=height, fps=fps)
                        if cap is None:
                            continue
                        state.reconnect_count += 1

                    if str(state.source).startswith("rtsp://") and state.last_capture_ts > 0:
                        elapsed_since_last = time.perf_counter() - state.last_capture_ts
                        if elapsed_since_last > (frame_interval * 1.5):
                            # Skip stale buffered frames so inference sees the live scene.
                            for _ in range(2):
                                cap.grab()

                    grabbed = cap.grab()

                    frame = None
                    if grabbed:
                        slot = state.pool[state.pool_write_index % len(state.pool)]
                        ret, decoded_img = cap.retrieve(image=slot.frame)
                        if ret and decoded_img is not None:
                            if decoded_img is not slot.frame:
                                # The stream resolution differs from the configured one; re-size the
                                # ring buffer once so subsequent decodes land in pre-allocated memory.
                                h, w = decoded_img.shape[:2]
                                logger.warning(
                                    "[%s] Stream delivers %dx%d (configured %dx%d). Re-allocating frame pool.",
                                    state.camera_id, w, h, width, height,
                                )
                                width, height = w, h
                                state.pool = [FrameSlot(h, w) for _ in range(pool_slots(h, w))]
                                state.pool_write_index = 0
                                slot = state.pool[0]
                                np.copyto(slot.frame, decoded_img)
                            frame = slot.frame
                            state.pool_write_index += 1

                    if frame is None:
                        cap.release()
                        cap = None
                        if not self._interruptible_sleep(reconnect_delay, state):
                            break
                        continue

                frame_id += 1
                state.frames_captured = frame_id
                state.last_capture_ts = time.perf_counter()
                state.last_heartbeat_ts = state.last_capture_ts

                self._dispatch_frame(state, camera_cfg, frame, frame_id, save_every, is_synthetic=is_synthetic,
                                     mock=mock_inference_cfg or state.force_mock_inference)

                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_interval:
                    if not self._interruptible_sleep(frame_interval - elapsed, state):
                        break
        except BaseException as e:
            state.failed = True
            state.failure_reason = f"{type(e).__name__}: {str(e)}"
            logger.error("[%s] Capture loop exception: %s", state.camera_id, state.failure_reason, exc_info=True)
            state.stop_event.set()
        finally:
            logger.info("[%s] Capture loop exited. shutdown=%s, stop=%s", state.camera_id, self.shutdown_event.is_set(), state.stop_event.is_set())
            if cap is not None:
                cap.release()

    def _dispatch_frame(self, state: CameraWorkerState, camera_cfg: dict, frame: np.ndarray, frame_id: int,
                        save_every: int, *, is_synthetic: bool, mock: bool) -> None:
        """Route a captured frame to live batched inference, or to the mock-inference worker."""
        if not mock:
            self._maybe_save_raw_frame(state.camera_id, frame, frame_id, save_every)
            new_events: tuple[list[str], list[str]] = ([], [])
            try:
                multiplexer = FactoryManager().get_instance("InferenceMultiplexer")
                result = None
                if multiplexer:
                    try:
                        request_id = multiplexer.submit_frame(state.camera_id, frame, block=False, timeout=None)
                        result = multiplexer.retrieve_result(request_id, timeout_sec=self.result_timeout_sec)
                    except queue.Full:
                        state.backpressure_warnings += 1
                        result = None

                if result is None:
                    if not is_synthetic:
                        state.dropped_frames += 1
                else:
                    boxes, _masks, xyxy, conf = result
                    boxes, xyxy, conf = self._filter_defect_detections(boxes, xyxy, conf)
                    new_events = self._handle_inference_result(
                        state, frame, frame_id, boxes, xyxy, conf,
                        defect_type="anomaly",
                        severity_fn=severity_from_conf,
                    )
            except Exception as e:
                logger.error("[%s] Multiplexer handoff failed: %s", state.camera_id, e, exc_info=True)
            self._track_part(state, frame, frame_id, *new_events)
            return

        packet = FramePacket(
            camera_id=state.camera_id,
            session_id=state.session_id,
            frame_id=frame_id,
            timestamp_ms=int(time.time() * 1000),
            frame=frame.copy(),
            full_image_path="",
        )
        try:
            state.frame_queue.put_nowait(packet)
            state.queue_overflow_streak = 0
        except queue.Full:
            state.queue_overflow_streak += 1
            if not is_synthetic:
                state.dropped_frames += 1

    def _mock_inference_loop(self, state: CameraWorkerState, camera_cfg: dict):
        # Process-driven synthetic scenes render real defects; injected mock boxes would
        # contaminate the labels, so they are only used for plain synthetic streams.
        process_scene = state.camera_id in self._part_feeds
        mock_event_every = max(1, int(camera_cfg.get("mock_event_every", 10) or 10))
        save_every = int(self._camera_setting(camera_cfg, "save_every", 0))
        manager = FactoryManager()

        try:
            while not (self.shutdown_event.is_set() or state.stop_event.is_set()):
                try:
                    packet: FramePacket = state.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                self._maybe_save_raw_frame(state.camera_id, packet.frame, packet.frame_id, save_every)

                multiplexer = manager.get_instance("InferenceMultiplexer")
                if not multiplexer:
                    self._interruptible_sleep(0.01, state)
                    continue

                try:
                    request_id = multiplexer.submit_frame(state.camera_id, packet.frame)
                    result = multiplexer.retrieve_result(request_id, timeout_sec=1.0)
                except queue.Full:
                    state.backpressure_warnings += 1
                    result = None

                if result is None:
                    is_synthetic = isinstance(state.source, str) and state.source.startswith("synthetic://")
                    if not is_synthetic:
                        state.dropped_frames += 1
                    self._track_part(state, packet.frame, packet.frame_id, [], [])
                    self._interruptible_sleep(0.01, state)
                    continue

                boxes, _masks, xyxy, conf = result
                boxes, xyxy, conf = self._filter_defect_detections(boxes, xyxy, conf)
                if not process_scene and len(boxes) == 0 and packet.frame_id % mock_event_every == 0:
                    offset = (packet.frame_id % 10) * 100.0
                    x1, y1, x2, y2 = 10.0 + offset, 10.0, 90.0 + offset, 90.0
                    boxes = np.array([[x1, y1, x2, y2, 0.99, 0.0]], dtype=np.float32)
                    xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                    conf = np.array([0.99], dtype=np.float32)

                new_events = self._handle_inference_result(
                    state, packet.frame, packet.frame_id, boxes, xyxy, conf,
                    defect_type="mock_defect",
                    severity_fn=lambda _c: 3,
                )
                self._track_part(state, packet.frame, packet.frame_id, *new_events)
        except Exception as e:
            state.failed = True
            state.failure_reason = f"{type(e).__name__}: {str(e)}"
            logger.error("[%s] Mock inference loop exception: %s", state.camera_id, state.failure_reason, exc_info=True)
            state.stop_event.set()

