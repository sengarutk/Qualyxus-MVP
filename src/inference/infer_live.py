import asyncio
import collections
import logging
import time
import json
import os
import sys
import queue
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np

from src.config.vault import ConfigVault
from src.core.factory_manager import FactoryManager
from src.api.pulse_server import PulseServer, broadcaster
from src.inference.tracker import DefectTracker

logger = logging.getLogger("QualyxusInference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class InferenceMultiplexer:
    """
    Centralized Batch Multiplexer for high-throughput multi-camera edge inference.
    Eliminates VRAM fragmentation, thread stalling, and hot-spinning event loops
    via static tensor allocation, isolated executors, and precise state management.
    """
    def __init__(self, config):
        self.config = config
        self.max_batch_size = 4
        self.timeout_sec = 0.005  # Tight 5.0ms floor latency budget
        self.max_queue_capacity = 60  # Strict VRAM backpressure limit
        
        self.queue = collections.deque()
        self.lock = asyncio.Lock()
        self.frame_ready_event = asyncio.Event()
        self.trackers = {}
        
        # Strong reference tracking registry to protect async tasks from garbage collection
        self._background_tasks = set()
        self.inference_executor = ThreadPoolExecutor(max_workers=1)
        
        logger.info("[Multiplexer] Allocating singular master YOLO segmentation model in VRAM.")
        model_path = os.environ.get("MODEL_ASSET_PATH", "models/best.pt")
        
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.model.to('cuda')
        
        try:
            accel_cfg = self.config.get("hardware_acceleration", default={})
            precision = accel_cfg.get("precision", "fp16")
            dtype = torch.float16 if precision == "fp16" else torch.float32
            
            self.static_batch_tensor = torch.zeros(
                (self.max_batch_size, 3, 640, 640), 
                dtype=dtype, 
                device='cuda'
            )
        except Exception as e:
            logger.critical("[HARDWARE_PANIC] Static tensor allocation failed: %s", e)
            logger.critical("Halting execution immediately to prevent CPU fallback performance degradation.")
            sys.exit(1)
            
        # Batch telemetry has its own file: CAMERA_STATUS_PATH belongs to CameraManager's
        # per-worker status (consumed by the healthcheck and dashboard).
        self.status_file = os.environ.get("INFERENCE_STATUS_PATH", "data/live/inference_status.json")
        self.total_processed = 0

    def enqueue_frame_from_thread(self, camera_id: str, tensor: torch.Tensor, timestamp_perf: float, loop: asyncio.AbstractEventLoop):
        """Thread-safe entrypoint processing pre-allocated GPU workload tensors into the async engine."""
        if not loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.enqueue_frame(camera_id, tensor, timestamp_perf), loop)

    async def enqueue_frame(self, camera_id: str, tensor: torch.Tensor, timestamp_perf: float):
        """Asynchronously processes incoming pre-allocated tensors with double-checked backpressure shields."""
        async with self.lock:
            if len(self.queue) >= self.max_queue_capacity:
                logger.warning("[Backpressure] Ingestion queue full (%d). Shedding frame from camera: %s", 
                               len(self.queue), camera_id)
                del tensor
                return

        async with self.lock:
            if len(self.queue) >= self.max_queue_capacity:
                logger.warning("[Backpressure] Post-Triton overflow trap triggered. Dropping payload.")
                del tensor
                return 
                
            self.queue.append((camera_id, tensor, timestamp_perf))
            self.frame_ready_event.set()

    async def run_batch_loop(self):
        """Event-driven window aggregation loop enforcing strict latency control without busy-waiting spikes."""
        logger.info("[Multiplexer] Non-spinning in-memory window aggregation engine active.")
        while True:
            batch = []
            async with self.lock:
                self.frame_ready_event.clear()
                if self.queue:
                    oldest_timestamp = self.queue[0][2]
                    elapsed = time.perf_counter() - oldest_timestamp
                    if elapsed >= self.timeout_sec or len(self.queue) >= self.max_batch_size:
                        while self.queue and len(batch) < self.max_batch_size:
                            batch.append(self.queue.popleft())
                            
            if batch:
                await self._execute_model_batch(batch)
            else:
                async with self.lock:
                    if self.queue:
                        rem_time = max(0.0005, self.timeout_sec - (time.perf_counter() - self.queue[0][2]))
                    else:
                        rem_time = 1.0
                try:
                    await asyncio.wait_for(self.frame_ready_event.wait(), timeout=rem_time)
                except asyncio.TimeoutError:
                    pass

    def _sync_forward_pass(self, count: int, tensors: list) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Isolated PyTorch forward pass execution pool.
        Flaw Fixed: Offloads PCIe GPU-to-CPU data transfers completely from the asyncio loop thread.
        """
        try:
            # Positive Mod: Enforce strict memory lockdown. Pre-empts graph allocation overhead entirely.
            with torch.inference_mode():
                for idx, t in enumerate(tensors):
                    if t.dim() == 4 and t.size(0) == 1:
                        t = t.squeeze(0)
                    if t.shape == self.static_batch_tensor[idx].shape:
                        self.static_batch_tensor[idx].copy_(t, non_blocking=True)
                    else:
                        logger.error("[Multiplexer] Frame shape mismatch: %s, skipping slot.", t.shape)
                        self.static_batch_tensor[idx].zero_()
                        
                torch.cuda.current_stream().synchronize()
                input_slice = self.static_batch_tensor[:count].contiguous()
                results = self.model.predict(input_slice, stream=False, verbose=False)
                torch.cuda.current_stream().synchronize()
                
                # Extract matrices to host CPU memory completely inside the background thread context
                parsed_batch_data = []
                for res in results:
                    # Hardened Fix: Stripped redundant .detach() to prevent PyTorch view linter panics
                    # Positive Mod: Enforce consistent float32 dtype for fallback empty arrays to prevent memory alignment panics
                    boxes = res.boxes.data.cpu().numpy() if res.boxes else np.empty((0, 6), dtype=np.float32)
                    masks = res.masks.data.cpu().numpy() if res.masks else np.empty((0, 640, 640), dtype=np.float32)
                    xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes else np.empty((0, 4), dtype=np.float32)
                    conf = res.boxes.conf.cpu().numpy() if res.boxes else np.empty((0,), dtype=np.float32)
                    parsed_batch_data.append((boxes, masks, xyxy, conf))
                    
                # Positive Mod: Explicitly force deterministic object deallocation inside the thread context
                del results
                del input_slice
                return parsed_batch_data
        except Exception as e:
            logger.error("[InferenceError] PyTorch forward pass failed: %s", e)
            return []
        finally:
            tensors.clear()

    def _sync_disk_telemetry(self, status: dict):
        """Atomically updates telemetry logs on disk with exponential backoff."""
        temp_path = None
        try:
            os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
            temp_path = f"{self.status_file}.{time.time_ns()}.tmp"
            
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(status, f)
                
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    os.replace(temp_path, self.status_file)
                    temp_path = None
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.005 * (2 ** attempt))
                    else:
                        logger.debug("[Telemetry] Lock contention threshold reached. Dropping payload slot.")
                        return
        except Exception as e:
            logger.error("Telemetry write failure on target: %s", self.status_file, exc_info=e)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _fast_payload_isolate(source_payloads: list) -> list:
        return [
            {
                "camera_id": item["camera_id"],
                "box": item["box"].copy(),
                "mask": item["mask"].copy() if item["mask"] is not None else None,
                "track_id": item["track_id"],
                "is_new_anomaly": item["is_new_anomaly"],
                "timestamp": item["timestamp"]
            }
            for item in source_payloads
        ]

    def _process_vibration_flow(self, camera_manager, camera_id, xyxy):
        try:
            _vib = camera_manager.vibration_analyzers.get(camera_id)
            if _vib is not None:
                _areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                _best = int(np.argmax(_areas))
                _cx = float((xyxy[_best, 0] + xyxy[_best, 2]) * 0.5)
                _cy = float((xyxy[_best, 1] + xyxy[_best, 3]) * 0.5)
                _prev_cx = getattr(_vib, "_prev_cx", _cx)
                _prev_cy = getattr(_vib, "_prev_cy", _cy)
                _u = _cx - _prev_cx
                _v = _cy - _prev_cy
                _vib._prev_cx = _cx
                _vib._prev_cy = _cy
                _vib.ingest(_u, _v)
        except Exception as e:
            logger.error("[VibrationFlow] Background optical flow failed: %s", e)

    @staticmethod
    def _batch_reflex_dispatch(fast_queue, signals: list):
        for sig in signals:
            try:
                fast_queue.put_nowait(sig)
            except queue.Full:
                pass

    async def _execute_model_batch(self, batch: list):
        start_time = time.perf_counter()
        loop = asyncio.get_running_loop()
        
        camera_ids = [item[0] for item in batch]
        tensors = [item[1] for item in batch]
        batch_size = len(batch)
        
        # Shift processing completely to execution worker thread
        batch_results = await loop.run_in_executor(self.inference_executor, self._sync_forward_pass, batch_size, tensors)
            
        if not batch_results:
            logger.error("[Main] Aborting batch dispatch due to upstream inference failure.")
            async with self.lock:
                current_queue_size = len(self.queue)
            status_payload = {
                "timestamp": time.time(),
                "queue_size": current_queue_size,
                "last_batch_size": 0,
                "last_batch_latency_ms": 0.0,
                "total_processed": self.total_processed,
                "status": "ERROR_INFERENCE_FAILED"
            }
            loop.run_in_executor(None, self._sync_disk_telemetry, status_payload)
            bt = asyncio.create_task(broadcaster.broadcast(status_payload))
            self._background_tasks.add(bt)
            bt.add_done_callback(self._background_tasks.discard)
            return

        detached_payloads = []
        reflex_signals = []
        frame_timestamp = time.time()
        
        manager = FactoryManager()
        camera_manager = manager.get_instance("CameraManager")
        
        for i, data_pack in enumerate(batch_results):
            camera_id = camera_ids[i]
            boxes, masks, xyxy, conf = data_pack
            
            tracker = self.trackers.get(camera_id)
            if tracker is None:
                tracker = DefectTracker(iou_threshold=0.45, max_lost_frames=30)
                self.trackers[camera_id] = tracker
            
            if camera_manager and len(boxes) > 0:
                if len(xyxy) > 0:
                    loop.run_in_executor(None, self._process_vibration_flow, camera_manager, camera_id, xyxy)
                
                for idx in range(len(boxes)):
                    try:
                        conf_val = float(conf[idx])
                        if conf_val >= 0.4:
                            reflex_signals.append({'ts': frame_timestamp, 'cid': camera_id, 'type': 'defect'})
                    except Exception:
                        pass
            
            track_ids, is_new_flags = tracker.update(boxes)
            
            for idx in range(len(boxes)):
                payload = {
                    "camera_id": camera_id,
                    "box": boxes[idx],
                    "mask": masks[idx] if len(masks) > idx else None,
                    "track_id": int(track_ids[idx]),
                    "is_new_anomaly": bool(is_new_flags[idx]),
                    "timestamp": frame_timestamp
                }
                detached_payloads.append(payload)
                
        if reflex_signals and camera_manager:
            loop.run_in_executor(None, self._batch_reflex_dispatch, camera_manager.fast_queue, reflex_signals)
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.total_processed += batch_size
        
        event_logger = manager.get_instance("EventLogger")
        historian = manager.get_instance("PredictiveHistorian")
        
        if detached_payloads:
            if event_logger:
                isolated_events = self._fast_payload_isolate(detached_payloads)
                t1 = asyncio.create_task(event_logger.log_batch_async(isolated_events))
                self._background_tasks.add(t1)
                t1.add_done_callback(self._background_tasks.discard)
                
            if historian:
                isolated_history = self._fast_payload_isolate(detached_payloads)
                t2 = asyncio.create_task(historian.process_history_async(isolated_history))
                self._background_tasks.add(t2)
                t2.add_done_callback(self._background_tasks.discard)
            
        async with self.lock:
            current_queue_size = len(self.queue)

        status_payload = {
            "timestamp": time.time(),
            "queue_size": current_queue_size,
            "last_batch_size": batch_size,
            "last_batch_latency_ms": round(latency_ms, 2),
            "total_processed": self.total_processed,
            "status": "OK"
        }
        
        loop.run_in_executor(None, self._sync_disk_telemetry, status_payload)
        bt_main = asyncio.create_task(broadcaster.broadcast(status_payload))
        self._background_tasks.add(bt_main)
        bt_main.add_done_callback(self._background_tasks.discard)

async def start_inference_node():
    config = ConfigVault.initialize()
    manager = FactoryManager()
    manager.boot_singletons()
    
    pulse_server = PulseServer()
    t_pulse = asyncio.create_task(pulse_server.run_async())
    
    multiplexer = InferenceMultiplexer(config)
    t_multi = asyncio.create_task(multiplexer.run_batch_loop())
    
    logger.info("Qualyxus Core Inference Multiplexer Running...")
    await asyncio.gather(t_pulse, t_multi)

if __name__ == "__main__":
    try:
        asyncio.run(start_inference_node())
    except KeyboardInterrupt:
        logger.info("Inference node terminated safely.")
        sys.exit(0)
