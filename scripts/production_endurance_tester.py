#!/usr/bin/env python3
import asyncio
import time
import os
import sys
import psutil
import logging
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

# Synchronize pathing for core modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config.vault import ConfigVault
from src.core.factory_manager import FactoryManager
from src.inference.infer_live import InferenceMultiplexer
from src.inference.postprocess import UnifiedPreprocessor
from src.logging.event_logger import EventLogger
from src.api.pulse_server import PulseServer, broadcaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EnduranceHarness")

class ProductionEnduranceTester:
    """
    Production-Grade End-to-End Endurance & Latency Profiler.
    Simulates multi-camera pipelines, high-frequency sensor DSP tracking, 
    network streams, and SQL transaction attacks against a strict 30ms budget.
    """
    def __init__(self, duration_sec: int = 60, target_fps: int = 30, num_cameras: int = 4):
        self.duration_sec = duration_sec
        self.target_fps = target_fps
        self.num_cameras = num_cameras
        self.shutdown_event = asyncio.Event()
        
        if not os.path.exists("models/best.pt") and os.environ.get("MODEL_ASSET_PATH") is None:
            logger.info("[Harness] Custom weights missing. Initializing real base yolov8n-seg.pt for un-mocked CUDA execution.")
            os.environ["MODEL_ASSET_PATH"] = "yolov8n-seg.pt"
            
        self.config = ConfigVault.initialize()
        self.manager = FactoryManager()
        self.manager.boot_singletons()
        
        self.logger_instance = EventLogger()
        self.multiplexer = InferenceMultiplexer(self.config)
        
        # Thread-Isolated Preprocessors to completely bypass lock contention
        self.preprocessors = {
            f"cam_{i}": UnifiedPreprocessor(use_triton=False, precision="fp16") 
            for i in range(self.num_cameras)
        }
        
        # Sized safely to accommodate parallel cameras and independent sensor DSP offloading tasks
        self.preprocessing_executor = ThreadPoolExecutor(max_workers=self.num_cameras + 2, thread_name_prefix="harness_workers")
        
        # Look-aside map and mutex to pass metadata across boundaries without breaking core contracts
        self._post_proc_timestamps = {}
        self._metadata_lock = threading.Lock()
        
        # Performance Tracking Matrices
        self.total_frames_injected = 0
        self.total_latency_breaches = 0
        self.cumulative_inference_latency_ms = 0.0
        self.cumulative_ete_latency_ms = 0.0
        self.measured_frames = 0
        self.db_events_fired = 0
        self.sensor_readings_captured = 0
        self.connected_network_clients = 0
        
        self.start_memory_mb = 0.0
        self._inject_profiler_hook()

    def _inject_profiler_hook(self):
        """Wraps the batch execution context to capture both pure inference and true end-to-end latencies."""
        original_execute = self.multiplexer._execute_model_batch
        
        async def monitored_execute(batch):
            try:
                await original_execute(batch)
            finally:
                now = time.perf_counter()
                for i in range(len(batch)):
                    cam_id, tensor, pre_proc_ts = batch[i]
                    
                    # Safely retrieve post-preprocessing timestamp from look-aside cache map
                    with self._metadata_lock:
                        post_proc_ts = self._post_proc_timestamps.pop(id(tensor), pre_proc_ts)
                    
                    inf_latency = (now - post_proc_ts) * 1000.0
                    ete_latency = (now - pre_proc_ts) * 1000.0
                    
                    self.cumulative_inference_latency_ms += inf_latency
                    self.cumulative_ete_latency_ms += ete_latency
                    self.measured_frames += 1
                    
                    if ete_latency > 30.0:
                        self.total_latency_breaches += 1
                        logger.warning(f"[LATENCY_BREACH] {cam_id} pipeline exceeded budget! ETE: {ete_latency:.2f}ms | INF: {inf_latency:.2f}ms")
                    
                    # Clear individual object markers inside the list container to
                    # release heavy GPU allocations instantly without mutating external array lengths.
                    batch[i] = (cam_id, None, pre_proc_ts)
                
                batch_size = len(batch)
                batch.clear()  # Sever the reference graph completely to ensure immediate deallocation
                
                if batch_size > 0:
                    try:
                        payload = {"status": "endurance_pulse", "timestamp": time.time(), "batch_size": batch_size}
                        asyncio.create_task(broadcaster.broadcast(payload))
                    except Exception as e:
                        logger.error(f"[Harness] Network injection hook failed: {e}")
                        
        self.multiplexer._execute_model_batch = monitored_execute

    def _thread_isolated_preprocess(self, raw_frame: np.ndarray, camera_id: str) -> torch.Tensor:
        """Runs image normalization on thread-isolated preprocessors with stream sync."""
        tensor = self.preprocessors[camera_id].process(raw_frame, device="cuda", camera_id=camera_id)
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        return tensor

    async def _camera_stream_worker(self, camera_id: str):
        """Generates continuous raw image matrices and offloads preprocessing to keep the main event loop unblocked."""
        logger.info(f"[{camera_id}] Camera stream ingestion active at {self.target_fps} FPS.")
        frame_interval = 1.0 / self.target_fps
        height, width = 640, 640
        
        raw_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        loop = asyncio.get_running_loop()
        
        try:
            while not self.shutdown_event.is_set():
                cycle_start = time.perf_counter()
                
                # Capture absolute data entry timestamp BEFORE pool execution
                pre_proc_timestamp = time.perf_counter()
                
                tensor = await loop.run_in_executor(
                    self.preprocessing_executor,
                    self._thread_isolated_preprocess,
                    raw_frame, camera_id
                )
                
                post_proc_timestamp = time.perf_counter()
                
                # Secure post-preprocessing timestamp inside look-aside cache map.
                # FLAW FIX: Bounded map to prevent catastrophic memory leaks from dropped queue frames.
                with self._metadata_lock:
                    if len(self._post_proc_timestamps) > 5000:
                        # Evict the absolute oldest key in dictionary order to prevent RAM bloat 
                        # caused by the multiplexer dropping frames under backpressure.
                        self._post_proc_timestamps.pop(next(iter(self._post_proc_timestamps)))
                    self._post_proc_timestamps[id(tensor)] = post_proc_timestamp
                
                await self.multiplexer.enqueue_frame(camera_id, tensor, pre_proc_timestamp)
                self.total_frames_injected += 1
                
                elapsed = time.perf_counter() - cycle_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(0)
                    
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _synchronous_dsp_math(samples: int):
        """Executes heavy NumPy vector generation completely isolated from the main loop thread."""
        for _ in range(samples):
            raw_signal = np.random.normal(0.0, 1.5, 100)
            _ = np.sqrt(np.mean(raw_signal ** 2))

    async def _sensor_dsp_worker(self):
        """Simulates continuous 1000Hz vibration sensor monitoring and RMS calculations."""
        logger.info("[SensorDSP] High-frequency accelerometer stream online (1000Hz emulation).")
        os_tick_rate = 0.015 
        samples_per_tick = int(1000 * os_tick_rate)
        loop = asyncio.get_running_loop()
        
        try:
            while not self.shutdown_event.is_set():
                start_cycle = time.perf_counter()
                
                await loop.run_in_executor(
                    self.preprocessing_executor,
                    self._synchronous_dsp_math,
                    samples_per_tick
                )
                
                self.sensor_readings_captured += samples_per_tick
                
                elapsed = time.perf_counter() - start_cycle
                sleep_time = os_tick_rate - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass

    async def _sqlite_contention_worker(self):
        """Simulates continuous high-frequency database commits to force transactional contention checks."""
        logger.info("[SQL_Hammer] Database transaction contention attack active.")
        payload = {
            "camera_id": "cam_endurance_stress",
            "box": [10, 20, 150, 200],
            "mask": None,
            "track_id": 9999,
            "is_new_anomaly": True,
            "timestamp": time.time()
        }
        
        try:
            while not self.shutdown_event.is_set():
                await self.logger_instance.log_batch_async([payload])
                self.db_events_fired += 1
                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            pass

    async def _network_client_worker(self, client_id: int):
        """Simulates an independent remote operator session consuming the real-time SSE telemetry feed."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", "http://localhost:8000/api/v1/telemetry/stream", timeout=60.0) as r:
                    self.connected_network_clients += 1
                    async for line in r.aiter_lines():
                        if self.shutdown_event.is_set():
                            break
        except Exception:
            pass
        finally:
            if self.connected_network_clients > 0:
                self.connected_network_clients -= 1

    async def _resource_monitor(self):
        """Periodically polls host system statistics to track stability profiles."""
        try:
            while not self.shutdown_event.is_set():
                await asyncio.sleep(5.0)
                mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
                vram_allocated = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
                
                avg_inf_latency = (self.cumulative_inference_latency_ms / self.measured_frames) if self.measured_frames > 0 else 0.0
                avg_ete_latency = (self.cumulative_ete_latency_ms / self.measured_frames) if self.measured_frames > 0 else 0.0
                
                logger.info("=" * 60)
                logger.info("[SYSTEM ENDURANCE TELEMETRY PROFILE]")
                logger.info(f"  Host RAM (RSS)   : {mem_mb:.2f} MB (Delta: {mem_mb - self.start_memory_mb:+.2f} MB)")
                logger.info(f"  Device VRAM Pool : {vram_allocated:.2f} MB")
                logger.info(f"  Multiplexer Queue: {len(self.multiplexer.queue)} / {self.multiplexer.max_queue_capacity}")
                logger.info(f"  Frames Processed : {self.measured_frames} (Injected: {self.total_frames_injected})")
                logger.info(f"  Avg Inf Latency  : {avg_inf_latency:.2f} ms")
                logger.info(f"  Avg ETE Latency  : {avg_ete_latency:.2f} ms (SLA Limit: 30.0ms)")
                logger.info(f"  SLA Breaches     : {self.total_latency_breaches}")
                logger.info(f"  DSP Ingestions   : {self.sensor_readings_captured} samples computed")
                logger.info(f"  SQL Commits Sent : {self.db_events_fired}")
                logger.info(f"  Active Network UI: {self.connected_network_clients} clients connected")
                logger.info("=" * 60)
        except asyncio.CancelledError:
            pass

    async def run(self):
        self.start_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Initializing Certified Endurance Harness. Window Duration: {self.duration_sec}s.")
        
        await self.logger_instance.start()
        pulse_server = PulseServer()
        server_task = asyncio.create_task(pulse_server.run_async())
        await asyncio.sleep(1.0)
        
        multi_task = asyncio.create_task(self.multiplexer.run_batch_loop())
        monitor_task = asyncio.create_task(self._resource_monitor())
        db_hammer_task = asyncio.create_task(self._sqlite_contention_worker())
        sensor_task = asyncio.create_task(self._sensor_dsp_worker())
        
        cam_tasks = [
            asyncio.create_task(self._camera_stream_worker(f"cam_{i}"))
            for i in range(self.num_cameras)
        ]
        
        client_tasks = [
            asyncio.create_task(self._network_client_worker(i))
            for i in range(20)
        ]
        
        try:
            await asyncio.sleep(self.duration_sec)
        except KeyboardInterrupt:
            logger.warning("Operator manual interrupt intercepted. Commencing graceful teardown matrix...")
            
        self.shutdown_event.set()
        
        for t in cam_tasks + client_tasks + [sensor_task]:
            t.cancel()
        monitor_task.cancel()
        db_hammer_task.cancel()
        multi_task.cancel()
        server_task.cancel()
        
        await asyncio.gather(*(cam_tasks + client_tasks + [sensor_task]), return_exceptions=True)
        await asyncio.gather(monitor_task, db_hammer_task, multi_task, server_task, return_exceptions=True)
        
        logger.info("[Harness] Draining residual database transaction queues...")
        while not self.logger_instance.queue.empty():
            await asyncio.sleep(0.1)
            
        if self.logger_instance.worker_task:
            self.logger_instance.worker_task.cancel()
            try:
                await self.logger_instance.worker_task
            except asyncio.CancelledError:
                pass
                
        self.preprocessing_executor.shutdown(wait=True)
        
        final_mem = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info("=" * 60)
        logger.info("ENDURANCE TEST SYSTEM ANALYSIS")
        logger.info(f"Total Frames Handled     : {self.measured_frames}")
        logger.info(f"Total Latency Breaches   : {self.total_latency_breaches} (Target: 0)")
        logger.info(f"Final Execution RAM RSS  : {final_mem:.2f} MB (Total Creep: {final_mem - self.start_memory_mb:+.2f} MB)")
        logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualyxus Production Endurance Harness")
    parser.add_argument("--duration", type=int, default=60, help="Test duration window in seconds")
    parser.add_argument("--cameras", type=int, default=4, help="Number of concurrent camera paths")
    parser.add_argument("--fps", type=int, default=30, help="Target frames per second per camera")
    args = parser.parse_args()
    
    try:
        asyncio.run(ProductionEnduranceTester(duration_sec=args.duration, num_cameras=args.cameras, target_fps=args.fps).run())
    except KeyboardInterrupt:
        logger.info("Harness process terminated.")
    except Exception as e:
        logger.critical(f"Unhandled exception during endurance execution: {e}", exc_info=True)
        sys.exit(1)
