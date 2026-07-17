import os
import time
import queue
import logging
import threading
import uuid
import numpy as np
import cv2
import torch
from ultralytics import YOLO

logger = logging.getLogger("InferenceMultiplexer")

class InferenceMultiplexer:
    """
    Centralized InferenceMultiplexer background daemon.
    Aggregates multi-camera streams to execute batch inference,
    reducing VRAM fragmentation and thread stalling.
    """
    def __init__(self, config):
        self.config = config
        self.max_batch_size = 4
        self.timeout_sec = 0.003  # 3ms timeout window
        self.max_queue_capacity = 60
        
        self.queue = queue.Queue(maxsize=self.max_queue_capacity)
        self.lock = threading.Lock()
        self.results = {}  # request_id -> (Event, result)
        
        # Load YOLO model
        model_path = self.config.get("inference", "model_path") or "models/checkpoints/default/best.pt"
        logger.info(f"[InferenceMultiplexer] Loading model from: {model_path}")
        
        # Hardware acceleration check
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if we should run in mock/simulation mode
        test_cfg = self.config.raw.get("test_flags", {}) if self.config.raw else {}
        self.mock_mode = bool(test_cfg.get("mock_inference", False)) or not os.path.exists(model_path)
        
        if self.mock_mode:
            logger.warning("[InferenceMultiplexer] Model file missing or mock_inference flag set. Running in mock mode.")
            self.model = None
        else:
            try:
                self.model = YOLO(model_path)
                self.model.to(self.device)
            except Exception as e:
                logger.error(f"[InferenceMultiplexer] Failed to load model: {e}. Falling back to mock mode.")
                self.model = None
                self.mock_mode = True
                
        # Start background processing daemon thread
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._batch_processing_loop, name="InferenceMultiplexerDaemon", daemon=True)
        self.worker_thread.start()
        
    def submit_frame(self, camera_id: str, frame: np.ndarray) -> str:
        """Registers a frame for batch inference and returns a unique request ID."""
        request_id = str(uuid.uuid4())
        event = threading.Event()
        
        with self.lock:
            self.results[request_id] = (event, None)
            
        try:
            self.queue.put((request_id, camera_id, frame), timeout=0.033)
        except queue.Full:
            # Handle queue overflow gracefully
            with self.lock:
                self.results.pop(request_id, None)
            raise queue.Full("InferenceMultiplexer buffer is fully saturated.")
            
        return request_id
        
    def retrieve_result(self, request_id: str, timeout_sec: float = 0.033) -> tuple | None:
        """Awaits and retrieves the inference result for a given request ID."""
        with self.lock:
            if request_id not in self.results:
                return None
            event, _ = self.results[request_id]
            
        # Wait for the background daemon to set the event
        finished = event.wait(timeout=timeout_sec)
        
        with self.lock:
            # Safely pop to avoid memory leak
            event, result = self.results.pop(request_id, (None, None))
            
        return result
        
    def stop(self):
        """Stops the background daemon thread."""
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
            
    def _batch_processing_loop(self):
        """Aggregates frames, performs batch inference, and fans out the results."""
        logger.info("[InferenceMultiplexer] Background batch processing daemon active.")
        
        while not self.stop_event.is_set():
            try:
                # Wait for the first frame in a batch
                first_item = self.queue.get(timeout=0.01)
            except queue.Empty:
                continue
                
            batch = [first_item]
            deadline = time.perf_counter() + self.timeout_sec
            
            # Aggregate frames up to max_batch_size within the 3ms window
            while len(batch) < self.max_batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    item = self.queue.get(timeout=timeout)
                    batch.append(item)
                except queue.Empty:
                    break
                    
            # Process the batch
            try:
                self._process_batch(batch)
            except Exception as e:
                logger.error(f"[InferenceMultiplexer] Error processing batch: {e}", exc_info=True)
                # Ensure waiting threads don't block forever on error and clean up tracking registries
                for item in batch:
                    req_id = item[0]
                    with self.lock:
                        event, _ = self.results.pop(req_id, (None, None))
                        if event is not None:
                            event.set()
                            
    def _process_batch(self, batch: list):
        """Executes YOLO predictions on the batch and signals the waiting threads."""
        frames = [item[2] for item in batch]
        
        if self.mock_mode or self.model is None:
            for idx, item in enumerate(batch):
                req_id = item[0]
                # Synthesize standard empty prediction arrays
                boxes = np.empty((0, 6), dtype=np.float32)
                masks = np.empty((0, 640, 640), dtype=np.float32)
                xyxy = np.empty((0, 4), dtype=np.float32)
                conf = np.empty((0,), dtype=np.float32)
                
                with self.lock:
                    if req_id in self.results:
                        event, _ = self.results[req_id]
                        self.results[req_id] = (event, (boxes, masks, xyxy, conf))
                        event.set()
            return
            
        # Standardize color layout and stack inputs natively
        h, w = frames[0].shape[:2]
        processed_frames = []
        for f in frames:
            # CRITICAL HARDENING: Normalize grayscale and single-channel layouts to BGR
            if len(f.shape) == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            elif len(f.shape) == 3 and f.shape[2] == 4:
                f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
            elif len(f.shape) == 3 and f.shape[2] == 1:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                
            # Direct dimension layout verification to handle multi-camera anomalies safely
            if len(f.shape) != 3 or f.shape[2] != 3:
                f = cv2.cvtColor(f[:, :, 0] if len(f.shape) == 3 else f, cv2.COLOR_GRAY2BGR)
                
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            processed_frames.append(f)
            
        stacked = np.stack(processed_frames, axis=0)
        
        # Execute YOLO batch inference
        results = self.model.predict(stacked, device=self.device, verbose=False)
        
        # Fan out predictions to the waiting threads via results lookup map
        for idx, res in enumerate(results):
            req_id = batch[idx][0]
            boxes = res.boxes.data.cpu().numpy() if res.boxes else np.empty((0, 6), dtype=np.float32)
            masks = res.masks.data.cpu().numpy() if res.masks else np.empty((0, 640, 640), dtype=np.float32)
            xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes else np.empty((0, 4), dtype=np.float32)
            conf = res.boxes.conf.cpu().numpy() if res.boxes else np.empty((0,), dtype=np.float32)
            
            with self.lock:
                if req_id in self.results:
                    event, _ = self.results[req_id]
                    self.results[req_id] = (event, (boxes, masks, xyxy, conf))
                    event.set()
