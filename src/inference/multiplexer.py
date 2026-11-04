import os
import time
import queue
import logging
import threading
import uuid
import numpy as np
import cv2
import torch

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
        
        self.class_names: dict[int, str] = {}
        # Load YOLO model: inference.model_path, falling back to the MODEL_ASSET_PATH
        # deployment variable (set by the Docker image / compose file) when that file is absent.
        model_path = self.config.get("inference", "model_path") or "models/checkpoints/default/best.pt"
        asset_path = os.environ.get("MODEL_ASSET_PATH")
        if not os.path.exists(model_path) and asset_path and os.path.exists(asset_path):
            model_path = asset_path
        logger.info(f"[InferenceMultiplexer] Loading model from: {model_path}")

        # Prediction thresholds come from the same config the offline tools use.
        self.conf = float(self.config.get("inference", "conf", default=0.35) or 0.35)
        self.iou = float(self.config.get("inference", "iou", default=0.5) or 0.5)
        
        # Hardware acceleration check (an explicit inference.device of "cpu" forces CPU)
        requested_device = str(self.config.get("inference", "device", default="") or "").lower()
        if requested_device == "cpu" or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda"
        
        # Check if we should run in mock/simulation mode
        test_cfg = (self.config.raw.get("test_flags", {}) or {}) if self.config.raw else {}
        # The frame detector is optional and off by default: parts are judged by the inspection
        # models, and Ultralytics is AGPL-3.0, so it is only loaded for lines that ask for it.
        classes = self.config.get("inference", "defect_classes", default=[])
        enabled = bool(self.config.get("inference", "frame_detector", default=bool(classes)))
        self.mock_mode = bool(test_cfg.get("mock_inference", False)) or not enabled
        self.load_failed = False
        if enabled and not self.mock_mode and not os.path.exists(model_path):
            logger.error("[InferenceMultiplexer] Frame detector %s not found.", model_path)
            self.mock_mode = self.load_failed = True

        if self.mock_mode:
            logger.info("[InferenceMultiplexer] Frame detector not in use.")
            self.model = None
        else:
            try:
                from ultralytics import YOLO

                self.model = YOLO(model_path)
                self.model.to(self.device)
                self.class_names = {int(k): str(v) for k, v in (getattr(self.model, "names", None) or {}).items()}
            except Exception as e:
                logger.error(f"[InferenceMultiplexer] Failed to load frame detector: {e}.")
                self.model = None
                self.mock_mode = self.load_failed = True
                
        # Start background processing daemon thread
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._batch_processing_loop, name="InferenceMultiplexerDaemon", daemon=True)
        self.worker_thread.start()
        
    def submit_frame(self, camera_id: str, frame: np.ndarray, block: bool = True, timeout: float | None = 0.033) -> str:
        """Registers a frame for batch inference and returns a unique request ID."""
        request_id = str(uuid.uuid4())
        event = threading.Event()
        
        with self.lock:
            self.results[request_id] = (event, None)
            
        try:
            self.queue.put((request_id, camera_id, frame), block=block, timeout=timeout)
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
        event.wait(timeout=timeout_sec)
        
        with self.lock:
            # Safely pop to avoid memory leak
            event, result = self.results.pop(request_id, (None, None))
            
        return result
        
    def stop(self):
        """Stops the background daemon thread and releases any request still waiting on it."""
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        with self.lock:
            pending = list(self.results.values())
        for event, _ in pending:
            event.set()
            
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
            
        # Standardize color layout. Frames are handed to Ultralytics as a list so that
        # cameras with different resolutions are each letterboxed correctly.
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
            processed_frames.append(f)
        
        # Execute YOLO batch inference
        results = self.model.predict(
            processed_frames,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        
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
