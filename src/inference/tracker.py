import numpy as np

class DefectTracker:
    """
    Vectorized spatial-temporal tracking class for industrial edge deployment.
    Maintains track identity persistence across lens flicker using explicit memory clones
    and clean NumPy transformations to optimize processing throughput.
    """
    def __init__(self, iou_threshold=0.45, max_lost_frames=30):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.active_tracks = {}
        self.next_track_id = 1
        self._max_id_limit = 1_000_000

    def _get_next_id(self) -> int:
        """Atomically returns the next available unallocated ID tracking coordinate."""
        start_id = self.next_track_id
        while True:
            candidate = self.next_track_id
            self.next_track_id += 1
            if self.next_track_id > self._max_id_limit:
                self.next_track_id = 1
                
            if candidate not in self.active_tracks:
                return candidate
                
            if self.next_track_id == start_id:
                # Hard Failure: Registry is entirely full. Raise a clear error instead of clobbering data.
                raise RuntimeError("[Tracker] Critical Error: Tracking registry completely saturated.")

    def compute_iou_matrix(self, boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        r"""
        Calculates the IoU intersection matrix using vectorized NumPy broadcasting.
        Formula:
        $$IoU = \frac{|A \cap B|}{|A \cup B|}$$
        """
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

        lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
        rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

        wh = np.maximum(0, rb - lt)
        intersection = wh[:, :, 0] * wh[:, :, 1]
        
        union = area_a[:, None] + area_b[None, :] - intersection
        return intersection / np.maximum(union, 1e-6)

    def update(self, detections: np.ndarray) -> tuple[list[int], list[bool]]:
        """
        Updates active tracks with new frame bounding matrices.
        Enforces strict age-based pruning to guarantee zero memory growth over time.
        """
        num_dets = len(detections)
        track_ids = [-1] * num_dets
        is_new_flags = [False] * num_dets

        # Increment loss counts across all existing entries at the start of the frame cycle
        for tid in list(self.active_tracks.keys()):
            self.active_tracks[tid]['lost_count'] += 1

        if num_dets == 0:
            # Hardening Fix: Explicitly prune dead tracks to seal the memory space
            for tid in list(self.active_tracks.keys()):
                if self.active_tracks[tid]['lost_count'] > self.max_lost_frames:
                    del self.active_tracks[tid]
            return track_ids, is_new_flags

        det_boxes = detections[:, :4]
        active_ids = list(self.active_tracks.keys())
        
        unmatched_dets = list(range(num_dets))
        unmatched_tracks = list(range(len(active_ids)))

        if len(active_ids) > 0:
            track_boxes = np.array([self.active_tracks[tid]['bbox'] for tid in active_ids])
            iou_matrix = self.compute_iou_matrix(track_boxes, det_boxes)
            
            if iou_matrix.size > 0:
                iou_flat = iou_matrix.flatten()
                sort_idx = np.argsort(iou_flat)[::-1]
                
                for idx in sort_idx:
                    t_idx = int(idx // iou_matrix.shape[1])
                    d_idx = int(idx % iou_matrix.shape[1])
                    
                    if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                        break
                        
                    if t_idx in unmatched_tracks and d_idx in unmatched_dets:
                        unmatched_tracks.remove(t_idx)
                        unmatched_dets.remove(d_idx)
                        
                        tid = active_ids[t_idx]
                        # Use explicit .copy() to break array reference chains and prevent memory leaks
                        self.active_tracks[tid]['bbox'] = det_boxes[d_idx].copy()
                        self.active_tracks[tid]['lost_count'] = 0
                        
                        track_ids[d_idx] = tid
                        is_new_flags[d_idx] = False

        for d_idx in unmatched_dets:
            tid = self._get_next_id()
            self.active_tracks[tid] = {
                'bbox': det_boxes[d_idx].copy(),
                'class_id': int(detections[d_idx, 5]),
                'lost_count': 0
            }
            track_ids[d_idx] = tid
            is_new_flags[d_idx] = True

        # Hardening Fix: Explicitly prune dead tracks to seal the memory space
        for tid in list(self.active_tracks.keys()):
            if self.active_tracks[tid]['lost_count'] > self.max_lost_frames:
                del self.active_tracks[tid]

        return track_ids, is_new_flags
