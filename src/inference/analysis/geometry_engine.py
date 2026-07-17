from __future__ import annotations

import os
import threading
import logging
import time
import cv2
import numpy as np

logger = logging.getLogger("QualyxusGeometryEngine")

class GeometryEngine:
    """
    High-Velocity, Sub-Pixel Accurate Golden Template Inspection (GTI) Engine.
    Uses vectorized NumPy and OpenCV operations to register live part matrices,
    calculate structural similarity coefficients, and audit component orientation.
    Optimized for execution within a strict <2.5ms CPU window.
    """
    def __init__(self, camera_id: str, template_path: str | os.PathLike | None = None):
        self.camera_id = camera_id
        self.cache_lock = threading.Lock()
        
        # Configure default inspection specifications
        self.structural_similarity_floor = 0.85
        self.max_spatial_drift_mm = 2.0
        self.max_orientation_drift_degrees = 5.0
        self.pixel_to_mm_scale = 0.1
        
        # Setup feature extraction components (optimized parameters for low latency)
        self.orb = cv2.ORB_create(nfeatures=85, scaleFactor=1.2, nlevels=3, edgeThreshold=15, patchSize=15)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.template: np.ndarray | None = None
        self.template_kp = None
        self.template_des = None
        self.template_theta_deg = None
        self.template_centroid: tuple[float, float] | None = None
        self.template_w: int | None = None
        self.template_h: int | None = None
        
        self._load_template(template_path)

    def _load_template(self, template_path: str | os.PathLike | None):
        if not template_path:
            self.template = None
            self.template_kp = None
            self.template_des = None
            self.template_theta_deg = None
            self.template_centroid = None
            self.template_w = None
            self.template_h = None
            return
            
        try:
            path_str = str(template_path)
            if os.path.exists(path_str):
                # Load as single-channel grayscale image
                img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.template = img
                    # Pre-compute template keypoints and descriptors once
                    self.template_kp, self.template_des = self.orb.detectAndCompute(img, None)
                    
                    # Pre-compute template moments and principal axis angle using fast fixed threshold
                    _, thresh_temp = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    
                    h, w = img.shape[:2]
                    self.template_w = w
                    self.template_h = h
                    
                    mu_temp = cv2.moments(thresh_temp)
                    if mu_temp['m00'] > 0:
                        self.template_centroid = (mu_temp['m10'] / mu_temp['m00'], mu_temp['m01'] / mu_temp['m00'])
                        theta_temp = 0.5 * np.arctan2(2 * mu_temp['mu11'], mu_temp['mu20'] - mu_temp['mu02'])
                        self.template_theta_deg = np.degrees(theta_temp)
                    else:
                        self.template_centroid = (w / 2.0, h / 2.0)
                        self.template_theta_deg = 0.0
                        
                    logger.info("[GeometryEngine] Loaded reference template and pre-computed features: %s (shape: %s)", path_str, img.shape)
                else:
                    self.template = None
                    self.template_kp = None
                    self.template_des = None
                    self.template_theta_deg = None
                    self.template_centroid = None
                    self.template_w = None
                    self.template_h = None
                    logger.error("[GeometryEngine] Failed to decode template image: %s", path_str)
            else:
                self.template = None
                self.template_kp = None
                self.template_des = None
                self.template_theta_deg = None
                self.template_centroid = None
                self.template_w = None
                self.template_h = None
                logger.warning("[GeometryEngine] Reference template path does not exist: %s", path_str)
        except Exception as e:
            self.template = None
            self.template_kp = None
            self.template_des = None
            self.template_theta_deg = None
            self.template_centroid = None
            self.template_w = None
            self.template_h = None
            logger.error("[GeometryEngine] Exception loading reference template: %s", e)

    def update_template_specifications(self, config: dict):
        """Dynamic parameter hot-swap interface called by the profile manager."""
        with self.cache_lock:
            self.structural_similarity_floor = config.get("structural_similarity_floor", self.structural_similarity_floor)
            self.max_spatial_drift_mm = config.get("max_spatial_drift_mm", self.max_spatial_drift_mm)
            self.max_orientation_drift_degrees = config.get("max_orientation_drift_degrees", self.max_orientation_drift_degrees)
            self.pixel_to_mm_scale = config.get("pixel_to_mm_scale", self.pixel_to_mm_scale)
            
            template_path = config.get("template_path")
            if template_path:
                self._load_template(template_path)

    def inspect_part_geometry(self, live_frame: np.ndarray, visual_defect_present: bool = False) -> dict:
        """
        Primary inspection gateway.
        Performs spatial registration, structural differencing, and centroid orientation audit.
        """
        # E. Upstream Defect Logical Union Override
        if visual_defect_present:
            return {
                "status": "GEOMETRIC_VIOLATION",
                "structural_similarity_score": 0.0,
                "spatial_drift_x_mm": 0.0,
                "spatial_drift_y_mm": 0.0,
                "orientation_deviation_degrees": 0.0,
                "justification": "Rejected due to upstream visual defect segmentation cascade."
            }

        # Guard against None and non-finite inputs (NaN / Inf)
        if live_frame is None or not isinstance(live_frame, np.ndarray) or not np.isfinite(live_frame).all():
            return {
                "status": "ALIGNMENT_FAULT",
                "structural_similarity_score": 0.0,
                "spatial_drift_x_mm": 0.0,
                "spatial_drift_y_mm": 0.0,
                "orientation_deviation_degrees": 0.0,
                "justification": "Alignment failed: Input frame contains non-finite values (NaN/Inf)."
            }

        with self.cache_lock:
            template = self.template
            template_kp = self.template_kp
            template_des = self.template_des
            template_theta_deg = self.template_theta_deg
            template_centroid = self.template_centroid
            template_w = self.template_w
            template_h = self.template_h
            sim_floor = self.structural_similarity_floor
            max_drift = self.max_spatial_drift_mm
            max_angle = self.max_orientation_drift_degrees
            scale = self.pixel_to_mm_scale

        if template is None or template_des is None or template_centroid is None or template_w is None or template_h is None:
            return {
                "status": "ALIGNMENT_FAULT",
                "structural_similarity_score": 0.0,
                "spatial_drift_x_mm": 0.0,
                "spatial_drift_y_mm": 0.0,
                "orientation_deviation_degrees": 0.0,
                "justification": "Alignment failed: Golden reference template not loaded."
            }

        try:
            # A. Channel-Agnostic Grayscale Ingestion Guard
            if len(live_frame.shape) == 3:
                channels = live_frame.shape[2]
                if channels == 3:
                    live_frame_gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
                elif channels == 4:
                    live_frame_gray = cv2.cvtColor(live_frame, cv2.COLOR_BGRA2GRAY)
                else:
                    live_frame_gray = live_frame[:, :, 0]
            else:
                live_frame_gray = live_frame

            # B. Spatial Registration & Affine Alignment Plane
            # Optimize: Detect and compute only on the live frame
            kp2, des2 = self.orb.detectAndCompute(live_frame_gray, None)

            if des2 is None or len(kp2) < 4:
                return {
                    "status": "ALIGNMENT_FAULT",
                    "structural_similarity_score": 0.0,
                    "spatial_drift_x_mm": 0.0,
                    "spatial_drift_y_mm": 0.0,
                    "orientation_deviation_degrees": 0.0,
                    "justification": "Alignment failed: Insufficient feature descriptors computed on live frame."
                }

            matches = self.bf.match(template_des, des2)
            if len(matches) < 4:
                return {
                    "status": "ALIGNMENT_FAULT",
                    "structural_similarity_score": 0.0,
                    "spatial_drift_x_mm": 0.0,
                    "spatial_drift_y_mm": 0.0,
                    "orientation_deviation_degrees": 0.0,
                    "justification": f"Alignment failed: Matching anchors ({len(matches)}) below threshold of 4."
                }

            # Sort matches and select best keypoint coordinates
            matches = sorted(matches, key=lambda x: x.distance)[:80]
            src_pts = np.float32([template_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate partial 2D transformation (dst_pts maps to src_pts)
            # Capping maxIters to 150 prevents CPU cycles from being wasted on high iteration RANSAC runs
            M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, maxIters=150)
            if M is None or inliers is None or np.sum(inliers) < 4:
                return {
                    "status": "ALIGNMENT_FAULT",
                    "structural_similarity_score": 0.0,
                    "spatial_drift_x_mm": 0.0,
                    "spatial_drift_y_mm": 0.0,
                    "orientation_deviation_degrees": 0.0,
                    "justification": "Alignment failed: Robust RANSAC estimation has fewer than 4 inliers."
                }

            # Invert M to map from template to live frame space
            M_inv = cv2.invertAffineTransform(M)
            if M_inv is None:
                return {
                    "status": "ALIGNMENT_FAULT",
                    "structural_similarity_score": 0.0,
                    "spatial_drift_x_mm": 0.0,
                    "spatial_drift_y_mm": 0.0,
                    "orientation_deviation_degrees": 0.0,
                    "justification": "Alignment failed: Affine transformation matrix is non-invertible."
                }

            # Warp live frame using bilinear interpolation to align with template space
            aligned_frame = cv2.warpAffine(live_frame_gray, M, (template_w, template_h), flags=cv2.INTER_LINEAR)

            # Decouple Translation Drift: Map the precomputed template centroid using M_inv to locate the centroid in live frame space
            tc_x, tc_y = template_centroid
            c_live_x = M_inv[0, 0] * tc_x + M_inv[0, 1] * tc_y + M_inv[0, 2]
            c_live_y = M_inv[1, 0] * tc_x + M_inv[1, 1] * tc_y + M_inv[1, 2]

            spatial_drift_x_mm = float(c_live_x - tc_x) * scale
            spatial_drift_y_mm = float(c_live_y - tc_y) * scale

            # C. High-Velocity Structural Similarity Alternative
            # Crop a 5% margin to prevent artificial boundary warping artifacts
            dy, dx = int(template_h * 0.05), int(template_w * 0.05)
            if dy > 0 and dx > 0 and template_h - 2*dy > 0 and template_w - 2*dx > 0:
                cropped_temp = template[dy:template_h-dy, dx:template_w-dx]
                cropped_aligned = aligned_frame[dy:template_h-dy, dx:template_w-dx]
            else:
                cropped_temp = template
                cropped_aligned = aligned_frame

            # Guard against zero variance in structural score
            std_temp = np.std(cropped_temp)
            std_aligned = np.std(cropped_aligned)
            if std_temp < 1e-4 or std_aligned < 1e-4:
                structural_similarity_score = 0.0
            else:
                # Downsample 2x using slicing stride for zero-copy 4x speedup
                cropped_aligned_ds = cropped_aligned[::2, ::2]
                cropped_temp_ds = cropped_temp[::2, ::2]
                res = cv2.matchTemplate(cropped_aligned_ds, cropped_temp_ds, cv2.TM_CCOEFF_NORMED)
                structural_similarity_score = float(res[0, 0])

            # D. Non-Discontinuous Angular Tracking & Centroid Auditing
            # Downsample 2x using slicing stride for scale-invariant 4x speedup on moment thresholding
            aligned_frame_ds = aligned_frame[::2, ::2]
            _, thresh_aligned = cv2.threshold(aligned_frame_ds, 127, 255, cv2.THRESH_BINARY)
            
            mu_aligned = cv2.moments(thresh_aligned)
            if mu_aligned['m00'] > 0:
                theta_aligned = 0.5 * np.arctan2(2 * mu_aligned['mu11'], mu_aligned['mu20'] - mu_aligned['mu02'])
                theta_aligned_deg = np.degrees(theta_aligned)
            else:
                theta_aligned_deg = 0.0

            # Derive relative rotation angle from the affine registration matrix M
            theta_M_rad = np.arctan2(M[1, 0], M[0, 0])
            theta_M_deg = np.degrees(theta_M_rad)

            # Mathematically construct measured orientation angle by accounting for rotation M
            theta_measured_deg = theta_aligned_deg - theta_M_deg
            theta_golden_deg = template_theta_deg if template_theta_deg is not None else 0.0

            # Apply non-discontinuous wrap-around correction:
            orientation_deviation_degrees = float(abs((theta_measured_deg - theta_golden_deg + 90) % 180 - 90))

            # Evaluate diagnostic specifications
            drift_x_abs = abs(spatial_drift_x_mm)
            drift_y_abs = abs(spatial_drift_y_mm)

            violation = False
            justification = "Geometry inspection completed successfully."

            if structural_similarity_score < sim_floor:
                violation = True
                justification = f"Structural similarity score {structural_similarity_score:.3f} below floor of {sim_floor:.3f}."
            elif drift_x_abs > max_drift or drift_y_abs > max_drift:
                violation = True
                justification = f"Spatial drift (x: {drift_x_abs:.2f}mm, y: {drift_y_abs:.2f}mm) exceeds threshold of {max_drift:.2f}mm."
            elif orientation_deviation_degrees > max_angle:
                violation = True
                justification = f"Orientation deviation {orientation_deviation_degrees:.2f} degrees exceeds threshold of {max_angle:.2f} degrees."

            status = "GEOMETRIC_VIOLATION" if violation else "OK"

            return {
                "status": status,
                "structural_similarity_score": structural_similarity_score,
                "spatial_drift_x_mm": spatial_drift_x_mm,
                "spatial_drift_y_mm": spatial_drift_y_mm,
                "orientation_deviation_degrees": orientation_deviation_degrees,
                "justification": justification
            }

        except Exception as e:
            logger.critical("[GeometryEngine] Exception in inspect_part_geometry: %s", e, exc_info=True)
            return {
                "status": "ALIGNMENT_FAULT",
                "structural_similarity_score": 0.0,
                "spatial_drift_x_mm": 0.0,
                "spatial_drift_y_mm": 0.0,
                "orientation_deviation_degrees": 0.0,
                "justification": f"Internal exception in inspection pipeline: {str(e)}"
            }
