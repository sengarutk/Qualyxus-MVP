from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import time
import shutil
import numpy as np
import cv2

from src.inference.analysis.geometry_engine import GeometryEngine

TEMPLATE_PATH = "data/reference_template.png"
PASS, FAIL = "✅", "❌"
_errors: list[str] = []

def _assert(cond: bool, msg: str) -> None:
    sym = PASS if cond else FAIL
    print(f"     {sym} {msg}")
    if not cond:
        _errors.append(msg)

def generate_synthetic_template(path: str):
    """Generate a synthetic calibration template with an asymmetric feature and anchor points."""
    # Create 300x300 dark gray canvas
    img = np.full((300, 300), 50, dtype=np.uint8)
    # Draw main circular body
    cv2.circle(img, (150, 150), 80, 180, -1)
    # Draw an asymmetric rectangular keyway/tab (at the top edge)
    cv2.rectangle(img, (135, 30), (165, 75), 255, -1)
    # Add robust corner anchor dots for descriptor anchoring
    cv2.circle(img, (70, 70), 12, 100, -1)
    cv2.circle(img, (230, 70), 12, 100, -1)
    cv2.circle(img, (70, 230), 12, 100, -1)
    cv2.circle(img, (230, 230), 12, 100, -1)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    return img

def main():
    print("[GeometryEngine Test] Initializing verification sequence...")
    
    # 1. Generate reference assets
    img = generate_synthetic_template(TEMPLATE_PATH)
    
    # 2. Instantiate Engine
    engine = GeometryEngine(camera_id="cam_test", template_path=TEMPLATE_PATH)
    
    # Default specifications
    engine.update_template_specifications({
        "structural_similarity_floor": 0.85,
        "max_spatial_drift_mm": 2.0,
        "max_orientation_drift_degrees": 5.0,
        "pixel_to_mm_scale": 0.1
    })

    # -------------------------------------------------------------
    # Test 1: Exact Template Match (Zero Drift / Perfect Alignment)
    # -------------------------------------------------------------
    print("\n[Test 1] Exact Template Match (Zero Drift / Perfect Alignment)")
    res = engine.inspect_part_geometry(img)
    print(f"     Diagnostic: {res}")
    _assert(res["status"] == "OK", "Status should be OK for exact copy match")
    _assert(res["structural_similarity_score"] >= 0.98, "SSIM score should be close to 1.0")
    _assert(abs(res["spatial_drift_x_mm"]) < 1e-2, "Spatial drift X should be negligible")
    _assert(abs(res["spatial_drift_y_mm"]) < 1e-2, "Spatial drift Y should be negligible")
    _assert(res["orientation_deviation_degrees"] < 0.1, "Orientation deviation should be negligible")

    # -------------------------------------------------------------
    # Test 2: Spatial Translation Drift (Drift > 2.0mm Violation)
    # -------------------------------------------------------------
    print("\n[Test 2] Spatial Translation Drift (Drift > 2.0mm)")
    # Shift image by 25 pixels in X, which maps to 2.5mm drift (threshold is 2.0mm)
    M_trans = np.float32([[1, 0, 25], [0, 1, 5]])
    shifted_img = cv2.warpAffine(img, M_trans, (300, 300), borderValue=50)
    res = engine.inspect_part_geometry(shifted_img)
    print(f"     Diagnostic: {res}")
    _assert(res["status"] == "GEOMETRIC_VIOLATION", "Status should be GEOMETRIC_VIOLATION for spatial drift exceedance")
    _assert(abs(res["spatial_drift_x_mm"] - 2.5) < 0.2, "Spatial drift X should be computed as ~2.5mm")
    _assert("exceeds threshold" in res["justification"], "Justification should cite threshold exceedance")

    # -------------------------------------------------------------
    # Test 3: Rotational Deviation (Rotation > 5.0deg Violation)
    # -------------------------------------------------------------
    print("\n[Test 3] Rotational Deviation (Rotation > 5.0deg)")
    # Rotate image by 12 degrees (threshold is 5.0 degrees)
    M_rot = cv2.getRotationMatrix2D((150, 150), 12.0, 1.0)
    rotated_img = cv2.warpAffine(img, M_rot, (300, 300), borderValue=50)
    res = engine.inspect_part_geometry(rotated_img)
    print(f"     Diagnostic: {res}")
    _assert(res["status"] == "GEOMETRIC_VIOLATION", "Status should be GEOMETRIC_VIOLATION for rotational drift exceedance")
    _assert(abs(res["orientation_deviation_degrees"] - 12.0) < 1.0, "Rotational drift should be computed as ~12.0 degrees")

    # -------------------------------------------------------------
    # Test 4: Insufficient Anchoring Features (Alignment Fault)
    # -------------------------------------------------------------
    print("\n[Test 4] Insufficient Anchoring Features (Alignment Fault)")
    # Pass a blank canvas, ORB will find 0 descriptors
    blank_img = np.full((300, 300), 50, dtype=np.uint8)
    res = engine.inspect_part_geometry(blank_img)
    print(f"     Diagnostic: {res}")
    _assert(res["status"] == "ALIGNMENT_FAULT", "Status should be ALIGNMENT_FAULT for featureless input")
    _assert("Alignment failed" in res["justification"], "Justification should indicate alignment failure")

    # -------------------------------------------------------------
    # Test 5: Upstream Visual Defect Override
    # -------------------------------------------------------------
    print("\n[Test 5] Upstream Visual Defect Override")
    res = engine.inspect_part_geometry(img, visual_defect_present=True)
    print(f"     Diagnostic: {res}")
    _assert(res["status"] == "GEOMETRIC_VIOLATION", "Status should be GEOMETRIC_VIOLATION when visual defect override is set")
    _assert("upstream visual defect segmentation cascade" in res["justification"], "Justification should report cascade rejection")

    # -------------------------------------------------------------
    # Test 6: Performance Benchmark (Latency Budget < 2.5ms)
    # -------------------------------------------------------------
    print("\n[Test 6] Performance Benchmark (Latency Budget < 2.5ms)")
    # Pre-warm ORB cache
    engine.inspect_part_geometry(img)
    
    num_runs = 200
    start_time = time.perf_counter()
    for _ in range(num_runs):
        engine.inspect_part_geometry(img)
    end_time = time.perf_counter()
    
    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0
    print(f"     Average Latency: {avg_latency_ms:.4f} ms")
    _assert(avg_latency_ms < 2.5, "Average CPU latency must be strictly under 2.5ms")

    print("\n" + "=" * 60)
    if _errors:
        print(f"❌ Verification completed with {len(_errors)} failure(s).")
        for err in _errors:
            print(f"   - {err}")
        sys.exit(1)
    else:
        print("✅ GeometryEngine verification completed successfully with 100% passes!")
        sys.exit(0)

if __name__ == "__main__":
    main()
