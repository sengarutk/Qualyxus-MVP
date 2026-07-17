import time
import queue
import numpy as np

import sys
from pathlib import Path

# Add src to path so we can import VibrationAnalyzer
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from inference.analysis.vibration_engine import VibrationAnalyzer


def make_signal(hz: float, duration_sec: float, fs: float = 120.0) -> np.ndarray:
    """Generate a clean synthetic AC signal at the specified frequency."""
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    # A = 50.0 (RMS ~ 35.3) + 100.0 DC offset to prevent hypot rectification
    return 100.0 + 50.0 * np.sin(2 * np.pi * hz * t)


def run_stream():
    """
    Simulate a continuous stream of (u, v) vectors feeding the 
    Vibration Engine to test the Pulse UI Dashboard.
    """
    print("=" * 60)
    print(" Starting Phase 5 UI Stream Generator (test_cam)")
    print(" VLM Narrative Simulation Enabled.")
    print(" Press Ctrl+C to stop.")
    print("=" * 60)

    # Note: Machine Group 2, Rigid foundation (C > 2.8, D > 4.5)
    # Our signals will have an RMS of ~35.3 mm/s (well above D threshold).
    fq = queue.Queue(maxsize=100)
    analyzer = VibrationAnalyzer(
        buffer_size=512,
        fast_queue=fq,
        machine_group=2,
        foundation_type="Rigid",
        camera_id="test_cam",  # Bound to test_cam for SSE /stream/test_cam
        pixel_to_mm_scale=1.0
    )

    t0 = time.perf_counter()
    sample_index = 0

    try:
        while True:
            # 1. 25 Hz (Spindle Misalignment) for 5 seconds
            print("\n>> Injecting 25 Hz (Spindle Misalignment) for 5 seconds...")
            sig_25 = make_signal(25.0, 5.0)
            for s in sig_25:
                ts = t0 + sample_index / 120.0
                analyzer.ingest(float(s), 0.0, ts=ts)
                sample_index += 1
                time.sleep(1.0 / 120.0)  # Real-time throttle

            # 2. Quiet period for 2 seconds
            print(">> Injecting 1 Hz quiet baseline for 2 seconds...")
            sig_quiet = make_signal(1.0, 2.0)
            # Make quiet amplitude tiny (e.g. 0.5) so RMS < 2.8 (ZONE_A_B)
            sig_quiet = 100.0 + 0.5 * np.sin(2 * np.pi * 1.0 * (np.arange(240)/120.0))
            for s in sig_quiet:
                ts = t0 + sample_index / 120.0
                analyzer.ingest(float(s), 0.0, ts=ts)
                sample_index += 1
                time.sleep(1.0 / 120.0)

            # 3. 50 Hz (Bearing Instability) for 5 seconds
            print(">> Injecting 50 Hz (Bearing Instability) for 5 seconds...")
            sig_50 = make_signal(50.0, 5.0)
            for s in sig_50:
                ts = t0 + sample_index / 120.0
                analyzer.ingest(float(s), 0.0, ts=ts)
                sample_index += 1
                time.sleep(1.0 / 120.0)

            print(">> Loop restart")

            # Drain fast queue to prevent memory leak since Brain worker isn't running
            while not fq.empty():
                fq.get_nowait()

    except KeyboardInterrupt:
        print("\nStopping UI stream.")


if __name__ == "__main__":
    run_stream()
