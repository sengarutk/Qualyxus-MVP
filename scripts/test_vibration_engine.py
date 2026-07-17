"""
Verification suite for src/inference/analysis/vibration_engine.py  v5
======================================================================

Changes vs. v4
--------------
- _verbose_feed(): prints every ingest() return dict for the first 600 samples
  so the exact sample# where PREWARMING -> Diagnostic transition is visible.
- _guard(): checks for EXCEPTION, CRITICAL_FLOW_FAILURE, and INTERNAL_NULL.
- 800-sample feeds for ample FFT window headroom.
- _feed() keys on "dominant_freq" presence.

Run from project root:
    python scripts/test_vibration_engine.py
"""

from __future__ import annotations

import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.analysis.vibration_engine import VibrationAnalyzer, _STRIDE

BUFFER_SIZE  = 512
TEST_DB      = "data/knowledge/knowledge_base.db"
TOL_HZ       = 2.0
LATENCY_MS   = 1.5
FEED_SAMPLES = 700    # 512 fill + 188 → (188 // 32) = 5 additional FFT windows

_errors: list[str] = []
PASS, FAIL = "✅", "❌"


def _assert(cond: bool, msg: str) -> None:
    sym = PASS if cond else FAIL
    print(f"     {sym} {msg}")
    if not cond:
        _errors.append(msg)


# Status values that indicate a code-path failure (not normal data-flow statuses)
_FAIL_STATUSES = {"EXCEPTION", "CRITICAL_FLOW_FAILURE", "INTERNAL_NULL"}


def _guard(r: dict, sample_num: int | None = None) -> bool:
    """Return False and print details if r contains a failure status."""
    if not r:
        return True
    status = r.get("status", "")
    if status in _FAIL_STATUSES:
        loc = f" (sample {sample_num})" if sample_num is not None else ""
        print(f"     {FAIL} {status} hit{loc}: {r.get('error', r.get('msg', ''))}")
        if "traceback" in r:
            print(r["traceback"])
        _errors.append(f"{status}: {r.get('error', r.get('msg', ''))}{loc}")
        return False
    return True


def _no_exception(r: dict) -> bool:
    """Legacy alias."""
    return _guard(r)


def _make_az(fq: queue.Queue | None = None) -> VibrationAnalyzer:
    return VibrationAnalyzer(buffer_size=BUFFER_SIZE, kap_db_path=TEST_DB, fast_queue=fq)


def _make_sine(hz: float, n: int = FEED_SAMPLES, fs: float = 120.0) -> np.ndarray:
    t = np.arange(n) / fs
    # Add DC offset so signal stays > 0. np.hypot(x,0) = abs(x).
    # If x crosses 0, it rectifies the sine wave and DOUBLES the frequency.
    return 100.0 + 50.0 * np.sin(2 * np.pi * hz * t)


def _feed(az: VibrationAnalyzer, signal: np.ndarray, fs: float = 120.0) -> dict | None:
    """Feed signal dynamically with perfect ts; return last true diagnostic."""
    last_diag = None
    t0 = time.perf_counter()
    dt = 1.0 / fs
    for i, s in enumerate(signal):
        r = az.ingest(float(s), 0.0, ts=t0 + i * dt)
        if not _guard(r, sample_num=i + 1):
            return None
        if r and r.get("status") == "OK":
            last_diag = r
    return last_diag


def _verbose_feed(az: VibrationAnalyzer, signal: np.ndarray, fs: float = 120.0, max_verbose: int = 600) -> dict | None:
    """
    Feed signal with per-sample console output for the first max_verbose samples.
    Shows the exact sample# where PREWARMING transitions to a Diagnostic.
    """
    last_diag = None
    t0 = time.perf_counter()
    dt = 1.0 / fs
    for i, s in enumerate(signal):
        r = az.ingest(float(s), 0.0, ts=t0 + i * dt)
        if not _guard(r, sample_num=i + 1):
            return None

        if i < max_verbose:
            status = r.get("status", "DIAGNOSTIC")
            if r.get("status") == "OK":
                print(
                    f"  sample {i+1:4d} -> DIAGNOSTIC "
                    f"freq={r['dominant_freq']:.3f} Hz  "
                    f"causal={r.get('causal_link','?')}  "
                    f"iso={r.get('iso_status','?')}"
                )
                print(
                    f"  CALIBRATION LOG: "
                    f"Detected: {r['dominant_freq']:.2f} Hz | "
                    f"f_s: {r.get('f_s_hz', '?'):.2f} Hz | "
                    f"buffered: {r.get('buffered','?')}/{r.get('required','?')}"
                )
            elif status == "PREWARMING":
                if (i + 1) % 64 == 0:   # suppress flood; print every 64th
                    print(f"  sample {i+1:4d} -> PREWARMING ({r.get('buffered','?')}/512 buffered)")
            else:
                print(f"  sample {i+1:4d} -> {status}  {r}")

        if r and r.get("status") == "OK":
            last_diag = r
    return last_diag


# ---------------------------------------------------------------------------
# Test 0 — PREWARMING status before buffer is full
# ---------------------------------------------------------------------------
def test_prewarming():
    print("\n[Test 0] PREWARMING status before buffer fills")
    az = _make_az()
    r = az.ingest(0.5, 0.0)
    _assert(r.get("status") == "PREWARMING", f"status={r.get('status')!r} (expected PREWARMING)")
    _assert("buffered" in r and "required" in r, "dict has 'buffered' and 'required' keys")


# ---------------------------------------------------------------------------
# Test 1 — 25 Hz → Spindle_Misalignment
# ---------------------------------------------------------------------------
def test_25hz():
    print("\n[Test 1] 25 Hz sine -> Spindle_Misalignment  (700 samples, truth table: 15-35 Hz)")
    az = _make_az()
    diag = _verbose_feed(az, _make_sine(25.0))

    _assert(diag is not None, "At least one diagnostic produced in 800 samples")
    if diag:
        print(f"          FULL DIAG: {diag}")
        _assert(diag.get("status") == "OK", f"status={diag.get('status')!r} (expected OK)")
        f   = diag["dominant_freq"]
        err = abs(f - 25.0)
        print(f"          detected={f:.3f} Hz  target=25.000 Hz  error={err:.3f} Hz  tol=\u00b1{TOL_HZ}")
        print(f"          f_s_hz={diag.get('f_s_hz', 'MISSING')}  rms={diag.get('rms', 'MISSING')}")
        _assert(err <= TOL_HZ,    f"dominant_freq within \u00b1{TOL_HZ} Hz")
        _assert(diag["causal_link"] == "Spindle_Misalignment",
                f"causal_link={diag['causal_link']!r}")
        _assert(len(diag["top_peaks"]) == 3, f"top_peaks count={len(diag['top_peaks'])}")
        _assert("iso_status" in diag,        "iso_status present")
        _assert(f >= 0.5,                    "dominant_freq not 0 Hz (DC guard)")
        _assert("f_s_hz" in diag,           "f_s_hz present in diagnostic")


# ---------------------------------------------------------------------------
# Test 2 — 50 Hz → Bearing_Instability
# ---------------------------------------------------------------------------
def test_50hz():
    print("\n[Test 2] 50 Hz sine -> Bearing_Instability  (700 samples, truth table: 35-60 Hz)")
    az = _make_az()
    az.reset()
    diag = _verbose_feed(az, _make_sine(50.0))

    _assert(diag is not None, "At least one diagnostic produced")
    if diag:
        print(f"          FULL DIAG: {diag}")
        _assert(diag.get("status") == "OK", f"status={diag.get('status')!r} (expected OK)")
        f   = diag["dominant_freq"]
        err = abs(f - 50.0)
        print(f"          detected={f:.3f} Hz  target=50.000 Hz  error={err:.3f} Hz  tol=\u00b1{TOL_HZ}")
        print(f"          f_s_hz={diag.get('f_s_hz', 'MISSING')}")
        _assert(err <= TOL_HZ, f"dominant_freq within \u00b1{TOL_HZ} Hz")
        _assert(diag["causal_link"] == "Bearing_Instability",
                f"causal_link={diag['causal_link']!r}")
        _assert("f_s_hz" in diag, "f_s_hz present in diagnostic")


# ---------------------------------------------------------------------------
# Test 3 — Latency benchmark (< 1.5 ms)
# ---------------------------------------------------------------------------
def test_latency():
    print("\n[Test 3] analyze_mechanical_state() mean latency < 1.5 ms")
    az   = _make_az()
    ts0  = time.perf_counter()
    t    = np.arange(BUFFER_SIZE) / 120.0
    sig  = 50.0 * np.sin(2 * np.pi * 25.0 * t)

    # Seed the (u, v, ts) buffer directly via public ingest to match the new model.
    dt   = 1.0 / 120.0
    fake_ts = time.perf_counter()
    for i, s in enumerate(sig):
        az.buffer.append((float(s), 0.0, fake_ts + i * dt))
    az._buffer_filled = True          # pretend buffer is "known full"

    az.analyze_mechanical_state()     # warm-up

    REPS   = 200
    t0     = time.perf_counter()
    for _  in range(REPS):
        az.analyze_mechanical_state()
    mean_ms = (time.perf_counter() - t0) * 1000.0 / REPS

    print(f"          → mean latency={mean_ms:.4f} ms  budget={LATENCY_MS} ms")
    _assert(mean_ms < LATENCY_MS, f"latency {mean_ms:.4f} ms < {LATENCY_MS} ms")


# ---------------------------------------------------------------------------
# Test 4 — RMS spike → VIBRATION_CRITICAL emitted BEFORE DB lookup
# ---------------------------------------------------------------------------
def test_rms_spike():
    print("\n[Test 4] RMS spike (>300% baseline) → VIBRATION_CRITICAL on fast_queue")
    fq: queue.Queue = queue.Queue(maxsize=50)
    az = _make_az(fq=fq)
    rng = np.random.default_rng(42)

    # Build quiet baseline over many strides (buffer + 30 × stride)
    n_quiet = BUFFER_SIZE + _STRIDE * 30
    quiet   = rng.uniform(-0.001, 0.001, n_quiet)
    t0      = time.perf_counter()
    for i, s in enumerate(quiet):
        r = az.ingest(float(s), 0.0, ts=t0 + i / 120.0)
        if not _no_exception(r):
            return

    # Drain spurious enqueues from baseline
    while not fq.empty():
        fq.get_nowait()

    # Inject 1000× burst to force spike
    t     = np.arange(BUFFER_SIZE + _STRIDE) / 120.0
    burst = 1000.0 + 1000.0 * np.sin(2 * np.pi * 25.0 * t)
    t1    = t0 + n_quiet / 120.0
    for i, s in enumerate(burst):
        r = az.ingest(float(s), 0.0, ts=t1 + i / 120.0)
        if not _no_exception(r):
            return

    _assert(not fq.empty(), "VIBRATION_CRITICAL enqueued on RMS spike")
    if not fq.empty():
        payload = fq.get_nowait()
        _assert(payload.get("type") == "VIBRATION_CRITICAL",
                f"queue payload type={payload.get('type')!r} (expected VIBRATION_CRITICAL)")
        print(f"          → payload keys: {list(payload.keys())}")


# ---------------------------------------------------------------------------
# Test 5 — JSON serialisability
# ---------------------------------------------------------------------------
def test_json():
    print("\n[Test 5] Diagnostic is JSON-serialisable")
    az   = _make_az()
    diag = _feed(az, _make_sine(25.0))
    _assert(diag is not None, "Diagnostic produced")
    if diag:
        try:
            json.dumps(diag)
            _assert(True, "json.dumps() succeeded")
        except (TypeError, ValueError) as exc:
            _assert(False, f"json.dumps() raised: {exc}")


# ---------------------------------------------------------------------------
# Test 6 — DC detrending: dominant_freq not 0 Hz despite large DC offset
# ---------------------------------------------------------------------------
def test_dc():
    print("\n[Test 6] DC detrending: dominant_freq ≥ 0.5 Hz despite large DC offset")
    az  = _make_az()
    t   = np.arange(FEED_SAMPLES) / 120.0
    sig = 100.0 + 0.5 * np.sin(2 * np.pi * 25.0 * t)
    diag = _feed(az, sig)

    _assert(diag is not None, "Diagnostic produced")
    if diag:
        f = diag["dominant_freq"]
        print(f"          → dominant_freq={f:.3f} Hz (must be ≥ 0.5)")
        _assert(f >= 0.5, f"dominant_freq={f:.3f} not 0 Hz")


# ---------------------------------------------------------------------------
# Test 7 — Rolling reflex fires multiple windows in 800 samples
# ---------------------------------------------------------------------------
def test_rolling_reflex():
    print("\n[Test 7] Rolling reflex: multiple FFT windows in 800 samples")
    az         = _make_az()
    fire_count = 0
    t0 = time.perf_counter()
    for i, s in enumerate(_make_sine(25.0, n=FEED_SAMPLES)):
        r = az.ingest(float(s), 0.0, ts=t0 + i / 120.0)
        if not _no_exception(r):
            return
        if r and r.get("status") == "OK":
            fire_count += 1

    # fire 1 at sample 512, then every 32: 512,544,576,...800
    # fires at: 512 + k*32 for k=0..9  but last is at 512+8*32=768 ≤ 800, and 512+9*32=800 = last sample
    # expected = (800-512)//32 + 1 = 9+1 = 10 (if 800th sample triggers)
    # safe floor: at least (800-512)//32 = 9
    expected_min = (FEED_SAMPLES - BUFFER_SIZE) // _STRIDE
    print(f"          → FFT windows fired: {fire_count}  expected: ≥{expected_min}")
    _assert(fire_count >= expected_min,
            f"rolling window fired {fire_count}× (expected ≥{expected_min})")


# ---------------------------------------------------------------------------
# Test 8 — EXCEPTION status is surfaced (not a silent None return)
# ---------------------------------------------------------------------------
def test_exception_visibility():
    print("\n[Test 8] EXCEPTION surfaced: analyze_mechanical_state never returns None on error")
    az = _make_az()
    # Temporarily corrupt the buffer with non-numeric data to force an exception.
    for _ in range(BUFFER_SIZE):
        az.buffer.append((1.0, 0.0, time.perf_counter()))
    az._buffer_filled = True

    # Corrupt the hann window to force a shape mismatch.
    az._hann = np.ones(1)   # wrong shape → rfft will still run but result shape changes
    result   = az.analyze_mechanical_state()

    _assert(result is not None, "analyze_mechanical_state returns a dict (not None/raises)")
    if result:
        # Either it succeeded (shape still OK) or it's an EXCEPTION dict — both are acceptable.
        is_valid = ("dominant_freq" in result) or (result.get("status") == "EXCEPTION")
        _assert(is_valid, f"result is a valid diagnostic or EXCEPTION dict (keys: {list(result.keys())})")


# ---------------------------------------------------------------------------
# Test 9 — ISO 10816 Thresholding and Reasoning Matrix
# ---------------------------------------------------------------------------
def test_iso_thresholding():
    print("\n[Test 9] ISO 10816 Thresholding (Group 2 Rigid, Target RMS=5.0 mm/s -> ZONE_D)")
    az = VibrationAnalyzer(buffer_size=BUFFER_SIZE, kap_db_path=TEST_DB,
                           machine_group=2, foundation_type="Rigid")
    
    # We want an exact AC RMS of 5.0. A pure AC sine A*sin(t) has RMS = A/sqrt(2).
    # A = 5.0 * sqrt(2) = 7.0710678. The 100.0 DC offset is detrended out.
    A = 5.0 * np.sqrt(2)
    t = np.arange(FEED_SAMPLES) / 120.0
    sig = 100.0 + A * np.sin(2 * np.pi * 50.0 * t)
    
    diag = _feed(az, sig)
    _assert(diag is not None, "Diagnostic produced")
    if diag:
        _assert(diag.get("status") == "OK", f"status={diag.get('status')}")
        _assert(diag.get("iso_status") == "ZONE_D", f"iso_status={diag.get('iso_status')}")
        _assert(diag.get("causal_link") == "Bearing_Instability", f"causal_link={diag.get('causal_link')}")
        
        just = diag.get("justification", "")
        print(f"          → Justification: {just}")
        _assert("Zone D" in just, "Justification string names Zone D")
        _assert("limit of 4.5" in just, "Justification string cites correct 4.5 mm/s threshold")


# ---------------------------------------------------------------------------
# Test 10 — Phase 5 Atomic Verification
# ---------------------------------------------------------------------------
def test_phase_5_atomic():
    print("\n[Test 10] Phase 5 Atomic Verification (Scaling = 2.0, Signal = 5.0 px)")
    az = VibrationAnalyzer(
        buffer_size=BUFFER_SIZE, kap_db_path=TEST_DB,
        machine_group=2, foundation_type="Rigid",
        pixel_to_mm_scale=2.0
    )
    
    # 5.0 pixel displacement RMS means Amplitude A = 5.0 * sqrt(2)
    A = 5.0 * np.sqrt(2)
    t = np.arange(FEED_SAMPLES) / 120.0
    sig = 100.0 + A * np.sin(2 * np.pi * 50.0 * t)
    
    diag = _feed(az, sig)
    _assert(diag is not None, "Diagnostic produced")
    if diag:
        _assert(diag.get("status") == "OK", f"status={diag.get('status')}")
        
        rms = diag.get("rms", 0.0)
        err = abs(rms - 10.0)
        print(f"          → detected RMS={rms:.3f} mm/s  (target=10.000, error={err:.3f})")
        _assert(err <= 0.1, f"RMS matches scaled amplitude (10.0 mm/s)")
        
        _assert(diag.get("iso_status") == "ZONE_D", f"iso_status={diag.get('iso_status')}")
        
        just = diag.get("justification", "")
        print(f"          → Justification: {just}")
        _assert("Unknown Vibration" in just or diag.get("causal_link") in just, f"causal link present in justification: {just}")
        _assert("limit of 4.5" in just, "justification string cites 4.5 mm/s limit")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print(" Qualyxus — Vibration Engine Verification Suite v4")
    print("=" * 65)

    test_prewarming()
    test_25hz()
    test_50hz()
    test_latency()
    test_rms_spike()
    test_json()
    test_dc()
    test_rolling_reflex()
    test_exception_visibility()
    test_iso_thresholding()
    test_phase_5_atomic()

    print("\n" + "=" * 65)
    if _errors:
        print(f"RESULT: {len(_errors)} assertion(s) FAILED:")
        for e in _errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("✅  All vibration engine tests passed.")
    print("=" * 65)
