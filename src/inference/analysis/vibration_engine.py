"""
Forensic Vibration Analysis & Macro-Drift Kernel
=================================================

Ingests per-frame (u, v) optical-flow vectors, maintains a 512-sample
rolling deque of (u, v, timestamp) tuples, and executes a Hanning-windowed
Real-FFT every 32 new samples once the buffer is full.  The PSD is
classified against ISO 10816-3 severity zones and cross-referenced with
the KAP SQLite database for causal labelling.

Design Constraints
------------------
- All numerical operations are vectorised NumPy — no Python-level for-loops
  in the hot path.
- The sampling rate is derived dynamically from stored perf_counter
  timestamps, making the FFT frequency axis accurate despite OS jitter.
- DC offset is removed (zero-centring) before windowing to prevent the
  0 Hz spike from masking machine harmonics.
- analyze_mechanical_state() is wrapped in a full try/except with traceback
  emission so silent crashes are never invisible.
- VIBRATION_CRITICAL is emitted to fast_queue BEFORE the slow DB lookup.
"""

from __future__ import annotations

import collections
import logging
import os
import queue
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Any

import sys
import json
import threading

import numpy as np

# Ensure narrator is available for Phase 5
try:
    from src.analytics.vlm_narrator import VLMNarrator
except ImportError:
    try:
        from analytics.vlm_narrator import VLMNarrator
    except ImportError:
        VLMNarrator = None

from src.fleet.historian import PredictiveHistorian

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_KAP_SCHEMA = """
CREATE TABLE IF NOT EXISTS mechanical_nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fault_label TEXT    NOT NULL,
    freq_low    REAL    NOT NULL,
    freq_high   REAL    NOT NULL,
    causal_link TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS iso_10816_limits (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_group     INTEGER NOT NULL,
    foundation_type   TEXT    NOT NULL,
    zone_c_threshold  REAL    NOT NULL,
    zone_d_threshold  REAL    NOT NULL
);
"""

_KAP_SEED_NODES: list[tuple[str, float, float, str]] = [
    ("Rotor_Unbalance",       0.5,  15.0, "Rotor_Unbalance"),
    ("Spindle_Misalignment", 15.0,  35.0, "Spindle_Misalignment"),
    ("Bearing_Instability",  35.0,  60.0, "Bearing_Instability"),
]

_ISO_LIMITS_SEED: list[tuple[int, str, float, float]] = [
    (1, "Rigid",    4.5,  7.1),
    (2, "Rigid",    2.8,  4.5),
    (1, "Flexible", 7.1, 11.0),
    (2, "Flexible", 4.5,  7.1),
]


_ZONE_ORDER = {
    "UNKNOWN": 0,
    "ZONE_A_B": 1,
    "ZONE_C": 2,
    "ZONE_D": 3,
    "DANGER": 4
}

_ISO_ZONES = [
    (1.12, "ZONE_A_B"),
    (2.8, "ZONE_C"),
    (7.1, "ZONE_D"),
    (11.2, "DANGER")
]

_STRIDE           = 32      # trigger FFT every N new samples once full
_RMS_SPIKE_FACTOR = 3.0     # RMS > N × baseline → VIBRATION_CRITICAL
_RMS_MA_LEN       = 16      # moving-average history depth (strides)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _zone_severity(zone: str) -> int:
    return _ZONE_ORDER.get(zone, 0)


def _iso_status(psd_peak: float) -> str:
    for threshold, label in _ISO_ZONES:
        if psd_peak < threshold:
            return label
    return "DANGER"


_KAP_INIT_LOCK = threading.Lock()


def _ensure_kap_db(db_path: Path) -> None:
    """
    Create knowledge_base.db and seed the reference tables when they are empty.

    Seeding is idempotent: tables are created if missing and only populated when empty,
    so concurrent analyzers (one per camera, re-created on worker restarts) never observe
    a dropped table mid-query and operator-curated KAP rows are preserved.
    """
    os.makedirs(str(db_path.parent), exist_ok=True)
    with _KAP_INIT_LOCK, sqlite3.connect(str(db_path), timeout=5.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.executescript(_KAP_SCHEMA)

        if conn.execute("SELECT COUNT(*) FROM mechanical_nodes").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO mechanical_nodes "
                "(fault_label, freq_low, freq_high, causal_link) "
                "VALUES (?, ?, ?, ?)",
                _KAP_SEED_NODES,
            )
        if conn.execute("SELECT COUNT(*) FROM iso_10816_limits").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO iso_10816_limits "
                "(machine_group, foundation_type, zone_c_threshold, zone_d_threshold) "
                "VALUES (?, ?, ?, ?)",
                _ISO_LIMITS_SEED,
            )
        conn.commit()  # atomic commit
        logger.info(
            "[VibrationEngine] KAP DB ready. WAL active. ISO seeds: %d",
            len(_ISO_LIMITS_SEED)
        )


# ---------------------------------------------------------------------------
# VibrationAnalyzer
# ---------------------------------------------------------------------------

class VibrationAnalyzer:
    """
    Rolling DSP kernel:  optical-flow (u, v) → PSD → ISO 10816 verdict.

    The internal buffer stores (u, v, timestamp) tuples so that the
    dynamic sampling rate can be computed directly from the data stream
    without a separate timestamp deque.
    """

    def __init__(
        self,
        *,
        buffer_size: int = 512,
        kap_db_path: str | Path = "data/knowledge/knowledge_base.db",
        fast_queue: queue.Queue | None = None,
        machine_group: int = 2,
        foundation_type: str = "Rigid",
        pixel_to_mm_scale: float = 1.0,
        camera_id: str = "cam_unknown",
        calibrated: bool = True,
    ) -> None:
        self.buffer_size       = buffer_size
        self.machine_group     = machine_group
        self.foundation_type   = foundation_type
        self.pixel_to_mm_scale = pixel_to_mm_scale
        self.camera_id         = camera_id
        # ISO 10816 limits are in mm/s of a physical vibration velocity. Uncalibrated input
        # (e.g. raw pixel displacement) is analysed spectrally but never classified into
        # ISO zones and never allowed to raise the advisory PLC flag.
        self.calibrated        = bool(calibrated)
        self.buffer_lock = threading.Lock()
        self.buffer: collections.deque[tuple[float, float, float]] = collections.deque(
            maxlen=buffer_size
        )
        self._fast_queue   = fast_queue
        self._kap_db_path  = Path(kap_db_path)
        _ensure_kap_db(self._kap_db_path)
        
        # Phase 5: Voice Singleton integration
        self.narrator = self._resolve_narrator()

        self.cache_lock = threading.Lock()
        self._forensic_cache = {}

        # Phase 7: Industrial Control PLC Bridge

        # Phase 8: Predictive Historian
        from src.core.factory_manager import FactoryManager
        hist_instance = FactoryManager().get_instance("PredictiveHistorian")
        if hist_instance is None or hist_instance.__class__.__name__ == "SafeServiceStub":
            from src.fleet.historian import PredictiveHistorian
            self.historian = PredictiveHistorian()
        else:
            self.historian = hist_instance
        self._last_history_snapshot = 0.0

        # Pre-compute Hanning window (fixed buffer length).
        self._hann: np.ndarray = np.hanning(buffer_size).astype(np.float64)

        # Sliding-window state.
        self._samples_since_fft: int  = 0
        self._buffer_filled:     bool = False

        # RMS moving-average history.
        self._rms_history: collections.deque[float] = collections.deque(maxlen=_RMS_MA_LEN)

        # Last diagnostic (for external polling / dashboards).
        self.last_diagnostic: dict[str, Any] | None = None
        
        # Phase 6: Core Factory Config
        self.active_profile = "default_vibration"
        self.ignore_vibration = False
        self.nominal_baseline = None
        
    @staticmethod
    def _resolve_narrator():
        """Reuse the orchestrator's configured narrator; otherwise build one from config."""
        from src.core.factory_manager import FactoryManager
        narrator = FactoryManager().get_instance("narrator")
        if narrator is not None:
            return narrator
        if VLMNarrator is None:
            return None
        try:
            from src.config.config import Config
            vlm_cfg = Config.load().get("vlm_reasoning", default={}) or {}
        except Exception:
            vlm_cfg = {}
        try:
            return VLMNarrator(config=vlm_cfg)
        except Exception as exc:
            logger.warning("[VibrationEngine] VLM narrator unavailable: %s", exc)
            return None

    def load_profile(self, config: dict):
        self.ignore_vibration = config.get("ignore_vibration", False)
        if "nominal_baseline" in config:
            self.nominal_baseline = float(config["nominal_baseline"])
            logger.info("[VibrationEngine] Nominal baseline locked to %.6f mm/s", self.nominal_baseline)
        else:
            self.nominal_baseline = None
            logger.info("[VibrationEngine] Reverting to rolling historical baseline.")
        self.reset()
        
        # Phase 5: Voice Singleton integration
        self.narrator = self._resolve_narrator()

    # ------------------------------------------------------------------
    # Schema helper  —  the single source of truth for key layout
    # ------------------------------------------------------------------

    def _create_base_diagnostic(self, status: str = "PROCESSING") -> dict[str, Any]:
        """
        Return a key-perfect diagnostic dict pre-filled with safe defaults.

        ALL return paths in ingest() and analyze_mechanical_state() must
        start from this template so the test suite never encounters a
        missing-key KeyError regardless of which branch executes.
        """
        with self.buffer_lock:
            buf_len = len(self.buffer)
        return {
            "status":         status,
            "buffered":       buf_len,
            "required":       self.buffer_size,
            "dominant_freq":  0.0,
            "psd_peak":       0.0,
            "iso_status":     "UNKNOWN",
            "causal_link":    "N/A",
            "justification":  "PENDING_ANALYSIS",
            "top_peaks":      [],
            "rms":            0.0,
            "rms_baseline":   0.0,
            "f_s_hz":         0.0,
            "spike_detected": False,
        }

    # ------------------------------------------------------------------
    # Public ingest API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all buffers and history to prevent spectral ghosting across tests."""
        with self.buffer_lock:
            self.buffer.clear()
        self._buffer_filled = False
        self._samples_since_fft = 0
        self._rms_history.clear()
        self.last_diagnostic = None
        with self.cache_lock:
            self._forensic_cache = {}
        if hasattr(self, "_last_telemetry_payload"):
            del self._last_telemetry_payload

    def add_displacement_vector(self, u: float, v: float, ts: float) -> None:
        """Lock-secured helper to append displacement coordinates to the rolling deque buffer."""
        with self.buffer_lock:
            self.buffer.append((u, v, ts))

    def _execute_spectral_estimation(self) -> list[tuple[float, float, float]]:
        """Lock-secured helper to safely extract a point-in-time list snapshot of the deque buffer."""
        with self.buffer_lock:
            return list(self.buffer)

    def _dispatch_asynchronous_reasoning(self, payload: dict, export_path: str | os.PathLike, visual_defect: bool) -> None:
        """
        Completely refactored asynchronous VLM reasoning dispatcher.
        Constructs a proper telemetry_context, maps the defect_type, 
        and calls generate_narrative inside a robust try/except error shield.
        """
        def worker():
            try:
                if self.narrator is None:
                    logger.warning("[VibrationEngine] VLMNarrator is not available/enabled.")
                    return
                telemetry_context = {
                    "machine_id": self.camera_id,
                    "shift_id": "unknown",
                    "temperature": 0.0,
                    "vibration": float(payload.get("rms", 0.0)),
                    "speed": 0.0,
                    "line_id": "unknown",
                    "station_id": self.camera_id,
                    "iso_clause": "Clause 8.5.1 Control of Production" if visual_defect else "Clause 8.7 Non-conforming Outputs"
                }
                
                defect_type = "visual_defect" if visual_defect else "vibration_anomaly"
                
                logger.info("[VibrationEngine] Spawning asynchronous reasoning pass on camera %s...", self.camera_id)
                
                narrative = self.narrator.generate_narrative(
                    image_path=str(export_path),
                    telemetry_context=telemetry_context,
                    defect_type=defect_type
                )
                
                if narrative:
                    logger.info("[VibrationEngine] VLM narrative completed successfully: %s", narrative)
                else:
                    logger.warning("[VibrationEngine] VLM narrator returned empty response or is disabled.")
            except Exception as e:
                logger.error("[VibrationEngine] Asynchronous reasoning pass failed: %s\n%s", e, traceback.format_exc())

        threading.Thread(target=worker, daemon=True).start()

    def ingest(self, u: float, v: float, ts: float = None, visual_defect: bool = False) -> dict:
        self.add_displacement_vector(u, v, ts or time.perf_counter())

        # Gate 1: Wait for initial fill
        with self.buffer_lock:
            buf_len = len(self.buffer)
        if buf_len < self.buffer_size:
            return self._create_base_diagnostic("PREWARMING")

        # Gate 2: The Stride Logic
        self._samples_since_fft += 1
        # Trigger if this is the FIRST fill, OR if we've hit the 32-sample stride
        if (not self._buffer_filled) or (self._samples_since_fft >= 32):
            self._buffer_filled = True
            self._samples_since_fft = 0
            return self.analyze_mechanical_state(visual_defect=visual_defect)

        return self._create_base_diagnostic("BUFFERING")

    # ------------------------------------------------------------------
    # DSP pipeline  —  fully vectorised, never returns None on exception
    # ------------------------------------------------------------------

    def analyze_mechanical_state(self, visual_defect: bool = False) -> dict[str, Any] | None:
        """
        Execute the full DSP pipeline on the current buffer snapshot.

        Steps
        -----
        1. Snapshot buffer → NumPy arrays.
        2. Compute L2 magnitude from (u, v) pairs.
        3. Dynamic f_s from perf_counter timestamps.
        4. Zero-centre (DC removal).
        5. Hanning window.
        6. rfft → one-sided PSD.
        7. Band-limit to [0.5, 60] Hz; dominant peak via argmax.
        8. Top-3 via argpartition.
        9. RMS spike detection → VIBRATION_CRITICAL emitted BEFORE DB call.
        10. ISO 10816 + KAP DB causal lookup.

        Any exception is caught; returns an EXCEPTION dict so callers never
        see a silent failure.
        """
        # Use the shared schema helper — key-perfect from first instruction.
        # Override buffered to reflect real buffer state at call time.
        diag_result: dict[str, Any] = self._create_base_diagnostic("PROCESSING")
        with self.buffer_lock:
            diag_result["buffered"] = len(self.buffer)

        try:
            # --- 1. Snapshot buffer (thread-safe copy) ---
            snapshot: list[tuple[float, float, float]] = self._execute_spectral_estimation()

            if len(snapshot) < self.buffer_size:
                diag_result["status"] = "BUFFER_UNDERRUN"
                return diag_result

            # --- 2. L2 magnitudes ---
            u_flow = np.array([sample[0] for sample in snapshot])
            v_flow = np.array([sample[1] for sample in snapshot])
            sig = np.hypot(u_flow, v_flow) * self.pixel_to_mm_scale

            # --- 3. Total Window f_s Accuracy ---
            t_start = snapshot[0][2]
            t_end   = snapshot[-1][2]
            delta_t = t_end - t_start
            f_s     = (len(snapshot) - 1) / delta_t if delta_t > 1e-6 else 120.0
            
            diag_result["f_s_hz"] = float(f_s)

            # --- 4. Detrending (DC removal) ---
            sig = sig - np.mean(sig)

            # --- 5. Hanning window ---
            sig_win = sig * self._hann

            # --- 6. rfft → one-sided PSD ---
            spectrum   = np.fft.rfft(sig_win)
            win_energy = float(np.sum(self._hann ** 2)) or 1.0
            psd        = (np.abs(spectrum) ** 2) / win_energy

            # --- 7. Frequency axis & band mask [0.5, 60] Hz ---
            # Use self.buffer_size (not len(sig)) to ensure the bin width is
            # always N/f_s regardless of any edge truncation.
            freqs = np.fft.rfftfreq(self.buffer_size, d=1.0 / f_s)
            # Frequencies above Nyquist (f_s / 2) are aliases, not machine harmonics.
            mask  = (freqs >= 0.5) & (freqs <= min(60.0, 0.5 * f_s))
            if not np.any(mask):
                diag_result["status"] = "NO_SIGNAL_IN_BAND"
                return diag_result

            psd_band   = psd[mask]
            freqs_band = freqs[mask]

            # --- 8. Dominant peak (O(N), deterministic) ---
            idx = int(np.argmax(psd_band))
            diag_result["dominant_freq"] = float(freqs_band[idx])
            diag_result["psd_peak"]      = float(psd_band[idx])

            # Top-3 peaks via argpartition (descending)
            n_peaks = min(3, psd_band.size)
            top_idx = np.argpartition(psd_band, -n_peaks)[-n_peaks:]
            top_idx = top_idx[np.argsort(psd_band[top_idx])[::-1]]
            diag_result["top_peaks"] = [
                {"freq": round(float(freqs_band[i]), 3), "psd": round(float(psd_band[i]), 6)}
                for i in top_idx
            ]

            # --- 9. RMS spike detection ---
            rms_current = float(np.sqrt(np.mean(sig ** 2)))
            diag_result["rms"] = round(rms_current, 6)
            
            if self.nominal_baseline is not None:
                rms_baseline = float(self.nominal_baseline)
            else:
                if len(self._rms_history) == 0:
                    self._rms_history.append(rms_current)   # warm-start
                self._rms_history.append(rms_current)
                rms_baseline = float(np.mean(
                    np.fromiter(self._rms_history, dtype=np.float64)
                ))
                
            diag_result["rms_baseline"]   = round(rms_baseline, 6)
            spike = rms_baseline > 0.0 and rms_current > _RMS_SPIKE_FACTOR * rms_baseline
            diag_result["spike_detected"] = spike

            # Emit CRITICAL BEFORE DB lookup — Reflex never gated on Brain latency.
            plc_trigger = False
            if spike and not self.ignore_vibration and self.calibrated:
                self._emit_vibration_critical_fast(
                    dominant_freq=diag_result["dominant_freq"],
                    psd_peak=diag_result["psd_peak"],
                    rms=rms_current,
                    peaks=diag_result["top_peaks"],
                )

            # --- 10. Decoupled ISO 10816 & KAP DB lookup ---
            # Physics is NEVER held hostage by the Knowledge Base.
            # dominant_freq and psd_peak are already written to diag_result above.
            if self.ignore_vibration:
                diag_result["iso_status"]    = "ZONE_A_B"
                diag_result["causal_link"]   = "SURFACE_INSPECTION_ACTIVE"
                diag_result["justification"] = "Mechanical DSP ignored; tracking visual anomalies."
            elif not self.calibrated:
                diag_result["iso_status"]    = "UNCALIBRATED"
                diag_result["causal_link"]   = "N/A"
                diag_result["justification"] = (
                    "Signal is not calibrated to mm/s; spectral trend only, no ISO 10816 zone."
                )
            else:
                try:
                    iso_status, causal_link, just_str = self.get_forensic_context(
                        rms_current, diag_result["dominant_freq"]
                    )
                    diag_result["iso_status"]    = iso_status
                    diag_result["causal_link"]   = causal_link
                    diag_result["justification"] = just_str
                except Exception as db_exc:
                    logger.warning(
                        "[VibrationEngine] KAP lookup failed; physics data preserved. %s", db_exc
                    )
                    diag_result["iso_status"]    = "UNKNOWN"
                    diag_result["causal_link"]   = "KAP_LOOKUP_FAILED"
                    diag_result["justification"] = "DB_ERROR"

            # Advisory only: the reflex path raises a flag, and the plant's PLC decides what to do.
            if diag_result["iso_status"] == "ZONE_D" and not self.ignore_vibration and self.calibrated:
                if not spike:  # a spike already raised the flag above
                    self._emit_vibration_critical_fast(
                        dominant_freq=diag_result["dominant_freq"],
                        psd_peak=diag_result["psd_peak"],
                        rms=rms_current,
                        peaks=diag_result["top_peaks"],
                        iso_status="ZONE_D",
                    )
                plc_trigger = True
                    
            # Phase 8: Historian Snapshot Hook
            curr_time = time.time()
            if (curr_time - self._last_history_snapshot) > 60.0:
                self._last_history_snapshot = curr_time
                self.historian.record_snapshot(
                    rms=rms_current, 
                    dominant_freq=diag_result["dominant_freq"], 
                    iso_status=diag_result["iso_status"]
                )

            diag_result["status"]   = "OK"
            diag_result["buffered"] = self.buffer_size   # schema invariant on OK
            diag_result["required"] = self.buffer_size
            self.last_diagnostic    = diag_result

            # --- 11. Atomic Telemetry Export ---
            export_dir = Path("data/telemetry")
            # Thread-safe creation of telemetry mount 
            os.makedirs(str(export_dir), exist_ok=True)
            export_path = export_dir / f"live_vibration_{self.camera_id}.json"
            tmp_path    = export_path.with_suffix(".tmp")
            
            payload = {
                "active_profile": self.active_profile,
                "status":        diag_result["status"],
                "iso_zone":      diag_result["iso_status"],
                "rms":           diag_result["rms"],
                "dominant_freq": diag_result["dominant_freq"],
                "justification": diag_result["justification"],
                "visual_defect": visual_defect,
                "plc_advisory_flag": plc_trigger,
                "trend_slope": self.historian.current_trend_slope,
                "estimated_rul_days": self.historian.current_rul_days
            }
            if not hasattr(self, "_last_telemetry_payload") or self._last_telemetry_payload != payload:
                self._last_telemetry_payload = payload
                try:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f)
                    # os.replace provides atomic writes on POSIX and Windows (Python 3.3+)
                    os.replace(tmp_path, export_path)
                    
                    # Phase 5: Voice Narrative Loop
                    should_narrate = False
                    if self.ignore_vibration and visual_defect:
                        should_narrate = True
                    elif not self.ignore_vibration and diag_result["iso_status"] in ("ZONE_C", "ZONE_D"):
                        should_narrate = True
                        
                    if should_narrate:
                        self._dispatch_asynchronous_reasoning(payload, export_path, visual_defect)
                        
                except OSError as io_exc:
                    logger.error("[VibrationEngine] Telemetry export failed: %s", io_exc)

            # Single unconditional return at the bottom of the try block.
            return diag_result

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("[VibrationEngine] analyze_mechanical_state EXCEPTION: %s\n%s", exc, tb)
            diag_result["status"]    = "EXCEPTION"
            diag_result["error"]     = str(exc)
            diag_result["traceback"] = tb
            # Still return diag_result: whatever physics was computed is preserved.
            return diag_result

        # Final Sentinel: Guaranteed to return a dict — Python can never fall off the end.
        return diag_result  # type: ignore[return]

    # ------------------------------------------------------------------
    # KAP DB lookup — atomic, never raises
    # ------------------------------------------------------------------

    def get_forensic_context(self, rms: float, f_peak: float) -> tuple[str, str, str]:
        """
        Query the KAP DB for ISO Limits matrix and forensic frequency match.

        Returns (iso_status, causal_link, justification).
        """
        # Cache lookup based on inputs and instance config
        cache_key = (rms, f_peak, self.machine_group, self.foundation_type)
        if hasattr(self, "_forensic_cache") and hasattr(self, "cache_lock"):
            with self.cache_lock:
                if cache_key in self._forensic_cache:
                    return self._forensic_cache[cache_key]

        iso      = "UNKNOWN"
        causal   = "Unknown Vibration"
        just_str = "PENDING_ANALYSIS"

        try:
            with sqlite3.connect(str(self._kap_db_path), timeout=0.020) as conn:
                # -------------------------------------------------------------
                # Step 1: Causal Link Lookup by Frequency
                # -------------------------------------------------------------
                try:
                    row = conn.execute(
                        """
                        SELECT causal_link,
                               ABS((freq_low + freq_high) / 2.0 - ?) AS dist
                        FROM   mechanical_nodes
                        WHERE  freq_low <= ? AND freq_high >= ?
                        ORDER  BY dist ASC
                        LIMIT  1
                        """,
                        (f_peak, f_peak, f_peak),
                    ).fetchone()
                except sqlite3.OperationalError as db_exc:
                    logger.critical("[VibrationEngine] KAP table error. %s", db_exc)
                    return iso, causal, just_str

                if row:
                    causal = str(row[0])
                else:
                    # Nearest-midpoint fallback
                    try:
                        row2 = conn.execute(
                            """
                            SELECT causal_link,
                                   ABS((freq_low + freq_high) / 2.0 - ?) AS dist
                            FROM   mechanical_nodes
                            ORDER  BY dist ASC
                            LIMIT  1
                            """,
                            (f_peak,),
                        ).fetchone()
                        if row2:
                            causal = str(row2[0])
                    except sqlite3.OperationalError:
                        pass

                # -------------------------------------------------------------
                # Step 2: ISO 10816 Broadband limits
                # -------------------------------------------------------------
                iso_row = conn.execute(
                    """
                    SELECT zone_c_threshold, zone_d_threshold
                    FROM iso_10816_limits
                    WHERE machine_group = ? AND foundation_type = ?
                    LIMIT 1
                    """,
                    (self.machine_group, self.foundation_type)
                ).fetchone()

                if iso_row:
                    zone_c, zone_d = iso_row
                    if rms >= zone_d:
                        iso = "ZONE_D"
                        just_str = f"Zone D: {causal} exceeding Group {self.machine_group} {self.foundation_type} limit of {zone_d} mm/s (RMS {rms:.2f} mm/s)."
                    elif rms >= zone_c:
                        iso = "ZONE_C"
                        just_str = f"Zone C: {causal} exceeding Group {self.machine_group} {self.foundation_type} threshold of {zone_c} mm/s (RMS {rms:.2f} mm/s)."
                    else:
                        iso = "ZONE_A_B"
                        just_str = f"Zones A/B: {causal} conditionally acceptable (limit {zone_c} mm/s, RMS {rms:.2f} mm/s)."
                else:
                    just_str = f"ISO_CONFIG_NOT_FOUND_FOR_G{self.machine_group}_{self.foundation_type}"

        except Exception as exc:
            logger.warning("[VibrationEngine] KAP DB connection failed: %s", exc)
            return iso, causal, "DB_CONNECTION_ERROR"

        res = (iso, causal, just_str)
        if hasattr(self, "_forensic_cache") and hasattr(self, "cache_lock"):
            with self.cache_lock:
                if len(self._forensic_cache) >= 1024:
                    # Evict oldest half of the entries to prevent growing without bounds
                    keys = list(self._forensic_cache.keys())
                    for k in keys[:512]:
                        self._forensic_cache.pop(k, None)
                self._forensic_cache[cache_key] = res
        return res

    # ------------------------------------------------------------------
    # Reflex emission helpers
    # ------------------------------------------------------------------

    def _emit_vibration_critical_fast(
        self,
        *,
        dominant_freq: float,
        psd_peak: float,
        rms: float,
        peaks: list[dict[str, float]],
        iso_status: str = "PENDING_KAP",
        causal_link: str = "PENDING_KAP",
    ) -> None:
        """
        non-blocking fast-path emit: fires BEFORE the DB lookup so the Reflex
        path is never gated on Brain latency.  iso_status and causal_link are
        filled in with PENDING_KAP as placeholders until the DB returns.
        """
        if self._fast_queue is None:
            return
        payload = {
            "type":         "VIBRATION_CRITICAL",
            "dominant_freq": round(dominant_freq, 3),
            "psd_peak":      round(psd_peak, 6),
            "iso_status":    iso_status,
            "causal_link":   causal_link,
            "rms":           round(rms, 6),
            "top_peaks":     peaks,
        }
        try:
            self._fast_queue.put_nowait(payload)
        except queue.Full:
            # Defect-storm protection: atomic clear + re-insert latest
            with self._fast_queue.mutex:
                self._fast_queue.queue.clear()
            try:
                self._fast_queue.put_nowait(payload)
            except Exception:
                pass
            logger.warning("[VibrationEngine] REFLEX_OVERLOAD: queue cleared for VIBRATION_CRITICAL.")
