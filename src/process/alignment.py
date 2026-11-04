"""Link every inspected part back to the machine cycle that produced it.

Defects are *created* at the machine but *seen* at the camera, usually some distance
(and time) downstream. Reading telemetry "now" when the camera sees a defect pairs the
image with the wrong cycle. This module resolves the producing cycle with one of three
strategies, configured per camera:

``part_id``  The line provides an explicit identifier (barcode, DMC, PLC part tracking)
             that equals or maps to the cycle id. Exact; always preferred when available.
``time``     Constant transport delay: the part seen at ``t`` left the machine at about
             ``t - transport_delay_s``. The nearest still-unlinked cycle end inside the
             tolerance window is chosen; confidence decays with the timing error.
``fifo``     Fixed number of parts in transit (e.g. an indexing conveyor holding N parts):
             the part seen now is the N-th most recent completed cycle.

:func:`estimate_transport_delay` calibrates the delay during commissioning from two
timestamp streams, so it does not have to be measured by hand.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.process.store import ProcessStore

logger = logging.getLogger("QualyxusAlignment")


@dataclass(frozen=True)
class StationLink:
    camera_id: str
    machine_id: str
    mode: str = "time"               # "time" | "fifo" | "part_id"
    transport_delay_s: float = 0.0
    tolerance_s: float = 5.0
    fifo_offset: int = 0              # parts between the machine and the camera
    parts_per_cycle: int = 1          # cavities: one shot makes this many parts

    @classmethod
    def from_config(cls, camera_id: str, payload: dict) -> "StationLink":
        mode = str(payload.get("mode", "time")).lower()
        if mode not in {"time", "fifo", "part_id"}:
            raise ValueError(f"Unknown alignment mode '{mode}' for camera {camera_id}")
        parts_per_cycle = max(1, int(payload.get("parts_per_cycle", 1)))
        if parts_per_cycle > 1 and mode == "fifo":
            raise ValueError(f"Camera {camera_id}: fifo alignment assumes one part per cycle; use mode: time")
        return cls(
            camera_id=camera_id,
            machine_id=str(payload.get("machine_id", "imm_1")),
            mode=mode,
            transport_delay_s=float(payload.get("transport_delay_s", 0.0)),
            tolerance_s=max(0.05, float(payload.get("tolerance_s", 5.0))),
            fifo_offset=max(0, int(payload.get("fifo_offset", 0))),
            parts_per_cycle=parts_per_cycle,
        )


@dataclass(frozen=True)
class AlignmentResult:
    cycle_id: str | None
    confidence: float
    method: str
    timing_error_s: float | None = None
    cavity: int = 0                   # which part of this cycle, in inspection order


class CycleAligner:
    def __init__(self, store: ProcessStore, links: Iterable[StationLink]):
        self.store = store
        self.links = {link.camera_id: link for link in links}

    def link_for(self, camera_id: str) -> StationLink | None:
        return self.links.get(camera_id)

    def align(self, camera_id: str, inspection_ts: float, part_id: str | None = None) -> AlignmentResult:
        link = self.links.get(camera_id)
        if link is None:
            return AlignmentResult(None, 0.0, "unlinked")

        if part_id:
            if self.store.get_features(part_id) is not None:
                return AlignmentResult(part_id, 1.0, "part_id")
            if link.mode == "part_id":
                return AlignmentResult(None, 0.0, "part_id_unknown")

        if link.mode == "fifo":
            candidates = self.store.cycles_ending_before(link.machine_id, inspection_ts, link.fifo_offset + 1)
            if len(candidates) <= link.fifo_offset:
                return AlignmentResult(None, 0.0, "fifo_underflow")
            cycle_id, _ = candidates[link.fifo_offset]
            return AlignmentResult(cycle_id, 0.9, "fifo")

        if link.mode == "part_id":
            return AlignmentResult(None, 0.0, "part_id_missing")

        expected_end = float(inspection_ts) - link.transport_delay_s
        candidates = self.store.cycles_between(
            link.machine_id, expected_end - link.tolerance_s, expected_end + link.tolerance_s,
        )
        if not candidates:
            return AlignmentResult(None, 0.0, "time_no_candidate")
        # Each cycle makes parts_per_cycle parts, each inspected once per camera: prefer
        # cycles that still have a part left to link.
        counts = self.store.link_counts(camera_id, [c for c, _ in candidates])
        already = {c for c, n in counts.items() if n >= link.parts_per_cycle}
        free = [(c, t) for c, t in candidates if c not in already] or candidates
        cycle_id, end_ts = min(free, key=lambda item: abs(item[1] - expected_end))
        error = end_ts - expected_end
        confidence = max(0.0, 1.0 - abs(error) / link.tolerance_s)
        if cycle_id in already:
            confidence *= 0.5  # every candidate was already linked: ambiguous
        cavity = counts.get(cycle_id, 0) % link.parts_per_cycle
        return AlignmentResult(cycle_id, round(confidence, 4), "time", round(error, 4), cavity)


def estimate_transport_delay(
    cycle_end_ts: Sequence[float],
    inspection_ts: Sequence[float],
    *,
    max_delay_s: float = 600.0,
    resolution_s: float = 0.05,
) -> dict[str, float]:
    """Estimate the machine->camera transport delay from two timestamp streams.

    For every candidate delay ``d`` each inspection is matched to its nearest cycle end at
    ``t - d``. The delay leaving the fewest unmatched inspections wins (median timing error
    breaks ties). Returns the delay, the residual jitter (median abs error), a suggested
    tolerance and whether the result is ambiguous by whole cycles.
    """
    ends = np.sort(np.asarray(cycle_end_ts, dtype=np.float64))
    seen = np.asarray(inspection_ts, dtype=np.float64)
    if ends.size < 3 or seen.size < 3:
        raise ValueError("Need at least 3 cycles and 3 inspections to estimate the transport delay")

    period = float(np.median(np.diff(ends))) if ends.size > 1 else 1.0
    match_tol = 0.2 * period

    def errors(delay: float) -> np.ndarray:
        target = seen - delay
        idx = np.clip(np.searchsorted(ends, target), 1, ends.size - 1)
        return np.minimum(np.abs(ends[idx] - target), np.abs(ends[idx - 1] - target))

    def objective(delay: float) -> tuple[int, float]:
        # Primary: how many inspections find no cycle end near them. Only irregular gaps
        # (stoppages, a tagged part) separate the true delay from one shifted by a whole
        # cycle, and those show up as mismatches - a median error would ignore them.
        err = errors(delay)
        return int(np.sum(err > match_tol)), float(np.median(err))

    def score(delay: float) -> float:
        mismatches, median_err = objective(delay)
        return mismatches + median_err / max(period, 1e-6)

    # Coarse sweep, then refine around the best coarse candidate.
    coarse_step = max(resolution_s, period / 20.0)
    coarse = np.arange(0.0, max_delay_s + coarse_step, coarse_step)
    best = float(coarse[int(np.argmin([score(d) for d in coarse]))])
    fine = np.arange(max(0.0, best - coarse_step), best + coarse_step + resolution_s, resolution_s)
    delay = float(fine[int(np.argmin([score(d) for d in fine]))])
    best_mismatches, jitter = objective(delay)

    neighbours = [objective(d)[0] for d in (delay - period, delay + period) if 0.0 <= d <= max_delay_s]
    ambiguous = bool(neighbours) and min(neighbours) - best_mismatches < 2
    if ambiguous:
        logger.warning(
            "[Alignment] Transport delay is ambiguous by whole cycles (period %.2fs). Record a "
            "stretch containing a machine stop or a tagged part, or use fifo/part_id alignment.", period,
        )
    if period > 0 and jitter > 0.25 * period:
        logger.warning(
            "[Alignment] Residual jitter %.2fs is large relative to the cycle period %.2fs; "
            "time-based alignment will be unreliable. Prefer part_id or fifo alignment.", jitter, period,
        )
    return {
        "transport_delay_s": round(delay, 3),
        "jitter_s": round(jitter, 3),
        "cycle_period_s": round(period, 3),
        "suggested_tolerance_s": round(min(0.45 * period, max(4.0 * jitter, resolution_s * 4)), 3),
        "ambiguous": ambiguous,
        "unmatched_inspections": best_mismatches,
    }
