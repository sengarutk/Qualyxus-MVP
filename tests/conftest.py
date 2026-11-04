from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def simulated_rows():
    """Labelled per-cycle feature rows from the simulator (labels = ground truth)."""
    from src.process.features import CycleFeatureExtractor
    from src.process.simulator import InjectionMoldingSimulator

    def _make(n: int, seed: int = 11):
        sim = InjectionMoldingSimulator(seed=seed, start_ts=1_000_000.0)
        extractor = CycleFeatureExtractor()
        rows = []
        for _ in range(n):
            cycle, outcome = sim.next_cycle()
            rows.append({
                "cycle_id": cycle.cycle_id,
                "end_ts": cycle.end_ts,
                "features": extractor.extract(cycle),
                "label": int(outcome.is_defective),
                "defect_type": outcome.defect_type,
                "cycle": cycle,
                "outcome": outcome,
            })
        return rows

    return _make
