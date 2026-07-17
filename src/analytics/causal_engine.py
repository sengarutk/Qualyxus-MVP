from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Any

import numpy as np


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


@dataclass(frozen=True)
class CausalScoreConfig:
    min_repeatability_count: int = 2


class CausalInferenceEngine:
    """Additive causal confidence layer over existing RCA signals."""

    def __init__(self, config: CausalScoreConfig | None = None):
        self.config = config or CausalScoreConfig()

    def score(
        self,
        *,
        correlations: dict[str, Any],
        temporal_rca: dict[str, Any],
        control_limits: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        factor_confidence: dict[str, float] = {}
        factor_details: list[dict[str, Any]] = []

        top_candidates: list[tuple[str, str, dict[str, Any]]] = []
        for dimension, rows in (correlations or {}).items():
            if not isinstance(rows, list) or not rows:
                continue
            top = rows[0]
            factor_name = str(top.get(dimension, top.get("value", "unknown")))
            top_candidates.append((dimension, factor_name, top))

        temporal_by_signal = {
            str(item.get("signal")): item for item in (temporal_rca or {}).get("insights", []) or []
        }
        control_by_signal = {
            key: value
            for key, value in ((control_limits or {}).get("metrics", {}) or {}).items()
            if isinstance(value, dict)
        }

        for dimension, factor_name, top in top_candidates:
            support = float(top.get("defect_rate", 0.0) or 0.0)
            lift = float(top.get("lift_ratio", 1.0) or 1.0)
            correlation_strength = float(np.clip(((lift - 1.0) / 2.0) + (0.35 * support), 0.0, 1.0))

            temporal_signal = temporal_by_signal.get(dimension.replace("_id", "")) or temporal_by_signal.get(dimension)
            temporal_alignment = 0.0
            repeatability = 0.0
            statistical_significance = 0.0
            if temporal_signal:
                corr = abs(float(temporal_signal.get("correlation", 0.0) or 0.0))
                drift_corr = abs(float(temporal_signal.get("drift_correlation", 0.0) or 0.0))
                temporal_alignment = float(np.clip(max(corr, drift_corr), 0.0, 1.0))
                repeatability = 1.0 if float(temporal_signal.get("spike_defect_delta", 0.0) or 0.0) > 0 else 0.5

            control_signal = control_by_signal.get(dimension.replace("_id", "")) or control_by_signal.get(dimension)
            if control_signal:
                outside_rate = float(control_signal.get("defect_rate_outside_control_limits", 0.0) or 0.0)
                trend_violations = float(control_signal.get("trend_violations", 0) or 0)
                repeatability = max(repeatability, float(np.clip(outside_rate + (0.1 * trend_violations), 0.0, 1.0)))

            sample_size = max(int(top.get("defect_count", 0) or 0), 1)
            z_like = max(lift - 1.0, 0.0) * np.sqrt(sample_size)
            p_value = float(max(1e-6, 2.0 * (1.0 - _norm_cdf(abs(z_like)))))
            statistical_significance = float(np.clip(1.0 - p_value, 0.0, 1.0))

            confidence = float(
                np.clip(
                    (0.35 * correlation_strength)
                    + (0.25 * temporal_alignment)
                    + (0.20 * repeatability)
                    + (0.20 * statistical_significance),
                    0.0,
                    1.0,
                )
            )
            factor_key = f"{dimension}:{factor_name}"
            factor_confidence[factor_key] = round(confidence, 4)
            factor_details.append(
                {
                    "factor": factor_name,
                    "dimension": dimension,
                    "confidence": round(confidence, 4),
                    "components": {
                        "correlation_strength": round(correlation_strength, 4),
                        "temporal_alignment": round(temporal_alignment, 4),
                        "repeatability": round(repeatability, 4),
                        "statistical_significance": round(statistical_significance, 4),
                    },
                    "statement": f"High-confidence root cause: {factor_name} ({confidence:.2f})" if confidence >= 0.7 else f"Candidate root cause: {factor_name} ({confidence:.2f})",
                }
            )

        factor_details.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        return {
            "factor_confidence": factor_confidence,
            "ranked_factors": factor_details[:10],
        }


def score_causal_factors(
    *,
    correlations: dict[str, Any],
    temporal_rca: dict[str, Any],
    control_limits: dict[str, Any] | None = None,
    config: CausalScoreConfig | None = None,
) -> dict[str, Any]:
    return CausalInferenceEngine(config=config).score(
        correlations=correlations,
        temporal_rca=temporal_rca,
        control_limits=control_limits,
    )