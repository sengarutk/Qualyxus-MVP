"""Decision-threshold calibration from good parts only.

Conformal (split) calibration gives a distribution-free guarantee: if the held-out good
scores and future good parts are exchangeable, then with threshold

    t = the ceil((n + 1) * (1 - alpha))-th smallest held-out good score

the probability that a future *good* part is rejected is at most ``alpha``. This needs
``n >= 1/alpha - 1`` held-out good images; with fewer, no such bound exists and the
fallback rule (max good score plus a margin) is reported as unguaranteed.

That guarantee is *marginal*: it holds on average over calibration draws, and one
deployment's realised false-reject rate - distributed Beta(n + 1 - k, k) for the k-th order
statistic - can land well above ``alpha`` when n is small (threshold = max of 100 scores
exceeds 3% FRR in ~5% of deployments). With ``confidence`` set, the threshold is instead
the smallest order statistic whose realised FRR is <= ``alpha`` with that probability
(a training-conditional / PAC bound), which needs more held-out images. Either way the
95% upper bound on this deployment's realised FRR is reported.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import beta as beta_dist


def _realised_frr_cdf(alpha: float, n: int, k: int) -> float:
    """P(realised FRR <= alpha) for threshold = k-th smallest of n exchangeable good scores."""
    return float(beta_dist.cdf(alpha, n + 1 - k, k))


def min_holdout(target_frr: float, confidence: float | None = None) -> int:
    if confidence is None:
        return int(math.ceil(1.0 / target_frr - 1.0))
    # k = n (the max) is the most conservative order statistic: P(FRR <= a) = 1 - (1 - a)^n.
    return int(math.ceil(math.log(1.0 - confidence) / math.log(1.0 - target_frr)))


def conformal_threshold(good_scores: Sequence[float], target_frr: float = 0.01,
                        fallback_margin: float = 0.02, confidence: float | None = None) -> dict[str, float | bool]:
    scores = np.sort(np.asarray(good_scores, dtype=np.float64))
    n = int(scores.size)
    if n == 0:
        raise ValueError("Need held-out good scores to calibrate a threshold")
    alpha = float(target_frr)
    if confidence is not None and not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if confidence is None:
        k = math.ceil((n + 1) * (1.0 - alpha))
    else:
        k = next((j for j in range(1, n + 1) if _realised_frr_cdf(alpha, n, j) >= confidence), n + 1)
    if k <= n:
        threshold = float(scores[k - 1])
        # Ties at the threshold would be rejected by a strict ">" comparison: nudge just above.
        threshold = float(np.nextafter(threshold, np.inf))
        guaranteed = True
        frr_upper_95 = float(beta_dist.ppf(0.95, n + 1 - k, k))
    else:
        threshold = float(scores[-1] * (1.0 + fallback_margin) if scores[-1] > 0 else scores[-1] + fallback_margin)
        guaranteed = False
        frr_upper_95 = None
    return {
        "threshold": threshold,
        "target_frr": alpha,
        "confidence": confidence,
        "frr_guaranteed": guaranteed,
        "frr_upper_bound_95": frr_upper_95,
        "n_holdout": n,
        "min_holdout_for_guarantee": min_holdout(alpha, confidence),
        "holdout_mean": float(scores.mean()),
        "holdout_std": float(scores.std()),
        "holdout_max": float(scores[-1]),
    }
