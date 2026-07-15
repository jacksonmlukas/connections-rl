"""Statistical primitives: bootstrap CIs, paired tests, calibration.

Stdlib-only (mirrors gvc-local's percentile bootstrap, without the numpy
dependency) so the eval harness runs anywhere, including CI smoke jobs.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile on pre-sorted values, q in [0, 1]."""
    if not sorted_vals:
        raise ValueError("empty sample")
    idx = q * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile-bootstrap CI of the mean. Returns (mean, lower, upper)."""
    if not values:
        raise ValueError("empty sample")
    rng = random.Random(seed)
    n = len(values)
    mean = sum(values) / n
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return mean, _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI on mean(a - b) resampling PAIRED observations (same puzzles).

    Returns (mean_diff, lower, upper). If the CI excludes 0, the arms differ.
    """
    if len(a) != len(b) or not a:
        raise ValueError("a and b must be equal-length, non-empty, paired samples")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    return bootstrap_ci(diffs, n_resamples=n_resamples, alpha=alpha, seed=seed)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar test p-value on paired binary outcomes.

    ``b`` = puzzles arm A solved and arm B missed; ``c`` = the reverse.
    Two-sided exact binomial on the discordant pairs (appropriate at the
    n<100 discordant-pair counts we expect).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def ece(
    confidences: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """Expected calibration error + per-bin reliability data.

    Returns (ece, bins) where each bin has keys: lo, hi, n, confidence, accuracy.
    Bins with no samples are omitted.
    """
    if len(confidences) != len(outcomes) or not confidences:
        raise ValueError("confidences and outcomes must be equal-length, non-empty")
    bins: list[dict[str, float]] = []
    total = len(confidences)
    err = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        members = [
            (conf, out)
            for conf, out in zip(confidences, outcomes, strict=True)
            if (lo <= conf < hi) or (i == n_bins - 1 and conf == 1.0)
        ]
        if not members:
            continue
        avg_conf = sum(m[0] for m in members) / len(members)
        acc = sum(m[1] for m in members) / len(members)
        err += (len(members) / total) * abs(avg_conf - acc)
        bins.append(
            {"lo": lo, "hi": hi, "n": len(members), "confidence": avg_conf, "accuracy": acc}
        )
    return err, bins
