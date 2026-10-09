"""Effect sizes and autocorrelation-robust intervals for two independent samples.

Cliff's delta, the Hodges-Lehmann shift, and block bootstrap percentile intervals. Consecutive
30-second CPU readings are autocorrelated (lag-1 r between 0.33 and 0.51 in this data), so an
i.i.d. bootstrap would understate the uncertainty; resampling whole blocks of consecutive readings
keeps the short-range dependence inside each resample. The circular variant wraps blocks around the
end of the series so every reading is equally likely to be drawn; the moving-block variant draws the
first and last readings less often. validate_instruments.py measures both.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]
Statistic = Callable[[Array, Array], float]

# Romano et al. (2006) thresholds, the ones used by the R effsize package and ScottKnottESD.
CLIFF_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.147, "negligible"),
    (0.33, "small"),
    (0.474, "medium"),
)


def cliffs_delta(x: Array, y: Array) -> float:
    """P(X > Y) - P(X < Y), from the Mann-Whitney U statistic (ties count one half)."""
    n, m = len(x), len(y)
    ranks = stats.rankdata(np.concatenate([x, y]))
    u = ranks[:n].sum() - n * (n + 1) / 2.0
    return float(2.0 * u / (n * m) - 1.0)


def cliffs_magnitude(delta: float) -> str:
    """Label |delta| with the Romano et al. thresholds."""
    size = abs(delta)
    for bound, label in CLIFF_THRESHOLDS:
        if size < bound:
            return label
    return "large"


def hodges_lehmann(x: Array, y: Array) -> float:
    """Median of all pairwise differences x_i - y_j: the typical shift, in the data's units."""
    return float(np.median(np.subtract.outer(x, y)))


def block_resample(series: Array, block: int, rng: np.random.Generator, circular: bool) -> Array:
    """One block bootstrap resample with the same length as `series`."""
    n = len(series)
    if not 1 <= block <= n:
        raise ValueError(f"block length {block} outside 1..{n}")
    n_blocks = -(-n // block)
    starts = rng.integers(0, n if circular else n - block + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return series[idx % n]


def block_bootstrap_draws(
    x: Array, y: Array, statistic: Statistic, block: int, n_boot: int, seed: int, circular: bool
) -> Array:
    """statistic(x*, y*) for n_boot pairs of independent block resamples."""
    rng = np.random.default_rng(seed)
    return np.array(
        [
            statistic(block_resample(x, block, rng, circular), block_resample(y, block, rng, circular))
            for _ in range(n_boot)
        ]
    )


def percentile_interval(draws: Array, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(low), float(high)


def block_bootstrap_ci(
    x: Array,
    y: Array,
    statistic: Statistic,
    block: int,
    n_boot: int,
    seed: int,
    level: float,
    circular: bool,
) -> tuple[float, float]:
    """Percentile interval for statistic(x, y) from a block bootstrap of each series."""
    draws = block_bootstrap_draws(x, y, statistic, block, n_boot, seed, circular)
    return percentile_interval(draws, level)


def epsilon_squared(h_statistic: float, n_total: int) -> float:
    """Kruskal-Wallis effect size, epsilon^2 = H / (n - 1) (Tomczak and Tomczak 2014)."""
    return float(h_statistic / (n_total - 1))
