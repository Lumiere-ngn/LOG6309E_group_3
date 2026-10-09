"""Non-parametric Scott-Knott ESD ranking, ported from the ScottKnottESD R package.

Source read on 2026-09-15: klainfo/ScottKnottESD, R/partition.R (PartitionNonParametric),
R/sk_esd.R and R/scottknott.R; the repository's last commit is dated 2023-02-09. R is not installed
on this machine, so the algorithm is ported rather than called.

Algorithm. R's control flow is a top-down binary partition. Sort the treatments by median, highest
first. A block of treatments is one group when its differences are negligible by Cliff's delta.
Otherwise the block is split at the cut that maximises the Kruskal-Wallis H statistic, and each part
is partitioned the same way. Groups are numbered from the highest median.

Three details of the R code, and what this port does with each:
  1. At each candidate cut R relabels only the block's rows as "left" and "right"; every treatment
     outside the block keeps its own label, so H is computed over 2 + (outside treatments) groups.
     Reproduced. The first version of this port pooled only the block, and a code review on
     2026-09-15 found it chose a different cut on 170 of 2,000 random 5-treatment inputs.
     `outside_as_groups=False` keeps that old rule for the regression check.
  2. R's negligibility loop indexes means[k] and means[g] on every iteration, so it only tests the
     block's two extreme treatments. Reproduced by default; `all_pairs=True` tests every pair, which
     is what the package documentation describes. The analysis reports whether the two agree.
  3. sk_esd() names the groups with rownames(m.inf), which is sorted by MEAN, although the partition
     is built in MEDIAN order, and R's rev(sort()) reverses tied medians. Not reproduced: this port
     labels groups in median order and keeps input order for ties. The analysis reports whether mean
     and median order coincide, in which case R returns the same labels.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from analysis.effect import cliffs_delta, cliffs_magnitude

Array = NDArray[np.float64]


def _block_is_negligible(samples: Mapping[str, Array], block: Sequence[str], all_pairs: bool) -> bool:
    if len(block) == 1:
        return True
    pairs = itertools.combinations(block, 2) if all_pairs else [(block[0], block[-1])]
    return all(
        cliffs_magnitude(cliffs_delta(samples[a], samples[b])) == "negligible" for a, b in pairs
    )


def _best_split(samples: Mapping[str, Array], block: Sequence[str], outside_as_groups: bool) -> int:
    """Cut index i maximising Kruskal-Wallis H of block[:i] vs block[i:]; the first maximum wins."""
    outside = [samples[name] for name in samples if name not in block] if outside_as_groups else []
    best_i, best_h = 1, -np.inf
    for i in range(1, len(block)):
        left = np.concatenate([samples[name] for name in block[:i]])
        right = np.concatenate([samples[name] for name in block[i:]])
        h = float(stats.kruskal(left, right, *outside).statistic)
        if h > best_h:
            best_i, best_h = i, h
    return best_i


def sk_esd_trace(
    samples: Mapping[str, Array], all_pairs: bool = False, outside_as_groups: bool = True
) -> list[dict[str, Any]]:
    """Every decision of the partition, in order: the block examined, and either kept or cut in two."""
    order = sorted(samples, key=lambda name: float(np.median(samples[name])), reverse=True)
    steps: list[dict[str, Any]] = []
    pending: list[list[str]] = [order]
    while pending:
        block = pending.pop(0)
        extreme = cliffs_delta(samples[block[0]], samples[block[-1]]) if len(block) > 1 else 0.0
        if _block_is_negligible(samples, block, all_pairs):
            steps.append({"block": block, "extreme_delta": extreme, "kept": True})
            continue
        cut = _best_split(samples, block, outside_as_groups)
        steps.append({"block": block, "extreme_delta": extreme, "kept": False, "left": block[:cut], "right": block[cut:]})
        pending[:0] = [block[:cut], block[cut:]]
    return steps


def sk_esd_np(
    samples: Mapping[str, Array], all_pairs: bool = False, outside_as_groups: bool = True
) -> dict[str, int]:
    """Map each treatment to its Scott-Knott ESD group; group 1 holds the highest medians."""
    kept = [step["block"] for step in sk_esd_trace(samples, all_pairs, outside_as_groups) if step["kept"]]
    return {name: rank for rank, block in enumerate(kept, start=1) for name in block}
