"""Step 5: remove correlated and redundant MCV features before training.

Three passes, fitted on the TRAINING split only (the test split never chooses features):
  1. drop constant columns (an event that never varies carries no signal);
  2. Spearman correlation clustering: average-linkage hierarchical clustering on 1 - |rho|, cut at
     |rho| = 0.7, keep one feature per cluster (the one with the highest training-set variance);
  3. VIF: drop the feature with the largest variance inflation factor while any VIF exceeds 5.

The paper (Wu, Li and Khomh, EMSE 2023) does none of these, so its numbers are our "all features"
condition.

Usage as a check: python src/collinearity.py {HDFS,BGL} [--seed 0] prints what each pass removes
on the training split of that seed.
"""

import argparse
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor

log = logging.getLogger("collinearity")


@dataclass(frozen=True)
class Reduction:
    """Features kept and what each pass removed."""
    kept: list[str]
    constant: list[str] = field(default_factory=list)
    correlated: list[str] = field(default_factory=list)
    high_vif: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"n_input": len(self.kept) + len(self.constant) + len(self.correlated) + len(self.high_vif),
                "removed_constant": len(self.constant), "removed_correlated": len(self.correlated),
                "removed_vif": len(self.high_vif), "n_kept": len(self.kept)}


def spearman_clusters(x: pd.DataFrame, rho: float) -> tuple[list[str], list[str]]:
    """Keep one feature per cluster of features whose |Spearman rho| exceeds `rho`."""
    if x.shape[1] < 2:
        return list(x.columns), []
    corr = x.rank().corr().abs().fillna(0).to_numpy(copy=True)  # Pearson on ranks = Spearman
    np.fill_diagonal(corr, 1.0)
    dist = squareform(1.0 - corr, checks=False)
    labels = fcluster(linkage(dist, method="average"), t=1.0 - rho, criterion="distance")
    variance = x.var()
    kept = [variance[x.columns[labels == c]].idxmax() for c in np.unique(labels)]
    dropped = [c for c in x.columns if c not in set(kept)]
    return [c for c in x.columns if c in set(kept)], dropped


def vif_filter(x: pd.DataFrame, threshold: float) -> tuple[list[str], list[str]]:
    """Drop the highest-VIF feature, one at a time, until every VIF is at most `threshold`."""
    cols, dropped = list(x.columns), []
    while len(cols) > 1:
        design = np.column_stack([np.ones(len(x)), x[cols].to_numpy(dtype=float)])
        with np.errstate(divide="ignore", invalid="ignore"):
            vifs = np.array([variance_inflation_factor(design, i + 1) for i in range(len(cols))])
        vifs = np.nan_to_num(vifs, nan=np.inf)
        worst = int(np.argmax(vifs))
        if vifs[worst] <= threshold:
            break
        dropped.append(cols.pop(worst))
    return cols, dropped


def reduce_features(x_train: pd.DataFrame, rho: float = 0.7, vif_max: float = 5.0) -> Reduction:
    constant = [c for c in x_train.columns if x_train[c].nunique() <= 1]
    x = x_train.drop(columns=constant)
    after_corr, correlated = spearman_clusters(x, rho)
    kept, high_vif = vif_filter(x[after_corr], vif_max)
    return Reduction(kept=kept, constant=constant, correlated=correlated, high_vif=high_vif)


def main() -> None:
    from pathlib import Path

    from sklearn.model_selection import train_test_split

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", choices=["HDFS", "BGL"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    root = Path(__file__).resolve().parents[1]
    mcv = pd.read_parquet(root / "data/features" / args.dataset / "mcv.parquet")
    test_size = 0.3 if args.dataset == "HDFS" else 0.2
    x_train, _ = train_test_split(mcv.drop(columns="label"), test_size=test_size,
                                  stratify=mcv["label"], random_state=args.seed)
    red = reduce_features(x_train)
    log.info("%s seed %d: %s", args.dataset, args.seed, red.summary())


if __name__ == "__main__":
    main()
