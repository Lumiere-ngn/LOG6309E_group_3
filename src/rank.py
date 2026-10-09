"""Step 8: rank the four step-7 models with non-parametric Scott-Knott ESD on F1 and on AUC, and box-plot each.

Each model contributes its 10 per-resample values (results/ext/rq2_runs.csv). analysis/sk_esd.py ports the
ScottKnottESD R package: models are sorted by median, a block whose extreme pair differs by a negligible Cliff's
delta (< 0.147) is one group, otherwise the block is cut where the Kruskal-Wallis H is largest. Group 1 is the best.

Outputs: results/ext/rq2_ranking.csv (model, metric, median, mean, group) and results/ext/rq2_{f1,auc}.png.
Usage: python src/rank.py [--runs results/ext/rq2_runs.csv]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.sk_esd import sk_esd_np  # noqa: E402

METRICS = {"f1": "F1", "auc": "AUC"}


def rank(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    samples = {m: g[metric].to_numpy(dtype=float) for m, g in runs.groupby("model")}
    groups = sk_esd_np(samples)
    both = sk_esd_np(samples, all_pairs=True)
    rows = [{"model": m, "metric": metric, "median": float(np.median(v)), "mean": float(np.mean(v)),
             "sd": float(np.std(v, ddof=1)), "n": len(v), "group": groups[m],
             "group_all_pairs": both[m]} for m, v in samples.items()]
    return pd.DataFrame(rows).sort_values("group")


def plot(runs: pd.DataFrame, table: pd.DataFrame, metric: str, out: Path) -> None:
    order = table["model"].tolist()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.boxplot([runs.loc[runs["model"] == m, metric] for m in order], tick_labels=order)
    for i, (_, row) in enumerate(table.iterrows(), start=1):
        ax.annotate(f"group {row['group']}", (i, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel(f"{METRICS[metric]} over 10 resamples")
    ax.set_title(f"HDFS, Scott-Knott ESD on {METRICS[metric]}", pad=18, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"rq2_{metric}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "ext" / "rq2_runs.csv")
    args = ap.parse_args()
    runs = pd.read_csv(args.runs)
    counts = runs.groupby("model").size()
    if counts.min() < 10:
        raise SystemExit(f"fewer than 10 resamples for some model: {counts.to_dict()}")
    tables = [rank(runs, metric) for metric in METRICS]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.runs.parent / "rq2_ranking.csv", index=False)
    for metric, t in zip(METRICS, tables):
        plot(runs, t, metric, args.runs.parent)
    print(table.round(4).to_string(index=False))
    if (table["group"] != table["group_all_pairs"]).any():
        print("NOTE: the extreme-pair and all-pairs negligibility rules disagree on some model")


if __name__ == "__main__":
    main()
