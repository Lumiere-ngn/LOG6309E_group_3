"""Step 8: rank the four step-7 models with non-parametric Scott-Knott ESD, and box-plot each metric.

AUC and average precision (AP) are the primary rankings, because neither depends on a threshold. F1 is secondary:
it depends on the cut each model uses. After the ranking, the script prints for every pair of models the median F1
gap and the number of test sessions behind it (mean over resamples of the test sessions, the anomalous ones, and
the difference in errors, false positives plus false negatives).

Each model contributes its 10 per-resample values (results/ext/rq2_runs.csv). analysis/sk_esd.py ports the
ScottKnottESD R package: models are sorted by median, a block whose extreme pair differs by a negligible Cliff's
delta (< 0.147) is one group, otherwise the block is cut where the Kruskal-Wallis H is largest. Group 1 is the best.

Outputs: results/ext/rq2_ranking.csv (model, metric, tier, median, mean, group), results/ext/rq2_gaps.csv and
results/ext/rq2_{auc,ap,f1}.png.
Usage: python src/rank.py [--runs results/ext/rq2_runs.csv]
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.sk_esd import sk_esd_np  # noqa: E402

METRICS = {"auc": "AUC", "ap": "AP", "f1": "F1"}  # primary first
TIER = {"auc": "primary", "ap": "primary", "f1": "secondary"}


def rank(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    samples = {m: g[metric].to_numpy(dtype=float) for m, g in runs.groupby("model")}
    groups = sk_esd_np(samples)
    both = sk_esd_np(samples, all_pairs=True)
    rows = [{"model": m, "metric": metric, "tier": TIER[metric], "median": float(np.median(v)),
             "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n": len(v), "group": groups[m],
             "group_all_pairs": both[m]} for m, v in samples.items()]
    return pd.DataFrame(rows).sort_values("group")


def f1_gaps(runs: pd.DataFrame) -> pd.DataFrame:
    """Per pair of models: the median F1 gap and the test sessions behind it.

    F1 = 2TP / (2TP + FP + FN), so the gap is carried by the difference in errors (FP + FN) on the same test set.
    A positive error_gap_sessions means model_a makes fewer errors than model_b.
    """
    has_counts = {"fp", "fn", "n_test_pos"} <= set(runs.columns)
    per_model = {m: g.sort_values("seed") for m, g in runs.groupby("model")}
    order = sorted(per_model, key=lambda m: -per_model[m]["f1"].median())
    rows = []
    for a, b in itertools.combinations(order, 2):
        ra, rb = per_model[a], per_model[b]
        row = {"model_a": a, "model_b": b, "median_f1_a": ra["f1"].median(), "median_f1_b": rb["f1"].median(),
               "f1_gap": ra["f1"].median() - rb["f1"].median(), "n_test": float(ra["n_test"].mean())}
        if has_counts:
            err_a, err_b = (ra["fp"] + ra["fn"]).to_numpy(), (rb["fp"] + rb["fn"]).to_numpy()
            row["n_test_anomalous"] = float(ra["n_test_pos"].mean())
            row["mean_errors_a"], row["mean_errors_b"] = float(err_a.mean()), float(err_b.mean())
            row["error_gap_sessions"] = float((err_b - err_a).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot(runs: pd.DataFrame, table: pd.DataFrame, metric: str, out: Path) -> None:
    import matplotlib  # imported here so the ranking and its tests run where matplotlib is absent

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = table["model"].tolist()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.boxplot([runs.loc[runs["model"] == m, metric] for m in order], tick_labels=order)
    for i, (_, row) in enumerate(table.iterrows(), start=1):
        ax.annotate(f"group {row['group']}", (i, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel(f"{METRICS[metric]} over 10 resamples")
    ax.set_title(f"HDFS, Scott-Knott ESD on {METRICS[metric]} ({TIER[metric]})", pad=18, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"rq2_{metric}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "ext" / "rq2_runs.csv")
    args = ap.parse_args()
    runs = pd.read_csv(args.runs)
    absent = [m for m in METRICS if m not in runs.columns]
    if absent:
        raise SystemExit(f"{args.runs} has no {absent} column; rerun src/extension.py (it writes average precision)")
    counts = runs.groupby("model").size()
    if counts.min() < 10:
        raise SystemExit(f"fewer than 10 resamples for some model: {counts.to_dict()}")
    tables = [rank(runs, metric) for metric in METRICS]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.runs.parent / "rq2_ranking.csv", index=False)
    for metric, t in zip(METRICS, tables):
        plot(runs, t, metric, args.runs.parent)
    print(table.round(4).to_string(index=False))
    gaps = f1_gaps(runs)
    gaps.to_csv(args.runs.parent / "rq2_gaps.csv", index=False)
    print("\nF1 gap per pair of models, with the test sessions behind it")
    print(gaps.round(4).to_string(index=False))
    if "fp" not in runs.columns:
        print("NOTE: no confusion counts in the runs file; only the test-set size is shown")
    if (table["group"] != table["group_all_pairs"]).any():
        print("NOTE: the extreme-pair and all-pairs negligibility rules disagree on some model")


if __name__ == "__main__":
    main()
