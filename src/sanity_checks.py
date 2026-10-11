"""Sanity checks on the step 6 results (plan/team-fixes-2026-10-11.md, item 1). BGL only, plus one HDFS count.

Why: the team found that a one-template rule reaches F1 0.976 on the BGL shuffled split, that RF falls to 0.698
under a chronological split, and that 99.95% of the HDFS test count vectors also occur in the training split.
This script reproduces each number from data/features/<DATASET>/mcv.parquet.

Outputs (results/sanity/):
  bgl_baselines.csv     BGL, the step 6 split (stratified 80/20, seeds 0 to 4). Baseline "all_anomalous" flags every
                        window. Baseline "best_template" flags a window when the count of one template is above
                        zero; the template is the one with the best F1 on the training split only.
  bgl_chronological.csv BGL windows ordered by start time (the window id is assigned in time order by
                        build_mcv.py), first 80% train, last 20% test. RF and MLP with the step 6 settings, model
                        seeds 0 to 4. One split, so the five rows per model differ only by model seed.
  hdfs_overlap.csv      HDFS, the step 6 split (stratified 70/30, seeds 0 to 4): how many test sessions have a count
                        vector that also appears in the training split. Counts only.
  hdfs_unseen_f1.csv    HDFS, same splits: RF and MLP (step 6 settings) scored on the test sessions whose count vector
                        never appears in training. One row per model and seed, plus a "distinct_union" row per model
                        that counts each unseen session once, with a 95% bootstrap interval over sessions.

Usage: python src/sanity_checks.py [--checks bgl_baselines bgl_chronological hdfs_overlap] [--seeds 0 1 2 3 4]
                                   [--out results/sanity]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from replicate import TEST_SIZE, fit_mlp, fit_rf, set_seed

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("sanity_checks")
CHECKS = ("bgl_baselines", "bgl_chronological", "hdfs_overlap", "hdfs_unseen_f1")
CHRONO_TEST_SIZE = 0.2


def load(dataset: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Feature table (index = session id), count matrix and labels of a dataset's MCV file."""
    mcv = pd.read_parquet(ROOT / "data/features" / dataset / "mcv.parquet")
    features = mcv.drop(columns="label")
    return features, features.to_numpy(dtype=np.float64), mcv["label"].to_numpy()


def prf(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return float(p), float(r), float(f1)


def best_template(x_tr: np.ndarray, y_tr: np.ndarray) -> int:
    """Column j whose rule `count_j > 0` has the best F1 on the training split (first column on ties)."""
    present = x_tr > 0
    tp = (present & (y_tr == 1)[:, None]).sum(axis=0)
    flagged = present.sum(axis=0)
    f1 = 2 * tp / np.maximum(flagged + y_tr.sum(), 1)  # 2TP / (TP + FP + TP + FN)
    return int(np.argmax(f1))


def baseline_rows(x_all: np.ndarray, y_all: np.ndarray, columns: list[str], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        x_tr, x_te, y_tr, y_te = train_test_split(x_all, y_all, test_size=TEST_SIZE["BGL"], stratify=y_all,
                                                  random_state=seed)
        j = best_template(x_tr, y_tr)
        for name, pred, template in (("all_anomalous", np.ones(len(y_te), dtype=int), ""),
                                     ("best_template", (x_te[:, j] > 0).astype(int), columns[j])):
            p, r, f1 = prf(y_te, pred)
            rows.append({"baseline": name, "seed": seed, "template": template, "n_train": len(y_tr),
                         "n_test": len(y_te), "n_test_pos": int(y_te.sum()), "precision": p, "recall": r, "f1": f1})
            log.info("BGL seed %d %s %s: P %.3f R %.3f F1 %.3f", seed, name, template, p, r, f1)
    return pd.DataFrame(rows)


def chronological_split(x_all: np.ndarray, y_all: np.ndarray, order: np.ndarray, test_size: float):
    """First (1 - test_size) of the sessions by `order` train, the rest test."""
    idx = np.argsort(order, kind="stable")
    cut = int(round(len(idx) * (1 - test_size)))
    tr, te = idx[:cut], idx[cut:]
    return x_all[tr], x_all[te], y_all[tr], y_all[te]


def chronological_rows(x_all: np.ndarray, y_all: np.ndarray, order: np.ndarray, seeds: list[int]) -> pd.DataFrame:
    x_tr, x_te, y_tr, y_te = chronological_split(x_all, y_all, order, CHRONO_TEST_SIZE)
    log.info("BGL chronological: train %d (%d anomalous), test %d (%d anomalous)", len(y_tr), y_tr.sum(), len(y_te),
             y_te.sum())
    if y_tr.sum() == 0:
        raise SystemExit("the chronological training split has no anomalous window; RF and MLP cannot be fitted")
    rows = []
    for name, fit in (("RF", fit_rf), ("MLP", fit_mlp)):
        for seed in seeds:
            set_seed(seed)
            pred, score = fit(x_tr, y_tr, x_te, seed)
            p, r, f1 = prf(y_te, pred)
            both_classes = len(np.unique(y_te)) == 2  # AUC and AP are undefined on a one-class test split
            rows.append({"model": name, "seed": seed, "n_train": len(y_tr), "n_train_pos": int(y_tr.sum()),
                         "n_test": len(y_te), "n_test_pos": int(y_te.sum()), "precision": p, "recall": r, "f1": f1,
                         "auc": roc_auc_score(y_te, score) if both_classes else float("nan"),
                         "ap": average_precision_score(y_te, score) if both_classes else float("nan")})
            log.info("BGL chronological %s seed %d: P %.3f R %.3f F1 %.3f", name, seed, p, r, f1)
    return pd.DataFrame(rows)


def row_ids(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer id per distinct count vector, shared by the two matrices."""
    both = np.ascontiguousarray(np.concatenate([train, test]).astype(np.int64))
    _, inverse = np.unique(both, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    return inverse[:len(train)], inverse[len(train):]


def hdfs_split(x_all: np.ndarray, y_all: np.ndarray, seed: int):
    """The step 6 HDFS split, the test rows' session indices, and which test sessions have a count vector that also
    occurs in training."""
    idx_tr, idx_te = train_test_split(np.arange(len(y_all)), test_size=TEST_SIZE["HDFS"], stratify=y_all,
                                      random_state=seed)
    x_tr, x_te, y_tr, y_te = x_all[idx_tr], x_all[idx_te], y_all[idx_tr], y_all[idx_te]
    ids_tr, ids_te = row_ids(x_tr, x_te)
    return x_tr, x_te, y_tr, y_te, idx_te, np.isin(ids_te, np.unique(ids_tr))


def weighted_prf(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    tp, fp, fn = (w * ((pred == 1) & (y == 1))).sum(), (w * ((pred == 1) & (y == 0))).sum(), \
        (w * ((pred == 0) & (y == 1))).sum()
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r, (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def unseen_f1_rows(x_all: np.ndarray, y_all: np.ndarray, seeds: list[int], n_boot: int = 2000) -> pd.DataFrame:
    """RF and MLP fitted on each seed's full training split, scored on the test sessions whose count vector never
    appears in that training split. Council decision 2026-10-11 (plan/council-thresholds): the summary row is the
    distinct union of unseen sessions, each session counted once (its predictions from the seeds where it was unseen
    share a weight of 1), with a 95% bootstrap interval resampling sessions. An unranked stress test."""
    rows, recs = [], []
    for name, fit in (("RF", fit_rf), ("MLP", fit_mlp)):
        for seed in seeds:
            x_tr, x_te, y_tr, y_te, idx_te, seen = hdfs_split(x_all, y_all, seed)
            set_seed(seed)
            pred, _ = fit(x_tr, y_tr, x_te, seed)
            u_true, u_pred = y_te[~seen], pred[~seen]
            p, r, f1 = weighted_prf(u_true, u_pred, np.ones(len(u_true)))
            rows.append({"model": name, "seed": str(seed), "n_unseen": len(u_true), "n_unseen_pos": int(u_true.sum()),
                         "precision": p, "recall": r, "f1": f1})
            recs.append(pd.DataFrame({"model": name, "session": idx_te[~seen], "y": u_true, "pred": u_pred}))
            log.info("HDFS seed %d %s on %d unseen sessions (%d anomalous): F1 %.3f", seed, name, len(u_true),
                     u_true.sum(), f1)
    rec = pd.concat(recs, ignore_index=True)
    rng = np.random.default_rng(0)
    for name, g in rec.groupby("model", sort=False):
        g = g.assign(w=1.0 / g.groupby("session")["session"].transform("size"))
        p, r, f1 = weighted_prf(g["y"].to_numpy(), g["pred"].to_numpy(), g["w"].to_numpy())
        g = g.assign(tp=g["w"] * ((g["pred"] == 1) & (g["y"] == 1)), fp=g["w"] * ((g["pred"] == 1) & (g["y"] == 0)),
                     fn=g["w"] * ((g["pred"] == 0) & (g["y"] == 1)))
        per_s = g.groupby("session")[["tp", "fp", "fn"]].sum().to_numpy()  # one row per distinct session
        sessions = np.arange(len(per_s))
        draws = per_s[rng.choice(sessions, (n_boot, len(sessions)))].sum(axis=1)  # resample sessions
        denom = 2 * draws[:, 0] + draws[:, 1] + draws[:, 2]
        boot = np.where(denom > 0, 2 * draws[:, 0] / np.where(denom > 0, denom, 1), 0.0)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        pos = int(g.drop_duplicates("session")["y"].sum())
        rows.append({"model": name, "seed": "distinct_union", "n_unseen": len(sessions), "n_unseen_pos": pos,
                     "precision": p, "recall": r, "f1": f1, "f1_ci_low": lo, "f1_ci_high": hi})
        log.info("HDFS %s distinct union: %d sessions (%d anomalous), F1 %.3f [%.3f, %.3f]", name, len(sessions), pos,
                 f1, lo, hi)
    return pd.DataFrame(rows)


def overlap_rows(x_all: np.ndarray, y_all: np.ndarray, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        _, _, y_tr, y_te, _, seen = hdfs_split(x_all, y_all, seed)
        rows.append({"seed": seed, "n_train": len(y_tr), "n_test": len(y_te), "n_test_pos": int(y_te.sum()),
                     "n_test_in_train": int(seen.sum()), "share_in_train": float(seen.mean()),
                     "n_unseen": int((~seen).sum()), "n_unseen_pos": int(y_te[~seen].sum())})
        log.info("HDFS seed %d: %d of %d test sessions (%.4f) also in train; %d unseen", seed, seen.sum(), len(seen),
                 seen.mean(), (~seen).sum())
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--checks", nargs="+", default=list(CHECKS), choices=CHECKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "sanity")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    if {"bgl_baselines", "bgl_chronological"} & set(args.checks):
        features, x_bgl, y_bgl = load("BGL")
        log.info("BGL: %d windows, %d anomalous, %d templates", len(y_bgl), y_bgl.sum(), x_bgl.shape[1])
        if "bgl_baselines" in args.checks:
            baseline_rows(x_bgl, y_bgl, list(features.columns), args.seeds).to_csv(
                args.out / "bgl_baselines.csv", index=False)
        if "bgl_chronological" in args.checks:
            chronological_rows(x_bgl, y_bgl, features.index.to_numpy(), args.seeds).to_csv(
                args.out / "bgl_chronological.csv", index=False)
    if {"hdfs_overlap", "hdfs_unseen_f1"} & set(args.checks):
        _, x_hdfs, y_hdfs = load("HDFS")
        log.info("HDFS: %d sessions, %d anomalous", len(y_hdfs), y_hdfs.sum())
        if "hdfs_overlap" in args.checks:
            overlap_rows(x_hdfs, y_hdfs, args.seeds).to_csv(args.out / "hdfs_overlap.csv", index=False)
        if "hdfs_unseen_f1" in args.checks:
            unseen_f1_rows(x_hdfs, y_hdfs, args.seeds).to_csv(args.out / "hdfs_unseen_f1.csv", index=False)


if __name__ == "__main__":
    main()
