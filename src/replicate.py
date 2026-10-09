"""Step 6: replicate RQ1 of Wu, Li and Khomh (EMSE 2023) with MCV features, Random Forest and MLP.

Grid: dataset {HDFS, BGL} x model {RF, MLP} x features {all, reduced (src/collinearity.py)} x seeds.
Each seed draws one stratified split (HDFS 70/30, BGL 80/20, the paper's ratios; the paper shuffled
without stratifying and fixed no seed). Feature reduction is fitted on that seed's training split.

Model settings copy the paper's code (github.com/mooselab/suppmaterial-LogRepForAnomalyDetection):
  RF   models/traditional.py: RandomForestClassifier(n_estimators=10, max_features="sqrt"), which is
       what "auto" meant for classifiers; the paper averages 5 runs.
  MLP  models/MLP.py: two hidden layers of 200 sigmoid units, Adam lr 0.01, cross-entropy, 2,000
       full-batch steps, raw counts, anomalous training rows repeated int(n/n_pos) - 1 extra times.
       The final model is evaluated; the paper's script prints test F1 every 10 steps instead.

Paper's RF caveat: traditional.py defines metrics(y_pred, y_true) but calls metrics(y_test, y_),
so its RF precision and recall are swapped. PAPER below keeps the published values; the summary
also lists them swapped so both readings sit beside ours.

Outputs (results/): rq1_runs.csv (one row per run), rq1_summary.csv (mean and sd over seeds, with the
paper's values), rq1_reduction.csv (features removed per pass, per dataset and seed).

Usage: python src/replicate.py [--datasets HDFS BGL] [--models RF MLP] [--seeds 0 1 2 3 4]
"""

import argparse
import logging
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from collinearity import reduce_features

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("replicate")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SIZE = {"HDFS": 0.3, "BGL": 0.2}
# Published MCV values (paper Tables 2 and 3): (precision, recall, F1).
PAPER = {("HDFS", "RF"): (0.998, 1.000, 0.999), ("HDFS", "MLP"): (0.999, 0.999, 0.999),
         ("BGL", "RF"): (0.830, 0.963, 0.891), ("BGL", "MLP"): (0.958, 0.840, 0.895)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def fit_rf(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rf = RandomForestClassifier(n_estimators=10, max_features="sqrt", random_state=seed, n_jobs=-1)
    rf.fit(x_tr, y_tr)
    return rf.predict(x_te), rf.predict_proba(x_te)[:, 1]


class MLP(torch.nn.Module):
    def __init__(self, n_in: int, n_hidden: int = 200) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_hidden), torch.nn.Sigmoid(),
            torch.nn.Linear(n_hidden, n_hidden), torch.nn.Sigmoid(),
            torch.nn.Linear(n_hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_mlp(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, seed: int,
            steps: int = 2000, lr: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    extra = int(len(y_tr) / y_tr.sum()) - 1  # paper: np.repeat(pos_items, ratio - 1)
    pos = x_tr[y_tr == 1]
    x = np.concatenate([x_tr, np.repeat(pos, extra, axis=0)])
    y = np.concatenate([y_tr, np.ones(len(pos) * extra, dtype=y_tr.dtype)])
    order = np.random.default_rng(seed).permutation(len(y))
    xt = torch.from_numpy(x[order]).float().to(DEVICE)
    yt = torch.from_numpy(y[order]).long().to(DEVICE)

    net = MLP(x.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            log.info("  MLP step %d loss %.4f", step, loss.item())
    with torch.no_grad():
        prob = torch.softmax(net(torch.from_numpy(x_te).float().to(DEVICE)), dim=1)[:, 1].cpu().numpy()
    return (prob >= 0.5).astype(int), prob


MODELS = {"RF": fit_rf, "MLP": fit_mlp}


def run(datasets: list[str], models: list[str], seeds: list[int], out: Path) -> None:
    rows, reductions = [], []
    for ds in datasets:
        mcv = pd.read_parquet(ROOT / "data/features" / ds / "mcv.parquet")
        x_all, y_all = mcv.drop(columns="label"), mcv["label"].to_numpy()
        log.info("%s: %d sessions, %d features, %d anomalous", ds, len(mcv), x_all.shape[1], y_all.sum())
        for seed in seeds:
            x_tr, x_te, y_tr, y_te = train_test_split(x_all, y_all, test_size=TEST_SIZE[ds],
                                                      stratify=y_all, random_state=seed)
            t0 = time.time()
            red = reduce_features(x_tr)
            reductions.append({"dataset": ds, "seed": seed, **red.summary(), "seconds": round(time.time() - t0, 1)})
            log.info("%s seed %d reduction: %s", ds, seed, red.summary())
            for cond, cols in (("all", list(x_all.columns)), ("reduced", red.kept)):
                for name in models:
                    set_seed(seed)
                    t0 = time.time()
                    pred, score = MODELS[name](x_tr[cols].to_numpy(dtype=np.float64), y_tr,
                                               x_te[cols].to_numpy(dtype=np.float64), seed)
                    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
                    row = {"dataset": ds, "model": name, "features": cond, "n_features": len(cols), "seed": seed,
                           "n_train": len(y_tr), "n_test": len(y_te), "precision": p, "recall": r, "f1": f1,
                           "auc": roc_auc_score(y_te, score), "seconds": round(time.time() - t0, 1)}
                    rows.append(row)
                    log.info("%s %s %s seed %d: P %.3f R %.3f F1 %.3f AUC %.3f (%.0fs)",
                             ds, name, cond, seed, p, r, f1, row["auc"], row["seconds"])
                    pd.DataFrame(rows).to_csv(out / "rq1_runs.csv", index=False)  # checkpoint every run

    pd.DataFrame(reductions).to_csv(out / "rq1_reduction.csv", index=False)
    summarize(pd.DataFrame(rows), out)


def summarize(runs: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Mean and sd over seeds per dataset, model and feature set, beside the paper's values."""
    summary = runs.groupby(["dataset", "model", "features"])[["n_features", "precision", "recall", "f1", "auc"]] \
        .agg(["mean", "std"]).round(3)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    for i, metric in enumerate(["precision", "recall", "f1"]):
        summary[f"paper_{metric}"] = [PAPER[(d, m)][i] for d, m in zip(summary["dataset"], summary["model"])]
    summary["paper_note"] = np.where(summary["model"] == "RF", "paper RF P and R likely swapped (metrics arg order)", "")
    summary["n_seeds"] = runs.groupby(["dataset", "model", "features"])["seed"].nunique().to_numpy()
    summary.to_csv(out / "rq1_summary.csv", index=False)
    log.info("\n%s", summary.to_string(index=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=["HDFS", "BGL"], choices=["HDFS", "BGL"])
    ap.add_argument("--models", nargs="+", default=["RF", "MLP"], choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--merge", nargs="+", type=Path,
                    help="only merge these rq1_runs.csv files (one per array task) into --out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        runs = pd.concat([pd.read_csv(p) for p in args.merge], ignore_index=True)
        dup = runs.duplicated(["dataset", "model", "features", "seed"])
        if dup.any():
            raise SystemExit(f"{int(dup.sum())} duplicate (dataset, model, features, seed) rows across {args.merge}")
        runs.to_csv(args.out / "rq1_runs.csv", index=False)
        summarize(runs, args.out)
        return
    log.info("python %s, numpy %s, pandas %s, sklearn %s, torch %s, threads %d, device %s", platform.python_version(),
             np.__version__, pd.__version__, sklearn.__version__, torch.__version__, torch.get_num_threads(), DEVICE)
    run(args.datasets, args.models, args.seeds, args.out)


if __name__ == "__main__":
    main()
