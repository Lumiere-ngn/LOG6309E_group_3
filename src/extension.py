"""Step 7: four models on HDFS MCV features over 10 stratified 70/30 resamples (Marwan-plan.md, extension).

Models:
  RF        supervised, as in replicate.py (10 trees, max_features="sqrt").
  MLP       supervised, as in replicate.py (paper settings, 2,000 full-batch steps).
  IForest   unsupervised Isolation Forest (scikit-learn), fitted on the fit part of the training split, no labels.
  AE        semi-supervised autoencoder on MCV, trained on the normal sessions of the fit part only; the anomaly
            score is the reconstruction error.

Thresholds for the two unsupervised models come from a held-out slice of the TRAINING split (10%, stratified),
never from the test set: the 99th percentile of the scores of the slice's normal sessions. AUC uses the raw score.
Features: all 48 MCV columns, the better condition in RQ1 (results/rq1_summary.csv).

Output: <out>/rq2_runs.csv, one row per model and resample (precision, recall, F1, AUC, threshold, seconds).
Usage: python src/extension.py [--seeds 0 1 ... 9] [--limit N] [--out results/ext]
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from replicate import DEVICE, fit_mlp, fit_rf, set_seed

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("extension")
QUANTILE = 0.99


def threshold_from_slice(scores: np.ndarray, labels: np.ndarray) -> float:
    """99th percentile of the held-out slice's normal scores."""
    return float(np.quantile(scores[labels == 0], QUANTILE))


def fit_iforest(x_fit: np.ndarray, x_slice: np.ndarray, y_slice: np.ndarray, x_te: np.ndarray,
                seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    model = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(x_fit)
    # score_samples is higher for normal points; negate so higher means more anomalous.
    thr = threshold_from_slice(-model.score_samples(x_slice), y_slice)
    score = -model.score_samples(x_te)
    return (score > thr).astype(int), score, thr


class Autoencoder(torch.nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.encode = torch.nn.Sequential(torch.nn.Linear(n_in, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))
        self.decode = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(8, 32), torch.nn.ReLU(),
                                          torch.nn.Linear(32, n_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def fit_autoencoder(x_fit_normal: np.ndarray, x_slice: np.ndarray, y_slice: np.ndarray, x_te: np.ndarray,
                    seed: int, epochs: int = 20, batch: int = 4096) -> tuple[np.ndarray, np.ndarray, float]:
    # log1p keeps a few very long sessions from dominating the reconstruction loss.
    def tensor(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.log1p(a)).float().to(DEVICE)

    xt = tensor(x_fit_normal)
    net = Autoencoder(xt.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(epochs):
        perm = torch.randperm(len(xt), generator=gen).to(DEVICE)
        for i in range(0, len(xt), batch):
            b = xt[perm[i:i + batch]]
            opt.zero_grad()
            loss = torch.mean((net(b) - b) ** 2)
            loss.backward()
            opt.step()
        if epoch % 5 == 0:
            log.info("  AE epoch %d loss %.5f", epoch, loss.item())

    def score(a: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = tensor(a)
            return torch.mean((net(x) - x) ** 2, dim=1).cpu().numpy()

    thr = threshold_from_slice(score(x_slice), y_slice)
    s = score(x_te)
    return (s > thr).astype(int), s, thr


def run(seeds: list[int], limit: int | None, out: Path) -> None:
    mcv = pd.read_parquet(ROOT / "data/features/HDFS/mcv.parquet")
    if limit:
        mcv = mcv.sample(n=limit, random_state=0)
    x_all, y_all = mcv.drop(columns="label").to_numpy(dtype=np.float64), mcv["label"].to_numpy()
    log.info("HDFS: %d sessions, %d features, %d anomalous; device %s", len(y_all), x_all.shape[1], y_all.sum(), DEVICE)
    rows = []
    for seed in seeds:
        x_tr, x_te, y_tr, y_te = train_test_split(x_all, y_all, test_size=0.3, stratify=y_all, random_state=seed)
        x_fit, x_slice, y_fit, y_slice = train_test_split(x_tr, y_tr, test_size=0.1, stratify=y_tr,
                                                          random_state=seed)
        jobs = {
            "RF": lambda: (*fit_rf(x_tr, y_tr, x_te, seed), None),
            "MLP": lambda: (*fit_mlp(x_tr, y_tr, x_te, seed), None),
            "IForest": lambda: fit_iforest(x_fit, x_slice, y_slice, x_te, seed),
            "AE": lambda: fit_autoencoder(x_fit[y_fit == 0], x_slice, y_slice, x_te, seed),
        }
        for name, job in jobs.items():
            set_seed(seed)
            t0 = time.time()
            pred, score, thr = job()
            p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
            row = {"model": name, "seed": seed, "n_train": len(y_tr), "n_test": len(y_te), "precision": p,
                   "recall": r, "f1": f1, "auc": roc_auc_score(y_te, score), "threshold": thr,
                   "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            log.info("seed %d %s: P %.3f R %.3f F1 %.3f AUC %.3f (%.0fs)", seed, name, p, r, f1, row["auc"],
                     row["seconds"])
            pd.DataFrame(rows).to_csv(out / "rq2_runs.csv", index=False)
    runs = pd.DataFrame(rows)
    summary = runs.groupby("model")[["precision", "recall", "f1", "auc"]].agg(["mean", "std"]).round(3)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(out / "rq2_summary.csv")
    log.info("\n%s", summary.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--limit", type=int, help="subsample N sessions (smoke test)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ext")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.seeds, args.limit, args.out)


if __name__ == "__main__":
    main()
