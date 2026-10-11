"""Step 7: four models on HDFS MCV features over 10 stratified 70/30 resamples (Marwan-plan.md, extension).

Models:
  RF        supervised, as in replicate.py (10 trees, max_features="sqrt").
  MLP       supervised, as in replicate.py (paper settings, 2,000 full-batch steps).
  IForest   unsupervised Isolation Forest (scikit-learn), fitted on the fit part of the training split, no labels.
  AE        semi-supervised autoencoder on MCV, trained on the normal sessions of the fit part only; the anomaly
            score is the reconstruction error.

Both unsupervised models use unsupervised scoring and a semi-supervised threshold, and two cuts are computed on a
labeled held-out slice of the TRAINING split (10%, stratified), never on the test set:
  primary    the 99th percentile of the slice's normal scores (a session is flagged when score > cut); columns
             precision, recall, f1, tp, fp, fn, threshold.
  F1-max     the score that maximises F1 on the slice (flagged when score >= cut); columns with the suffix
             _f1max (precision, recall, f1, threshold). Empty for RF and MLP.
AUC and average precision use the raw score. A council decides which cut is primary; the code supports both.
The autoencoder trains for up to 100 epochs and stops when the reconstruction loss of the slice's normal sessions
has not improved for 5 epochs; the weights of the best epoch are kept.
Features: all 48 MCV columns, as in RQ1.

Output: <out>/rq2_runs.csv, one row per model and resample (precision, recall, F1, AUC, average precision,
confusion counts, threshold, the _f1max columns, seconds).
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
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split

from replicate import DEVICE, fit_mlp, fit_rf, set_seed

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("extension")
MAX_EPOCHS = 100
PATIENCE = 5
QUANTILE = 0.99
Alt = tuple[np.ndarray, float] | None  # (predictions, cut) under the F1-max cut; None for supervised models


def threshold_from_slice(scores: np.ndarray, labels: np.ndarray) -> float:
    """99th percentile of the held-out slice's normal scores."""
    return float(np.quantile(scores[labels == 0], QUANTILE))


def f1_max_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Score cut that maximises F1 on the labeled held-out slice; a session is flagged when score >= cut.

    Only the slice's scores and labels enter. Ties on F1 go to the lowest cut.
    """
    if labels.sum() == 0:
        raise ValueError("the held-out slice has no anomalous session, so no F1-maximising cut exists")
    prec, rec, cuts = precision_recall_curve(labels, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(cuts[int(np.argmax(f1))])


def both_cuts(slice_scores: np.ndarray, y_slice: np.ndarray,
              test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, Alt]:
    """Predictions under the primary 99th-percentile cut and under the F1-max cut. Both cuts read only the slice."""
    thr = threshold_from_slice(slice_scores, y_slice)
    thr_f1 = f1_max_threshold(slice_scores, y_slice)
    return (test_scores > thr).astype(int), test_scores, thr, ((test_scores >= thr_f1).astype(int), thr_f1)


def fit_iforest(x_fit: np.ndarray, x_slice: np.ndarray, y_slice: np.ndarray, x_te: np.ndarray,
                seed: int) -> tuple[np.ndarray, np.ndarray, float, Alt]:
    model = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(x_fit)
    # score_samples is higher for normal points; negate so higher means more anomalous.
    return both_cuts(-model.score_samples(x_slice), y_slice, -model.score_samples(x_te))


class Autoencoder(torch.nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.encode = torch.nn.Sequential(torch.nn.Linear(n_in, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))
        self.decode = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(8, 32), torch.nn.ReLU(),
                                          torch.nn.Linear(32, n_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def log_tensor(a: np.ndarray) -> torch.Tensor:
    # log1p keeps a few very long sessions from dominating the reconstruction loss.
    return torch.from_numpy(np.log1p(a)).float().to(DEVICE)


def train_autoencoder(x_fit_normal: np.ndarray, x_val_normal: np.ndarray, seed: int, max_epochs: int = MAX_EPOCHS,
                      patience: int = PATIENCE, batch: int = 4096) -> tuple[Autoencoder, int]:
    """Train on normal sessions; stop when the validation reconstruction loss has not improved for `patience`
    epochs. Returns the network with the best epoch's weights and the number of epochs run."""
    xt, xv = log_tensor(x_fit_normal), log_tensor(x_val_normal)
    net = Autoencoder(xt.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, stale, epochs_run = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        perm = torch.randperm(len(xt), generator=gen).to(DEVICE)
        for i in range(0, len(xt), batch):
            b = xt[perm[i:i + batch]]
            opt.zero_grad()
            loss = torch.mean((net(b) - b) ** 2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = torch.mean((net(xv) - xv) ** 2).item()
        epochs_run = epoch + 1
        if val < best:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if epoch % 5 == 0:
            log.info("  AE epoch %d train loss %.5f val loss %.5f", epoch, loss.item(), val)
        if stale >= patience:
            break
    net.load_state_dict(best_state)
    log.info("  AE stopped after %d epochs, best val loss %.5f", epochs_run, best)
    return net, epochs_run


def fit_autoencoder(x_fit_normal: np.ndarray, x_slice: np.ndarray, y_slice: np.ndarray, x_te: np.ndarray,
                    seed: int) -> tuple[np.ndarray, np.ndarray, float, Alt]:
    net, _ = train_autoencoder(x_fit_normal, x_slice[y_slice == 0], seed)

    def score(a: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = log_tensor(a)
            return torch.mean((net(x) - x) ** 2, dim=1).cpu().numpy()

    return both_cuts(score(x_slice), y_slice, score(x_te))


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
            "RF": lambda: (*fit_rf(x_tr, y_tr, x_te, seed), None, None),
            "MLP": lambda: (*fit_mlp(x_tr, y_tr, x_te, seed), None, None),
            "IForest": lambda: fit_iforest(x_fit, x_slice, y_slice, x_te, seed),
            "AE": lambda: fit_autoencoder(x_fit[y_fit == 0], x_slice, y_slice, x_te, seed),
        }
        for name, job in jobs.items():
            set_seed(seed)
            t0 = time.time()
            pred, score, thr, alt = job()
            p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
            row = {"model": name, "seed": seed, "n_train": len(y_tr), "n_test": len(y_te),
                   "n_test_pos": int(y_te.sum()), "precision": p, "recall": r, "f1": f1,
                   "auc": roc_auc_score(y_te, score), "ap": average_precision_score(y_te, score),
                   "tp": int(tp), "fp": int(fp), "fn": int(fn), "threshold": thr}
            p2, r2, f2 = (np.nan, np.nan, np.nan)
            if alt is not None:
                p2, r2, f2, _ = precision_recall_fscore_support(y_te, alt[0], average="binary", zero_division=0)
            row.update({"precision_f1max": p2, "recall_f1max": r2, "f1_f1max": f2,
                        "threshold_f1max": np.nan if alt is None else alt[1],
                        "seconds": round(time.time() - t0, 1)})
            rows.append(row)
            log.info("seed %d %s: P %.3f R %.3f F1 %.3f AUC %.3f AP %.3f (%.0fs)", seed, name, p, r, f1,
                     row["auc"], row["ap"], row["seconds"])
            pd.DataFrame(rows).to_csv(out / "rq2_runs.csv", index=False)
    runs = pd.DataFrame(rows)
    summary = runs.groupby("model")[["precision", "recall", "f1", "auc", "ap", "precision_f1max", "recall_f1max",
                                     "f1_f1max"]].agg(["mean", "std"]).round(3)
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
