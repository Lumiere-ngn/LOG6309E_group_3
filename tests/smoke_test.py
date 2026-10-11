"""Smoke test on synthetic data whose answer is known.

Step 6: on data where one column separates the classes, RF and a short MLP must both reach F1 > 0.9.
Step 7: the F1-maximising threshold reads only the held-out slice and never test labels or test features; the
  autoencoder stops before the epoch cap on easy data and runs to the cap when early stopping is disabled.
Step 8: the pairwise F1 gap table reports the errors (FP + FN) behind each gap.
Sanity checks: the template rule is chosen on the training split only, the chronological split puts the latest
  windows in test, and the HDFS overlap count finds the planted duplicates.

Usage: python tests/smoke_test.py   (exit 0 when every assertion holds; runs in seconds on CPU)
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import extension  # noqa: E402
import rank  # noqa: E402
import sanity_checks  # noqa: E402
from replicate import fit_mlp, fit_rf, set_seed  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402


def check_step6() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    x_sep = rng.poisson(3, (n, 6)).astype(float)
    y = (rng.random(n) < 0.05).astype(int)
    x_sep[:, 0] += 20 * y  # one column separates the classes
    tr, te = slice(0, 1500), slice(1500, n)
    for name, fit in (("RF", fit_rf), ("MLP", lambda a, b, c, s: fit_mlp(a, b, c, s, steps=300))):
        pred, score = fit(x_sep[tr], y[tr], x_sep[te], 0)
        f1 = f1_score(y[te], pred)
        print(f"{name}: F1 {f1:.3f}, score range {score.min():.2f}-{score.max():.2f}")
        assert f1 > 0.9, f"{name} failed on separable data"


def unsupervised_data(seed: int = 0, n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Normal sessions lie on a 2-dimensional pattern; anomalous ones break it in one column."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 6, (n, 2)).astype(float)
    x = np.column_stack([base[:, 0], base[:, 1], base[:, 0] + base[:, 1], 2 * base[:, 0], base[:, 1], base[:, 0]])
    y = (rng.random(n) < 0.08).astype(int)
    x[:, 2] += 15 * y
    return x, y


def check_f1_max_threshold() -> None:
    rng = np.random.default_rng(1)
    labels = (rng.random(400) < 0.2).astype(int)
    scores = rng.normal(0, 1, 400) + 2.0 * labels
    thr = extension.f1_max_threshold(scores, labels)

    def f1_at(t: float) -> float:
        return f1_score(labels, (scores >= t).astype(int), zero_division=0)

    best = max(f1_at(t) for t in np.unique(scores))  # brute force over every candidate cut
    assert abs(f1_at(thr) - best) < 1e-12, f"cut {thr} gives F1 {f1_at(thr)}, best is {best}"
    p99 = float(np.quantile(scores[labels == 0], 0.99))
    assert f1_at(thr) > f1_at(p99), "F1-max cut must beat the old 99th-percentile cut on this slice"
    try:
        extension.f1_max_threshold(scores, np.zeros_like(labels))
    except ValueError:
        pass
    else:
        raise AssertionError("a slice without anomalies must raise, not return a cut")
    print(f"f1_max_threshold: cut {thr:.3f}, F1 {f1_at(thr):.3f} (p99 cut {f1_at(p99):.3f})")


def check_threshold_never_reads_test() -> None:
    for fn in (extension.fit_iforest, extension.fit_autoencoder, extension.f1_max_threshold,
               extension.threshold_from_slice, extension.both_cuts):
        names = set(inspect.signature(fn).parameters)
        assert not names & {"y_te", "y_test", "y_all"}, f"{fn.__name__} accepts test labels: {names}"
    x, y = unsupervised_data()
    x_fit, y_fit, x_slice, y_slice = x[:2000], y[:2000], x[2000:2500], y[2000:2500]
    x_te_a, x_te_b = x[2500:], np.random.default_rng(9).integers(0, 50, x[2500:].shape).astype(float)
    runs = {"IForest": lambda te: extension.fit_iforest(x_fit, x_slice, y_slice, te, 0),
            "AE": lambda te: extension.fit_autoencoder(x_fit[y_fit == 0], x_slice, y_slice, te, 0)}
    for name, fn in runs.items():
        set_seed(0)
        _, _, thr_a, alt_a = fn(x_te_a)
        set_seed(0)
        _, _, thr_b, alt_b = fn(x_te_b)  # a different test matrix must not move either cut
        assert thr_a == thr_b and alt_a[1] == alt_b[1], f"{name} cuts depend on the test set"
        assert thr_a != alt_a[1], f"{name}: the two cuts must differ on this slice"
        if name == "IForest":
            model = IsolationForest(n_estimators=100, random_state=0, n_jobs=-1).fit(x_fit)
            slice_scores = -model.score_samples(x_slice)
            assert extension.f1_max_threshold(slice_scores, y_slice) == alt_a[1], "F1-max cut is not the slice's"
            assert extension.threshold_from_slice(slice_scores, y_slice) == thr_a, "primary cut is not the slice's p99"
        print(f"{name}: p99 cut {thr_a:.5f}, F1-max cut {alt_a[1]:.5f}, both identical for two test matrices")


def check_both_cuts() -> None:
    rng = np.random.default_rng(4)
    labels = (rng.random(500) < 0.2).astype(int)
    scores = rng.normal(0, 1, 500) + 2.5 * labels
    test = rng.normal(0, 1, 300) + 2.5 * (rng.random(300) < 0.2)
    pred, score, thr, alt = extension.both_cuts(scores, labels, test)
    assert np.array_equal(score, test) and thr == float(np.quantile(scores[labels == 0], 0.99))
    assert np.array_equal(pred, (test > thr).astype(int)) and np.array_equal(alt[0], (test >= alt[1]).astype(int))
    assert alt[1] < thr, "on this slice the F1-max cut lies below the p99 cut, so it flags more sessions"
    assert alt[0].sum() > pred.sum()
    print(f"both_cuts: p99 flags {pred.sum()}, F1-max flags {alt[0].sum()} of {len(test)} test sessions")


def check_early_stopping() -> None:
    x, y = unsupervised_data()
    # Poisson noise on every column sets a floor the reconstruction loss reaches, as real count vectors do.
    normal = x[y == 0] + np.random.default_rng(5).poisson(2.0, x[y == 0].shape)
    set_seed(0)
    _, epochs = extension.train_autoencoder(normal[:2000], normal[2000:2400], seed=0, max_epochs=100, patience=5)
    assert 6 <= epochs < 100, f"expected a stop between epoch 6 and the cap of 100, got {epochs}"
    set_seed(0)
    _, epochs_off = extension.train_autoencoder(normal[:2000], normal[2000:2400], seed=0, max_epochs=20,
                                                patience=10_000)
    assert epochs_off == 20, f"with patience disabled the run must reach its cap of 20, got {epochs_off}"
    print(f"AE early stopping: stopped after {epochs} of 100 epochs; patience disabled ran {epochs_off} of 20")


def check_rank_gaps() -> None:
    rng = np.random.default_rng(2)
    rows = []
    for model, fp, fn in (("A", 5, 5), ("B", 40, 20), ("C", 100, 60)):
        for seed in range(10):
            tp = 200 - fn
            rows.append({"model": model, "seed": seed, "n_test": 1000, "n_test_pos": 200, "fp": fp, "fn": fn,
                         "f1": 2 * tp / (2 * tp + fp + fn) + rng.normal(0, 1e-3),
                         "auc": {"A": 0.99, "B": 0.95, "C": 0.90}[model] + rng.normal(0, 1e-3),
                         "ap": {"A": 0.98, "B": 0.90, "C": 0.80}[model] + rng.normal(0, 1e-3)})
    runs = pd.DataFrame(rows)
    gaps = rank.f1_gaps(runs).set_index(["model_a", "model_b"])
    assert gaps.loc[("A", "B"), "error_gap_sessions"] == 50 and gaps.loc[("B", "C"), "error_gap_sessions"] == 100
    assert gaps.loc[("A", "C"), "n_test"] == 1000 and gaps.loc[("A", "B"), "f1_gap"] > 0
    for metric in ("auc", "ap", "f1"):
        groups = rank.rank(runs, metric).set_index("model")["group"]
        assert groups["A"] < groups["B"] < groups["C"], f"{metric} groups {groups.to_dict()}"
    print("rank: pair gaps carry the error counts; AUC, AP and F1 all rank A > B > C")


def check_sanity_checks() -> None:
    rng = np.random.default_rng(3)
    n, k = 1000, 8
    x = rng.poisson(2, (n, k)).astype(float)
    y = (rng.random(n) < 0.3).astype(int)
    x[:, 5] = (y == 1) * rng.integers(1, 4, n)  # template 5 occurs only in anomalous windows
    assert sanity_checks.best_template(x, y) == 5
    flipped = 1 - y  # the choice reads training labels only, so test labels cannot enter its signature
    assert list(inspect.signature(sanity_checks.best_template).parameters) == ["x_tr", "y_tr"]
    assert sanity_checks.best_template(x, flipped) != 5
    base = sanity_checks.baseline_rows(x, y, [f"E{i}" for i in range(k)], [0, 1])
    rule = base[base["baseline"] == "best_template"]
    assert (rule["template"] == "E5").all() and (rule["f1"] > 0.99).all(), rule
    allpos = base[base["baseline"] == "all_anomalous"]
    assert (allpos["recall"] == 1.0).all() and (allpos["precision"] < 0.5).all()

    order = rng.permutation(n)  # rows shuffled; `order` holds each row's time
    x_tr, x_te, y_tr, y_te = sanity_checks.chronological_split(x, y, order, 0.2)
    idx = np.argsort(order)
    assert len(y_te) == 200 and np.array_equal(x_te, x[idx[800:]]) and np.array_equal(x_tr, x[idx[:800]])

    dup = np.array([[1, 0], [2, 0], [3, 1], [4, 4]] * 25 + [[9, 9]] * 4 + [[8, 8]] * 4, dtype=float)
    labels = np.array([0, 0, 1, 0] * 25 + [1] * 4 + [0] * 4)
    ids_tr, ids_te = sanity_checks.row_ids(dup[:100], dup[100:])
    seen = np.isin(ids_te, np.unique(ids_tr))
    assert not seen.any() and labels[100:].sum() == 4, "rows absent from train must read as unseen"
    ids_tr, ids_te = sanity_checks.row_ids(dup[:50], dup[50:100])
    assert np.isin(ids_te, np.unique(ids_tr)).all(), "rows repeated from train must read as seen"
    chrono = sanity_checks.chronological_rows(x, y, order, [0])
    assert set(chrono["model"]) == {"RF", "MLP"} and len(chrono) == 2
    xs = np.concatenate([rng.poisson(2, (600, 4)), rng.integers(100, 200, (400, 4))]).astype(float)
    ys = np.concatenate([rng.random(600) < 0.1, rng.random(400) < 0.5]).astype(int)  # the second block is unique rows
    unseen = sanity_checks.unseen_f1_rows(xs, ys, [0, 1])
    per, union = unseen[unseen["seed"] != "distinct_union"], unseen[unseen["seed"] == "distinct_union"].set_index("model")
    for model in ("RF", "MLP"):
        sub = per[per["model"] == model]
        n_u = union.loc[model, "n_unseen"]
        # a session unseen in both seeds is counted once, so the union is below the per-seed sum but not below the max
        assert sub["n_unseen"].max() <= n_u < sub["n_unseen"].sum(), (n_u, sub["n_unseen"].tolist())
        assert union.loc[model, "f1_ci_low"] <= union.loc[model, "f1"] <= union.loc[model, "f1_ci_high"]
    print("sanity_checks: template rule, chronological split and overlap count behave as planted")


def main() -> None:
    check_step6()
    check_f1_max_threshold()
    check_threshold_never_reads_test()
    check_both_cuts()
    check_early_stopping()
    check_rank_gaps()
    check_sanity_checks()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
