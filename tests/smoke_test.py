"""Smoke test for steps 5 and 6 on synthetic data whose answer is known.

Collinearity, both directions:
  - a duplicated column and a column that is 3x another must be removed, and so must a constant;
  - independent columns must all be kept.
Models: on data where one column separates the classes, RF and a short MLP must both reach F1 > 0.9.

Usage: python tests/smoke_test.py   (exit 0 when every assertion holds; runs in seconds on CPU)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from collinearity import reduce_features  # noqa: E402
from replicate import fit_mlp, fit_rf  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    x = pd.DataFrame({f"e{i}": rng.poisson(3, n) for i in range(6)})
    x["dup_e0"] = x["e0"]
    x["triple_e1"] = 3 * x["e1"]
    x["const"] = 7

    red = reduce_features(x)
    print("reduction:", red.summary(), "kept:", red.kept)
    assert "const" in red.constant, "constant column not removed"
    assert len({"e0", "dup_e0"} & set(red.kept)) == 1, "duplicate pair not reduced to one"
    assert len({"e1", "triple_e1"} & set(red.kept)) == 1, "scaled pair not reduced to one"
    assert {f"e{i}" for i in range(2, 6)} <= set(red.kept), "an independent column was removed"

    y = (rng.random(n) < 0.05).astype(int)
    x_sep = x[[f"e{i}" for i in range(6)]].to_numpy(dtype=float, copy=True)
    x_sep[:, 0] += 20 * y  # one column separates the classes
    tr, te = slice(0, 1500), slice(1500, n)
    for name, fit in (("RF", fit_rf), ("MLP", lambda a, b, c, s: fit_mlp(a, b, c, s, steps=300))):
        pred, score = fit(x_sep[tr], y[tr], x_sep[te], 0)
        f1 = f1_score(y[te], pred)
        print(f"{name}: F1 {f1:.3f}, score range {score.min():.2f}-{score.max():.2f}")
        assert f1 > 0.9, f"{name} failed on separable data"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
