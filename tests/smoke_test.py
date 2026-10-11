"""Smoke test for step 6 on synthetic data whose answer is known.

On data where one column separates the classes, RF and a short MLP must both reach F1 > 0.9.

Usage: python tests/smoke_test.py   (exit 0 when every assertion holds; runs in seconds on CPU)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replicate import fit_mlp, fit_rf  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402


def main() -> None:
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
    print("SMOKE OK")


if __name__ == "__main__":
    main()
