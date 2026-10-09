"""Step 9: explain the Random Forest from resample 0 with SHAP TreeExplainer; each feature is one log template.

Refits the RF of src/extension.py on resample 0 (same split, same seed, so the same model), explains up to 4,000 test
sessions (all test anomalies, then normal sessions up to the cap), and writes the top 10 templates by mean |SHAP|
for the anomalous class with their text.

Outputs in results/ext/: shap_top10.csv (EventId, template, mean |SHAP|, mean SHAP on anomalous and on normal
sessions) and shap_summary.png (beeswarm of the top 10).
Usage: python src/explain.py [--cap 4000]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--cap", type=int, default=4000)
    args = ap.parse_args()
    out = ROOT / "results" / "ext"
    out.mkdir(parents=True, exist_ok=True)

    mcv = pd.read_parquet(ROOT / "data/features/HDFS/mcv.parquet")
    x, y = mcv.drop(columns="label"), mcv["label"].to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, stratify=y, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=10, max_features="sqrt", random_state=SEED, n_jobs=-1)
    rf.fit(x_tr.to_numpy(dtype=np.float64), y_tr)

    anom = np.flatnonzero(y_te == 1)
    normal = np.random.default_rng(SEED).permutation(np.flatnonzero(y_te == 0))[: max(0, args.cap - len(anom))]
    idx = np.concatenate([anom[: args.cap], normal])
    sample = x_te.iloc[idx]
    values = shap.TreeExplainer(rf).shap_values(sample.to_numpy(dtype=np.float64))
    # Recent shap returns (n, features, classes); older versions a list per class.
    sv = values[1] if isinstance(values, list) else values[..., 1]

    templates = pd.read_csv(ROOT / "data/parsed/HDFS/HDFS.log_templates.csv").set_index("EventId")["EventTemplate"]
    is_anom = y_te[idx] == 1
    table = pd.DataFrame({
        "EventId": sample.columns,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        "mean_shap_anomalous": sv[is_anom].mean(axis=0),
        "mean_shap_normal": sv[~is_anom].mean(axis=0),
    })
    table["template"] = table["EventId"].map(templates)
    top = table.sort_values("mean_abs_shap", ascending=False).head(10)
    top.to_csv(out / "shap_top10.csv", index=False)

    shap.summary_plot(sv, sample, feature_names=[f"{e}" for e in sample.columns], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=200)
    plt.close()
    print(f"explained {len(idx)} test sessions ({int(is_anom.sum())} anomalous)")
    print(top.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
