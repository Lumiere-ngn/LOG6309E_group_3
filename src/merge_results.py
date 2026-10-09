"""Rebuild results/ from per-dataset run files: HDFS from one set of runs, BGL from another.

Used after BGL was rerun alone with line-opened windows (commit 9eb343c). The clock-window BGL rows move to
results/bgl_clock_windows/ so the two windowings can be compared in the report.

Usage: python src/merge_results.py --hdfs results/rq1_runs.csv --bgl results/per_seed_bgl_line/seed_*/rq1_runs.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replicate import summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--hdfs", type=Path, required=True, help="runs file holding the HDFS rows to keep")
    ap.add_argument("--bgl", type=Path, nargs="+", required=True, help="runs files holding the new BGL rows")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    old = pd.read_csv(args.hdfs)
    clock = old[old["dataset"] == "BGL"]
    if len(clock):
        (args.out / "bgl_clock_windows").mkdir(parents=True, exist_ok=True)
        clock.to_csv(args.out / "bgl_clock_windows" / "rq1_runs.csv", index=False)
        summarize(clock, args.out / "bgl_clock_windows")
    bgl = pd.concat([pd.read_csv(p) for p in args.bgl], ignore_index=True)
    runs = pd.concat([old[old["dataset"] == "HDFS"], bgl], ignore_index=True)
    if runs.duplicated(["dataset", "model", "features", "seed"]).any():
        raise SystemExit("duplicate (dataset, model, features, seed) rows")
    runs.to_csv(args.out / "rq1_runs.csv", index=False)
    summarize(runs, args.out)


if __name__ == "__main__":
    main()
