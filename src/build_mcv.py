"""Step 4: group parsed log lines into sessions and build the Message Count Vector (MCV) matrix.

Grouping follows the paper (Wu, Li and Khomh, EMSE 2023, Table 1):
  HDFS  one session per block ID. A line that mentions several blocks counts once in each block's
        session, as in loghub's Event_traces.csv. Labels come from anomaly_label.csv.
  BGL   fixed 6-hour windows on the epoch Timestamp, counted from the first line. A window is
        anomalous if any of its lines carries an alert label (first column other than "-").

Input:  data/parsed/<DATASET>/<DATASET>.log_structured.parquet   (src/parse_drain.py)
Output: data/features/<DATASET>/mcv.parquet
        rows = sessions (index "session"), one int32 column per EventId, plus "label" (0/1).

Usage: python src/build_mcv.py {HDFS,BGL} [--window-hours 6]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("build_mcv")


def count_matrix(sessions: pd.Series, events: pd.Series) -> pd.DataFrame:
    """Session-by-event count table from two aligned Series."""
    counts = pd.crosstab(sessions, events).astype("int32")
    counts.index.name = "session"
    counts.columns = counts.columns.astype(str)
    return counts


def hdfs_mcv() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/parsed/HDFS/HDFS.log_structured.parquet",
                         columns=["BlockIds", "EventId"])
    # dict.fromkeys drops a block repeated within one line, matching validate_parse.py.
    df["Block"] = df["BlockIds"].map(lambda s: list(dict.fromkeys(s.split())))
    df = df.explode("Block").dropna(subset=["Block"])
    mcv = count_matrix(df["Block"], df["EventId"])

    labels = pd.read_csv(ROOT / "data/raw/HDFS_v1/preprocessed/anomaly_label.csv")
    labels = labels.set_index("BlockId")["Label"].eq("Anomaly").astype("int8")
    missing = mcv.index.difference(labels.index)
    if len(missing):
        raise ValueError(f"{len(missing)} parsed blocks have no label, e.g. {list(missing[:3])}")
    mcv["label"] = labels.reindex(mcv.index).to_numpy()
    return mcv


def bgl_mcv(window_hours: float) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/parsed/BGL/BGL.log_structured.parquet",
                         columns=["Label", "Timestamp", "EventId"])
    width = int(window_hours * 3600)
    df["Window"] = (df["Timestamp"] - df["Timestamp"].min()) // width
    mcv = count_matrix(df["Window"], df["EventId"])
    mcv["label"] = df["Label"].ne("-").groupby(df["Window"]).any().astype("int8").reindex(mcv.index).to_numpy()
    return mcv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", choices=["HDFS", "BGL"])
    ap.add_argument("--window-hours", type=float, default=6.0, help="BGL fixed window size")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    mcv = hdfs_mcv() if args.dataset == "HDFS" else bgl_mcv(args.window_hours)
    out = ROOT / "data/features" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    mcv.to_parquet(out / "mcv.parquet")
    n_anom = int(mcv["label"].sum())
    log.info("%s: %d sessions, %d anomalous (%.2f%%), %d event columns -> %s",
             args.dataset, len(mcv), n_anom, 100 * n_anom / len(mcv), mcv.shape[1] - 1, out)


if __name__ == "__main__":
    main()
