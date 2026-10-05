"""Parse raw HDFS / BGL logs into event templates with Drain (drain3).

Settings follow the Drain configuration of the logpai/logparser benchmark
(depth 4, similarity threshold 0.5). Masking regexes: HDFS uses the benchmark
regexes (block IDs, IP:port); BGL uses the refined regexes from the README of
the paper's replication package (mooselab/suppmaterial-LogRepForAnomalyDetection).

Outputs, in data/parsed/<DATASET>/:
  <DATASET>.log_structured.parquet  one row per log line (header fields + EventId)
  <DATASET>.log_templates.csv       EventId, EventTemplate, Occurrences

Post-processing: each line's template is that of its final Drain cluster, read
once parsing has finished, so lines parsed early get the cluster's final (most
general) template rather than the template at the time they were seen. Clusters
with identical template text are then merged: EventId is the first 8 hex digits
of the template's MD5, as in logpai/logparser. Without this, messages made mostly
of parameters (e.g. BGL register dumps "fpr<*>=<*> <*> <*> <*>") open a new
cluster each time, because Drain ignores wildcards when scoring similarity.

Usage: python src/parse_drain.py {HDFS,BGL} [--limit N] [--sim-th X --tag _stX]
"""

import argparse
import hashlib
import re
import time
from pathlib import Path

import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

ROOT = Path(__file__).resolve().parents[1]
BLK_RE = re.compile(r"blk_-?\d+")
EMPTY_TEMPLATE = "<EMPTY>"

DATASETS = {
    "HDFS": {
        "path": ROOT / "data/raw/HDFS_v1/HDFS.log",
        # <Date> <Time> <Pid> <Level> <Component>: <Content>
        "n_header": 5,
        "sim_th": 0.5,
        "regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
    },
    "BGL": {
        "path": ROOT / "data/raw/BGL/BGL.log",
        # <Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>
        "n_header": 9,
        "sim_th": 0.5,
        "regex": [
            r"core\.\d+",
            r"(?<=:)(\ [A-Z][+-]?)+(?![a-z])",
            r"(?<=r)\d{1,2}",
            r"(?<=fpr)\d{1,2}",
            r"(0x)?[0-9a-fA-F]{8}",
            r"(?<=\.\.)0[xX][0-9a-fA-F]+",
            r"(?<=\.\.)\d+(?!x)",
            r"\d+(?=:)",
            r"^\d+$",
            r"(?<=\=)\d+(?!x)",
            r"(?<=\=)0[xX][0-9a-fA-F]+",
        ],
    },
}


def event_id(template):
    return hashlib.md5(template.encode("utf-8")).hexdigest()[:8]


def make_miner(regexes, sim_th):
    config = TemplateMinerConfig()
    config.drain_sim_th = sim_th
    config.drain_depth = 4
    config.masking_instructions = [MaskingInstruction(r, "*") for r in regexes]
    config.parametrize_numeric_tokens = True
    return TemplateMiner(config=config)


def parse(name, limit=None, sim_th=None, tag=""):
    spec = DATASETS[name]
    miner = make_miner(spec["regex"], sim_th or spec["sim_th"])
    n_header = spec["n_header"]

    cluster_ids, headers = [], []
    t0 = time.time()
    with open(spec["path"], encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            parts = line.rstrip("\n").split(None, n_header)
            content = parts[n_header] if len(parts) > n_header else ""
            headers.append(parts[:n_header])
            # Lines with no message content (BGL only) get their own event
            # instead of being dropped, so their labels are kept.
            cluster_ids.append(miner.add_log_message(content)["cluster_id"] if content else 0)
            if (i + 1) % 1_000_000 == 0:
                print(f"{name}: {i + 1:,} lines, {len(miner.drain.clusters)} templates, "
                      f"{time.time() - t0:.0f}s", flush=True)

    cluster_template = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    cluster_template[0] = EMPTY_TEMPLATE
    event_of = {cid: event_id(t) for cid, t in cluster_template.items()}
    templates = {event_of[cid]: t for cid, t in cluster_template.items()}
    n_clusters = len(miner.drain.clusters)

    if name == "HDFS":
        df = pd.DataFrame(headers, columns=["Date", "Time", "Pid", "Level", "Component"])
        # A line can mention several blocks; it then belongs to each block's session.
        with open(spec["path"], encoding="utf-8", errors="replace") as f:
            df["BlockIds"] = [" ".join(BLK_RE.findall(l)) for l, _ in zip(f, range(len(df)))]
        df = df[["Date", "Time", "BlockIds"]]
    else:
        df = pd.DataFrame(headers, columns=["Label", "Timestamp", "Date", "Node", "Time",
                                            "NodeRepeat", "Type", "Component", "Level"])
        df["Timestamp"] = df["Timestamp"].astype("int64")
        df = df[["Label", "Timestamp", "Node", "Level"]]
    df.insert(0, "LineId", range(1, len(df) + 1))
    df["EventId"] = [event_of[c] for c in cluster_ids]

    out = ROOT / "data/parsed" / (name + tag)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{limit}" if limit else ""
    df.to_parquet(out / f"{name}{suffix}.log_structured.parquet", index=False)

    counts = df["EventId"].value_counts()
    tmpl = pd.DataFrame({"EventId": list(templates), "EventTemplate": list(templates.values())})
    tmpl["Occurrences"] = tmpl["EventId"].map(counts).fillna(0).astype(int)
    tmpl = tmpl[tmpl["Occurrences"] > 0]
    tmpl.sort_values("Occurrences", ascending=False).to_csv(
        out / f"{name}{suffix}.log_templates.csv", index=False)

    print(f"{name}: done, {len(df):,} lines, {n_clusters} Drain clusters -> {len(tmpl)} events, "
          f"{time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("--limit", type=int, help="parse only the first N lines (smoke test)")
    ap.add_argument("--sim-th", type=float, help="override the Drain similarity threshold")
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    args = ap.parse_args()
    parse(args.dataset, args.limit, args.sim_th, args.tag)
