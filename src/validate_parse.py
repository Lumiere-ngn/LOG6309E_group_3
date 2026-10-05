"""Check the HDFS Drain parse against loghub's ground-truth event traces.

loghub ships preprocessed/Event_traces.csv: for each block, the ordered list of
ground-truth events (E1..E29) of the lines that mention it. Rebuilding the same
per-block sequences from our parse aligns our EventIds with the ground-truth
ones position by position, which gives a line-level contingency table.

Grouping Accuracy (GA, Zhu et al. 2019): a line counts as correctly parsed when
its event groups exactly the same lines as its ground-truth event, i.e. the
event maps one-to-one onto a ground-truth event. Lines that mention several
blocks are counted once per block, as in the traces.

Usage: python src/validate_parse.py
"""

from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parsed = pd.read_parquet(ROOT / "data/parsed/HDFS/HDFS.log_structured.parquet",
                             columns=["BlockIds", "EventId"])
    ours = {}
    for blocks, ev in zip(parsed["BlockIds"], parsed["EventId"]):
        for b in dict.fromkeys(blocks.split()):
            ours.setdefault(b, []).append(ev)

    traces = pd.read_csv(ROOT / "data/raw/HDFS_v1/preprocessed/Event_traces.csv",
                         usecols=["BlockId", "Features"])
    pairs, aligned, mismatched = Counter(), 0, 0
    for block, feats in zip(traces["BlockId"], traces["Features"]):
        gt = feats.strip("[]").split(",")
        mine = ours.get(block, [])
        if len(mine) != len(gt):
            mismatched += 1
            continue
        aligned += 1
        pairs.update(zip(mine, gt))

    print(f"blocks: {len(traces):,} in traces, {len(ours):,} in our parse, "
          f"{aligned:,} aligned, {mismatched:,} with a different length")

    gt_per_ours, ours_per_gt = {}, {}
    for (o, g), n in pairs.items():
        gt_per_ours.setdefault(o, set()).add(g)
        ours_per_gt.setdefault(g, set()).add(o)
    total = sum(pairs.values())
    correct = sum(n for (o, g), n in pairs.items()
                  if len(gt_per_ours[o]) == 1 and len(ours_per_gt[g]) == 1)
    print(f"events: {len(gt_per_ours)} ours, {len(ours_per_gt)} ground truth")
    print(f"Grouping Accuracy: {correct / total:.4f} ({correct:,} of {total:,} line-block pairs)")

    tmpl = pd.read_csv(ROOT / "data/parsed/HDFS/HDFS.log_templates.csv").set_index("EventId")
    print("\nGround-truth events split across several of our events:")
    for g, os_ in sorted(ours_per_gt.items(), key=lambda x: int(x[0][1:])):
        if len(os_) > 1:
            print(f"  {g}: {len(os_)} events, e.g. {tmpl.loc[sorted(os_)[0], 'EventTemplate'][:90]}")
    merged = {o: gs for o, gs in gt_per_ours.items() if len(gs) > 1}
    print(f"Our events merging several ground-truth events: {len(merged)}")
    for o, gs in merged.items():
        print(f"  {o} -> {sorted(gs)}: {tmpl.loc[o, 'EventTemplate'][:90]}")


if __name__ == "__main__":
    main()
