"""Which BGL 6-hour windowing gives the paper's 718 sessions (Wu, Li and Khomh, EMSE 2023, Table 1, Sec. 4.3.2)?

The paper reports 718 sessions averaging 6,565 lines (about 4.71M of BGL's 4,747,963). Our build_mcv.py anchors
windows at the first timestamp and keeps every non-empty window: 826. This prints the count, mean lines and
anomaly share under each candidate rule, so the rule that reproduces 718 can be named rather than guessed.

Usage: python tests/bgl_window_count.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 6 * 3600


def describe(name: str, window: pd.Series, label: pd.Series) -> None:
    sizes = window.value_counts()
    anom = label.ne("-").groupby(window).any()
    print(f"{name:<52} {len(sizes):>5} windows  mean {sizes.mean():>7,.0f} lines  "
          f"anomalous {anom.mean():.1%}")


def main() -> None:
    df = pd.read_parquet(ROOT / "data/parsed/BGL/BGL.log_structured.parquet", columns=["Label", "Timestamp"])
    ts, label = df["Timestamp"], df["Label"]
    print(f"{len(df):,} lines, timestamps {ts.min()} to {ts.max()}, "
          f"{(ts.diff() < 0).sum():,} out-of-order steps, span {(ts.max() - ts.min()) / 86400:.1f} days")
    describe("anchored at first timestamp (ours)", (ts - ts.min()) // WIDTH, label)
    describe("aligned to epoch 6 h boundaries", ts // WIDTH, label)
    # A window that starts at a line and closes 6 h later, the next opening at the next line (loglizer style).
    starts, start = [], None
    for t in ts.sort_values().to_numpy():
        if start is None or t - start >= WIDTH:
            start = t
        starts.append(start)
    describe("opened at a line, closed 6 h later (gap-skipping)", pd.Series(starts, index=ts.sort_values().index).reindex(ts.index), label)
    sizes = ((ts - ts.min()) // WIDTH).value_counts()
    for k in (10, 50, 100, 200, 500, 1000):
        kept = sizes[sizes >= k]
        print(f"  ours, keeping windows with >= {k:>4} lines: {len(kept):>4} windows, mean {kept.mean():,.0f}")


if __name__ == "__main__":
    main()
