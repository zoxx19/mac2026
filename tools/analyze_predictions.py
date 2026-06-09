"""
Quick statistics for an AdaTAD / fused prediction JSON.

Prints proposal counts, per-video distribution, score range, segment-duration
stats, and the most frequent predicted labels. Useful for sanity-checking a
prediction file before fusion or submission (e.g. spotting an empty stream, a
degenerate score scale, or runaway proposal counts).

Usage
-----
    python tools/analyze_predictions.py predictions/large_test.json
    python tools/analyze_predictions.py predictions/fused_val.json --top 15
"""

import argparse
import json
from collections import Counter

import numpy as np


def load(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("results", data)


def main():
    ap = argparse.ArgumentParser(description="Summarise a prediction JSON.")
    ap.add_argument("input", help="Prediction JSON ({results: {vid: [{segment,label,score}]}}).")
    ap.add_argument("--top", type=int, default=10, help="How many top labels to list.")
    args = ap.parse_args()

    results = load(args.input)
    n_videos = len(results)
    counts = [len(v) for v in results.values()]
    scores, durations, labels = [], [], Counter()
    for props in results.values():
        for p in props:
            scores.append(float(p["score"]))
            seg = p["segment"]
            durations.append(float(seg[1]) - float(seg[0]))
            labels[p["label"]] += 1

    total = sum(counts)
    print(f"File: {args.input}")
    print(f"  videos               : {n_videos:,}")
    print(f"  total proposals      : {total:,}")
    if n_videos:
        print(f"  proposals/video      : min {min(counts)}  median {int(np.median(counts))}  "
              f"mean {total / n_videos:.1f}  max {max(counts)}")
    if scores:
        s = np.array(scores)
        print(f"  score range          : {s.min():.4f} .. {s.max():.4f}  (mean {s.mean():.4f})")
    if durations:
        d = np.array(durations)
        print(f"  segment duration (s) : min {d.min():.2f}  median {np.median(d):.2f}  max {d.max():.2f}")
        neg = int((d <= 0).sum())
        if neg:
            print(f"  WARNING: {neg} proposals with non-positive duration")

    print(f"\n  top {args.top} labels:")
    for label, c in labels.most_common(args.top):
        print(f"    {c:>8,}  {label}")


if __name__ == "__main__":
    main()
