"""
Multi-stream ensemble fusion for MAC 2026 Track 2.

Prediction files are discovered at runtime from the canonical location:
  predictions/{stream}_{split}.json

Streams: large, rgb, body, skeleton, tg

Usage examples
--------------
# Auto-discover all available streams, weight by val mAP
python src/fusion.py --split val --mode auto

# Restrict to subset
python src/fusion.py --split val --mode auto --streams large,skeleton,tg

# Override a stream's val mAP used for weighting
python src/fusion.py --split test --mode auto --streams large,skeleton \
    --large_map 24.5 --skeleton_map 7.11

# See what prediction files exist without running fusion
python src/fusion.py --split val --list_streams

# Evaluate fused output against ground truth
python src/fusion.py --split val --mode auto --evaluate
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────

PRED_BASE = "predictions"
OPENTAD_DIR = "../../OpenTAD"
DEFAULT_ANNO = (
    "."
    "/data/MMA-52/Annotations/adatad/mma52_anno.json"
)

ALL_STREAMS = ["large", "rgb", "body", "skeleton", "tg"]

# Default val mAPs — override with --{stream}_map
DEFAULT_MAPS = {
    "large":    21.61,
    "rgb":       2.12,
    "body":      2.08,
    "skeleton":  7.11,
    "tg":        0.0,
}


# ── skeleton format converter ─────────────────────────────────────────────────

def convert_skeleton_preds(data):
    """
    Normalise skeleton predictions to the standard AdaTAD format:
      {"results": {video_id: [{"segment": [t_s, t_e], "label": str, "score": float}]}}

    Handles three source formats:
      1. Standard — already correct, pass through.
      2. Frame-based — {"fps": N, "results": {vid: [{"segment": [f,f], ...}]}}
         Divides segment values by fps.
      3. Old flat list — [{video_id, start/t_start, end/t_end, label/class, score}]
         Reshapes into the standard dict.
    """
    # Format 1: standard (has "results", no "fps")
    if isinstance(data, dict) and "results" in data and "fps" not in data:
        return data

    # Format 2: frame-based with fps
    if isinstance(data, dict) and "fps" in data and "results" in data:
        fps = float(data["fps"])
        results = {}
        for vid, proposals in data["results"].items():
            converted = []
            for p in proposals:
                seg = p["segment"]
                converted.append({
                    "segment": [seg[0] / fps, seg[1] / fps],
                    "label": p["label"],
                    "score": float(p["score"]),
                })
            results[vid] = converted
        return {"results": results}

    # Format 3: old flat list (from legacy fusion.py era)
    if isinstance(data, list):
        results = defaultdict(list)
        for p in data:
            vid = str(p.get("video_id", ""))
            label = p.get("label", str(p.get("class", "")))
            start = float(p.get("t_start", p.get("start", 0.0)))
            end = float(p.get("t_end", p.get("end", 0.0)))
            score = float(p.get("score", 0.0))
            results[vid].append({"segment": [start, end], "label": label, "score": score})
        return {"results": dict(results)}

    return data


# ── loading helpers ───────────────────────────────────────────────────────────

def pred_path(stream, split):
    return os.path.join(PRED_BASE, f"{stream}_{split}.json")


def available_streams(split, candidates):
    """Return subset of candidates that have prediction files on disk."""
    return [s for s in candidates if os.path.exists(pred_path(s, split))]


def load_stream(stream, split):
    """Load and normalise one stream's predictions. Returns results dict or None."""
    path = pred_path(stream, split)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if stream == "skeleton":
        data = convert_skeleton_preds(data)
    results = data.get("results", {})
    return results


def normalize_scores(results):
    """Scale all scores in a stream to [0, 1] in-place; returns results."""
    all_scores = [p["score"] for proposals in results.values() for p in proposals]
    if not all_scores:
        return results
    max_score = max(all_scores)
    if max_score <= 0:
        return results
    for proposals in results.values():
        for p in proposals:
            p["score"] /= max_score
    return results


# ── soft-NMS ──────────────────────────────────────────────────────────────────

def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def soft_nms(proposals, sigma=0.5):
    """
    Gaussian soft-NMS on a list of proposals sharing the same (video, label).
    Each proposal: {"segment": [t_s, t_e], "label": str, "score": float}.
    Returns list sorted by descending score.
    """
    segs = sorted(proposals, key=lambda x: x["score"], reverse=True)
    kept = []
    while segs:
        best = segs.pop(0)
        kept.append(best)
        remaining = []
        for s in segs:
            ov = _iou(best["segment"], s["segment"])
            if ov > 0:
                s = dict(s)
                s["score"] *= math.exp(-(ov ** 2) / sigma)
            remaining.append(s)
        segs = sorted(remaining, key=lambda x: x["score"], reverse=True)
    return kept


# ── weighting ─────────────────────────────────────────────────────────────────

def compute_weights(mode, maps, manual_weights, active_streams):
    if mode == "equal":
        return {s: 1.0 for s in active_streams}
    if mode == "manual":
        return {s: manual_weights.get(s, 1.0) for s in active_streams}
    # auto: proportional to val mAP
    total = sum(maps.get(s, 0.0) for s in active_streams)
    if total == 0:
        return {s: 1.0 / len(active_streams) for s in active_streams}
    return {s: maps.get(s, 0.0) / total for s in active_streams}


# ── fusion ────────────────────────────────────────────────────────────────────

def fuse(stream_results, weights, sigma=0.5):
    """
    Merge all streams' weighted proposals and apply soft-NMS per (video, label).
    Returns {video_id: [proposals sorted by descending score]}.
    """
    groups = defaultdict(list)
    for stream, results in stream_results.items():
        w = weights.get(stream, 1.0)
        for vid, proposals in results.items():
            for p in proposals:
                groups[(vid, p["label"])].append({
                    "segment": p["segment"],
                    "label": p["label"],
                    "score": p["score"] * w,
                })

    output = defaultdict(list)
    for (vid, _label), props in groups.items():
        output[vid].extend(soft_nms(props, sigma=sigma))

    return {vid: sorted(props, key=lambda x: x["score"], reverse=True)
            for vid, props in output.items()}


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_fused(results, anno_path, split):
    if OPENTAD_DIR not in sys.path:
        sys.path.insert(0, OPENTAD_DIR)
    try:
        from opentad.evaluations import mAP as mAPEvaluator
    except ImportError as e:
        print(f"[WARN] Cannot import opentad.evaluations.mAP: {e}")
        return

    subset = {"val": "validation", "test": "testing"}[split]
    tiou_thresholds = [0.2, 0.5, 0.7]
    pred_dict = {"results": results}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pred_dict, f)
        pred_path_tmp = f.name

    try:
        evaluator = mAPEvaluator(
            ground_truth_filename=anno_path,
            prediction_filename=pred_path_tmp,
            subset=subset,
            tiou_thresholds=tiou_thresholds,
        )
        result = evaluator.evaluate()
        print("\n--- Evaluation Results ---")
        for tiou in tiou_thresholds:
            key = f"mAP@{tiou}"
            if key in result:
                print(f"  {key}: {result[key]:.4f}")
        if "average_mAP" in result:
            print(f"  avg mAP: {result['average_mAP']:.4f}")
    finally:
        os.unlink(pred_path_tmp)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse multi-stream TAD predictions for MAC 2026 Track 2.")

    parser.add_argument("--split", choices=["val", "test"], required=True,
                        help="Dataset split to fuse")
    parser.add_argument("--streams", default=None,
                        help="Comma-separated stream subset, e.g. large,skeleton,tg "
                             "(default: auto-discover all with prediction files)")
    parser.add_argument("--list_streams", action="store_true",
                        help="Show which streams have prediction files and exit")
    parser.add_argument("--mode", choices=["auto", "manual", "equal"], default="auto",
                        help="auto=weight by val mAP, manual=explicit weights, equal=1/N")

    # Per-stream manual weights
    for s in ALL_STREAMS:
        parser.add_argument(f"--{s}_weight", type=float, default=1.0,
                            help=f"Weight for {s} stream (used in --mode manual)")

    # Per-stream val mAP overrides (used in --mode auto)
    for s in ALL_STREAMS:
        parser.add_argument(f"--{s}_map", type=float, default=DEFAULT_MAPS[s],
                            help=f"Val mAP for {s} stream (used in --mode auto, "
                                 f"default {DEFAULT_MAPS[s]})")

    parser.add_argument("--output", default=None,
                        help="Output path (default: PRED_BASE/fused_{split}.json)")
    parser.add_argument("--nms_sigma", type=float, default=0.5,
                        help="Gaussian soft-NMS sigma (default 0.5)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Score fused output against ground-truth annotation")
    parser.add_argument("--anno", default=DEFAULT_ANNO,
                        help="Ground-truth annotation JSON for evaluation")

    return parser.parse_args()


def main():
    args = parse_args()
    split = args.split

    # Determine candidate streams
    if args.streams:
        candidates = [s.strip() for s in args.streams.split(",") if s.strip()]
        invalid = [s for s in candidates if s not in ALL_STREAMS]
        if invalid:
            print(f"[ERROR] Unknown stream(s): {invalid}. Valid: {ALL_STREAMS}")
            raise SystemExit(1)
    else:
        candidates = ALL_STREAMS

    # --list_streams: just show availability and exit
    if args.list_streams:
        print(f"\nStream availability for split='{split}':")
        print(f"  {'Stream':<12} {'Available':>10}  {'Path'}")
        print("  " + "-" * 70)
        for s in candidates:
            path = pred_path(s, split)
            exists = os.path.exists(path)
            tag = "YES" if exists else "no"
            print(f"  {s:<12} {tag:>10}  {path}")
        return

    # Collect per-stream mAP and weight overrides from args
    val_maps = {s: getattr(args, f"{s}_map") for s in ALL_STREAMS}
    manual_weights = {s: getattr(args, f"{s}_weight") for s in ALL_STREAMS}

    # Load available streams
    stream_results = {}
    loaded_streams = []
    for s in candidates:
        results = load_stream(s, split)
        if results is None:
            print(f"[SKIP] {s}: no prediction file at {pred_path(s, split)}")
            continue
        results = normalize_scores(results)
        stream_results[s] = results
        loaded_streams.append(s)

    if not loaded_streams:
        print("\nNo prediction files found. Nothing to fuse.")
        raise SystemExit(1)

    # Compute fusion weights over loaded streams only
    weights = compute_weights(args.mode, val_maps, manual_weights, loaded_streams)

    # Print fusion table
    print(f"\nFusion  mode={args.mode}  split={split}  sigma={args.nms_sigma}")
    print(f"  {'Stream':<12} {'Weight':>8}  {'valMAP':>8}  {'Proposals':>10}")
    print("  " + "-" * 50)
    for s in loaded_streams:
        n = sum(len(v) for v in stream_results[s].values())
        print(f"  {s:<12} {weights[s]:>8.4f}  {val_maps[s]:>8.2f}%  {n:>10,}")

    # Fuse
    fused = fuse(stream_results, weights, sigma=args.nms_sigma)

    total_props = sum(len(v) for v in fused.values())
    n_vids = len(fused)
    avg_props = total_props / n_vids if n_vids > 0 else 0.0
    print(f"\n  Videos: {n_vids:,}  |  "
          f"Total proposals after NMS: {total_props:,}  |  "
          f"Avg/video: {avg_props:.1f}")

    # Save
    out_path = args.output or os.path.join(PRED_BASE, f"fused_{split}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": fused}, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Optional evaluation
    if args.evaluate:
        evaluate_fused(fused, args.anno, split)


if __name__ == "__main__":
    main()
