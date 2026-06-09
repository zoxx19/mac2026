"""
Flexible multi-checkpoint ensemble for MAC 2026 Track 2.

Fuses N arbitrary prediction JSON files with configurable per-input weights.
Unlike fusion.py (which is limited to the known stream names
large/rgb/body/skeleton/tg discovered by convention), this takes explicit input
paths and weights, so any set of checkpoints or post-processed predictions can
be combined.

Algorithm
---------
  1. Load all N prediction JSONs.
  2. Per input, normalise scores to [0, 1] (divide by that input's max score),
     so inputs on different score scales contribute comparably.
  3. Pool every video's proposals across all inputs (tagging each with its
     input's weight).
  4. Merge per (video, label) with one of:
       softnms - Gaussian Soft-NMS on weight-scaled scores (default).
       wbf     - Weighted Boxes Fusion: cluster overlapping proposals
                 (IoU > --wbf_iou) and emit one weight-averaged segment per
                 cluster, down-weighted when supported by fewer inputs.
  5. Sort by score and keep the top_k proposals per video (top_k<=0 keeps all).
  6. Save the fused result.

Usage
-----
python multi_ensemble.py \
    --inputs pred1.json pred2.json pred3.json \
    --weights 0.4 0.4 0.2 \
    --output fused.json --split val --evaluate \
    --merge softnms --sigma 0.5 --top_k 200

python multi_ensemble.py \
    --inputs pred1.json pred2.json \
    --weights 0.5 0.5 \
    --output fused.json --split val --evaluate \
    --merge wbf --wbf_iou 0.5 --top_k 0
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────

PRED_BASE = "predictions"
OPENTAD_DIR = "../../OpenTAD"
DEFAULT_ANNO = (
    "."
    "/data/MMA-52/Annotations/adatad/mma52_anno.json"
)


# ── IO helpers ────────────────────────────────────────────────────────────────

def resolve_pred_path(path):
    """Allow bare filenames (resolved under PRED_BASE) or absolute paths."""
    if os.path.isabs(path):
        return path
    candidate = os.path.join(PRED_BASE, path)
    return candidate if os.path.exists(candidate) else path


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("results", data)


def normalize_scores(results):
    """Scale all scores in one input to [0, 1] in-place; returns results."""
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


# ── soft-NMS (Gaussian, numpy-vectorised) ─────────────────────────────────────

def soft_nms(proposals, sigma=0.5):
    """
    Gaussian Soft-NMS on proposals sharing one (video, label).

    Vectorised with numpy: each iteration picks the current max-score proposal
    and decays the score of every remaining proposal by exp(-IoU^2 / sigma).
    Identical semantics to the classic per-element implementation, but O(n^2)
    array ops instead of O(n^3) repeated Python sorts. Returns proposals in
    selection order (descending effective score).
    """
    n = len(proposals)
    if n == 0:
        return []
    if n == 1:
        return [dict(proposals[0])]

    starts = np.array([p["segment"][0] for p in proposals], dtype=np.float64)
    ends = np.array([p["segment"][1] for p in proposals], dtype=np.float64)
    scores = np.array([p["score"] for p in proposals], dtype=np.float64)
    active = np.ones(n, dtype=bool)
    order = []
    eff_score = np.empty(n, dtype=np.float64)

    for _ in range(n):
        masked = np.where(active, scores, -np.inf)
        i = int(np.argmax(masked))
        if masked[i] == -np.inf:
            break
        active[i] = False
        order.append(i)
        eff_score[i] = scores[i]

        inter = np.clip(np.minimum(ends[i], ends) - np.maximum(starts[i], starts), 0.0, None)
        union = np.maximum(ends[i], ends) - np.minimum(starts[i], starts)
        iou = np.where(union > 0, inter / union, 0.0)
        scores = np.where(active, scores * np.exp(-(iou ** 2) / sigma), scores)

    return [{"segment": proposals[i]["segment"],
             "label": proposals[i]["label"],
             "score": float(eff_score[i])} for i in order]


# ── weighted boxes fusion (1-D temporal) ──────────────────────────────────────

def _iou_1d(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def weighted_boxes_fusion(proposals, iou_thr=0.5, n_models=1):
    """
    Weighted Boxes Fusion (Solovyev et al.) adapted to 1-D temporal segments,
    on proposals that share one (video, label).

    Each proposal must carry: segment [t_s, t_e], label, score, weight (the
    per-input weight of the model it came from).

      * Sort proposals by score descending.
      * Greedily assign each to the first cluster whose current fused segment
        has IoU > iou_thr; else start a new cluster.
      * A cluster's fused segment is the (weight*score)-weighted average of its
        members' endpoints; its fused score is the weight-weighted average of
        member scores.
      * Finally rescale each cluster's score by min(n_models, n_members)/n_models
        so proposals backed by fewer inputs are down-weighted.

    Returns a list of fused proposals (unsorted).
    """
    if not proposals:
        return []

    def fuse_cluster(members):
        wsum = sum(m["weight"] for m in members)
        swsum = sum(m["weight"] * m["score"] for m in members)
        cw = swsum if swsum > 0 else 1.0
        start = sum(m["weight"] * m["score"] * m["segment"][0] for m in members) / cw
        end = sum(m["weight"] * m["score"] * m["segment"][1] for m in members) / cw
        score = swsum / wsum if wsum > 0 else 0.0
        return {"segment": [start, end], "label": members[0]["label"], "score": score}

    order = sorted(proposals, key=lambda p: p["score"], reverse=True)
    clusters = []  # list[list[proposal]]
    reps = []      # list[fused proposal], kept in sync with clusters

    for p in order:
        best, best_iou = -1, iou_thr
        for ci, rep in enumerate(reps):
            iou = _iou_1d(p["segment"], rep["segment"])
            if iou > best_iou:
                best_iou, best = iou, ci
        if best == -1:
            clusters.append([p])
            reps.append(fuse_cluster([p]))
        else:
            clusters[best].append(p)
            reps[best] = fuse_cluster(clusters[best])

    denom = max(n_models, 1)
    fused = []
    for ci, rep in enumerate(reps):
        rep = dict(rep)
        rep["score"] *= min(denom, len(clusters[ci])) / denom
        fused.append(rep)
    return fused


# ── ensemble ──────────────────────────────────────────────────────────────────

def ensemble(inputs_results, weights, merge="softnms", sigma=0.5, wbf_iou=0.5, top_k=200):
    """
    Pool score-normalised proposals from all inputs and merge per (video, label)
    using `merge` ("softnms" or "wbf"). Returns {video_id: [proposals]} truncated
    to top_k (top_k<=0 keeps all).
    """
    n_models = len(weights)
    groups = defaultdict(list)  # (vid, label) -> pooled proposals (raw score + weight)
    for results, w in zip(inputs_results, weights):
        for vid, proposals in results.items():
            for p in proposals:
                groups[(vid, p["label"])].append({
                    "segment": p["segment"],
                    "label": p["label"],
                    "score": p["score"],
                    "weight": w,
                })

    output = defaultdict(list)
    for (vid, _label), props in groups.items():
        if merge == "wbf":
            merged = weighted_boxes_fusion(props, iou_thr=wbf_iou, n_models=n_models)
        else:  # softnms: pre-scale score by input weight, then Soft-NMS
            scaled = [{"segment": p["segment"], "label": p["label"],
                       "score": p["score"] * p["weight"]} for p in props]
            merged = soft_nms(scaled, sigma=sigma)
        output[vid].extend(merged)

    fused = {}
    for vid, props in output.items():
        props.sort(key=lambda x: x["score"], reverse=True)
        fused[vid] = props[:top_k] if top_k and top_k > 0 else props
    return fused


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(results, anno_path, split):
    if OPENTAD_DIR not in sys.path:
        sys.path.insert(0, OPENTAD_DIR)
    try:
        from opentad.evaluations import mAP as mAPEvaluator
    except ImportError as e:
        print(f"[WARN] Cannot import opentad.evaluations.mAP: {e}")
        return

    subset = {"val": "validation", "test": "testing"}[split]
    tiou_thresholds = [0.2, 0.5, 0.7]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"results": results}, f)
        tmp = f.name
    try:
        evaluator = mAPEvaluator(
            ground_truth_filename=anno_path,
            prediction_filename=tmp,
            subset=subset,
            tiou_thresholds=tiou_thresholds,
        )
        result = evaluator.evaluate()
        print("\n--- Ensemble Evaluation Results ---")
        for tiou in tiou_thresholds:
            key = f"mAP@{tiou}"
            if key in result:
                print(f"  {key}: {result[key] * 100:.2f}%")
        if "average_mAP" in result:
            print(f"  avg mAP: {result['average_mAP'] * 100:.2f}%")
    finally:
        os.unlink(tmp)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Flexible N-way prediction ensemble.")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Prediction JSONs (bare names resolved under PRED_BASE).")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-input weights (default: equal). Must match --inputs length.")
    p.add_argument("--output", default="multi_ensemble.json",
                   help="Output JSON (bare name written under PRED_BASE).")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--evaluate", action="store_true", help="Run mAP evaluation.")
    p.add_argument("--anno", default=DEFAULT_ANNO, help="Ground-truth annotation JSON.")
    p.add_argument("--merge", choices=["softnms", "wbf"], default="softnms",
                   help="Per-(video,label) merge strategy (default: softnms).")
    p.add_argument("--sigma", "--nms_sigma", dest="sigma", type=float, default=0.5,
                   help="Gaussian Soft-NMS sigma (merge=softnms).")
    p.add_argument("--wbf_iou", type=float, default=0.5,
                   help="IoU threshold for clustering proposals (merge=wbf).")
    p.add_argument("--top_k", type=int, default=200,
                   help="Keep top_k proposals per video after fusion (<=0 = keep all).")
    return p.parse_args()


def main():
    args = parse_args()

    n = len(args.inputs)
    if args.weights is None:
        weights = [1.0 / n] * n
    else:
        if len(args.weights) != n:
            print(f"[ERROR] --weights ({len(args.weights)}) must match --inputs ({n}).")
            raise SystemExit(1)
        weights = args.weights

    in_paths = [resolve_pred_path(p) for p in args.inputs]
    out_path = args.output if os.path.isabs(args.output) else os.path.join(PRED_BASE, args.output)

    merge_desc = (f"wbf(iou={args.wbf_iou})" if args.merge == "wbf"
                  else f"softnms(sigma={args.sigma})")
    print(f"Multi-ensemble  split={args.split}  merge={merge_desc}  top_k={args.top_k}")
    print(f"  {'Input':<55} {'Weight':>8}  {'Proposals':>10}")
    print("  " + "-" * 80)

    inputs_results = []
    for path, w in zip(in_paths, weights):
        if not os.path.exists(path):
            print(f"[ERROR] Missing input: {path}")
            raise SystemExit(1)
        results = normalize_scores(load_results(path))
        inputs_results.append(results)
        n_props = sum(len(v) for v in results.values())
        print(f"  {os.path.basename(path):<55} {w:>8.4f}  {n_props:>10,}")

    fused = ensemble(inputs_results, weights, merge=args.merge, sigma=args.sigma,
                     wbf_iou=args.wbf_iou, top_k=args.top_k)

    n_props = sum(len(v) for v in fused.values())
    n_vids = len(fused)
    print(f"\n  Fused: {n_vids:,} videos  |  {n_props:,} proposals  "
          f"|  avg/video: {n_props / max(n_vids, 1):.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": fused}, f)
    print(f"\nSaved -> {out_path}")

    if args.evaluate:
        evaluate(fused, args.anno, args.split)


if __name__ == "__main__":
    main()
