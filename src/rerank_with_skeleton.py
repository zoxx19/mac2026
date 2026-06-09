"""
Re-rank Large TAD predictions using skeleton class confidence as a boost signal.

For each Large proposal (video, t_start, t_end, label, score):
    skeleton_conf = max skeleton score for that label in that video (0 if absent)
    new_score     = score * (1 + alpha * skeleton_conf)

Segment boundaries are unchanged; only scores are re-weighted.

Usage
-----
Single alpha:
    python src/rerank_with_skeleton.py --split val --alpha 0.3 --evaluate

Alpha sweep (with baseline at alpha=0):
    python src/rerank_with_skeleton.py --split val --sweep
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict

PRED_BASE = "predictions"
OPENTAD_DIR = "../../OpenTAD"
DEFAULT_ANNO = (
    "."
    "/data/MMA-52/Annotations/adatad/mma52_anno.json"
)

SWEEP_ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


# ── core algorithm ────────────────────────────────────────────────────────────

def build_skeleton_conf(skeleton_results):
    """
    Per-video, per-label maximum skeleton confidence.
    Returns {video_id: {label: max_score}}.
    """
    conf = {}
    for vid, proposals in skeleton_results.items():
        per_label = defaultdict(float)
        for p in proposals:
            lbl = p["label"]
            s = float(p["score"])
            if s > per_label[lbl]:
                per_label[lbl] = s
        conf[vid] = dict(per_label)
    return conf


def rerank(large_results, skeleton_conf, alpha):
    """
    Apply skeleton boost to Large proposals.
    Returns new results dict in same format; boundaries unchanged.
    """
    out = {}
    for vid, proposals in large_results.items():
        skel = skeleton_conf.get(vid, {})
        boosted = []
        for p in proposals:
            sc = skel.get(p["label"], 0.0)
            new_score = float(p["score"]) * (1.0 + alpha * sc)
            boosted.append({
                "segment": p["segment"],
                "label": p["label"],
                "score": new_score,
            })
        boosted.sort(key=lambda x: x["score"], reverse=True)
        out[vid] = boosted
    return out


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(results, anno_path, split, tiou_thresholds=None):
    """
    Evaluate results dict using OpenTAD's mAP evaluator.
    Returns {tiou: mAP, 'average_mAP': float} or None on import error.
    """
    if tiou_thresholds is None:
        tiou_thresholds = [0.2, 0.5, 0.7]

    if OPENTAD_DIR not in sys.path:
        sys.path.insert(0, OPENTAD_DIR)
    try:
        from opentad.evaluations import mAP as mAPEval
    except ImportError as e:
        print(f"[WARN] Cannot import opentad.evaluations.mAP: {e}")
        return None

    subset = {"val": "validation", "test": "testing"}[split]
    pred_dict = {"results": results}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pred_dict, f)
        tmp = f.name

    try:
        ev = mAPEval(
            ground_truth_filename=anno_path,
            prediction_filename=tmp,
            subset=subset,
            tiou_thresholds=tiou_thresholds,
        )
        return ev.evaluate()
    finally:
        os.unlink(tmp)


# ── CLI helpers ───────────────────────────────────────────────────────────────

def default_large_path(split):
    return os.path.join(PRED_BASE, f"large_{split}_original.json")


def default_skeleton_path(split):
    return os.path.join(PRED_BASE, f"skeleton_{split}.json")


def load_results(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("results", d)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-rank Large TAD predictions with skeleton class confidence.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--large_preds", default=None,
                        help="Large prediction JSON (default: large_{split}_original.json)")
    parser.add_argument("--skeleton_preds", default=None,
                        help="Skeleton prediction JSON (default: skeleton_{split}.json)")
    parser.add_argument("--output", default=None,
                        help="Output path (default: predictions/reranked_{split}.json)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Skeleton boost factor (default 0.3)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Score output with opentad mAP evaluator")
    parser.add_argument("--anno", default=DEFAULT_ANNO,
                        help="Annotation JSON for evaluation")
    parser.add_argument("--sweep", action="store_true",
                        help=f"Sweep alphas {SWEEP_ALPHAS} and print mAP table; "
                             "saves best-alpha output to --output")
    args = parser.parse_args()

    split = args.split
    large_path = args.large_preds or default_large_path(split)
    skel_path = args.skeleton_preds or default_skeleton_path(split)
    out_path = args.output or os.path.join(PRED_BASE, f"reranked_{split}.json")

    # Load
    print(f"Loading large   : {large_path}")
    large_results = load_results(large_path)
    print(f"  {len(large_results)} videos, "
          f"{sum(len(v) for v in large_results.values()):,} proposals")

    print(f"Loading skeleton: {skel_path}")
    skel_results = load_results(skel_path)
    print(f"  {len(skel_results)} videos, "
          f"{sum(len(v) for v in skel_results.values()):,} proposals")

    skeleton_conf = build_skeleton_conf(skel_results)

    # ── sweep mode ──
    if args.sweep:
        print(f"\nAlpha sweep on split={split}")
        print(f"  {'alpha':>7}  {'mAP@0.2':>9}  {'mAP@0.5':>9}  {'mAP@0.7':>9}  {'avg_mAP':>9}")
        print("  " + "-" * 55)

        best_alpha = 0.0
        best_avg = -1.0
        best_results = None

        for alpha in SWEEP_ALPHAS:
            reranked = rerank(large_results, skeleton_conf, alpha)
            ev = evaluate(reranked, args.anno, split)
            if ev is None:
                print(f"  {alpha:>7.2f}  [evaluation failed]")
                continue
            m02 = ev.get("mAP@0.2", float("nan"))
            m05 = ev.get("mAP@0.5", float("nan"))
            m07 = ev.get("mAP@0.7", float("nan"))
            avg = ev.get("average_mAP", float("nan"))
            marker = " ← best" if avg > best_avg else ""
            print(f"  {alpha:>7.2f}  {m02:>9.4f}  {m05:>9.4f}  {m07:>9.4f}  {avg:>9.4f}{marker}")
            if avg > best_avg:
                best_avg = avg
                best_alpha = alpha
                best_results = reranked

        print(f"\n  Best alpha: {best_alpha}  →  avg_mAP={best_avg:.4f}")

        if best_results is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"results": best_results}, f, indent=2)
            print(f"  Saved best-alpha result → {out_path}")
        return

    # ── single alpha mode ──
    print(f"\nRe-ranking with alpha={args.alpha} ...")
    reranked = rerank(large_results, skeleton_conf, args.alpha)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": reranked}, f, indent=2)
    print(f"Saved → {out_path}")

    if args.evaluate:
        ev = evaluate(reranked, args.anno, split)
        if ev:
            print("\n--- Evaluation ---")
            for tiou in [0.2, 0.5, 0.7]:
                k = f"mAP@{tiou}"
                if k in ev:
                    print(f"  {k}: {ev[k]:.4f}")
            if "average_mAP" in ev:
                print(f"  avg mAP: {ev['average_mAP']:.4f}")


if __name__ == "__main__":
    main()
