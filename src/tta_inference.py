"""
Test-Time Augmentation (TTA) for MAC 2026 Track 2 AdaTAD predictions.

Pure post-processing TTA — no GPU, no retraining. Instead of re-running the
model on temporally augmented videos, we synthesise the augmented hypotheses
directly from an existing prediction JSON and merge them with Soft-NMS.

Variants
--------
  original   pass-through (the current test pipeline output)
  reverse    temporal flip:  [t_s, t_e] -> [dur - t_e, dur - t_s]
  scale09    speed perturbation 0.9x:  [t_s, t_e] -> [0.9*t_s, 0.9*t_e]
  scale11    speed perturbation 1.1x:  [t_s, t_e] -> [1.1*t_s, 1.1*t_e]
  scale095   speed perturbation 0.95x
  scale105   speed perturbation 1.05x

All requested variants are pooled per (video, label) and merged with Gaussian
Soft-NMS, which sharpens boundaries and improves recall.

Durations (needed for temporal reversal) come from the annotation JSON's
`duration` field. For videos absent from the annotation (e.g. the test split),
the duration is estimated as max(t_end) over that video's proposals, falling
back to the median known duration.

Usage
-----
python tta_inference.py --split val --evaluate \
    --variants original,reverse,scale09,scale11
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

ALL_VARIANTS = ["original", "reverse", "scale09", "scale11", "scale095", "scale105"]
SCALE_FACTORS = {
    "scale09": 0.9,
    "scale11": 1.1,
    "scale095": 0.95,
    "scale105": 1.05,
}


# ── path / IO helpers ─────────────────────────────────────────────────────────

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


def load_durations(anno_path):
    """video_id -> duration (seconds) from the annotation database."""
    with open(anno_path) as f:
        anno = json.load(f)
    db = anno.get("database", anno)
    return {vid: float(entry["duration"]) for vid, entry in db.items() if "duration" in entry}


def duration_for(vid, proposals, duration_map, median_dur):
    """Resolve a video's duration: annotation > max(t_end) > median fallback."""
    if vid in duration_map:
        return duration_map[vid]
    if proposals:
        return max(p["segment"][1] for p in proposals)
    return median_dur


# ── variant transforms ────────────────────────────────────────────────────────

def transform_proposals(proposals, variant, duration):
    """Return a new list of proposals with the variant's temporal transform applied."""
    out = []
    for p in proposals:
        t_s, t_e = p["segment"]
        if variant == "original":
            new_seg = [t_s, t_e]
        elif variant == "reverse":
            new_seg = [duration - t_e, duration - t_s]
        elif variant in SCALE_FACTORS:
            f = SCALE_FACTORS[variant]
            new_seg = [t_s * f, t_e * f]
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Clamp to a sane temporal range and drop degenerate spans
        lo = max(0.0, min(new_seg))
        hi = max(new_seg)
        if duration > 0:
            hi = min(hi, duration)
        if hi <= lo:
            continue
        out.append({"segment": [lo, hi], "label": p["label"], "score": float(p["score"])})
    return out


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


# ── TTA merge ─────────────────────────────────────────────────────────────────

def run_tta(results, variants, duration_map, sigma=0.5, max_per_video=2000):
    """
    Build augmented hypotheses for each variant and merge with Soft-NMS.

    Soft-NMS decays overlapping scores but does not delete proposals, so pooling
    K variants multiplies the proposal count by ~K. We therefore keep only the
    top `max_per_video` proposals per video after the merge — matching AdaTAD's
    max_seg_num and the baseline density, which keeps evaluation fast and the
    comparison fair (set max_per_video<=0 to keep everything).
    """
    durations = list(duration_map.values())
    median_dur = sorted(durations)[len(durations) // 2] if durations else 0.0

    groups = defaultdict(list)  # (vid, label) -> pooled proposals
    for vid, proposals in results.items():
        dur = duration_for(vid, proposals, duration_map, median_dur)
        for variant in variants:
            for p in transform_proposals(proposals, variant, dur):
                groups[(vid, p["label"])].append(p)

    output = defaultdict(list)
    for (vid, _label), props in groups.items():
        output[vid].extend(soft_nms(props, sigma=sigma))

    merged = {}
    for vid, props in output.items():
        props.sort(key=lambda x: x["score"], reverse=True)
        merged[vid] = props[:max_per_video] if max_per_video and max_per_video > 0 else props
    return merged


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
        print("\n--- TTA Evaluation Results ---")
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
    p = argparse.ArgumentParser(description="Post-processing TTA for AdaTAD predictions.")
    p.add_argument("--input", default="large_val_original.json",
                   help="Input prediction JSON (bare name resolved under PRED_BASE).")
    p.add_argument("--output", default="tta_val.json",
                   help="Output prediction JSON (bare name written under PRED_BASE).")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--evaluate", action="store_true", help="Run mAP evaluation.")
    p.add_argument("--anno", default=DEFAULT_ANNO, help="Ground-truth annotation JSON.")
    p.add_argument("--variants", default="original,reverse,scale09,scale11",
                   help=f"Comma-separated subset of {ALL_VARIANTS}.")
    p.add_argument("--nms_sigma", type=float, default=0.5, help="Gaussian Soft-NMS sigma.")
    p.add_argument("--max_per_video", type=int, default=2000,
                   help="Keep top-K proposals per video after merge (<=0 = keep all).")
    return p.parse_args()


def main():
    args = parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    invalid = [v for v in variants if v not in ALL_VARIANTS]
    if invalid:
        print(f"[ERROR] Unknown variant(s): {invalid}. Valid: {ALL_VARIANTS}")
        raise SystemExit(1)

    in_path = resolve_pred_path(args.input)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(PRED_BASE, args.output)

    print(f"TTA  split={args.split}  sigma={args.nms_sigma}")
    print(f"  input  : {in_path}")
    print(f"  variants: {variants}")

    results = load_results(in_path)
    duration_map = load_durations(args.anno)
    n_in = sum(len(v) for v in results.values())
    print(f"  videos : {len(results):,}  |  input proposals: {n_in:,}")

    tta = run_tta(results, variants, duration_map, sigma=args.nms_sigma,
                  max_per_video=args.max_per_video)

    n_out = sum(len(v) for v in tta.values())
    n_vids = len(tta)
    print(f"\n  After TTA merge: {n_vids:,} videos  |  {n_out:,} proposals  "
          f"|  avg/video: {n_out / max(n_vids, 1):.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": tta}, f)
    print(f"\nSaved -> {out_path}")

    if args.evaluate:
        evaluate(tta, args.anno, args.split)


if __name__ == "__main__":
    main()
