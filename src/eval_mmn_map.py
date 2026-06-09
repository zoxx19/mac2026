"""
eval_mmn_map_v2.py — Improved sliding-window inference + mAP for 52-class MMAD.

Key fixes over v1:
  1. Threshold sweep: finds best threshold automatically (0.02 → 0.50)
  2. Segment score = MAX (not mean) over constituent windows → better ranking
  3. Gap filling: bridges up to --max_gap consecutive below-threshold windows
     (prevents one weak window from splitting one real action into two segments)
  4. Frame-level mAP: --mode frame computes per-frame AP (no tIoU needed)
     Use this if MAC 2026 submission metric is frame-based mAP

Usage — threshold sweep (detection mAP, find best threshold):
    python eval_mmn_map_v2.py --mode sweep

Usage — single-threshold eval (after sweep):
    python eval_mmn_map_v2.py --mode eval --threshold 0.10

Usage — frame-level mAP (if challenge metric is frame-based):
    python eval_mmn_map_v2.py --mode frame
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

MMN_REPO = "../Micro-action-skeleton/MMN"
sys.path.insert(0, os.path.join(MMN_REPO, "torchlight"))
sys.path.insert(0, os.path.join(MMN_REPO, "torchpack"))
sys.path.insert(0, MMN_REPO)

from model.MMN import MMN

NUM_CLASSES = 52
JOINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 3, 4]

DEFAULT_CHECKPOINT = (
    "../Micro-action-skeleton/MMN"
    "/work_dir/train/mmad_52class_J/runs-30-148800.pt"
)
DEFAULT_SKELETON_DIR = "data/skeleton/val"
DEFAULT_ANN_CSV = (
    "."
    "/data/MMA-52/Annotations/val.csv"
)


# ─────────────────────────────────────────────────────────── model loading ──

def load_model(checkpoint_path, device):
    model = MMN(
        in_channels=2, num_classes=NUM_CLASSES, num_people=1,
        num_frames=32, num_points=17, kernel_size=3, num_heads=4,
        drop=0.0, head_drop=0.1, drop_path=0.3, mlp_ratio=2.0, index_t=True,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


# ────────────────────────────────────────────────────── preprocessing / inference ──

def preprocess_window(window):
    """Val-branch preprocessing. window: (T, V, C) → (1,C,T,V,1), (1,T)"""
    value = window.copy()
    T, V, C = value.shape
    center = (value[0, 9, :] + value[0, 10, :]) / 2.0
    value -= center
    flat = value.reshape(-1, C)
    eps = 1e-6
    flat = (flat - flat.min(0)) / (flat.max(0) - flat.min(0) + eps) * 2 - 1
    value = flat.reshape(T, V, C)
    idx = np.linspace(0, T - 1, T).astype(int)
    index_t = (2 * idx.astype(np.float32) / T - 1)
    data = np.transpose(value[idx], (2, 0, 1))[:, :, :, np.newaxis]
    return (torch.from_numpy(data).unsqueeze(0).float(),
            torch.from_numpy(index_t).unsqueeze(0).float())


def sliding_window_scores(npy_path, model, window_size, stride, device):
    """Returns list of (start_frame, end_frame, scores_52)."""
    kp = np.load(npy_path)[:, JOINTS, :2].astype(np.float32)  # (N,17,2)
    N = kp.shape[0]
    results = []
    for t in range(0, max(1, N - window_size + 1), stride):
        if t + window_size > N:
            break
        d, ix = preprocess_window(kp[t:t + window_size])
        with torch.no_grad():
            scores = torch.sigmoid(model(d.to(device), ix.to(device)))[0].cpu().numpy()
        results.append((t, t + window_size, scores))
    return results


# ─────────────────────────────────────────────────── segment merging ──

def merge_segments(window_scores, class_idx, threshold, max_gap=2):
    """
    Merge consecutive windows where score[class_idx] >= threshold.
    - Bridges gaps of up to max_gap below-threshold windows.
    - Segment score = MAX of constituent window scores (better ranking signal).

    max_gap=2 means two consecutive weak windows won't break the segment.
    At stride=8 that's a 16-frame bridge — reasonable for micro-actions.
    """
    segments = []
    seg_start = None
    seg_end = None
    seg_max = 0.0
    gap = 0

    for start, end, scores in window_scores:
        score = float(scores[class_idx])
        if score >= threshold:
            if seg_start is None:
                seg_start = start
            seg_end = end
            seg_max = max(seg_max, score)
            gap = 0
        else:
            if seg_start is not None:
                gap += 1
                if gap > max_gap:
                    segments.append({"start": seg_start, "end": seg_end, "score": seg_max})
                    seg_start = seg_end = None
                    seg_max = 0.0
                    gap = 0
                # else: bridge the gap — seg_end stays, we wait for next positive

    if seg_start is not None:
        segments.append({"start": seg_start, "end": seg_end, "score": seg_max})
    return segments


# ──────────────────────────────────────────────── AP / mAP utilities ──

def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def compute_detection_ap(predictions, ground_truths, iou_threshold):
    """Standard detection AP with tIoU matching."""
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
    total_gt = sum(len(v) for v in ground_truths.values())
    if total_gt == 0:
        return float("nan")
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    matched = {vid: set() for vid in ground_truths}
    for i, pred in enumerate(predictions):
        gts = ground_truths.get(pred["video_id"], [])
        best_iou, best_j = 0.0, -1
        for j, (gs, ge) in enumerate(gts):
            iou = compute_iou(pred["start"], pred["end"], gs, ge)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j not in matched[pred["video_id"]]:
            tp[i] = 1
            matched[pred["video_id"]].add(best_j)
        else:
            fp[i] = 1
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec = np.concatenate([[0], tp_c / total_gt, [1]])
    prec = np.concatenate([[1], tp_c / (tp_c + fp_c + 1e-6), [0]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def compute_frame_ap(video_window_scores, gt_per_class, cls, video_lengths):
    """
    Frame-level AP for class `cls`.
    For each frame in each video, score = max over all windows covering that frame.
    GT = 1 if frame is inside any GT segment for this class.
    Returns AP of per-frame (score, label) pairs.
    """
    gt_cls = gt_per_class[cls]
    all_scores = []
    all_labels = []

    for video_id, win_scores in video_window_scores.items():
        nframes = video_lengths.get(video_id, 0)
        if nframes == 0:
            continue
        frame_scores = np.zeros(nframes, dtype=np.float32)
        for t_start, t_end, scores in win_scores:
            s = float(scores[cls])
            for f in range(t_start, min(t_end, nframes)):
                if s > frame_scores[f]:
                    frame_scores[f] = s

        gt_frames = np.zeros(nframes, dtype=np.int32)
        for gs, ge in gt_cls.get(video_id, []):
            gt_frames[gs:min(ge, nframes)] = 1

        all_scores.append(frame_scores)
        all_labels.append(gt_frames)

    if not all_scores:
        return float("nan")

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    if all_labels.sum() == 0:
        return float("nan")

    # Compute AP via precision-recall
    order = np.argsort(-all_scores)
    tp_c = np.cumsum(all_labels[order])
    fp_c = np.cumsum(1 - all_labels[order])
    rec = tp_c / all_labels.sum()
    prec = tp_c / (tp_c + fp_c + 1e-6)
    rec = np.concatenate([[0], rec, [1]])
    prec = np.concatenate([[1], prec, [0]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


# ──────────────────────────────────────────────── inference runner ──

def run_inference(args, device):
    model = load_model(args.checkpoint, device)
    npy_files = sorted(Path(args.skeleton_dir).glob("*_keypoints.npy"))
    print(f"Running inference on {len(npy_files)} videos...")
    video_window_scores = {}
    video_lengths = {}
    for i, nf in enumerate(npy_files):
        if i % 50 == 0:
            print(f"  {i}/{len(npy_files)}", flush=True)
        vid = nf.name.replace("_keypoints.npy", "")
        kp = np.load(str(nf))
        video_lengths[vid] = kp.shape[0]
        video_window_scores[vid] = sliding_window_scores(
            str(nf), model, args.window_size, args.stride, device)
    return video_window_scores, video_lengths


def load_gt(ann_csv):
    df = pd.read_csv(ann_csv)
    gt_per_class = [{} for _ in range(NUM_CLASSES)]
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        cls = int(row["class"])
        gt_per_class[cls].setdefault(vid, []).append(
            (int(row["start_frame"]), int(row["end_frame"])))
    return gt_per_class


# ─────────────────────────────────────────────────────── modes ──

def mode_sweep(args, video_window_scores, gt_per_class, device):
    """Sweep threshold 0.02→0.50 with gap_fill=2. Print mAP at each."""
    iou_thresholds = args.iou_thresholds
    thresholds = np.round(np.arange(0.02, 0.52, 0.02), 3)

    print(f"\n{'Thresh':>8} | {'mAP@0.2':>8} {'mAP@0.5':>8} {'mAP@0.7':>8} | {'avg mAP':>8}")
    print("-" * 58)

    best_avg = 0.0
    best_thresh = 0.1

    for thresh in thresholds:
        ap_matrix = np.full((NUM_CLASSES, len(iou_thresholds)), np.nan)
        valid_mask = np.array([
            sum(len(v) for v in gt_per_class[c].values()) > 0
            for c in range(NUM_CLASSES)])

        for cls in np.where(valid_mask)[0]:
            for t_idx, iou_t in enumerate(iou_thresholds):
                preds = []
                for vid, wins in video_window_scores.items():
                    for seg in merge_segments(wins, cls, thresh, max_gap=args.max_gap):
                        preds.append({"video_id": vid, **seg})
                ap_matrix[cls, t_idx] = compute_detection_ap(preds, gt_per_class[cls], iou_t)

        per_iou = [float(np.nanmean(ap_matrix[valid_mask, t])) for t in range(len(iou_thresholds))]
        avg = float(np.nanmean(ap_matrix[valid_mask]))
        print(f"{thresh:>8.3f} | {per_iou[0]:>8.4f} {per_iou[1]:>8.4f} {per_iou[2]:>8.4f} | {avg:>8.4f}")

        if avg > best_avg:
            best_avg = avg
            best_thresh = thresh

    print(f"\n>>> Best threshold: {best_thresh:.3f}  →  avg mAP: {best_avg:.4f}")
    return best_thresh


def mode_eval(args, video_window_scores, gt_per_class, device):
    """Detailed eval at a single threshold. Reports per-class AP."""
    thresh = args.threshold
    iou_thresholds = args.iou_thresholds
    ap_matrix = np.full((NUM_CLASSES, len(iou_thresholds)), np.nan)
    valid_mask = np.array([
        sum(len(v) for v in gt_per_class[c].values()) > 0
        for c in range(NUM_CLASSES)])

    for cls in np.where(valid_mask)[0]:
        for t_idx, iou_t in enumerate(iou_thresholds):
            preds = []
            for vid, wins in video_window_scores.items():
                for seg in merge_segments(wins, cls, thresh, max_gap=args.max_gap):
                    preds.append({"video_id": vid, **seg})
            ap_matrix[cls, t_idx] = compute_detection_ap(preds, gt_per_class[cls], iou_t)

    print(f"\n--- Per-IoU mAP @ threshold={thresh} ---")
    for t_idx, iou_t in enumerate(iou_thresholds):
        aps = ap_matrix[valid_mask, t_idx]
        aps = aps[~np.isnan(aps)]
        print(f"  mAP@{iou_t:.1f}: {np.mean(aps):.4f}  ({len(aps)} classes)")
    print(f"\navg mAP: {float(np.nanmean(ap_matrix[valid_mask])):.4f}")

    # Show worst 10 classes
    avg_per_class = np.nanmean(ap_matrix, axis=1)
    worst = np.argsort(avg_per_class[valid_mask])[:10]
    worst_classes = np.where(valid_mask)[0][worst]
    print("\nWorst 10 classes by avg AP:")
    for c in worst_classes:
        gt_count = sum(len(v) for v in gt_per_class[c].values())
        print(f"  class {c:3d}: avg AP={avg_per_class[c]:.4f}  GT instances={gt_count}")


def mode_frame(args, video_window_scores, gt_per_class, video_lengths):
    """
    Frame-level mAP — no tIoU threshold needed.
    Per-frame score = max over all windows covering that frame.
    This matches the 'frame-based mAP' language in the MAC 2026 overview.
    """
    valid_mask = np.array([
        sum(len(v) for v in gt_per_class[c].values()) > 0
        for c in range(NUM_CLASSES)])

    aps = []
    for cls in np.where(valid_mask)[0]:
        ap = compute_frame_ap(video_window_scores, gt_per_class, cls, video_lengths)
        if not np.isnan(ap):
            aps.append(ap)

    mean_ap = float(np.mean(aps)) if aps else 0.0
    print(f"\nFrame-level mAP: {mean_ap:.4f}  ({len(aps)} classes)")
    return mean_ap


# ──────────────────────────────────────────────────── main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--skeleton_dir", default=DEFAULT_SKELETON_DIR)
    parser.add_argument("--ann_csv", default=DEFAULT_ANN_CSV)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Detection threshold (used in --mode eval)")
    parser.add_argument("--max_gap", type=int, default=2,
                        help="Max consecutive below-threshold windows to bridge (default 2)")
    parser.add_argument("--iou_thresholds", type=float, nargs="+", default=[0.2, 0.5, 0.7])
    parser.add_argument("--mode", choices=["sweep", "eval", "frame", "all"], default="sweep",
                        help="sweep: find best threshold | eval: single-threshold detail | "
                             "frame: frame-level mAP | all: run all three")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    video_window_scores, video_lengths = run_inference(args, device)
    gt_per_class = load_gt(args.ann_csv)

    if args.mode in ("sweep", "all"):
        best_thresh = mode_sweep(args, video_window_scores, gt_per_class, device)
        if args.mode == "all":
            args.threshold = best_thresh

    if args.mode in ("eval", "all"):
        mode_eval(args, video_window_scores, gt_per_class, device)

    if args.mode in ("frame", "all"):
        mode_frame(args, video_window_scores, gt_per_class, video_lengths)


if __name__ == "__main__":
    main()
