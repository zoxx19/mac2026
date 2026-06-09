"""
Sliding window inference + avg mAP evaluation for hand detection stream.

For each val video:
1. Load skeleton .npy (N, 33, 4) → extract 6 arm joints → (N, 6, 2)
2. Slide 32-frame window with stride 8
3. Preprocess each window (val branch: center, normalize, linspace sample)
4. Run MMN binary classifier → softmax score for class 1
5. Merge consecutive positive windows into segments {start, end, score}
6. Compute AP@0.2, AP@0.5, AP@0.7 → avg mAP

Usage:
python src/eval_hand_map.py \
    --checkpoint ../Micro-action-skeleton/MMN/work_dir/train/hand_detection_J/runs-25-124000.pt \
    --skeleton_dir data/skeleton/val \
    --ann_csv .\ /data/MMA-52/Annotations/hand_binary/val.csv \
    --window_size 32 \
    --stride 8 \
    --iou_thresholds 0.2 0.5 0.7
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import argparse
import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

# Inject MMN repo into path — mirrors SLURM PYTHONPATH
MMN_REPO = "../Micro-action-skeleton/MMN"
sys.path.insert(0, os.path.join(MMN_REPO, "torchlight"))
sys.path.insert(0, os.path.join(MMN_REPO, "torchpack"))
sys.path.insert(0, MMN_REPO)

from model.MMN import MMN


HAND_JOINTS = [11, 12, 13, 14, 15, 16]  # left/right shoulder, elbow, wrist


def load_model(checkpoint_path, device):
    model = MMN(
        in_channels=2,
        num_classes=2,
        num_people=1,
        num_frames=32,
        num_points=6,
        kernel_size=3,
        num_heads=4,
        drop=0.0,
        head_drop=0.1,
        drop_path=0.3,
        mlp_ratio=2.0,
        index_t=True,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # Handle both raw state_dict and wrapped checkpoints
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess_window(window):
    """
    Val-branch preprocessing matching feeder_hand_detection.py exactly.
    window: (T, V, C) = (32, 6, 2) float32
    Returns: data (1, C, T, V, 1), index_t (1, T)
    """
    value = window.copy()
    T, V, C = value.shape
    time_steps = T

    # Center on midpoint of joints 0,1 (left/right shoulder in 6-joint space)
    center = (value[0, 0, :] + value[0, 1, :]) / 2.0
    value = value - center

    # Normalize to [-1, 1]
    scalar = value.reshape(-1, C)
    epsilon = 1e-6
    scalar = (scalar - np.min(scalar, axis=0)) / (
        np.max(scalar, axis=0) - np.min(scalar, axis=0) + epsilon
    )
    scalar = scalar * 2 - 1
    value = scalar.reshape(T, V, C)

    # Linspace sampling
    length = value.shape[0]
    idx = np.linspace(0, length - 1, time_steps).astype(int)
    data = value[idx, :, :]                                      # (T, V, C)
    index_t = (2 * idx.astype(np.float32) / length - 1)         # (T,)

    # (T, V, C) → (C, T, V) → (C, T, V, 1)
    data = np.transpose(data, (2, 0, 1))
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

    # Add batch dim
    data = torch.from_numpy(data).unsqueeze(0)                   # (1, C, T, V, 1)
    index_t = torch.from_numpy(index_t).unsqueeze(0)             # (1, T)
    return data, index_t


def sliding_window_scores(npy_path, model, window_size, stride, device):
    """
    Returns list of (start_frame, end_frame, score).
    score = softmax probability for class 1 (action present).
    """
    kp = np.load(npy_path)                                       # (N, 33, 4)
    kp = kp[:, HAND_JOINTS, :2].astype(np.float32)              # (N, 6, 2)
    N = kp.shape[0]

    results = []
    for t_start in range(0, max(1, N - window_size + 1), stride):
        t_end = t_start + window_size
        if t_end > N:
            break
        window = kp[t_start:t_end]                               # (32, 6, 2)
        data, index_t = preprocess_window(window)
        data = data.float().to(device)
        index_t = index_t.float().to(device)
        with torch.no_grad():
            logits = model(data, index_t)                        # (1, 2)
            score = F.softmax(logits, dim=-1)[0, 1].item()
        results.append((t_start, t_end, score))
    return results


def merge_segments(window_scores, threshold):
    """Merge consecutive windows above threshold into segments."""
    segments = []
    current = None
    current_scores = []

    for start, end, score in window_scores:
        if score >= threshold:
            if current is None:
                current = [start, end]
                current_scores = [score]
            else:
                current[1] = end
                current_scores.append(score)
        else:
            if current is not None:
                segments.append({
                    "start": current[0],
                    "end": current[1],
                    "score": float(np.mean(current_scores))
                })
                current = None
                current_scores = []

    if current is not None:
        segments.append({
            "start": current[0],
            "end": current[1],
            "score": float(np.mean(current_scores))
        })
    return segments


def compute_iou(pred_start, pred_end, gt_start, gt_end):
    inter = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return inter / union if union > 0 else 0.0


def compute_ap(predictions, ground_truths, iou_threshold):
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
    total_gt = sum(len(v) for v in ground_truths.values())
    if total_gt == 0:
        return 0.0

    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    matched = {vid: set() for vid in ground_truths}

    for i, pred in enumerate(predictions):
        vid = pred["video_id"]
        gts = ground_truths.get(vid, [])
        best_iou = 0.0
        best_j = -1
        for j, (gt_start, gt_end) in enumerate(gts):
            iou = compute_iou(pred["start"], pred["end"], gt_start, gt_end)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j not in matched.get(vid, set()):
            tp[i] = 1
            matched[vid].add(best_j)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[1], precision, [0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return float(ap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--skeleton_dir", required=True)
    parser.add_argument("--ann_csv", required=True)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou_thresholds", type=float, nargs="+", default=[0.2, 0.5, 0.7])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MMN checkpoint...")
    model = load_model(args.checkpoint, device)
    print("Model loaded.")

    # Load ground truth
    df = pd.read_csv(args.ann_csv)
    ground_truths = {}
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        if vid not in ground_truths:
            ground_truths[vid] = []
        ground_truths[vid].append((int(row["start_frame"]), int(row["end_frame"])))

    # Run inference
    npy_files = sorted(Path(args.skeleton_dir).glob("*_keypoints.npy"))
    print(f"Running inference on {len(npy_files)} videos...")

    all_predictions = []
    for i, nf in enumerate(npy_files):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(npy_files)}", flush=True)
        video_id = nf.name.replace("_keypoints.npy", "")
        window_scores = sliding_window_scores(
            str(nf), model, args.window_size, args.stride, device
        )
        segments = merge_segments(window_scores, args.threshold)
        for seg in segments:
            all_predictions.append({
                "video_id": video_id,
                "start": seg["start"],
                "end": seg["end"],
                "score": seg["score"]
            })

    print(f"Total predictions: {len(all_predictions)}")
    print(f"Total GT instances: {sum(len(v) for v in ground_truths.values())}")

    # Compute mAP
    aps = []
    for iou_thresh in args.iou_thresholds:
        ap = compute_ap(all_predictions, ground_truths, iou_thresh)
        aps.append(ap)
        print(f"AP@{iou_thresh:.1f}: {ap:.4f}")

    avg_map = np.mean(aps)
    print(f"\navg mAP: {avg_map:.4f}")


if __name__ == "__main__":
    main()
