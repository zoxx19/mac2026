"""
Sliding window inference + avg mAP evaluation for head detection stream.

For each val video:
1. Load head-cropped video from head_crops/val/
2. Slide 16-frame window with stride 8
3. Run VideoMAE binary classifier on each window
4. Get confidence score for class 1 (movement) per window
5. Apply threshold → binary predictions
6. Merge consecutive positive windows into segments {start, end, score}
7. Compare against ground truth from head_binary/val.csv
8. Compute AP@0.2, AP@0.5, AP@0.7 and avg mAP

Usage:
python src/eval_head_map.py \
    --model_dir data/head_detector/best_model \
    --video_dir data/head_crops/val \
    --ann_csv data/MMA-52/Annotations/head_binary/val.csv \
    --window_frames 16 \
    --stride 8 \
    --iou_thresholds 0.2 0.5 0.7
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor


def read_all_frames(video_path, size=224):
    """Read all frames from video sequentially into memory."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
    cap.release()
    return frames, fps


def sliding_window_scores(video_path, model, processor, window_frames, stride, device):
    """Run sliding window inference, return per-window (start, end, score)."""
    frames, fps = read_all_frames(video_path)
    total = len(frames)

    results = []
    for start in range(0, max(1, total - window_frames + 1), stride):
        end = start + window_frames
        if end > total:
            break
        window = frames[start:end]
        inputs = processor(window, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits
            score = F.softmax(logits, dim=-1)[0, 1].item()
        results.append((start, end, score))
    return results, fps


def merge_segments(window_scores, threshold, stride):
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
    """
    predictions: list of {video_id, start, end, score}
    ground_truths: dict {video_id: [(start, end), ...]}
    """
    # Sort by score descending
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

        if best_iou >= iou_threshold and best_j not in matched[vid]:
            tp[i] = 1
            matched[vid].add(best_j)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)

    # AP via trapezoidal rule
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[1], precision, [0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return float(ap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--ann_csv", required=True)
    parser.add_argument("--window_frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou_thresholds", type=float, nargs="+", default=[0.2, 0.5, 0.7])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    processor = VideoMAEImageProcessor.from_pretrained("data/videomae-base")
    model = VideoMAEForVideoClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Load ground truth
    df = pd.read_csv(args.ann_csv)
    ground_truths = {}
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        if vid not in ground_truths:
            ground_truths[vid] = []
        ground_truths[vid].append((int(row["start_frame"]), int(row["end_frame"])))

    # Run inference
    all_predictions = []
    video_files = sorted(Path(args.video_dir).glob("*_head.mp4"))
    print(f"Running inference on {len(video_files)} videos...")

    
    for i, vf in enumerate(video_files):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(video_files)}", flush=True)
        video_id = vf.stem.replace("_head", "")
        window_scores, fps = sliding_window_scores(
            str(vf), model, processor, args.window_frames, args.stride, device
        )
        segments = merge_segments(window_scores, args.threshold, args.stride)
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
    print(f"avg mAP: {avg_map:.4f}")


if __name__ == "__main__":
    main()
