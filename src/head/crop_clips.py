"""
crop_clips.py v2
Crops head+neck region from each video using saved keypoint JSON files.
Uses dynamic per-frame bounding box with temporal smoothing to follow
head movement throughout the clip.

Changes from v1:
    - Per-frame dynamic box (follows head movement)
    - Temporal smoothing with moving average to reduce jitter
    - Falls back to median box if keypoints are missing for a frame

Usage:
    python src/head/crop_clips.py \
        --csv        data/head_dataset/train.csv \
        --video_dir  data/videos/train/train \
        --kp_dir     data/keypoints/train \
        --out_dir    data/head_crops_dynamic/train \
        --out_csv    data/head_dataset/train_crops_dynamic.csv \
        --size       224 \
        --split      train
"""

import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

# COCO keypoint indices
LEFT_SHLDR  = 5
RIGHT_SHLDR = 6
LEFT_HIP    = 11
RIGHT_HIP   = 12
NOSE        = 0


def compute_head_box(kps, frame_w, frame_h):
    """Compute head+neck bounding box from keypoints for one frame."""
    l_shldr = np.array(kps[LEFT_SHLDR][:2])
    r_shldr = np.array(kps[RIGHT_SHLDR][:2])
    l_hip   = np.array(kps[LEFT_HIP][:2])
    r_hip   = np.array(kps[RIGHT_HIP][:2])
    nose    = np.array(kps[NOSE][:2])

    neck    = (l_shldr + r_shldr) / 2
    mid_hip = (l_hip + r_hip) / 2
    L       = np.linalg.norm(neck - mid_hip)

    if L < 10:
        return None

    head_size = L / 1.4
    hw = int(head_size / 2)
    cx, cy = int(nose[0]), int(nose[1])

    x1 = max(0, cx - hw)
    y1 = max(0, cy - hw)
    x2 = min(frame_w, cx + hw)
    y2 = min(frame_h, cy + hw)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    return (x1, y1, x2, y2)


def smooth_boxes(boxes, window=5):
    """
    Apply moving average smoothing to box coordinates.
    Reduces jitter from frame-to-frame keypoint variation.
    boxes: list of (x1,y1,x2,y2) or None
    Returns smoothed list of (x1,y1,x2,y2)
    """
    n = len(boxes)
    arr = np.array([b if b is not None else [np.nan]*4 for b in boxes], dtype=float)

    # fill NaN with nearest valid box
    for col in range(4):
        mask = np.isnan(arr[:, col])
        if mask.all():
            arr[:, col] = 0
            continue
        # forward fill
        valid_idx = np.where(~mask)[0]
        arr[:, col] = np.interp(np.arange(n), valid_idx, arr[valid_idx, col])

    # moving average
    smoothed = np.zeros_like(arr)
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end   = min(n, i + half + 1)
        smoothed[i] = arr[start:end].mean(axis=0)

    return [tuple(smoothed[i].astype(int)) for i in range(n)]


def crop_video(video_path, kp_path, out_path, target_size=(224, 224)):
    """
    Crop head+neck region from every frame using dynamic smoothed boxes.
    """
    with open(kp_path) as f:
        kp_data = json.load(f)

    cap     = cv2.VideoCapture(str(video_path))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # compute per-frame boxes
    raw_boxes = []
    for frame_data in kp_data:
        if frame_data["keypoints"] is None:
            raw_boxes.append(None)
        else:
            box = compute_head_box(frame_data["keypoints"], frame_w, frame_h)
            raw_boxes.append(box)

    if all(b is None for b in raw_boxes):
        cap.release()
        return False

    # smooth boxes to reduce jitter
    smoothed_boxes = smooth_boxes(raw_boxes, window=5)

    # compute median box as fallback for missing frames
    valid = [b for b in raw_boxes if b is not None]
    median_box = tuple(np.median(valid, axis=0).astype(int)) if valid else None

    if median_box is None:
        cap.release()
        return False

    # write cropped video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, target_size)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # use smoothed box for this frame, fallback to median
        if frame_idx < len(smoothed_boxes):
            box = smoothed_boxes[frame_idx]
        else:
            box = median_box

        x1, y1, x2, y2 = box
        # clamp to frame boundaries
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(x1 + 1, min(x2, frame_w))
        y2 = max(y1 + 1, min(y2, frame_h))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        crop_resized = cv2.resize(crop, target_size)
        writer.write(crop_resized)
        frame_idx += 1

    cap.release()
    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        required=True)
    parser.add_argument("--video_dir",  required=True)
    parser.add_argument("--kp_dir",     required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--out_csv",    required=True)
    parser.add_argument("--size",       type=int, default=224)
    parser.add_argument("--split",      default="train")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    kp_dir    = Path(args.kp_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df          = pd.read_csv(args.csv)
    target_size = (args.size, args.size)
    results     = []
    skipped     = 0

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Cropping {args.split} (dynamic)"):
        video_name = row["video"]
        head_label = row["head_label"]
        fine_label = row["fine_label"]

        video_path = video_dir / video_name
        kp_path    = kp_dir / (Path(video_name).stem + ".json")
        out_path   = out_dir / video_name

        if out_path.exists():
            results.append({
                "video":      video_name,
                "crop_path":  str(out_path),
                "fine_label": fine_label,
                "head_label": head_label,
            })
            continue

        if not video_path.exists() or not kp_path.exists():
            skipped += 1
            continue

        success = crop_video(video_path, kp_path, out_path, target_size)

        if success:
            results.append({
                "video":      video_name,
                "crop_path":  str(out_path),
                "fine_label": fine_label,
                "head_label": head_label,
            })
        else:
            skipped += 1

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nDone — {len(results)} clips saved, {skipped} skipped")
    print(f"CSV saved to {args.out_csv}")


if __name__ == "__main__":
    main()
