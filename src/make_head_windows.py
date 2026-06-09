"""
make_head_windows.py
====================
Uses pre-computed bbox .npy files + MMA-52 annotation CSVs
to generate cropped head window clips with binary labels:
  1 = head movement (B1-B6)
  0 = no head movement (B7)

Output:
  data/head_windows/
    clips/
      1/   <- head movement
      0/   <- no movement
    metadata.csv
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Head action classes (B1-B6)
# ─────────────────────────────────────────────
HEAD_CLASSES = {9, 10, 8, 7, 5, 6}  # nodding, shaking, turning, tilting, bowing, head up

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BBOX_DIR = Path("outputs/mma_pose_eye_anchor")
VIDEO_DIR    = Path("data/MMA-52/extracted/train/train")
ANN_PATH     = Path("data/MMA-52/Annotations/train.csv")
OUTPUT_DIR   = Path("data/head_windows")

# ─────────────────────────────────────────────
# Window parameters
# ─────────────────────────────────────────────
WINDOW_SEC        = 1.0
STRIDE_SEC        = 0.25
OVERLAP_THRESHOLD = 0.50   # min overlap ratio to assign positive label
B7_RATIO          = 1.5    # max B7 = B7_RATIO × count of largest positive class
CROP_SIZE         = (112, 112)  # resize head crop to this


def load_annotations(ann_path):
    """Load annotations, return dict: video_id -> list of (start, end, class)"""
    df = pd.read_csv(ann_path)
    ann = {}
    for _, row in df.iterrows():
        vid = row["video_id"]
        if vid not in ann:
            ann[vid] = []
        ann[vid].append((int(row["start_frame"]), int(row["end_frame"]), int(row["class"])))
    return ann


def get_label(win_start, win_end, annotations):
    """
    Return 1 if window overlaps any head action enough, else 0.
    If multiple head actions overlap, pick the one with most overlap.
    """
    win_len = win_end - win_start
    if win_len <= 0:
        return 0

    best_overlap = 0.0
    for (ann_start, ann_end, cls) in annotations:
        if cls not in HEAD_CLASSES:
            continue
        overlap = max(0, min(win_end, ann_end) - max(win_start, ann_start))
        ratio = overlap / win_len
        if ratio > best_overlap:
            best_overlap = ratio

    return 1 if best_overlap >= OVERLAP_THRESHOLD else 0


def crop_head_frame(frame, bbox):
    """Crop head region from frame, resize to CROP_SIZE."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, CROP_SIZE)


def process_video(video_id, ann_dict, output_dirs, window_frames, stride_frames):
    """
    Process one video:
    - load bboxes
    - slide window
    - crop head frames
    - label window
    - save clip
    Returns list of metadata dicts.
    """
    bbox_path  = BBOX_DIR / f"{video_id}_bboxes.npy"
    video_path = VIDEO_DIR / f"{video_id}.mp4"

    if not bbox_path.exists():
        print(f"  [SKIP] no bbox file: {bbox_path}")
        return []
    if not video_path.exists():
        print(f"  [SKIP] no video file: {video_path}")
        return []

    bboxes = np.load(bbox_path, allow_pickle=True)
    total_frames = len(bboxes)

    annotations = ann_dict.get(video_id, [])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # read all frames into memory (videos are short ~5-10s)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return []

    # use min of bbox length and actual frame count
    n_frames = min(len(frames), total_frames)

    metadata = []
    win_idx = 0
    start = 0

    while start + window_frames <= n_frames:
        end = start + window_frames
        label = get_label(start, end, annotations)

        # crop head region for each frame in window
        crops = []
        for f in range(start, end):
            bbox_dict = bboxes[f]
            head_bbox = bbox_dict.get("head") if isinstance(bbox_dict, dict) else None
            crop = crop_head_frame(frames[f], head_bbox)
            if crop is not None:
                crops.append(crop)

        if len(crops) < window_frames * 0.5:
            # too many missing head crops, skip window
            start += stride_frames
            win_idx += 1
            continue

        # save as mp4 clip
        label_dir = output_dirs[label]
        clip_name = f"{video_id}_w{win_idx:04d}.mp4"
        clip_path = label_dir / clip_name

        h, w = CROP_SIZE[1], CROP_SIZE[0]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))

        for crop in crops:
            writer.write(crop)
        writer.release()

        metadata.append({
            "video_id":    video_id,
            "window_idx":  win_idx,
            "start_frame": start,
            "end_frame":   end,
            "label":       label,
            "clip_path":   str(clip_path),
        })

        start += stride_frames
        win_idx += 1

    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_sec",        type=float, default=WINDOW_SEC)
    parser.add_argument("--stride_sec",        type=float, default=STRIDE_SEC)
    parser.add_argument("--overlap_threshold", type=float, default=OVERLAP_THRESHOLD)
    parser.add_argument("--b7_ratio",          type=float, default=B7_RATIO)
    parser.add_argument("--output_dir",        type=str,   default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dirs = {
        0: output_dir / "clips" / "0",
        1: output_dir / "clips" / "1",
    }
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # load annotations
    print("Loading annotations...")
    ann_dict = load_annotations(ANN_PATH)

    # find available video IDs from bbox files
    bbox_files = sorted(BBOX_DIR.glob("*_bboxes.npy"))
    video_ids  = [f.stem.replace("_bboxes", "") for f in bbox_files]
    print(f"Found {len(video_ids)} videos with bbox files.")

    # default fps=30 for window sizing
    window_frames = int(args.window_sec * 30)
    stride_frames = int(args.stride_sec * 30)
    print(f"Window: {window_frames} frames | Stride: {stride_frames} frames")

    all_metadata = []

    for vid in video_ids:
        print(f"Processing {vid}...")
        meta = process_video(
            video_id=vid,
            ann_dict=ann_dict,
            output_dirs=output_dirs,
            window_frames=window_frames,
            stride_frames=stride_frames,
        )
        all_metadata.extend(meta)
        pos = sum(1 for m in meta if m["label"] == 1)
        neg = sum(1 for m in meta if m["label"] == 0)
        print(f"  windows: {len(meta)} | pos(1): {pos} | neg(0): {neg}")

    # ── class distribution before sampling ──
    total_pos = sum(1 for m in all_metadata if m["label"] == 1)
    total_neg = sum(1 for m in all_metadata if m["label"] == 0)
    print(f"\n── Before B7 sampling ──")
    print(f"  Label 1 (movement):    {total_pos}")
    print(f"  Label 0 (no movement): {total_neg}")
    print(f"  Ratio neg/pos:         {total_neg / max(1, total_pos):.1f}x")

    # ── sample B7 to reduce imbalance ──
    max_neg = int(total_pos * args.b7_ratio)
    pos_meta = [m for m in all_metadata if m["label"] == 1]
    neg_meta = [m for m in all_metadata if m["label"] == 0]

    if len(neg_meta) > max_neg:
        rng = np.random.default_rng(42)
        neg_meta = list(rng.choice(neg_meta, size=max_neg, replace=False))
        print(f"\nSampled B7 down to {max_neg} (ratio {args.b7_ratio}x pos)")

    final_metadata = pos_meta + neg_meta

    # ── save metadata CSV ──
    meta_path = output_dir / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_metadata[0].keys())
        writer.writeheader()
        writer.writerows(final_metadata)

    print(f"\n── Final dataset ──")
    print(f"  Label 1: {len(pos_meta)}")
    print(f"  Label 0: {len(neg_meta)}")
    print(f"  Total:   {len(final_metadata)}")
    print(f"  Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()