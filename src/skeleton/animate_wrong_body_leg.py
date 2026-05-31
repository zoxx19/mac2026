"""
animate_wrong_body_leg.py
Creates annotated MP4 videos for wrong predictions.
Reads wrong_predictions.csv from evaluate_body_leg.py.

For each true class, randomly picks n_per_class wrong predictions.
Draws the COCO skeleton on the original video frames.
Adds TRUE/PREDICTED label overlay and red border.
Saves slowed-down MP4 for easy inspection.

Usage:
    python src/skeleton/animate_wrong_body_leg.py \
        --track      body \
        --wrong_csv  outputs/eval_body/wrong_predictions.csv \
        --output_dir outputs/eval_body/animated_wrong \
        --n_per_class 5

    python src/skeleton/animate_wrong_body_leg.py \
        --track      leg \
        --wrong_csv  outputs/eval_leg/wrong_predictions.csv \
        --output_dir outputs/eval_leg/animated_wrong \
        --n_per_class 5
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from collections import Counter

random.seed(42)

TRACK_CFG = {
    'body': {
        'class_names': ['A1','A2','A3','A4','A5','no-body'],
        'vid_dir':     'data/videos/val/val',
        'kp_dir':      'data/keypoints/val',
    },
    'leg': {
        'class_names': ['D1','D2','D3','D4','D5','D6','D7','D8','no-leg'],
        'vid_dir':     'data/videos/val/val',
        'kp_dir':      'data/keypoints/val',
    },
}

# COCO skeleton edges
COCO_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# BGR colours per body segment
def edge_color(i, j):
    if i<=4 or j<=4:                                          return (86,180,233)
    if i in [7,8,9,10] or j in [7,8,9,10]:                   return (230,159,0)
    if i in [11,12,13,14,15,16] or j in [11,12,13,14,15,16]: return (0,94,213)
    return (0,158,115)

GREEN=(0,200,0); RED=(0,0,220); WHITE=(255,255,255); BLACK=(0,0,0)


def draw_skeleton(frame, kps17, conf_thresh=0.2):
    """Draw COCO skeleton using raw pixel coordinates from keypoint JSON."""
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    for (i, j) in COCO_EDGES:
        if kps17[i,2] < conf_thresh or kps17[j,2] < conf_thresh:
            continue
        p1 = (int(kps17[i,0]), int(kps17[i,1]))
        p2 = (int(kps17[j,0]), int(kps17[j,1]))
        if all(0 <= p[0] < w and 0 <= p[1] < h for p in [p1,p2]):
            cv2.line(overlay, p1, p2, edge_color(i,j), 2)
    for k in range(17):
        if kps17[k,2] < conf_thresh: continue
        pt = (int(kps17[k,0]), int(kps17[k,1]))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(overlay, pt, 4, WHITE, -1)
            cv2.circle(overlay, pt, 4, BLACK, 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


def draw_label_overlay(frame, true_label, pred_label, frame_idx, total):
    """Same style as head/animate_wrong.py."""
    h, w    = frame.shape[:2]
    banner  = frame.copy()
    cv2.rectangle(banner, (0,0), (w,80), BLACK, -1)
    cv2.addWeighted(banner, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, f"TRUE:  {true_label}",
                (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN, 2)
    cv2.putText(frame, f"PRED:  {pred_label}",
                (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, RED, 2)
    cv2.putText(frame, f"Frame {frame_idx+1}/{total}",
                (w-140, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    cv2.rectangle(frame, (0,0), (w-1,h-1), RED, 3)
    return frame


def animate_clip(video_id, vid_dir, kp_dir, true_label, pred_label,
                 dst_path, slow_factor=3):
    vid_path = os.path.join(vid_dir, f"{video_id}.mp4")
    kp_path  = os.path.join(kp_dir,  f"{video_id}.json")

    if not os.path.exists(vid_path):
        print(f"  Video not found: {vid_path}")
        return False
    if not os.path.exists(kp_path):
        print(f"  Keypoints not found: {kp_path}")
        return False

    # Load keypoints (raw pixel coordinates)
    with open(kp_path) as f:
        kp_data = json.load(f)
    kps_raw = [np.array(fr['keypoints'], dtype=np.float32)
               for fr in kp_data]   # list of (17,3)

    # Load video frames
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"  Cannot open: {vid_path}")
        return False
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (w, h))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    total = len(frames)
    for idx, frame in enumerate(frames):
        # Match keypoint frame index to video frame index
        kp_idx = min(int(idx * len(kps_raw) / max(total,1)),
                     len(kps_raw)-1)
        frame  = draw_skeleton(frame, kps_raw[kp_idx])
        frame  = draw_label_overlay(frame, true_label, pred_label,
                                    idx, total)
        for _ in range(slow_factor):
            writer.write(frame)

    writer.release()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--track',       choices=['body','leg'], required=True)
    p.add_argument('--wrong_csv',   required=True)
    p.add_argument('--output_dir',  required=True)
    p.add_argument('--n_per_class', type=int, default=5)
    p.add_argument('--slow_factor', type=int, default=3)
    p.add_argument('--seed',        type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    cfg        = TRACK_CFG[args.track]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_wrong = pd.read_csv(args.wrong_csv)
    print(f"Total wrong predictions: {len(df_wrong)}")

    # Show top confused pairs
    pairs = Counter(zip(df_wrong['true'], df_wrong['predicted']))
    print("\nTop confused pairs (true → predicted):")
    for (t, p), cnt in pairs.most_common(10):
        print(f"  {t} → {p}: {cnt}")

    total_saved = 0
    for cls_name in cfg['class_names']:
        cls_df = df_wrong[df_wrong['true'] == cls_name]
        if len(cls_df) == 0:
            print(f"\n{cls_name}: no wrong predictions ✅")
            continue

        print(f"\n{cls_name}: {len(cls_df)} wrong predictions")
        confused = cls_df['predicted'].value_counts()
        for pred_name, cnt in confused.items():
            print(f"  → predicted as {pred_name}: {cnt}x")

        # Random sample
        n       = min(args.n_per_class, len(cls_df))
        samples = cls_df.sample(n=n, random_state=args.seed)

        cls_dir = output_dir / cls_name
        cls_dir.mkdir(exist_ok=True)

        for i, (_, row) in enumerate(samples.iterrows()):
            video_id  = row['video_id']
            pred_name = row['predicted']
            dst_name  = (f"{i+1:02d}_true_{cls_name}_"
                         f"pred_{pred_name}_{video_id}.mp4")
            ok = animate_clip(
                video_id, cfg['vid_dir'], cfg['kp_dir'],
                cls_name, pred_name,
                cls_dir / dst_name,
                slow_factor=args.slow_factor
            )
            if ok:
                print(f"  Saved: {dst_name}")
                total_saved += 1

    print(f"\nTotal videos saved: {total_saved}")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
