"""
crop_upperbody.py
Crops upper body region (nose to hips) from videos for hand fine model.

Crop definition per video:
  - Top:    mean(nose_y across frames) - 40px
  - Bottom: mean(hip_y across frames)  + 80px
  - Left:   0 (full width)
  - Right:  frame_width (full width)
  - Resized to 224x224

Usage:
    python src/hand_fine/crop_upperbody.py --split train
    python src/hand_fine/crop_upperbody.py --split val
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

PAD_TOP    = 40
PAD_BOTTOM = 80
CROP_SIZE  = 224


def get_crop_box(kp_frames, frame_h, frame_w):
    nose_ys, hip_ys = [], []
    for frame in kp_frames:
        kps = np.array(frame['keypoints'], dtype=np.float32)
        if kps[0][2] > 0.1:
            nose_ys.append(kps[0][1])
        hip_y = [kps[i][1] for i in [11, 12] if kps[i][2] > 0.1]
        if hip_y:
            hip_ys.append(float(np.mean(hip_y)))
    if not nose_ys or not hip_ys:
        return 0, 0, frame_w, int(frame_h * 0.70)
    y1 = max(0, int(np.mean(nose_ys)) - PAD_TOP)
    y2 = min(frame_h, int(np.mean(hip_ys)) + PAD_BOTTOM)
    x1 = 0
    x2 = frame_w
    return x1, y1, x2, y2


def crop_video(video_path, kp_path, out_path):
    with open(kp_path) as f:
        kp_frames = json.load(f)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    x1, y1, x2, y2 = get_crop_box(kp_frames, frame_h, frame_w)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (CROP_SIZE, CROP_SIZE))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
        writer.write(resized)
    cap.release()
    writer.release()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['train', 'val'], required=True)
    args = p.parse_args()

    cfg = {
        'train': {
            'csv_in':  'data/hand_dataset/hand_fine_train.csv',
            'kp_dir':  'data/keypoints/train',
            'vid_dir': 'data/videos/train/train',
            'out_dir': 'data/upperbody_crops/train',
            'csv_out': 'data/hand_dataset/hand_fine_upperbody_train.csv',
        },
        'val': {
            'csv_in':  'data/hand_dataset/hand_fine_val.csv',
            'kp_dir':  'data/keypoints/val',
            'vid_dir': 'data/videos/val/val',
            'out_dir': 'data/upperbody_crops/val',
            'csv_out': 'data/hand_dataset/hand_fine_upperbody_val.csv',
        },
    }[args.split]

    os.makedirs(cfg['out_dir'], exist_ok=True)
    df = pd.read_csv(cfg['csv_in'])
    print(f"Processing {len(df)} videos — split={args.split}")

    rows, skipped = [], 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row['video_id']
        kp_path  = os.path.join(cfg['kp_dir'],  f"{video_id}.json")
        vid_path = os.path.join(cfg['vid_dir'],  f"{video_id}.mp4")
        out_path = os.path.join(cfg['out_dir'],  f"{video_id}.mp4")

        if not os.path.exists(kp_path) or not os.path.exists(vid_path):
            skipped += 1
            continue

        if not os.path.exists(out_path):
            ok = crop_video(vid_path, kp_path, out_path)
            if not ok:
                skipped += 1
                continue

        rows.append({
            'video_id':      video_id,
            'fine_label':    row['fine_label'],
            'local_label':   row['local_label'],
            'video_path':    out_path,
            'has_keypoints': True,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(cfg['csv_out'], index=False)
    print(f"Done. {len(df_out)} crops saved, {skipped} skipped.")
    print(f"CSV: {cfg['csv_out']}")


if __name__ == '__main__':
    main()