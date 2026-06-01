"""
crop_body.py
Crops full body region (nose to ankles) for body track VideoMAE training.

Crop definition:
  - Top:    mean(nose_y) - 40px
  - Bottom: mean(ankle_y) + 60px
  - Left:   mean(min shoulder/hip x) - 60px
  - Right:  mean(max shoulder/hip x) + 60px
  - Resized to 224x224

Output:
  data/body_crops/train/<video_id>.mp4
  data/body_crops/val/<video_id>.mp4
  data/skeleton_dataset/body_train_videomae.csv
  data/skeleton_dataset/body_val_videomae.csv

Usage:
  python src/body_leg/crop_body.py --split train
  python src/body_leg/crop_body.py --split val
"""

import os, json, argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

PAD_TOP    = 40
PAD_BOTTOM = 60
PAD_SIDE   = 60
CROP_SIZE  = 224

# COCO joints
NOSE         = 0
L_SHOULDER   = 5
R_SHOULDER   = 6
L_HIP        = 11
R_HIP        = 12
L_ANKLE      = 15
R_ANKLE      = 16


def get_crop_box(kp_frames, frame_h, frame_w):
    tops, bottoms, lefts, rights = [], [], [], []
    for frame in kp_frames:
        kps = np.array(frame['keypoints'], dtype=np.float32)
        if kps[NOSE][2] > 0.1:
            tops.append(kps[NOSE][1])
        ankles = [kps[i][1] for i in [L_ANKLE, R_ANKLE] if kps[i][2] > 0.1]
        if ankles:
            bottoms.append(max(ankles))
        xs = [kps[i][0] for i in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
              if kps[i][2] > 0.1]
        if xs:
            lefts.append(min(xs))
            rights.append(max(xs))
    if not tops or not bottoms:
        return 0, 0, frame_w, frame_h
    y1 = max(0, int(np.mean(tops)) - PAD_TOP)
    y2 = min(frame_h, int(np.mean(bottoms)) + PAD_BOTTOM)
    x1 = max(0, int(np.mean(lefts)) - PAD_SIDE)
    x2 = min(frame_w, int(np.mean(rights)) + PAD_SIDE)
    return x1, y1, x2, y2


def crop_video(vid_path, kp_path, out_path):
    with open(kp_path) as f:
        kp_frames = json.load(f)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return False
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    x1, y1, x2, y2 = get_crop_box(kp_frames, frame_h, frame_w)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (CROP_SIZE, CROP_SIZE))
    while True:
        ret, frame = cap.read()
        if not ret: break
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: crop = frame
        writer.write(cv2.resize(crop, (CROP_SIZE, CROP_SIZE)))
    cap.release(); writer.release()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['train','val'], required=True)
    args = p.parse_args()

    cfg = {
        'train': {
            'csv_in':  'data/skeleton_dataset/body_train.csv',
            'kp_dir':  'data/keypoints/train',
            'vid_dir': 'data/videos/train/train',
            'out_dir': 'data/body_crops/train',
            'csv_out': 'data/skeleton_dataset/body_train_videomae.csv',
            'label_col': 'body_label',
        },
        'val': {
            'csv_in':  'data/skeleton_dataset/body_val.csv',
            'kp_dir':  'data/keypoints/val',
            'vid_dir': 'data/videos/val/val',
            'out_dir': 'data/body_crops/val',
            'csv_out': 'data/skeleton_dataset/body_val_videomae.csv',
            'label_col': 'body_label',
        },
    }[args.split]

    os.makedirs(cfg['out_dir'], exist_ok=True)
    df = pd.read_csv(cfg['csv_in'])
    print(f"Processing {len(df)} videos — split={args.split}")

    rows, skipped = [], 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid_id   = str(row['video']).replace('.mp4','')
        kp_path  = os.path.join(cfg['kp_dir'],  f"{vid_id}.json")
        vid_path = os.path.join(cfg['vid_dir'],  f"{vid_id}.mp4")
        out_path = os.path.join(cfg['out_dir'],  f"{vid_id}.mp4")
        if not os.path.exists(kp_path) or not os.path.exists(vid_path):
            skipped += 1; continue
        if not os.path.exists(out_path):
            if not crop_video(vid_path, kp_path, out_path):
                skipped += 1; continue
        rows.append({'video': row['video'], 'crop_path': out_path,
                     'body_label': row[cfg['label_col']]})

    pd.DataFrame(rows).to_csv(cfg['csv_out'], index=False)
    print(f"Done. {len(rows)} crops saved, {skipped} skipped.")
    print(f"CSV: {cfg['csv_out']}")

if __name__ == '__main__':
    main()
