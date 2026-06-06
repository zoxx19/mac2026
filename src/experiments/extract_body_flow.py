"""
extract_body_flow.py
Extracts optical flow from body crop videos using Farneback method.
Same approach as extract_optical_flow.py but for body crops.

Usage:
  python src/experiments/extract_body_flow.py --split train
  python src/experiments/extract_body_flow.py --split val
"""

import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse


def extract_flow_video(video_path, out_path, size=224):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (size, size))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (size, size))
    writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (size, size))
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv = np.zeros((size, size, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = mag.astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        writer.write(bgr)
        prev_gray = gray
    cap.release()
    writer.release()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['train', 'val'], required=True)
    args = p.parse_args()

    cfg = {
        'train': {
            'csv':     'data/skeleton_dataset/body_train_videomae.csv',
            'out_dir': 'data/body_flow/train',
            'csv_out': 'data/skeleton_dataset/body_train_flow.csv',
        },
        'val': {
            'csv':     'data/skeleton_dataset/body_val_videomae.csv',
            'out_dir': 'data/body_flow/val',
            'csv_out': 'data/skeleton_dataset/body_val_flow.csv',
        },
    }[args.split]

    os.makedirs(cfg['out_dir'], exist_ok=True)
    df = pd.read_csv(cfg['csv'])
    print(f"Extracting flow for {len(df)} videos — split={args.split}")

    rows, skipped = [], 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid_id   = str(row['video']).replace('.mp4', '')
        out_path = os.path.join(cfg['out_dir'], f"{vid_id}.mp4")
        if not os.path.exists(row['crop_path']):
            skipped += 1; continue
        if not os.path.exists(out_path):
            ok = extract_flow_video(row['crop_path'], out_path)
            if not ok:
                skipped += 1; continue
        rows.append({'video': row['video'], 'flow_path': out_path,
                     'body_label': row['body_label']})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(cfg['csv_out'], index=False)
    print(f"Done. {len(rows)} flows saved, {skipped} skipped.")
    print(f"CSV: {cfg['csv_out']}")

if __name__ == '__main__':
    main()
