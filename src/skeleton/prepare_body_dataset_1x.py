"""
src/skeleton/prepare_body_dataset.py

6-class body movement dataset:
    body_label  meaning          fine labels
    ----------  ---------------  -----------
    0           A1               0
    1           A2               1
    2           A3               2
    3           A4               3
    4           A5               4
    5           A6 no body move  all other coarse (1,2,3,4,5,6)
                capped at 2x largest A class

Output:
    data/skeleton_dataset/body_train_1x.csv
    data/skeleton_dataset/body_val.csv
"""

import random
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/home/woody/iwso/iwso226h/ma52/data")
ANN_DIR  = DATA_DIR / "annotations"
OUT_DIR  = DATA_DIR / "skeleton_dataset"
OUT_DIR.mkdir(exist_ok=True)

KP_TRAIN = DATA_DIR / "keypoints/train"
KP_VAL   = DATA_DIR / "keypoints/val"

# fine label 0-4 → A1-A5 → body_label 0-4
# everything else → A6   → body_label 5
FINE_TO_BODY = {0:0, 1:1, 2:2, 3:3, 4:4}

LABEL_NAMES = {
    0:"A1", 1:"A2", 2:"A3", 3:"A4", 4:"A5",
    5:"A6 no body move",
}

def fine_to_coarse(fine):
    fine = int(fine)
    if   fine <= 4:  return 0
    elif fine <= 10: return 1
    elif fine <= 23: return 2
    elif fine <= 31: return 3
    elif fine <= 37: return 4
    elif fine <= 47: return 5
    else:            return 6

def body_label(fine):
    return FINE_TO_BODY.get(int(fine), 5)

def cap_no_body(df, multiplier=1.0, seed=42):
    body_df    = df[df["body_label"] != 5]
    no_body_df = df[df["body_label"] == 5]
    if len(body_df) == 0:
        return df
    largest = body_df["body_label"].value_counts().max()
    cap     = int(multiplier * largest)
    if len(no_body_df) > cap:
        no_body_df = no_body_df.sample(n=cap, random_state=42)
        print(f"  no_body_move capped → {cap} (2x largest A class={largest})")
    return pd.concat([body_df, no_body_df]).reset_index(drop=True)

def process_split(ann_file, kp_dir, split_name, apply_cap=False):
    rows = []
    missing = 0
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            video_name = parts[0]
            fine       = int(parts[1])
            coarse     = fine_to_coarse(fine)
            kp_path    = kp_dir / (Path(video_name).stem + ".json")
            if not kp_path.exists():
                missing += 1
                continue
            rows.append({
                "video":      video_name,
                "kp_path":    str(kp_path),
                "fine_label": fine,
                "coarse":     coarse,
                "body_label": body_label(fine),
            })

    df = pd.DataFrame(rows)
    if apply_cap:
        df = cap_no_body(df)

    print(f"\n{'='*55}")
    print(f"{split_name} split  ({len(df)} samples, {missing} missing kp)")
    print(f"{'='*55}")
    counts = df["body_label"].value_counts().sort_index()
    for lbl, name in LABEL_NAMES.items():
        print(f"  {lbl} {name:<22}: {counts.get(lbl,0)}")

    out_path = OUT_DIR / f"body_{split_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    return df

if __name__ == "__main__":
    train_df = process_split(ANN_DIR/"train_list_videos.txt", KP_TRAIN, "train", apply_cap=True)
    val_df   = process_split(ANN_DIR/"val_list_videos.txt",   KP_VAL,   "val",   apply_cap=False)
