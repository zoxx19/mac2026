"""
src/skeleton/prepare_leg_dataset.py

5-class leg movement dataset:
    leg_label  meaning          fine labels
    ---------  ---------------  -----------
    0          D1               24
    1          D2               25
    2          D3               26
    3          D4               27
    4          D5               28
    5          D6               29
    6          D7               30
    7          D8               31
    8          D9 no leg move   all other coarse (0,1,2,4,5,6)
               capped at 2x largest D class

Output:
    data/skeleton_dataset/leg_train_1x.csv
    data/skeleton_dataset/leg_val.csv
"""

import random
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")
ANN_DIR  = DATA_DIR / "annotations"
OUT_DIR  = DATA_DIR / "skeleton_dataset"
OUT_DIR.mkdir(exist_ok=True)

KP_TRAIN = DATA_DIR / "keypoints/train"
KP_VAL   = DATA_DIR / "keypoints/val"

# fine label 24-31 → D1-D8 → leg_label 0-7
# everything else  → D9    → leg_label 8
FINE_TO_LEG = {24:0, 25:1, 26:2, 27:3, 28:4, 29:5, 30:6, 31:7}

LABEL_NAMES = {
    0:"D1", 1:"D2", 2:"D3", 3:"D4",
    4:"D5", 5:"D6", 6:"D7", 7:"D8",
    8:"D9 no leg move",
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

def leg_label(fine):
    return FINE_TO_LEG.get(int(fine), 8)

def cap_no_leg(df, multiplier=1.0, seed=42):
    leg_df    = df[df["leg_label"] != 8]
    no_leg_df = df[df["leg_label"] == 8]
    if len(leg_df) == 0:
        return df
    largest = leg_df["leg_label"].value_counts().max()
    cap     = int(multiplier * largest)
    if len(no_leg_df) > cap:
        no_leg_df = no_leg_df.sample(n=cap, random_state=seed)
        print(f"  no_leg_move capped → {cap} (2x largest D class={largest})")
    return pd.concat([leg_df, no_leg_df]).reset_index(drop=True)

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
                "video":     video_name,
                "kp_path":   str(kp_path),
                "fine_label": fine,
                "coarse":    coarse,
                "leg_label": leg_label(fine),
            })

    df = pd.DataFrame(rows)
    if apply_cap:
        df = cap_no_leg(df)

    print(f"\n{'='*55}")
    print(f"{split_name} split  ({len(df)} samples, {missing} missing kp)")
    print(f"{'='*55}")
    counts = df["leg_label"].value_counts().sort_index()
    for lbl, name in LABEL_NAMES.items():
        print(f"  {lbl} {name:<22}: {counts.get(lbl,0)}")

    out_path = OUT_DIR / f"leg_{split_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    return df

if __name__ == "__main__":
    train_df = process_split(ANN_DIR/"train_list_videos.txt", KP_TRAIN, "train", apply_cap=True)
    val_df   = process_split(ANN_DIR/"val_list_videos.txt",   KP_VAL,   "val",   apply_cap=False)
