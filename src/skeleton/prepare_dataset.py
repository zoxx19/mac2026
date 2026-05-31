"""
prepare_dataset.py
Prepares 5-class labels for skeleton-based hand movement recognition.

Label mapping:
    0 → coarse 2  (C: upper limb)
    1 → coarse 4  (E: body-hand)
    2 → coarse 5  (F: head-hand)
    3 → coarse 6  (G: leg-hand)
    4 → no hand movement  (coarse 0, 1, 3) — capped at 2× largest hand class

fine2coarse:
    0-4   → 0 (body A)
    5-10  → 1 (head B)
    11-23 → 2 (upper limb C)
    24-31 → 3 (lower limb D)
    32-37 → 4 (body-hand E)
    38-47 → 5 (head-hand F)
    48-51 → 6 (leg-hand G)

Output:
    data/skeleton_dataset/hand_train.csv
    data/skeleton_dataset/hand_val.csv

Each CSV has columns:
    video       — original video filename
    kp_path     — path to keypoint JSON file
    fine_label  — original fine label (0-51)
    coarse      — coarse label (0-6)
    hand_label  — 0=C, 1=E, 2=F, 3=G, 4=no_hand_move
"""

import random
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/woody/iwso/iwso226h/ma52/data")
ANN_DIR  = DATA_DIR / "annotations"
OUT_DIR  = DATA_DIR / "skeleton_dataset"
OUT_DIR.mkdir(exist_ok=True)

KP_TRAIN = DATA_DIR / "keypoints/train"
KP_VAL   = DATA_DIR / "keypoints/val"

LABEL_NAMES = {
    0: "C upper limb",
    1: "E body-hand",
    2: "F head-hand",
    3: "G leg-hand",
    4: "no hand move",
}

# ── fine label → coarse label ─────────────────────────────────────────────────
def fine_to_coarse(fine):
    fine = int(fine)
    if   fine <= 4:  return 0   # body A
    elif fine <= 10: return 1   # head B
    elif fine <= 23: return 2   # upper limb C
    elif fine <= 31: return 3   # lower limb D
    elif fine <= 37: return 4   # body-hand E
    elif fine <= 47: return 5   # head-hand F
    else:            return 6   # leg-hand G


# ── coarse → 5-class hand label ───────────────────────────────────────────────
def hand_label(coarse):
    if   coarse == 2: return 0   # C upper limb
    elif coarse == 4: return 1   # E body-hand
    elif coarse == 5: return 2   # F head-hand
    elif coarse == 6: return 3   # G leg-hand
    else:             return 4   # no hand movement (coarse 0, 1, 3)


# ── Cap no-hand class (train only) ────────────────────────────────────────────
def cap_no_hand(df, multiplier=2.0, seed=42):
    hand_df    = df[df["hand_label"] != 4]
    no_hand_df = df[df["hand_label"] == 4]

    largest = hand_df["hand_label"].value_counts().max()
    cap     = int(multiplier * largest)

    if len(no_hand_df) > cap:
        no_hand_df = no_hand_df.sample(n=cap, random_state=seed)
        print(f"  no_hand_move capped → {cap} samples (2× largest hand class={largest})")

    return pd.concat([hand_df, no_hand_df]).reset_index(drop=True)


# ── Process one split ─────────────────────────────────────────────────────────
def process_split(ann_file, kp_dir, split_name, apply_cap=False):
    rows = []
    missing_kp = 0

    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            video_name = parts[0]
            fine_label = int(parts[1])
            coarse     = fine_to_coarse(fine_label)

            kp_path = kp_dir / (Path(video_name).stem + ".json")
            if not kp_path.exists():
                missing_kp += 1
                continue

            rows.append({
                "video":      video_name,
                "kp_path":    str(kp_path),
                "fine_label": fine_label,
                "coarse":     coarse,
                "hand_label": hand_label(coarse),
            })

    df = pd.DataFrame(rows)

    if apply_cap:
        df = cap_no_hand(df)

    print(f"\n{'='*55}")
    print(f"{split_name} split")
    print(f"{'='*55}")
    print(f"Total videos with keypoints : {len(df)}")
    print(f"Missing keypoints skipped   : {missing_kp}")

    print(f"\nHand label distribution:")
    counts = df["hand_label"].value_counts().sort_index()
    for lbl, name in LABEL_NAMES.items():
        print(f"  {lbl} {name:<20}: {counts.get(lbl, 0)}")

    out_path = OUT_DIR / f"hand_{split_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_df = process_split(ANN_DIR / "train_list_videos.txt", KP_TRAIN, "train", apply_cap=True)
    val_df   = process_split(ANN_DIR / "val_list_videos.txt",   KP_VAL,   "val",   apply_cap=False)

    print(f"\n{'='*55}")
    print("Summary")
    print(f"{'='*55}")
    counts_tr = train_df["hand_label"].value_counts().sort_index()
    counts_val = val_df["hand_label"].value_counts().sort_index()
    for lbl, name in LABEL_NAMES.items():
        print(f"  {name:<20}  train={counts_tr.get(lbl,0):>5}  val={counts_val.get(lbl,0):>5}")