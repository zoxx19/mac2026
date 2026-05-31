"""
prepare_dataset.py
Reads MA-52 annotations and creates train/val CSVs for head recognition.
Labels: B1-B6 (fine labels 5-10) + B7 (no head movement, all other classes)
B7 is undersampled to 2x the largest B class to handle imbalance.
"""

import pandas as pd
from pathlib import Path
import random

random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("/home/woody/iwso/iwso226h/ma52/data")
ANN_DIR   = DATA_DIR / "annotations"
OUT_DIR   = DATA_DIR / "head_dataset"
OUT_DIR.mkdir(exist_ok=True)

# ── Label mapping ─────────────────────────────────────────────────────────────
FINE_TO_HEAD = {
    5: "B1",   # nodding
    6: "B2",   # shaking head
    7: "B3",   # turning head
    8: "B4",   # tilting head
    9: "B5",   # bowing head
    10: "B6",  # head up
}

def fine_to_head_label(fine_label):
    return FINE_TO_HEAD.get(int(fine_label), "B7")

# ── Process one split ─────────────────────────────────────────────────────────
def process_split(ann_file, split_name, b7_cap=None):
    rows = []
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            video_name = parts[0]
            fine_label = int(parts[1])
            head_label = fine_to_head_label(fine_label)
            rows.append({
                "video":      video_name,
                "fine_label": fine_label,
                "head_label": head_label,
            })

    df = pd.DataFrame(rows)

    print(f"\n{'='*50}")
    print(f"{split_name} — before B7 cap ({len(df)} total)")
    print(df["head_label"].value_counts().sort_index())

    # undersample B7
    if b7_cap is not None:
        b7_df    = df[df["head_label"] == "B7"].sample(
                       min(b7_cap, len(df[df["head_label"] == "B7"])),
                       random_state=42)
        other_df = df[df["head_label"] != "B7"]
        df       = pd.concat([other_df, b7_df]).reset_index(drop=True)
        df       = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n{split_name} — after B7 cap ({len(df)} total)")
    print(df["head_label"].value_counts().sort_index())

    out_path = OUT_DIR / f"{split_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    return df

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # first pass — get class counts to compute B7 cap
    train_df_full = process_split(ANN_DIR / "train_list_videos.txt", "train_full")

    # compute cap = 2x largest B class (B1-B6 only)
    b_counts  = train_df_full[train_df_full["head_label"] != "B7"]["head_label"].value_counts()
    largest_b = b_counts.max()
    b7_cap    = int(largest_b * 2)
    print(f"\nLargest B class: {largest_b} → B7 cap: {b7_cap}")

    # second pass — apply cap
    process_split(ANN_DIR / "train_list_videos.txt", "train", b7_cap=b7_cap)

    # val — no cap (we want full val for honest evaluation)
    process_split(ANN_DIR / "val_list_videos.txt", "val")