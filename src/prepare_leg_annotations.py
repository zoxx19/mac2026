"""
prepare_leg_annotations.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/prepare_leg_annotations.py --help` for options where applicable.
"""
import os
import pandas as pd

LEG_CLASSES = set(range(24, 32))  # D: 24-31

SPLITS = ["train", "val", "test"]
SRC_DIR = "data/MMA-52/Annotations"
DST_DIR = "data/MMA-52/Annotations/leg_binary"

def process_split(split):
    src = os.path.join(SRC_DIR, f"{split}.csv")
    dst = os.path.join(DST_DIR, f"{split}.csv")
    df = pd.read_csv(src)
    total_instances = len(df)
    total_videos = df["video_id"].nunique()
    leg_df = df[df["class"].isin(LEG_CLASSES)].copy()
    leg_df["class"] = 1
    leg_instances = len(leg_df)
    leg_videos = leg_df["video_id"].nunique()
    leg_df.to_csv(dst, index=False)
    print(f"[{split}]")
    print(f"  Total instances : {total_instances:,}  |  Leg instances : {leg_instances:,}")
    print(f"  Total videos    : {total_videos:,}  |  Videos w/ leg  : {leg_videos:,}")
    print(f"  Saved → {dst}")

def main():
    os.makedirs(DST_DIR, exist_ok=True)
    for split in SPLITS:
        process_split(split)

if __name__ == "__main__":
    main()
