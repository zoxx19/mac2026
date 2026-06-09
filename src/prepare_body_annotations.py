"""
prepare_body_annotations.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/prepare_body_annotations.py --help` for options where applicable.
"""
import os
import pandas as pd

BODY_CLASSES = set(range(0, 5))  # A: 0-4

SPLITS = ["train", "val", "test"]
SRC_DIR = "data/MMA-52/Annotations"
DST_DIR = "data/MMA-52/Annotations/body_binary"

def process_split(split):
    src = os.path.join(SRC_DIR, f"{split}.csv")
    dst = os.path.join(DST_DIR, f"{split}.csv")
    df = pd.read_csv(src)
    total_instances = len(df)
    total_videos = df["video_id"].nunique()
    body_df = df[df["class"].isin(BODY_CLASSES)].copy()
    body_df["class"] = 1
    body_instances = len(body_df)
    body_videos = body_df["video_id"].nunique()
    body_df.to_csv(dst, index=False)
    print(f"[{split}]")
    print(f"  Total instances : {total_instances:,}  |  Body instances : {body_instances:,}")
    print(f"  Total videos    : {total_videos:,}  |  Videos w/ body  : {body_videos:,}")
    print(f"  Saved → {dst}")

def main():
    os.makedirs(DST_DIR, exist_ok=True)
    for split in SPLITS:
        process_split(split)

if __name__ == "__main__":
    main()
