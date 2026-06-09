"""
prepare_hand_annotations.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/prepare_hand_annotations.py --help` for options where applicable.
"""
import os
import pandas as pd

HAND_CLASSES = (
    set(range(11, 24))   # C: 11-23
    | set(range(32, 38)) # E: 32-37
    | set(range(38, 48)) # F: 38-47
    | set(range(48, 52)) # G: 48-51
)

SPLITS = ["train", "val", "test"]
SRC_DIR = "data/MMA-52/Annotations"
DST_DIR = "data/MMA-52/Annotations/hand_binary"


def process_split(split):
    src = os.path.join(SRC_DIR, f"{split}.csv")
    dst = os.path.join(DST_DIR, f"{split}.csv")

    df = pd.read_csv(src)
    total_instances = len(df)
    total_videos = df["video_id"].nunique()

    hand_df = df[df["class"].isin(HAND_CLASSES)].copy()
    hand_df["class"] = 1

    hand_instances = len(hand_df)
    hand_videos = hand_df["video_id"].nunique()

    hand_df.to_csv(dst, index=False)

    print(f"[{split}]")
    print(f"  Total instances : {total_instances:,}  |  Hand instances : {hand_instances:,}")
    print(f"  Total videos    : {total_videos:,}  |  Videos w/ hand  : {hand_videos:,}")
    print(f"  Saved → {dst}")


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()
