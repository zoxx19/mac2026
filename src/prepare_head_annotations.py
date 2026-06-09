"""
Filters MMA-52 annotations to head-movement classes (5-10) and relabels them to 1.
Saves per-split CSVs to data/MMA-52/Annotations/head_binary/.
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import pandas as pd
from pathlib import Path

ANN_DIR      = Path("data/MMA-52/Annotations")
OUT_DIR      = ANN_DIR / "head_binary"
HEAD_CLASSES = {5, 6, 7, 8, 9, 10}
SPLITS       = ["train", "val", "test"]


def process_split(split):
    df = pd.read_csv(ANN_DIR / f"{split}.csv")
    total_before = len(df)
    class_counts = df[df["class"].isin(HEAD_CLASSES)]["class"].value_counts().sort_index()

    df_head = df[df["class"].isin(HEAD_CLASSES)].copy()
    df_head["class"] = 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_head.to_csv(OUT_DIR / f"{split}.csv", index=False)

    return total_before, len(df_head), class_counts


def main():
    print(f"Head classes: {sorted(HEAD_CLASSES)}\n")
    for split in SPLITS:
        total, kept, class_counts = process_split(split)
        print(f"[{split}]")
        print(f"  Total rows before filtering: {total}")
        print(f"  Head rows kept:              {kept}")
        print(f"  Per-class counts before relabeling:")
        for cls, cnt in class_counts.items():
            print(f"    class {cls}: {cnt}")
        print()


if __name__ == "__main__":
    main()
