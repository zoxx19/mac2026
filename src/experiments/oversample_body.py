"""
oversample_body.py
Oversamples rare body classes (A3, A4) by repeating them in the training CSV.
Creates data/skeleton_dataset/body_train_oversample.csv

Oversample factors (tuned to reach ~200 samples per class):
  A1 (309): 1x
  A2 (150): 1x
  A3 (20):  10x → 200
  A4 (9):   22x → 198
  A5 (156): 1x
  no-body (618): 1x (already capped at 1x)

Usage:
  python src/experiments/oversample_body.py
"""

import pandas as pd
import numpy as np
import os

TRAIN_CSV = 'data/skeleton_dataset/body_train.csv'
OUT_CSV   = 'data/skeleton_dataset/body_train_oversample.csv'

OVERSAMPLE = {
    0: 1,   # A1: 309
    1: 1,   # A2: 150
    2: 10,  # A3: 20 → 200
    3: 22,  # A4: 9  → 198
    4: 1,   # A5: 156
    5: 1,   # no-body: 618
}

df = pd.read_csv(TRAIN_CSV)
print("Original distribution:")
print(df['body_label'].value_counts().sort_index())

parts = []
for label, factor in OVERSAMPLE.items():
    subset = df[df['body_label'] == label]
    if factor > 1:
        repeated = pd.concat([subset] * factor, ignore_index=True)
        parts.append(repeated)
        print(f"  Label {label}: {len(subset)} → {len(repeated)} ({factor}x)")
    else:
        parts.append(subset)

out_df = pd.concat(parts, ignore_index=True).sample(
    frac=1, random_state=42).reset_index(drop=True)

print(f"\nOversampled distribution:")
print(out_df['body_label'].value_counts().sort_index())
print(f"\nTotal: {len(out_df)} samples")

out_df.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
