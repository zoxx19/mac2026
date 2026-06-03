"""
oversample_leg.py
Oversamples rare leg classes (D2, D5) by repeating them in the training CSV.
Creates data/skeleton_dataset/leg_train_oversample.csv

Class counts: D1=238, D2=66, D3=92, D4=97, D5=60, D6=231, D7=259, D8=398, no-leg=398
Target: ~200 samples per active class

Oversample factors:
  D1 (238): 1x
  D2 (66):  3x → 198
  D3 (92):  2x → 184
  D4 (97):  2x → 194
  D5 (60):  3x → 180
  D6 (231): 1x
  D7 (259): 1x
  D8 (398): 1x
  no-leg:   1x

Usage:
  python src/experiments/oversample_leg.py
"""

import pandas as pd
import numpy as np
import os

TRAIN_CSV = 'data/skeleton_dataset/leg_train.csv'
OUT_CSV   = 'data/skeleton_dataset/leg_train_oversample.csv'

OVERSAMPLE = {
    0: 1,  # D1: 238
    1: 3,  # D2: 66 → 198
    2: 2,  # D3: 92 → 184
    3: 2,  # D4: 97 → 194
    4: 3,  # D5: 60 → 180
    5: 1,  # D6: 231
    6: 1,  # D7: 259
    7: 1,  # D8: 398
    8: 1,  # no-leg
}

df = pd.read_csv(TRAIN_CSV)
print("Original distribution:")
print(df['leg_label'].value_counts().sort_index())

parts = []
for label, factor in OVERSAMPLE.items():
    subset = df[df['leg_label'] == label]
    if factor > 1:
        repeated = pd.concat([subset] * factor, ignore_index=True)
        parts.append(repeated)
        print(f"  Label {label}: {len(subset)} → {len(repeated)} ({factor}x)")
    else:
        parts.append(subset)

out_df = pd.concat(parts, ignore_index=True).sample(
    frac=1, random_state=42).reset_index(drop=True)

print(f"\nOversampled distribution:")
print(out_df['leg_label'].value_counts().sort_index())
print(f"\nTotal: {len(out_df)} samples")

out_df.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
