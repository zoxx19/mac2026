"""
prepare_twostage.py
Prepares datasets for the two-stage hand recognition pipeline.

Stage 1: 5-class coarse (C/E/F/G/no-hand), no-hand capped at 1x largest class
Stage 2: Per-group fine-grained CSVs
  - hand_C_train.csv  (C1-C13, labels 0-12)
  - hand_E_train.csv  (E1-E6,  labels 0-5)
  - hand_F_train.csv  (F1-F10, labels 0-9)
  - hand_G_train.csv  (G1-G4,  labels 0-3)

Usage:
  python src/hand/prepare_twostage.py
"""

import pandas as pd
import numpy as np
import os

# Paths
ANNOT_DIR  = 'data/annotations'
KP_DIR_TR  = 'data/keypoints/train'
KP_DIR_VAL = 'data/keypoints/val'
VIDEO_TR   = 'data/videos/train/train'
VIDEO_VAL  = 'data/videos/val/val'
OUT_DIR    = 'data/hand_dataset'
os.makedirs(OUT_DIR, exist_ok=True)

# Fine2coarse mapping
# 0-4=body(0), 5-10=head(1), 11-23=upper_limb(2=C),
# 24-31=lower_limb(3=D), 32-37=body_hand(4=E),
# 38-47=head_hand(5=F), 48-51=leg_hand(6=G)
HAND_COARSE = {2: 0, 4: 1, 5: 2, 6: 3}  # C=0,E=1,F=2,G=3
NO_HAND_COARSE = {0, 1, 3}

def fine_to_coarse(fine):
    if 11 <= fine <= 23: return 2  # C
    if 32 <= fine <= 37: return 4  # E
    if 38 <= fine <= 47: return 5  # F
    if 48 <= fine <= 51: return 6  # G
    return -1  # no hand

def fine_to_local(fine):
    """Convert fine label to local label within group (0-based)"""
    if 11 <= fine <= 23: return fine - 11   # C: 0-12
    if 32 <= fine <= 37: return fine - 32   # E: 0-5
    if 38 <= fine <= 47: return fine - 38   # F: 0-9
    if 48 <= fine <= 51: return fine - 48   # G: 0-3
    return -1

def load_split(ann_file, kp_dir, split):
    rows = []
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            video = parts[0]
            fine  = int(parts[1])
            vid_id = video.replace('.mp4','')
            kp_path = os.path.join(kp_dir, f"{vid_id}.json")
            if not os.path.exists(kp_path): continue
            coarse = fine_to_coarse(fine)
            rows.append({
                'video':       video,
                'fine_label':  fine,
                'coarse_hand': coarse,  # 2=C,4=E,5=F,6=G,-1=no-hand
                'local_label': fine_to_local(fine),
            })
    return pd.DataFrame(rows)

print("Loading annotations...")
train_df = load_split(f'{ANNOT_DIR}/train_list_videos.txt', KP_DIR_TR, 'train')
val_df   = load_split(f'{ANNOT_DIR}/val_list_videos.txt',   KP_DIR_VAL, 'val')
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ── Stage 1 dataset ────────────────────────────────────────────────────────
# coarse_label: C=0, E=1, F=2, G=3, no-hand=4
def make_stage1_label(coarse):
    mapping = {2:0, 4:1, 5:2, 6:3}
    return mapping.get(coarse, 4)

train_df['stage1_label'] = train_df['coarse_hand'].apply(make_stage1_label)
val_df['stage1_label']   = val_df['coarse_hand'].apply(make_stage1_label)

# Add upperbody crop path
train_df['crop_path'] = train_df['video'].apply(
    lambda v: f"data/upperbody_crops/train/{v.replace('.mp4','')}.mp4")
val_df['crop_path'] = val_df['video'].apply(
    lambda v: f"data/upperbody_crops/val/{v.replace('.mp4','')}.mp4")

# Cap no-hand at 1x largest hand class
hand_train = train_df[train_df['stage1_label'] < 4]
largest_hand = hand_train['stage1_label'].value_counts().max()
no_hand_train = train_df[train_df['stage1_label'] == 4]
cap = int(largest_hand * 1.0)
no_hand_capped = no_hand_train.sample(n=min(cap, len(no_hand_train)),
                                       random_state=42)
stage1_train = pd.concat([hand_train, no_hand_capped]).sample(
    frac=1, random_state=42).reset_index(drop=True)

print(f"\nStage 1 train distribution (no-hand capped at {cap}):")
print(stage1_train['stage1_label'].value_counts().sort_index())
print(f"Total Stage 1 train: {len(stage1_train)}")

stage1_train.to_csv(f'{OUT_DIR}/hand_stage1_train.csv', index=False)
val_df.to_csv(f'{OUT_DIR}/hand_stage1_val.csv', index=False)
print(f"Saved: {OUT_DIR}/hand_stage1_train.csv")
print(f"Saved: {OUT_DIR}/hand_stage1_val.csv")

# ── Stage 2 datasets ───────────────────────────────────────────────────────
groups = {
    'C': {'coarse': 2, 'n_classes': 13, 'fine_range': (11, 23)},
    'E': {'coarse': 4, 'n_classes': 6,  'fine_range': (32, 37)},
    'F': {'coarse': 5, 'n_classes': 10, 'fine_range': (38, 47)},
    'G': {'coarse': 6, 'n_classes': 4,  'fine_range': (48, 51)},
}

for grp, cfg in groups.items():
    # Train
    grp_train = train_df[train_df['coarse_hand'] == cfg['coarse']].copy()
    grp_val   = val_df[val_df['coarse_hand'] == cfg['coarse']].copy()

    # Add head crop path for F group
    if grp == 'F':
        grp_train['crop_path'] = grp_train['video'].apply(
            lambda v: f"data/head_crops/train/{v.replace('.mp4','')}.mp4")
        grp_val['crop_path'] = grp_val['video'].apply(
            lambda v: f"data/head_crops/val/{v.replace('.mp4','')}.mp4")

    grp_train = grp_train.rename(columns={'local_label': 'label'})
    grp_val   = grp_val.rename(columns={'local_label': 'label'})

    grp_train.to_csv(f'{OUT_DIR}/hand_{grp}_train.csv', index=False)
    grp_val.to_csv(f'{OUT_DIR}/hand_{grp}_val.csv',     index=False)

    print(f"\nGroup {grp} train: {len(grp_train)} samples, "
          f"{cfg['n_classes']} classes")
    print(grp_train['label'].value_counts().sort_index().to_string())
    print(f"Saved: {OUT_DIR}/hand_{grp}_train.csv")

print("\nDone!")
