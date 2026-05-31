"""
prepare_hand_fine_dataset.py
Builds train/val CSVs for the fine-grained hand fusion model.

Label schema (34 classes):
  C  fine 11-23  →  local  0-12  (13 classes)
  E  fine 32-37  →  local 13-18  ( 6 classes)
  F  fine 38-47  →  local 19-28  (10 classes)
  G  fine 48-51  →  local 29-32  ( 4 classes)
  no-hand        →  local    33  ( 1 class )

No-hand = coarse 0 (A), 1 (B), 3 (D)
No-hand capped at 2x largest hand class in train.

Output:
  data/hand_dataset/hand_fine_train.csv
  data/hand_dataset/hand_fine_val.csv

Columns: video_id, fine_label, local_label, video_path, has_keypoints
"""

import os
import random
import pandas as pd

random.seed(42)

ROOT       = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
ANNO_TRAIN = os.path.join(ROOT, 'annotations', 'train_list_videos.txt')
ANNO_VAL   = os.path.join(ROOT, 'annotations', 'val_list_videos.txt')
KP_TRAIN   = os.path.join(ROOT, 'keypoints', 'train')
KP_VAL     = os.path.join(ROOT, 'keypoints', 'val')
VID_TRAIN  = os.path.join(ROOT, 'videos', 'train', 'train')
VID_VAL    = os.path.join(ROOT, 'videos', 'val', 'val')
OUT_DIR    = os.path.join(ROOT, 'hand_dataset')
os.makedirs(OUT_DIR, exist_ok=True)

# fine2coarse
FINE2COARSE = {}
for f in range(0,  5):  FINE2COARSE[f] = 0
for f in range(5,  11): FINE2COARSE[f] = 1
for f in range(11, 24): FINE2COARSE[f] = 2
for f in range(24, 32): FINE2COARSE[f] = 3
for f in range(32, 38): FINE2COARSE[f] = 4
for f in range(38, 48): FINE2COARSE[f] = 5
for f in range(48, 52): FINE2COARSE[f] = 6

NO_HAND_COARSE = {0, 1, 3}   # A, B, D → no hand move


def fine_to_local(fine_label):
    if 11 <= fine_label <= 23: return fine_label - 11          # C: 0-12
    if 32 <= fine_label <= 37: return fine_label - 32 + 13    # E: 13-18
    if 38 <= fine_label <= 47: return fine_label - 38 + 19    # F: 19-28
    if 48 <= fine_label <= 51: return fine_label - 48 + 29    # G: 29-32
    if FINE2COARSE.get(fine_label) in NO_HAND_COARSE: return 33
    return None  # D (leg) excluded


def build_split(anno_path, kp_dir, vid_dir):
    rows = []
    with open(anno_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts      = line.split()
            video_file = parts[0]
            fine_label = int(parts[1])
            local      = fine_to_local(fine_label)
            if local is None: continue
            video_id   = os.path.splitext(video_file)[0]
            rows.append({
                'video_id':      video_id,
                'fine_label':    fine_label,
                'local_label':   local,
                'video_path':    os.path.join(vid_dir, video_file),
                'has_keypoints': os.path.exists(os.path.join(kp_dir, f"{video_id}.json")),
            })
    return pd.DataFrame(rows)


def cap_nohand(df):
    hand_df   = df[df['local_label'] < 33]
    nohand_df = df[df['local_label'] == 33]
    cap       = int(hand_df['local_label'].value_counts().max()) * 2
    print(f"  no-hand: {len(nohand_df)} → cap {cap}")
    if len(nohand_df) > cap:
        nohand_df = nohand_df.sample(cap, random_state=42)
    return pd.concat([hand_df, nohand_df]).sample(frac=1, random_state=42).reset_index(drop=True)


LABEL_NAMES = {
    **{i:   f'C{i+1}'   for i in range(0,  13)},
    **{i:   f'E{i-12}'  for i in range(13, 19)},
    **{i:   f'F{i-18}'  for i in range(19, 29)},
    **{i:   f'G{i-28}'  for i in range(29, 33)},
    33: 'no-hand',
}


def print_dist(df, name):
    print(f"\n[{name}] {len(df)} samples")
    for lbl, cnt in df['local_label'].value_counts().sort_index().items():
        print(f"  {lbl:2d} {LABEL_NAMES.get(lbl,'?'):>10s}: {cnt}")


if __name__ == '__main__':
    print("Building fine-grained hand dataset (34 classes)...")
    train_df = build_split(ANNO_TRAIN, KP_TRAIN, VID_TRAIN)
    val_df   = build_split(ANNO_VAL,   KP_VAL,   VID_VAL)
    print(f"Raw — train: {len(train_df)}, val: {len(val_df)}")
    train_df = cap_nohand(train_df)
    print_dist(train_df, 'train')
    print_dist(val_df,   'val')
    train_df.to_csv(os.path.join(OUT_DIR, 'hand_fine_train.csv'), index=False)
    val_df.to_csv(  os.path.join(OUT_DIR, 'hand_fine_val.csv'),   index=False)
    print("\nDone.")
