#!/bin/bash -l
#SBATCH --job-name=crop_nohand
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/crop_nohand.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/crop_nohand.err
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON - << 'PYEOF'
import pandas as pd, os, sys
sys.path.insert(0, 'src/hand')
from crop_upperbody import crop_video
from tqdm import tqdm

df = pd.read_csv('data/hand_dataset/hand_stage1_train.csv')
missing = df[~df['crop_path'].apply(os.path.exists)]
print(f"Cropping {len(missing)} missing no-hand videos...")

skipped = 0
for _, row in tqdm(missing.iterrows(), total=len(missing)):
    vid_id   = row['video'].replace('.mp4','')
    vid_path = f"data/videos/train/train/{vid_id}.mp4"
    kp_path  = f"data/keypoints/train/{vid_id}.json"
    out_path = row['crop_path']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.exists(kp_path):
        skipped += 1; continue
    crop_video(vid_path, kp_path, out_path)

print(f"Done. Skipped: {skipped}")
PYEOF
