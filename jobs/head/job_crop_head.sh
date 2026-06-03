#!/bin/bash -l
#SBATCH --job-name=crop_dynamic
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/crop_dynamic_%j.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/crop_dynamic_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python

cd /home/woody/iwso/iwso226h/ma52

# crop train
$PYTHON src/head/crop_clips.py \
    --csv       data/head_dataset/train.csv \
    --video_dir data/videos/train/train \
    --kp_dir    data/keypoints/train \
    --out_dir   data/head_crops_dynamic/train \
    --out_csv   data/head_dataset/train_crops_dynamic.csv \
    --size      224 \
    --split     train

# crop val
$PYTHON src/head/crop_clips.py \
    --csv       data/head_dataset/val.csv \
    --video_dir data/videos/val/val \
    --kp_dir    data/keypoints/val \
    --out_dir   data/head_crops_dynamic/val \
    --out_csv   data/head_dataset/val_crops_dynamic.csv \
    --size      224 \
    --split     val
