#!/bin/bash -l
#SBATCH --job-name=ma52_pose
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/pose_%j.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/pose_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
export YOLO_CONFIG_DIR=/home/woody/iwso/iwso226h/ultralytics_config

cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/extraction/extract_pose.py \
    --video_dir data/videos/train/train \
    --kp_dir    data/keypoints/train_v2 \
    --vis_dir   data/visualizations/train_v2 \
    --n_vis     20 \
    --max_videos 50 \
    --split     train
