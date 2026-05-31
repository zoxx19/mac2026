#!/bin/bash -l
#SBATCH --job-name=pose_new
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/pose_new.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/pose_new.err
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/extraction/extract_pose.py \
    --video_dir /home/woody/iwso/iwso226h/new_dataset \
    --kp_dir    /home/woody/iwso/iwso226h/new_dataset_keypoints \
    --vis_dir   /home/woody/iwso/iwso226h/new_dataset_vis \
    --n_vis     1000 \
    --split     new
