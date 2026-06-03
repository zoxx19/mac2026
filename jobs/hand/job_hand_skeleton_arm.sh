#!/bin/bash -l
#SBATCH --job-name=hand_skeleton_arm
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/hand_skeleton_arm_%j.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/hand_skeleton_arm_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python

cd /home/woody/iwso/iwso226h/ma52
mkdir -p logs

echo "=== Run B: arm only (6 joints) ==="
echo "Started: $(date)"

$PYTHON src/skeleton/train_mmn.py \
    --mode arm \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --num_frames 64

echo "Finished: $(date)"
