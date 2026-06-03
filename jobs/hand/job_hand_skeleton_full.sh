#!/bin/bash -l
#SBATCH --job-name=hand_skeleton_full
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/hand_skeleton_full_%j.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/hand_skeleton_full_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python

cd /home/woody/iwso/iwso226h/ma52
mkdir -p logs

echo "=== Run A: full body (17 joints) ==="
echo "Started: $(date)"

$PYTHON src/skeleton/train_mmn.py \
    --mode full \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --num_frames 64

echo "Finished: $(date)"
