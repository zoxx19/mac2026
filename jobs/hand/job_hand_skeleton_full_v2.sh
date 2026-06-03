#!/bin/bash -l
#SBATCH --job-name=hand_full_v2
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/hand_full_v2_%j.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/hand_full_v2_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python

cd /home/woody/iwso/iwso226h/ma52
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "=== Run C: full body (17 joints) — paper hyperparams + STCA ==="
echo "Started: $(date)"

$PYTHON src/skeleton/train_mmn_v2.py \
    --mode full \
    --epochs 80 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_frames 64

echo "Finished: $(date)"
