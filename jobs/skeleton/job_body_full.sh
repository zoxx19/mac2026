#!/bin/bash -l
#SBATCH --job-name=body_full
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/body_full_%j.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/body_full_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "=== body track | full joints | 6 classes ==="
echo "Started: $(date)"

$PYTHON src/skeleton/train_mmn_generic.py \
    --track body \
    --mode full \
    --num_classes 6 \
    --epochs 80 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_frames 64

echo "Finished: $(date)"
