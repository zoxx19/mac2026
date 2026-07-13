#!/bin/bash -l
# NOTE: Update the PYTHON= path and cd path below to match your cluster environment.
#SBATCH --job-name=hand_full_bone
#SBATCH --output=logs/hand_full_bone_%j.log
#SBATCH --error=logs/hand_full_bone_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52
mkdir -p logs

echo "=== hand_full_bone ==="
echo "Started: $(date)"

$PYTHON src/skeleton/train_mmn_generic_v2.py \
    --track hand \
    --mode full \
    --modality bone \
    --num_classes 5 \
    --augment \
    --epochs 80 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_frames 64

echo "Finished: $(date)"
