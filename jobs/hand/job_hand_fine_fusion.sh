#!/bin/bash -l
#SBATCH --job-name=hand_fine_fusion
#SBATCH --output=logs/hand_fine_fusion.log
#SBATCH --error=logs/hand_fine_fusion.err
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/hand_fine/train_hand_fine.py \
    --videomae_path models/videomae-ssv2 \
    --output_dir    outputs/hand_fine_fusion \
    --augment \
    --epochs        80 \
    --batch_size    16 \
    --lr            1e-4 \
    --weight_decay  0.1
