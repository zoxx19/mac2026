#!/bin/bash -l
# NOTE: Update the PYTHON= path and cd path below to match your cluster environment.
#SBATCH --job-name=body_1x
#SBATCH --output=logs/body_1x.log
#SBATCH --error=logs/body_1x.err
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/skeleton/train_mmn_generic_v2.py \
    --track body --mode full --num_classes 6 \
    --modality joint --augment \
    --epochs 80 --batch_size 32 --lr 1e-4 \

