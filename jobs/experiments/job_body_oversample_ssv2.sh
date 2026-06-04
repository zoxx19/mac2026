#!/bin/bash -l
#SBATCH --job-name=body_ovs2
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/body_oversample_ssv2.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/body_oversample_ssv2.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/experiments/oversample_body.py

$PYTHON src/experiments/train_body_oversample.py \
    --train_csv  data/skeleton_dataset/body_train_oversample.csv \
    --val_csv    data/skeleton_dataset/body_val_videomae.csv \
    --output_dir outputs/body_oversample_ssv2 \
    --model_path models/videomae-ssv2 \
    --batch_size    8 \
    --stage1_epochs 10 \
    --stage2_epochs 25 \
    --lr_stage1     1e-3 \
    --lr_stage2     1e-5 \
    --patience      7 \
    --focal_gamma   2.0
