#!/bin/bash -l
#SBATCH --job-name=body_ovs
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/body_oversample.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/body_oversample.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

# Step 1: generate oversampled CSV
$PYTHON src/experiments/oversample_body.py

# Step 2: crop body videos (if not already done)
$PYTHON src/body_leg/crop_body.py --split train
$PYTHON src/body_leg/crop_body.py --split val

# Step 3: train with oversampling + focal loss + heavy aug
$PYTHON src/experiments/train_body_oversample.py \
    --train_csv  data/skeleton_dataset/body_train_oversample.csv \
    --val_csv    data/skeleton_dataset/body_val_videomae.csv \
    --output_dir outputs/body_oversample \
    --model_path models/videomae-large-kinetics \
    --batch_size    8 \
    --stage1_epochs 10 \
    --stage2_epochs 25 \
    --lr_stage1     1e-3 \
    --lr_stage2     1e-5 \
    --patience      7 \
    --focal_gamma   2.0
