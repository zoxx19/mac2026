#!/bin/bash -l
#SBATCH --job-name=leg_tsf
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/leg_timesformer.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/leg_timesformer.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/experiments/train_timesformer.py \
    --track        leg \
    --train_csv    data/skeleton_dataset/leg_train_videomae.csv \
    --val_csv      data/skeleton_dataset/leg_val_videomae.csv \
    --video_col    crop_path \
    --label_col    leg_label \
    --num_classes  9 \
    --output_dir   outputs/leg_timesformer \
    --model_path   models/timesformer-ssv2 \
    --img_size     224 \
    --batch_size   8 \
    --stage1_epochs 10 \
    --stage2_epochs 20 \
    --lr_stage1    1e-3 \
    --lr_stage2    1e-5 \
    --patience     7
