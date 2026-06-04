#!/bin/bash -l
#SBATCH --job-name=body_tshr
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/body_timesformer_hr.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/body_timesformer_hr.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/experiments/train_timesformer.py \
    --track        body \
    --train_csv    data/skeleton_dataset/body_train_videomae.csv \
    --val_csv      data/skeleton_dataset/body_val_videomae.csv \
    --video_col    crop_path \
    --label_col    body_label \
    --num_classes  6 \
    --output_dir   outputs/body_timesformer_hr \
    --model_path   models/timesformer-hr-ssv2 \
    --img_size     448 \
    --batch_size   4 \
    --stage1_epochs 10 \
    --stage2_epochs 20 \
    --lr_stage1    1e-3 \
    --lr_stage2    1e-5 \
    --patience     7
