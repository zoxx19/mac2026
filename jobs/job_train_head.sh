#!/bin/bash -l
#SBATCH --job-name=head_v4a
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/head_v4a_%j.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/head_v4a_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/head/train.py \
    --train_csv     data/head_dataset/train_crops.csv \
    --val_csv       data/head_dataset/val_crops.csv \
    --output_dir    outputs/head_v4a \
    --model_name    /home/woody/iwso/iwso226h/ma52/models/videomae-kinetics \
    --model_type    videomae \
    --batch_size    8 \
    --stage1_epochs 5 \
    --stage2_epochs 20 \
    --lr_stage1     1e-3 \
    --lr_stage2     1e-5 \
    --dropout       0.0 \
    --patience      10 \
    --num_workers   4
