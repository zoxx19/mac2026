#!/bin/bash -l
#SBATCH --job-name=head_long
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/head_long.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/head_long.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/head/train.py \
    --train_csv     data/head_dataset/train_crops.csv \
    --val_csv       data/head_dataset/val_crops.csv \
    --output_dir    outputs/head_long \
    --model_name    models/videomae-ssv2 \
    --model_type    videomae \
    --batch_size    16 \
    --stage1_epochs 10 \
    --stage2_epochs 20 \
    --patience      7
