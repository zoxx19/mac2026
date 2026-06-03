#!/bin/bash -l
#SBATCH --job-name=head_large
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/head_large.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/head_large.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/experiments/train_videomae_large.py \
    --track head --train_csv data/head_dataset/train_crops.csv \
    --val_csv data/head_dataset/val_crops.csv \
    --video_col crop_path --label_col head_label --num_classes 7 \
    --output_dir outputs/head_large --model_path models/videomae-large-kinetics \
    --batch_size 8 --accum_steps 2 --stage1_epochs 10 --stage2_epochs 20 \
    --lr_stage1 1e-3 --lr_stage2 5e-6 --patience 7 --mixup_alpha 0.4 --label_smooth 0.1
