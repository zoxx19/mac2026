#!/bin/bash -l
#SBATCH --job-name=leg_large
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/leg_large.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/leg_large.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/experiments/train_videomae_large.py \
    --track leg --train_csv data/skeleton_dataset/leg_train_videomae.csv \
    --val_csv data/skeleton_dataset/leg_val_videomae.csv \
    --video_col crop_path --label_col leg_label --num_classes 9 \
    --output_dir outputs/leg_large --model_path models/videomae-large-kinetics \
    --batch_size 8 --accum_steps 2 --stage1_epochs 10 --stage2_epochs 20 \
    --lr_stage1 1e-3 --lr_stage2 5e-6 --patience 7 --mixup_alpha 0.4 --label_smooth 0.1
