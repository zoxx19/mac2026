#!/bin/bash -l
#SBATCH --job-name=body_vmae
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/body_videomae.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/body_videomae.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

# Step 1: crop
echo "Cropping body — train..."
$PYTHON src/body_leg/crop_body.py --split train
echo "Cropping body — val..."
$PYTHON src/body_leg/crop_body.py --split val

# Step 2: train
$PYTHON src/body_leg/train_body_videomae.py \
    --train_csv  data/skeleton_dataset/body_train_videomae.csv \
    --val_csv    data/skeleton_dataset/body_val_videomae.csv \
    --output_dir outputs/body_videomae \
    --model_path models/videomae-ssv2 \
    --batch_size    8 \
    --stage1_epochs 10 \
    --stage2_epochs 20 \
    --lr_stage1     1e-3 \
    --lr_stage2     1e-5 \
    --patience      5
