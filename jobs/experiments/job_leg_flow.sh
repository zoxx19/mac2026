#!/bin/bash -l
#SBATCH --job-name=leg_flow
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/leg_flow.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/leg_flow.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

# Step 1: extract optical flow from leg crops
echo "Extracting optical flow — train..."
$PYTHON src/experiments/extract_optical_flow.py --split train
echo "Extracting optical flow — val..."
$PYTHON src/experiments/extract_optical_flow.py --split val

# Step 2: train on flow videos
$PYTHON src/experiments/train_leg_flow.py \
    --train_csv  data/skeleton_dataset/leg_train_flow.csv \
    --val_csv    data/skeleton_dataset/leg_val_flow.csv \
    --output_dir outputs/leg_flow \
    --model_path models/videomae-large-kinetics \
    --batch_size    8 \
    --stage1_epochs 10 \
    --stage2_epochs 20 \
    --lr_stage1     1e-3 \
    --lr_stage2     5e-6 \
    --patience      7
