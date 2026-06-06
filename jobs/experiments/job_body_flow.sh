#!/bin/bash -l
#SBATCH --job-name=body_flow
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/body_flow.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/body_flow.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

echo "Extracting body flow — train..."
$PYTHON src/experiments/extract_body_flow.py --split train
echo "Extracting body flow — val..."
$PYTHON src/experiments/extract_body_flow.py --split val

$PYTHON src/experiments/train_body_flow.py \
    --train_csv  data/skeleton_dataset/body_train_flow.csv \
    --val_csv    data/skeleton_dataset/body_val_flow.csv \
    --output_dir outputs/body_flow \
    --model_path models/videomae-ssv2 \
    --batch_size    8 \
    --stage1_epochs 10 \
    --stage2_epochs 20 \
    --lr_stage1     1e-3 \
    --lr_stage2     1e-5 \
    --patience      7
