#!/bin/bash -l
# NOTE: Update the PYTHON= path and cd path below to match your cluster environment.
#SBATCH --job-name=leg_flssv2
#SBATCH --output=logs/leg_flow_ssv2.log
#SBATCH --error=logs/leg_flow_ssv2.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/experiments/train_leg_flow.py \
    --train_csv  data/skeleton_dataset/leg_train_flow.csv \
    --val_csv    data/skeleton_dataset/leg_val_flow.csv \
    --output_dir outputs/leg_flow_ssv2 \
    --model_path models/videomae-ssv2 \
    --batch_size    8 \
    --stage1_epochs 10 \
    --stage2_epochs 20 \
    --lr_stage1     1e-3 \
    --lr_stage2     1e-5 \
    --patience      7
