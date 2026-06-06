#!/bin/bash -l
#SBATCH --job-name=ens_leg
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/ensemble_leg.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/ensemble_leg.err
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/experiments/ensemble_leg.py \
    --flow_model outputs/leg_flow_ssv2/best_model.pt \
    --mmn_model  outputs/leg_leg_joint_aug/best_model.pt \
    --flow_csv   data/skeleton_dataset/leg_val_flow.csv \
    --mmn_csv    data/skeleton_dataset/leg_val.csv \
    --model_path models/videomae-ssv2
