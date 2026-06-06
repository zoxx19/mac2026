#!/bin/bash -l
#SBATCH --job-name=ens_body
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/ensemble_body.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/ensemble_body.err
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/experiments/ensemble_body.py \
    --flow_model outputs/body_flow/best_model.pt \
    --mmn_model  outputs/body_full_joint_aug/best_model.pt \
    --flow_csv   data/skeleton_dataset/body_val_flow.csv \
    --mmn_csv    data/skeleton_dataset/body_val.csv \
    --model_path models/videomae-ssv2
