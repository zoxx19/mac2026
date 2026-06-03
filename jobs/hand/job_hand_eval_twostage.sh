#!/bin/bash -l
#SBATCH --job-name=hand_eval
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/hand_eval_twostage.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/hand_eval_twostage.err
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/hand/evaluate_twostage.py \
    --stage1_model outputs/hand_stage1/best_model.pt \
    --stage2_C     outputs/hand_stage2_C/best_model.pt \
    --stage2_E     outputs/hand_stage2_E/best_model.pt \
    --stage2_F     outputs/hand_stage2_F/best_model.pt \
    --stage2_G     outputs/hand_stage2_G/best_model.pt \
    --val_csv      data/hand_dataset/hand_stage1_val.csv \
    --output_dir   outputs/eval_hand_twostage \
    --model_path   models/videomae-ssv2
