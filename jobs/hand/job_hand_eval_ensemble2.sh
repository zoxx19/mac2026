#!/bin/bash -l
# NOTE: Update the PYTHON= path and cd path below to match your cluster environment.
#SBATCH --job-name=hand_ens2
#SBATCH --output=logs/hand_eval_ensemble2.log
#SBATCH --error=logs/hand_eval_ensemble2.err
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/hand/evaluate_twostage_ensemble.py \
    --stage1_videomae  outputs/hand_stage1/best_model.pt \
    --stage1_mmn_joint outputs/hand_skeleton_full/best_model.pt \
    --stage1_mmn_bone  outputs/hand_full_bone_aug/best_model.pt \
    --stage2_C         outputs/hand_stage2_C_ssv2_os/best_model.pt \
    --stage2_E         outputs/hand_stage2_E_ssv2_os/best_model.pt \
    --stage2_F         outputs/hand_stage2_F_ssv2_os/best_model.pt \
    --stage2_G         outputs/hand_stage2_G_ssv2_os/best_model.pt \
    --val_csv          data/hand_dataset/hand_stage1_val.csv \
    --model_path       models/videomae-ssv2 \
    --output_dir       outputs/eval_hand_twostage_ensemble2
