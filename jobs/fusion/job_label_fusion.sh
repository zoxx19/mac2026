#!/bin/bash -l
#SBATCH --job-name=fusion
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/label_fusion.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/label_fusion.err
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/fusion/label_fusion.py \
    --val_csv       data/annotations/val_list_videos.txt \
    --model_path    models/videomae-ssv2 \
    --output_dir    outputs/label_fusion \
    --head_model    outputs/head_best_run1/best_model.pt \
    --body_flow     outputs/body_flow/best_model.pt \
    --body_mmn      outputs/body_full_joint_aug/best_model.pt \
    --leg_flow      outputs/leg_flow_ssv2/best_model.pt \
    --leg_mmn       outputs/leg_leg_joint_aug/best_model.pt \
    --hand_stage1   outputs/hand_stage1/best_model.pt \
    --hand_mmn_joint outputs/hand_skeleton_full/best_model.pt \
    --hand_mmn_bone  outputs/hand_full_bone_aug/best_model.pt \
    --hand_s2_C     outputs/hand_stage2_C_B/best_model.pt \
    --hand_s2_E     outputs/hand_stage2_E_B/best_model.pt \
    --hand_s2_F     outputs/hand_stage2_F_B/best_model.pt \
    --hand_s2_G     outputs/hand_stage2_G/best_model.pt \
    --body_flow_w   0.7 \
    --body_mmn_w    0.3 \
    --leg_flow_w    0.6 \
    --leg_mmn_w     0.4 \
    --hand_vmae_w   0.4 \
    --hand_joint_w  0.3 \
    --hand_bone_w   0.3
