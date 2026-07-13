#!/bin/bash -l
# NOTE: Update the PYTHON= path and cd path below to match your cluster environment.
#SBATCH --job-name=hs2_E_S
#SBATCH --output=logs/hand_stage2_E_ssv2_os.log
#SBATCH --error=logs/hand_stage2_E_ssv2_os.err
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/hand/train_stage2.py \
    --group         E \
    --train_csv     data/hand_dataset/hand_E_train_oversample.csv \
    --val_csv       data/hand_dataset/hand_E_val.csv \
    --output_dir    outputs/hand_stage2_E_ssv2_os \
    --model_path    models/videomae-ssv2 \
    --init_model    outputs/hand_stage1/best_model.pt \
    --batch_size    8 \
    --stage1_epochs 10 \
    --stage2_epochs 40 \
    --lr_stage1     1e-3 \
    --lr_stage2     1e-5 \
    --patience      10
