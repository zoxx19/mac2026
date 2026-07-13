#!/bin/bash -l
# NOTE: Update the PYTHON= path and cd path below to match your cluster environment.
#SBATCH --job-name=eval_head
#SBATCH --output=logs/eval_head_%j.log
#SBATCH --error=logs/eval_head_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/head/evaluate.py \
    --val_csv    data/head_dataset/val_crops.csv \
    --model_path outputs/head_recognition/best_model.pt \
    --model_name models/videomae-ssv2 \
    --model_type videomae \
    --output_dir outputs/head_recognition/evaluation \
    --n_wrong    5
