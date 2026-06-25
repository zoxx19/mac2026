#!/bin/bash -l
#SBATCH --job-name=eval_body
#SBATCH --output=logs/eval_body.log
#SBATCH --error=logs/eval_body.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/skeleton/evaluate_body_leg.py \
    --track body \
    --model_path outputs/body_skeleton_full/best_model.pt \
    --output_dir outputs/eval_body
