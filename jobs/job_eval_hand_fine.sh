#!/bin/bash -l
#SBATCH --job-name=eval_hf
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/eval_hand_fine.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/eval_hand_fine.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52
$PYTHON src/hand_fine/eval_hand_fine.py
