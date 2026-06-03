#!/bin/bash -l
#SBATCH --job-name=test_large
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/test_large.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/test_large.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52
$PYTHON src/experiments/test_large_model.py
