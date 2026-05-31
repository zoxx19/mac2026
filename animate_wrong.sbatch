#!/bin/bash
#SBATCH --job-name=animate_wrong
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/animate_wrong_%j.log

module load python
conda run -n ma52 python src/skeleton/animate_wrong.py \
    --track       leg \
    --output_dir  outputs/eval_leg_animated \
    --n_per_class 5

conda run -n ma52 python src/skeleton/animate_wrong.py \
    --track       body \
    --output_dir  outputs/eval_body_animated \
    --n_per_class 5
