#!/bin/bash -l
#SBATCH --job-name=hand_s2_F
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/hand_stage2_F.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/hand_stage2_F.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/hand/train_stage2.py     --group      F     --train_csv  data/hand_dataset/hand_F_train.csv     --val_csv    data/hand_dataset/hand_F_val.csv     --output_dir outputs/hand_stage2_F     --model_path models/videomae-ssv2     --batch_size    8     --stage1_epochs 10     --stage2_epochs 30     --lr_stage1     1e-3     --lr_stage2     1e-5     --patience      7
