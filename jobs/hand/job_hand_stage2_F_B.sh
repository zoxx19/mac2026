#!/bin/bash -l
#SBATCH --job-name=hs2_F_B
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/hand_stage2_F_B.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/hand_stage2_F_B.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

$PYTHON src/hand/train_stage2.py     --group      F     --train_csv  data/hand_dataset/hand_F_train.csv     --val_csv    data/hand_dataset/hand_F_val.csv     --output_dir outputs/hand_stage2_F_B     --model_path models/videomae-ssv2     --init_model outputs/hand_stage1/best_model.pt     --batch_size    8     --stage1_epochs 10     --stage2_epochs 30     --lr_stage1     1e-3     --lr_stage2     1e-5     --patience      7
