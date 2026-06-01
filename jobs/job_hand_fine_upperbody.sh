#!/bin/bash -l
#SBATCH --job-name=hand_fine_crop
#SBATCH --output=/home/woody/iwso/iwso226h/ma52/logs/hand_fine_upperbody.log
#SBATCH --error=/home/woody/iwso/iwso226h/ma52/logs/hand_fine_upperbody.err
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
cd /home/woody/iwso/iwso226h/ma52

# Step 1: crop upper body
if [ ! -f data/hand_dataset/hand_fine_upperbody_train.csv ]; then
    echo "Cropping upper body — train..."
    $PYTHON src/hand_fine/crop_upperbody.py --split train
    echo "Cropping upper body — val..."
    $PYTHON src/hand_fine/crop_upperbody.py --split val
fi

# Step 2: train on upper body crops (uses crop script)
$PYTHON src/hand_fine/train_hand_fine_crop.py \
    --videomae_path models/videomae-ssv2 \
    --output_dir    outputs/hand_fine_upperbody \
    --augment \
    --epochs        80 \
    --batch_size    16 \
    --lr            1e-4 \
    --weight_decay  0.1
