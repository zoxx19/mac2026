#!/bin/bash
#SBATCH --job-name=train_large2
#SBATCH --account=iwso
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/train_large2_%j.log

# ============================ USER CONFIGURATION =============================
# Edit these to match your cluster. Everything below this block is generic.
PROJECT_DIR="${PROJECT_DIR:-$PWD}"          # this repo (micro_challenge)
OPENTAD_DIR="${OPENTAD_DIR:-$PWD/../../OpenTAD}"   # OpenTAD checkout
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/MMA-52}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/work_dirs/large2}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-$PROJECT_DIR/predictions}"
CONDA_ENV="${CONDA_ENV:-opentad}"
CONFIG="$OPENTAD_DIR/configs/adatad/mma52/e2e_mma52_videomae_l_adapter2.py"   # copy from configs/adatad/large2.py
# NOTE: SLURM flags are FAU-NHR specific. Do NOT add --mem (it errors here).
# ============================================================================

set -x
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$OPENTAD_DIR"
export PYTHONPATH="$OPENTAD_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=8
mkdir -p "$CHECKPOINT_DIR" "$PREDICTIONS_DIR" logs

# Resume the newest in-progress checkpoint if one exists (modification-time
# sort, NOT `sort -V` -- version sort picks the wrong epoch, see TECHNICAL_NOTES).
LATEST=$(find "$CHECKPOINT_DIR" -name 'epoch_*.pth' -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | tail -1 | awk '{print $2}')

if [ -n "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    RESUME_ARG=(--resume "$LATEST")
else
    echo "Starting from scratch"
    RESUME_ARG=()
fi

torchrun --nproc_per_node=2 --master_port=29501 tools/train.py "$CONFIG" \
    "${RESUME_ARG[@]}" \
    --cfg-options work_dir="$CHECKPOINT_DIR"
