#!/bin/bash
#SBATCH --job-name=train_dsta
#SBATCH --account=iwso
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/train_dsta_%j.log

# ============================ USER CONFIGURATION =============================
# Edit these to match your cluster. Everything below this block is generic.
PROJECT_DIR="${PROJECT_DIR:-$PWD}"          # this repo (micro_challenge)
OPENTAD_DIR="${OPENTAD_DIR:-$PWD/../../OpenTAD}"   # OpenTAD checkout
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/MMA-52}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/work_dirs/dsta}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-$PROJECT_DIR/predictions}"
CONDA_ENV="${CONDA_ENV:-opentad}"
CONFIG="$OPENTAD_DIR/configs/adatad/mma52/e2e_mma52_videomae_l_dsta.py"   # copy from configs/adatad/dsta.py
# NOTE: SLURM flags are FAU-NHR specific. Do NOT add --mem (it errors here).
# ============================================================================

set -x
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$OPENTAD_DIR"
export PYTHONPATH="$OPENTAD_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=8
mkdir -p "$CHECKPOINT_DIR" "$PREDICTIONS_DIR" logs

# DSTA warm-starts (weights only) from a trained adapter checkpoint, then trains
# the fresh dual-path adapters. amp/fp16 are DISABLED in the config (fp16 -> Loss=inf).
WARM_START="${WARM_START:-$PROJECT_DIR/work_dirs/large1/checkpoint/epoch_22.pth}"
STRIPPED="$CHECKPOINT_DIR/warmstart_weights_only.pth"
mkdir -p "$CHECKPOINT_DIR"

# Build the weights-only (epoch=-1) warm-start checkpoint once.
if [ ! -f "$STRIPPED" ] && [ -f "$WARM_START" ]; then
    echo "Creating weights-only warm-start checkpoint from $WARM_START"
    python "$PROJECT_DIR/tools/strip_checkpoint.py" "$WARM_START" "$STRIPPED"
fi

# Prefer resuming an in-progress DSTA run (full state); else fresh warm start.
LATEST=$(find "$CHECKPOINT_DIR" -name 'epoch_*.pth' -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | tail -1 | awk '{print $2}')
if [ -n "$LATEST" ]; then
    RESUME="$LATEST";  echo "Resuming in-progress DSTA run from: $RESUME"
else
    RESUME="$STRIPPED"; echo "Fresh warm start from: $RESUME"
fi

torchrun --nproc_per_node=4 --master_port=29560 tools/train.py "$CONFIG" \
    --resume "$RESUME" \
    --cfg-options work_dir="$CHECKPOINT_DIR"
