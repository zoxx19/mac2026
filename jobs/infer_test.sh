#!/bin/bash
#SBATCH --job-name=infer_test
#SBATCH --account=iwso
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/infer_test_%j.log

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

SPLIT="test"
SUBSET="testing"
OUT_NAME="${OUT_NAME:-large2_test}"          # -> $PREDICTIONS_DIR/$OUT_NAME.json
TMP="$PREDICTIONS_DIR/tmp_${OUT_NAME}"
mkdir -p "$TMP"

# Newest checkpoint by modification time.
CKPT="${CKPT:-$(find "$CHECKPOINT_DIR" -name 'epoch_*.pth' -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | tail -1 | awk '{print $2}')}"
echo "Using checkpoint: $CKPT"

torchrun --nproc_per_node=1 --master_port=29542 tools/test.py "$CONFIG" \
    --checkpoint "$CKPT" \
    --not_eval \
    --cfg-options \
        work_dir="$TMP" \
        post_processing.save_dict=True \
        dataset.test.subset_name="$SUBSET" \
        dataset.test.data_path="$DATA_DIR/extracted/$SPLIT/$SPLIT"

# Collect the detection JSON into the canonical predictions location.
find "$TMP" -name 'result_detection.json' | head -1 | \
    xargs -I{} cp {} "$PREDICTIONS_DIR/$OUT_NAME.json"
echo "Done: $PREDICTIONS_DIR/$OUT_NAME.json"
