#!/bin/bash
# One-shot environment setup for MAC 2026 Track 2 (Multi-label Micro-Action Detection).
#
# What it does:
#   1. Creates / activates a conda env and installs Python deps.
#   2. Clones OpenTAD (the AdaTAD/ActionFormer detector framework).
#   3. Installs our custom configs + DSTA backbone + losses INTO OpenTAD.
#   4. Creates the expected data/ and work_dirs/ directory skeleton.
#   5. Runs the environment smoke test.
#
# Usage:
#   bash setup.sh
#
# Override any of these on the command line, e.g.:
#   OPENTAD_DIR=~/code/OpenTAD CONDA_ENV=mac26 bash setup.sh
set -u

# ============================ USER CONFIGURATION =============================
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
OPENTAD_DIR="${OPENTAD_DIR:-$PROJECT_DIR/../OpenTAD}"
CONDA_ENV="${CONDA_ENV:-opentad}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
OPENTAD_REPO="${OPENTAD_REPO:-https://github.com/sming256/OpenTAD.git}"
# ============================================================================

echo "==> Project : $PROJECT_DIR"
echo "==> OpenTAD : $OPENTAD_DIR"
echo "==> Conda   : $CONDA_ENV (python $PYTHON_VERSION)"

# --- 1. conda env + deps -----------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | grep -qE "^\s*$CONDA_ENV\s"; then
        echo "==> Creating conda env '$CONDA_ENV'"
        conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
    fi
    conda activate "$CONDA_ENV"
else
    echo "!! conda not found — using the current Python environment."
fi

echo "==> Installing Python requirements"
pip install -r "$PROJECT_DIR/requirements.txt"

# --- 2. OpenTAD --------------------------------------------------------------
if [ ! -d "$OPENTAD_DIR" ]; then
    echo "==> Cloning OpenTAD into $OPENTAD_DIR"
    git clone "$OPENTAD_REPO" "$OPENTAD_DIR"
fi
echo "==> Installing OpenTAD (editable)"
pip install -e "$OPENTAD_DIR" || echo "!! OpenTAD editable install reported issues — check mmcv/torch compatibility."

# --- 3. install our custom files into OpenTAD --------------------------------
echo "==> Installing custom configs + DSTA backbone + losses into OpenTAD"
mkdir -p "$OPENTAD_DIR/configs/adatad/mma52"
# AdaTAD configs (rename to the OpenTAD-style filenames the job scripts expect)
cp "$PROJECT_DIR/configs/adatad/large1.py" "$OPENTAD_DIR/configs/adatad/mma52/e2e_mma52_videomae_l_adapter.py"
cp "$PROJECT_DIR/configs/adatad/large2.py" "$OPENTAD_DIR/configs/adatad/mma52/e2e_mma52_videomae_l_adapter2.py"
cp "$PROJECT_DIR/configs/adatad/dsta.py"   "$OPENTAD_DIR/configs/adatad/mma52/e2e_mma52_videomae_l_dsta.py"
cp "$PROJECT_DIR/configs/adatad/asl.py"    "$OPENTAD_DIR/configs/adatad/mma52/e2e_mma52_videomae_l_adapter_asl.py"
# base dataset config
mkdir -p "$OPENTAD_DIR/configs/_base_/datasets/mma52"
cp "$PROJECT_DIR/configs/base/mma52_dataset.py" "$OPENTAD_DIR/configs/_base_/datasets/mma52/e2e_train_trunc_test_sw.py"
# novel DSTA backbone (registers VisionTransformerDSTA)
cp "$PROJECT_DIR/src/models/dsta_adapter.py" "$OPENTAD_DIR/opentad/models/backbones/videomae_dsta.py"
echo "   NOTE: register VisionTransformerDSTA and AsymmetricLoss in OpenTAD's"
echo "         backbones/__init__.py and losses/__init__.py (see docs/architecture.md)."

# --- 4. directory skeleton ---------------------------------------------------
echo "==> Creating data / work_dirs / predictions skeleton"
mkdir -p "$PROJECT_DIR"/data/MMA-52/{videos/{train,val,test},Annotations/adatad,extracted/{train,val,test}}
mkdir -p "$PROJECT_DIR"/{work_dirs,predictions,pretrained,logs}

# --- 5. smoke test -----------------------------------------------------------
echo "==> Environment smoke test"
python "$PROJECT_DIR/tools/check_env.py" || true

cat <<'EOF'

==> Setup complete.
Next steps:
  1. Download MMA-52 into data/MMA-52/ (see README "Data" section).
  2. Download VideoMAE-Large weights into pretrained/videomae_large.pth.
  3. Generate annotations:  python src/data/prepare_annotations.py
                            python src/data/prepare_augmentation.py
  4. Train:                 sbatch jobs/train_large2.sh
EOF
