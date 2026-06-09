#!/bin/bash
# MAC 2026 Track 2 — Full pipeline runner
# Usage: bash run_pipeline.sh [stage]
# Stages: annotations, train, infer, ensemble, submit, all
#
# ===== CONFIGURE THESE =====
OPENTAD_DIR="${OPENTAD_DIR:-/path/to/OpenTAD}"
DATA_DIR="${DATA_DIR:-data/MMA-52}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-./predictions}"
CONDA_ENV="${CONDA_ENV:-opentad}"
# ===========================

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

STAGE=${1:-"help"}

run_annotations() {
    echo "=== Stage 1: Preparing annotations ==="
    python src/data/prepare_annotations.py \
        --csv_dir "$DATA_DIR/Annotations" \
        --output_dir "$DATA_DIR/Annotations/adatad"
    python src/data/prepare_augmentation.py \
        --anno "$DATA_DIR/Annotations/adatad/mma52_anno.json" \
        --output "$DATA_DIR/Annotations/adatad/mma52_augmented_anno.json"
    echo "Done."
}

run_ensemble() {
    echo "=== Stage 6: Ensemble fusion ==="
    mkdir -p "$PREDICTIONS_DIR"
    python src/multi_ensemble.py \
        --inputs "$PREDICTIONS_DIR/large1_test.json" \
                 "$PREDICTIONS_DIR/large2_test.json" \
        --weights 0.5 0.5 \
        --merge softnms --sigma 0.5 \
        --output "$PREDICTIONS_DIR/fused_test.json" \
        --split test
    echo "Fused predictions saved to $PREDICTIONS_DIR/fused_test.json"
}

run_submit() {
    echo "=== Stage 7: Generate submission CSV ==="
    python src/prepare_submission_csv.py \
        --input "$PREDICTIONS_DIR/fused_test.json" \
        --output "$PREDICTIONS_DIR/submission.csv" \
        --threshold 0.0 --max_per_video 1000
    echo "Submission ready: $PREDICTIONS_DIR/submission.csv"
    echo "Upload to: https://www.kaggle.com/competitions/the-3rd-micro-action-analysis-grand-challenge-track-2"
}

case $STAGE in
  annotations)
    run_annotations
    ;;

  train)
    echo "=== Stage 3: Training ==="
    echo "Training runs on a SLURM cluster — submit the job templates:"
    echo "  sbatch jobs/train_large2.sh    # best single model"
    echo "  sbatch jobs/train_large1.sh    # second ensemble member"
    echo "  sbatch jobs/train_dsta.sh      # novel DSTA adapter"
    echo "Edit each script's USER CONFIGURATION block first."
    ;;

  infer)
    echo "=== Stage 4: Running inference ==="
    echo "Submit the inference templates (edit checkpoint paths first):"
    echo "  sbatch jobs/infer_val.sh       # -> $PREDICTIONS_DIR/large2_val.json"
    echo "  sbatch jobs/infer_test.sh      # -> $PREDICTIONS_DIR/large2_test.json"
    ;;

  ensemble)
    run_ensemble
    ;;

  submit)
    run_submit
    ;;

  all)
    echo "=== Running full data->submission chain ==="
    echo "(training + inference run on the cluster; this chains the CPU stages)"
    run_annotations
    echo
    echo ">> Now train (sbatch jobs/train_*.sh) and infer (sbatch jobs/infer_*.sh)"
    echo ">> Once predictions/large{1,2}_test.json exist, re-run: bash run_pipeline.sh ensemble && bash run_pipeline.sh submit"
    ;;

  help|*)
    echo "Usage: bash run_pipeline.sh [stage]"
    echo "Stages:"
    echo "  annotations  — prepare annotation JSONs from CSVs"
    echo "  train        — show the SLURM training commands"
    echo "  infer        — show the SLURM inference commands"
    echo "  ensemble     — fuse predictions from multiple models"
    echo "  submit       — generate the submission CSV"
    echo "  all          — run the CPU stages and print next steps"
    ;;
esac
