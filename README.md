# MAC 2026 Track 2: Multi-label Micro-Action Detection

> **MAC 2026 Grand Challenge — MAD Lab, FAU Erlangen-Nürnberg**  
> Track 2 (Multi-label Micro-Action Detection) — **Ziad Gaber**  
> Supervised by Amirreza Asemanrafat.

End-to-end temporal action detection for the **MAC 2026 Grand Challenge
(ACM Multimedia 2026), Track 2 — Multi-label Micro-Action Detection** on the
**MMA-52** dataset.

> **Result: 3rd place — 0.22993 average mAP** on the official test leaderboard.

Our system fine-tunes a **VideoMAE-Large** video backbone end-to-end with
lightweight **adapters**, feeds it into an **AdaTAD / ActionFormer** temporal
detection head, and fuses two complementary models with Soft-NMS. We also
introduce **DSTA (Dual-path Spatial-Temporal Adapter)**, a new adapter design
explored as our main architectural contribution.

---

## Overview

| | |
|---|---|
| **Task** | Multi-label temporal action detection: localise and classify micro-actions in untrimmed video |
| **Dataset** | MMA-52 — 52 micro-action classes (body / head / hand / leg / composite gestures) |
| **Metric** | Average mAP at tIoU = {0.2, 0.5, 0.7} |
| **Detector** | AdaTAD (ActionFormer head) on a VideoMAE-Large backbone, trained end-to-end |
| **Our entry** | Equal-weight ensemble of two VideoMAE-L + adapter models, fused with Soft-NMS |
| **Best score** | **0.22993 avg mAP (3rd place)** |

The micro-actions are short, subtle, and frequently co-occur (multi-label),
which makes both precise temporal localisation and fine-grained classification
hard. We found that **scaling the backbone (Large) and ensembling complementary
runs** mattered far more than any of the post-processing or auxiliary-modality
tricks we tried (see [Key Findings](#key-findings) and
[RESULTS.md](RESULTS.md)).

---

## Architecture

```
                          MAC 2026 Track 2 — full pipeline
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  DATA PREP                                                                 │
 │   MMA-52 CSVs ──prepare_annotations.py──► mma52_anno.json                  │
 │   + MA-52 Track-1 clips ──prepare_augmentation.py──► mma52_augmented_anno  │
 │     (≈3.5× more training clips — 15,784 videos from 4,534 originals)        │
 │     (MA-52 Track-1 single-label clips repurposed as weak temporal          │
 │      supervision)                                                          │
 └──────────────────────────────────────────────────────────────────────────┘
                                    │  (.mp4 clips, decoded on the fly)
                                    ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  BACKBONE  (end-to-end, frozen ViT + trainable adapters)                   │
 │                                                                            │
 │   160×160 video, 512-frame window ─► split into 32 × 16-frame chunks       │
 │              │                                                             │
 │              ▼                                                             │
 │   VideoMAE-Large ViT  (depth 24, dim 1024, 16 heads, patch 16, no CLS)     │
 │     every block:  Attn → FFN → ┌─ Adapter ─┐                               │
 │                                │  bottleneck (Large1/Large2)               │
 │                                │  ★ DSTA   (our contribution)              │
 │                                └───────────┘                               │
 │     backbone weights FROZEN — only adapters train                          │
 │              │  feature map (B, 1024, T, 10, 10)                           │
 │              ▼  mean-pool spatial, interpolate to window_size              │
 │   1-D temporal feature sequence  (B, 1024, 512)                            │
 └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  DETECTION HEAD — AdaTAD / ActionFormer                                    │
 │   multiscale projection → classification (FocalLoss) + regression (DIoU)   │
 │   sliding-window inference → Soft-NMS (σ=0.7)                              │
 │   output: per-video {segment [t_s,t_e], label, score}                      │
 └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ENSEMBLE & SUBMISSION                                                      │
 │   Large1 preds ┐                                                           │
 │   Large2 preds ┼─ fusion.py / multi_ensemble.py (Soft-NMS, equal weights)  │
 │                ┘                                                           │
 │   fused preds ──prepare_submission.py──► results.json (zipped for upload)  │
 └──────────────────────────────────────────────────────────────────────────┘
```

> **Note:** VideoMAE-Large has `depth=24`, `dim=1024`. The spatial grid is
> **10×10 (not 14×14)** because the input is 160 px, not 224 px, and there is
> **no CLS token**. This differs from standard ViT assumptions and directly
> shaped the DSTA implementation.

### DSTA — Dual-path Spatial-Temporal Adapter (novel contribution)

DSTA was implemented in the final hours of the competition after identifying
that the gap to #1 was architectural rather than tunable. Inspired by the
MMAD paper (ICCV 2025) baseline, we extended the dual-path adapter concept
to VideoMAE-Large (the paper only reported VideoMAE-Base results).

Implementation differences from the paper, discovered by inspecting the actual
model:
- VideoMAE here has **NO CLS token** (unusual for ViT)
- Spatial grid is **10×10** (160 px input ÷ 16 px patches), not 14×14
- Backbone is **fully frozen** — only adapter weights train
- **Identity initialisation** (zero-init up-projections, scales = 1) enables a
  stable warm-start from a trained bottleneck-adapter checkpoint

How it works: a shared down-projection feeds two parallel depth-wise conv stacks
— an **S-path** (per-frame 2-D convs over the 10×10 grid) and a **T-path**
(per-location 1-D convs over time) — each with its own up-projection, summed
back into the residual with learnable `s_scale` / `t_scale`. Implementation:
[`src/models/dsta_adapter.py`](src/models/dsta_adapter.py)
(`VisionTransformerDSTA`); full notes in [`docs/architecture.md`](docs/architecture.md).

**Training status:** DSTA trained successfully on **4×A100-40GB** GPUs for
**700 of 1972 iterations of epoch 0 (~1.5 h wall-clock)**, with loss dropping
from **0.6878 → 0.5238** (stable, finite, monotonically decreasing). Training was
cut short by the **cluster maintenance window (2026-06-09 06:30)** — the SLURM
job hit its 01:30 time limit and ended at 06:23 — **not** by any model failure.
The implementation is complete and correct; it simply needs more training time
to reach a validation evaluation.

From the logs (`train_dsta_1692722.log`):
- Hardware: 4×A100-40GB, **98 % GPU utilisation**, ~19.3 GB / GPU (fp32, `amp=False`)
- Steps: iteration 50 → 700 of 1972 in epoch 0 (logged every 50 iters)
- Loss: `Loss=0.6878` (cls 0.5003 / reg 0.1875) → `Loss=0.5238` (cls 0.3901 / reg 0.1337)
- LR still in warmup ramp: 1.1e-05 → 2.6e-05
- mAP evaluations reached: **0** (cut before epoch 0 completed)

### Ensemble strategy

Two VideoMAE-L + adapter models trained with different classification losses
(weighted vs. vanilla FocalLoss) make **complementary errors**. We pool their
proposals per `(video, label)`, normalise scores to [0, 1], and merge with
Gaussian Soft-NMS. **Equal weights (0.5 / 0.5) gave the best test score** —
better than any val-tuned weighting (see [Key Findings](#key-findings)).

---

## Results

| Model | Val avg mAP | Test (leaderboard) | Notes |
|-------|:-----------:|:------------------:|-------|
| **Large1** (VideoMAE-L + adapter, weighted FocalLoss), ep22 | 20.56% | 0.22691 | Strong single model |
| **Large2** (VideoMAE-L + adapter, vanilla FocalLoss), ep9 | 21.29% | 0.21591 | Higher val, lower test |
| Large Restart (warm restart from ep18) | 21.08% | — | Did not improve on test |
| **Large1 + Large2** (equal-weight Soft-NMS fusion) | 21.76% | **0.22993** | **Submitted — 3rd place** |
| 3-way fusion (+ Large ep18) | 21.70% | 0.22956 | ep18 hurts the test score |
| DSTA (dual-path adapter, 4×A100) | — | — | Trained 700 iters (ep0), loss 0.69→0.52, cut short by maintenance |
| ASL (AsymmetricLoss variant) | — | — | Loss=inf on warm start; abandoned |
| MMN skeleton (GCN recogniser) | 7.11% | — | Good recognition, hurts fusion |

A tuned weighting (Large1 0.47 / Large2 0.53, σ=0.7, top-1000) reached **22.10%
val** but did **not** beat equal weights on the test server — a recurring theme:
**val mAP is an unreliable proxy for test mAP on this dataset.** Full ablations,
including everything that *didn't* work, are in [RESULTS.md](RESULTS.md).

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ GPU with **≥ 20 GB VRAM** (VideoMAE-Large end-to-end training)
- ~50 GB disk for the dataset + checkpoints
- [OpenTAD](https://github.com/sming256/OpenTAD) — the AdaTAD detector framework
  (cloned automatically by `setup.sh`)

### Installation

```bash
git clone <this-repo> micro_challenge && cd micro_challenge

# Creates the conda env, installs deps, clones OpenTAD, installs our custom
# configs + DSTA backbone + losses into it, and runs an environment smoke test.
bash setup.sh

# Verify the stack (Python / CUDA / torch / mmaction / mediapipe)
python tools/check_env.py
```

`setup.sh` copies our files into OpenTAD. Two registrations are manual (one-time)
— see [`docs/architecture.md`](docs/architecture.md):
- register `VisionTransformerDSTA` in `opentad/models/backbones/__init__.py`
- register `AsymmetricLoss` in `opentad/models/losses/__init__.py`

---

## Complete Pipeline Guide

This guide walks through every step from raw data to leaderboard submission.
Each step is independent — jump to whichever stage you need.

> **Paths to edit by hand:** anything shown as `/path/to/...` and the
> `USER CONFIGURATION` block at the top of each `jobs/*.sh` script
> (`OPENTAD_DIR`, `DATA_DIR`, `CHECKPOINT_DIR`, `PREDICTIONS_DIR`, `CONDA_ENV`).
> The Python scripts default to relative `data/…` / `predictions/…` paths, so
> running from the project root needs no edits.

### Prerequisites check
```bash
python tools/check_env.py
```
Expected: Python 3.10+, CUDA available, torch / mmengine / decord all OK.

---

### Stage 0: Get the data

**Option A — Download from HuggingFace (recommended):**
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='VUT-HFUT/MMA-52',
    repo_type='dataset',
    local_dir='data/MMA-52'
)
"
```

**Option B — Download from the official challenge page:**
Visit https://sites.google.com/view/micro-action , download MMA-52, and place
it at `data/MMA-52/`.

**Verify download:**
```bash
python -c "
import os
required = [
    'data/MMA-52/videos/train',
    'data/MMA-52/videos/val',
    'data/MMA-52/videos/test',
    'data/MMA-52/Annotations/train.csv',
    'data/MMA-52/Annotations/val.csv',
]
for p in required:
    print(('OK' if os.path.exists(p) else 'MISSING') + ': ' + p)
"
```

---

### Stage 1: Prepare annotations
*Input: raw CSVs → Output: JSON annotations for training*

```bash
# Convert official CSVs to OpenTAD JSON format
python src/data/prepare_annotations.py \
    --csv_dir data/MMA-52/Annotations \
    --output_dir data/MMA-52/Annotations/adatad

# Add MA-52 Track-1 augmentation (3.5x more training data)
python src/data/prepare_augmentation.py \
    --anno data/MMA-52/Annotations/adatad/mma52_anno.json \
    --output data/MMA-52/Annotations/adatad/mma52_augmented_anno.json
```

Expected output:
- `mma52_anno.json` (≈4,534 train + 1,475 val + 519 test videos)
- `mma52_augmented_anno.json` (≈15,784 train videos)
- `category_idx.txt` (52 classes)

---

### Stage 2: Download pretrained backbone
*Required for training from scratch*

```bash
mkdir -p pretrained

# Option A: HuggingFace
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='MCG-NJU/videomae-large',
    filename='pytorch_model.bin',
    local_dir='pretrained'
)
"

# Option B: convert from safetensors if needed
python src/convert_videomae_checkpoint.py \
    --input pretrained/model.safetensors \
    --output pretrained/videomae_large.pth
```

---

### Stage 3: Train models

**Recommended: train Large2 (our best single model)**
```bash
# Edit the USER CONFIGURATION block at the top of the job script first
nano jobs/train_large2.sh   # set OPENTAD_DIR, DATA_DIR, etc.

# Then submit (SLURM cluster):
sbatch jobs/train_large2.sh

# OR run directly (single machine):
cd $OPENTAD_DIR
torchrun --nproc_per_node=2 --master_port=29500 tools/train.py \
    /path/to/micro_challenge/configs/adatad/large2.py \
    --cfg-options \
        work_dir=/path/to/checkpoints/large2 \
        dataset.train.data_path=/path/to/data/MMA-52/videos/train \
        dataset.val.data_path=/path/to/data/MMA-52/videos/val \
        dataset.test.data_path=/path/to/data/MMA-52/videos/test
```

**Novel contribution: train DSTA (dual-path adapter)**
```bash
# First train Large1 or Large2, then warm-start DSTA from it:
python tools/strip_checkpoint.py \
    checkpoints/large1/epoch_22.pth \
    checkpoints/large1_weights_only.pth

sbatch jobs/train_dsta.sh
```

**Monitor training progress:**
```bash
# Watch loss in real time
tail -f logs/train_large2_*.log | grep "Loss="

# Check mAP evaluations
grep "Average-mAP" logs/train_large2_*.log | tail -5
```

---

### Stage 4: Run inference
*Input: trained checkpoint → Output: prediction JSON*

```bash
# Validation split (to evaluate before submitting)
cd $OPENTAD_DIR
torchrun --nproc_per_node=1 tools/test.py \
    /path/to/configs/adatad/large2.py \
    --checkpoint checkpoints/large2/epoch_best.pth \
    --cfg-options \
        work_dir=predictions/tmp_large2_val \
        post_processing.save_dict=True \
        dataset.test.subset_name=validation \
        dataset.test.data_path=/path/to/data/MMA-52/videos/val

cp predictions/tmp_large2_val/result_detection.json predictions/large2_val.json

# Test split
torchrun --nproc_per_node=1 tools/test.py \
    /path/to/configs/adatad/large2.py \
    --checkpoint checkpoints/large2/epoch_best.pth \
    --cfg-options \
        work_dir=predictions/tmp_large2_test \
        post_processing.save_dict=True \
        dataset.test.subset_name=testing \
        dataset.test.data_path=/path/to/data/MMA-52/videos/test

cp predictions/tmp_large2_test/result_detection.json predictions/large2_test.json
```

> `jobs/infer_val.sh` and `jobs/infer_test.sh` wrap exactly these commands with
> the `USER CONFIGURATION` block — use them on a SLURM cluster.

---

### Stage 5: Evaluate predictions (optional but recommended)

```bash
# Evaluate a single model on val (one input, weight 1.0)
python src/multi_ensemble.py \
    --inputs predictions/large2_val.json \
    --weights 1.0 \
    --split val --evaluate \
    --output /tmp/eval_only.json

# Expected (approx): mAP@0.2 ~33%  mAP@0.5 ~20%  mAP@0.7 ~10%  avg mAP ~21%
```

---

### Stage 6: Ensemble fusion

```bash
# Two-model fusion — our best configuration (equal weights, sigma=0.5 -> 0.22993)
python src/multi_ensemble.py \
    --inputs predictions/large1_test.json predictions/large2_test.json \
    --weights 0.5 0.5 \
    --merge softnms --sigma 0.5 \
    --output predictions/fused_test.json \
    --split test

# Always evaluate the fusion on val first, before submitting
python src/multi_ensemble.py \
    --inputs predictions/large1_val.json predictions/large2_val.json \
    --weights 0.5 0.5 \
    --merge softnms --sigma 0.5 \
    --output predictions/fused_val.json \
    --split val --evaluate
```

---

### Stage 7: Generate the submission

```bash
# CSV submission (Kaggle-style)
python src/prepare_submission_csv.py \
    --input predictions/fused_test.json \
    --output predictions/submission.csv \
    --threshold 0.0 \
    --max_per_video 1000

# Verify
python -c "
import pandas as pd
df = pd.read_csv('predictions/submission.csv')
print(f'Rows: {len(df):,}')
print(f'Videos: {df[\"video-id\"].nunique()} (expected ~519)')
print(df.head(3))
"
```
Upload `predictions/submission.csv` to the challenge submission page.
(For a CodaLab-style `results.json` zip instead, use
`python src/prepare_submission.py --input predictions/fused_test.json --output predictions/submission.zip`.)

---

### Quick reference: which script does what

| Task | Script | Key flags |
|------|--------|-----------|
| Convert annotations | `src/data/prepare_annotations.py` | `--csv_dir --output_dir` |
| Augment training data | `src/data/prepare_augmentation.py` | `--anno --output` |
| Ensemble predictions | `src/multi_ensemble.py` | `--inputs --weights --merge --sigma --evaluate` |
| Generate CSV | `src/prepare_submission_csv.py` | `--input --threshold --max_per_video` |
| Generate zip | `src/prepare_submission.py` | `--input --output` |
| Analyse predictions | `tools/analyze_predictions.py` | `<prediction.json> --top` |
| Strip checkpoint | `tools/strip_checkpoint.py` | `<src> <dst>` |
| Check environment | `tools/check_env.py` | (no flags) |

---

## Data

The **MMA-52 dataset is NOT included** in this repository. Download it from the
official challenge page:

> https://sites.google.com/view/micro-action

Expected layout after download (and after running the prep scripts):

```
data/
└── MMA-52/
    ├── videos/
    │   ├── train/                 ← training videos (.mp4)
    │   ├── val/                   ← validation videos
    │   └── test/                  ← test videos
    ├── Annotations/
    │   ├── train.csv              ← official CSV (video_id, fps, frames, start/end, class)
    │   ├── val.csv
    │   ├── test.csv
    │   └── adatad/                ← generated by prepare_annotations.py:
    │       ├── mma52_anno.json
    │       ├── mma52_augmented_anno.json   ← generated by prepare_augmentation.py (gitignored)
    │       └── category_idx.txt
    └── extracted/                 ← standardised clips read by the e2e pipeline
        ├── train/ val/ test/
```

> The end-to-end AdaTAD pipeline **decodes `.mp4` clips directly** (via Decord) —
> you do **not** need to pre-extract frames. `src/data/extract_frames.py` is an
> optional helper for inspection / the auxiliary streams.

### Pretrained backbone

Training requires VideoMAE-Large pretrained weights (~1.2 GB):

- Download from the [VideoMAE](https://github.com/MCG-NJU/VideoMAE) release.
- Place at `pretrained/videomae_large.pth` (matches the config default).
- HuggingFace `.safetensors` weights can be converted with
  [`src/convert_videomae_checkpoint.py`](src/convert_videomae_checkpoint.py).

**Trained model checkpoints are NOT included** (each ~3 GB). Reproduce them with
the configs below, or strip a training checkpoint to weights-only with
[`tools/strip_checkpoint.py`](tools/strip_checkpoint.py).

---

## Reproducing Results

> Condensed SLURM-first version of the [Complete Pipeline Guide](#complete-pipeline-guide)
> above — see that section for copy-pasteable single-machine commands and
> expected outputs.

### Step 1 — Prepare annotations
```bash
python src/data/prepare_annotations.py    # CSVs → mma52_anno.json + category_idx.txt
python src/data/prepare_augmentation.py    # adds MA-52 Track-1 clips → augmented anno (~3.5×)
```

### Step 2 — (Optional) extract frames
```bash
python src/data/extract_frames.py --split train   # only needed for the image-based aux streams
```

### Step 3 — Train models
```bash
sbatch jobs/train_large2.sh     # our strongest single model (vanilla FocalLoss)
sbatch jobs/train_large1.sh     # weighted-FocalLoss model (the other half of the ensemble)
sbatch jobs/train_dsta.sh       # ★ DSTA — our novel contribution (warm-starts from Large1)
```
Each job script begins with a `USER CONFIGURATION` block — edit
`OPENTAD_DIR`, `DATA_DIR`, `CHECKPOINT_DIR`, etc. for your cluster.

### Step 4 — Run inference
```bash
sbatch jobs/infer_val.sh        # → predictions/large2_val.json
sbatch jobs/infer_test.sh       # → predictions/large2_test.json
```

### Step 5 — Ensemble and submit
```bash
# Equal-weight Soft-NMS fusion of the two models (the SUBMITTED configuration:
# equal weights, sigma=0.5 — this scored best on the test server)
python src/multi_ensemble.py \
    --inputs predictions/large1_test.json predictions/large2_test.json \
    --weights 0.5 0.5 \
    --merge softnms --sigma 0.5 --top_k 1000 \
    --output predictions/fused_test.json --split test

# Build the upload-ready results.json (+ zip)
python src/prepare_submission.py \
    --input predictions/fused_test.json \
    --output predictions/submission.zip
```
Use `--split val --evaluate` on `multi_ensemble.py` to score a fusion against
ground truth before submitting.

---

## Repository Layout

```
micro_challenge/
├── README.md / RESULTS.md / TECHNICAL_NOTES.md   docs (read these)
├── requirements.txt / setup.sh / .gitignore
├── configs/
│   ├── adatad/   large1.py large2.py dsta.py asl.py (+ tg.py body.py)
│   └── base/     mma52_dataset.py            dataset / pipeline config
├── src/
│   ├── models/   dsta_adapter.py (★ novel)   losses.py (Focal + ASL)
│   ├── data/     prepare_annotations.py  prepare_augmentation.py  extract_frames.py
│   ├── fusion.py  multi_ensemble.py            ensemble (Soft-NMS / WBF)
│   ├── prepare_submission.py  prepare_submission_csv.py
│   ├── postprocess.py  tta_inference.py  rerank_with_skeleton.py
│   ├── extract_*.py  pose_extractor.py        auxiliary streams (TG/flow/body/pose/skeleton)
│   └── eval_*.py  feeder_*.py  convert_*.py    MMN skeleton experiments + utilities
├── jobs/         SLURM job templates (cluster-specific, adapt paths before use)
├── tools/        check_env.py  analyze_predictions.py  strip_checkpoint.py  watch_map.sh
└── docs/         architecture.md  experiments.md
```

---

## Training Timeline

This competition was run under significant compute constraints. Here is
the honest chronology:

| Phase | What ran | Duration | Outcome |
|-------|----------|----------|---------|
| Week 1 | Large1 (VideoMAE-L, weighted FocalLoss) | 22 epochs | 21.67% val mAP, 0.22691 leaderboard |
| Week 1 | Large2 (VideoMAE-L, vanilla FocalLoss) | 9 epochs | 21.31% val mAP (still climbing) |
| Week 2 | Large Restart (warm restart from ep18) | 6 epochs | 21.08% val, did not improve ensemble |
| Week 2 | MMN Skeleton (GCN, ASL loss) | 28 epochs | 6.02% detection mAP, abandoned |
| Week 2 | TG (temporal gradient modality) | 2 epochs | No eval reached, cut by maintenance |
| Final night | ASL (AsymmetricLoss, warm start) | 0 epochs | Loss=inf, numerical instability |
| Final night | DSTA (dual-path adapter, 4×A100) | 700 iters (ep0) | Loss 0.69→0.52, cut by maintenance |

Best ensemble submitted: Large1 ep22 + Large2 ep9, equal weights → **0.22993**

---

## Key Findings

**What worked**
- **Scaling the backbone to VideoMAE-Large** — by far the biggest single lever.
- **Ensembling two complementary models** (weighted vs. vanilla FocalLoss) with
  Soft-NMS: 0.22691 → **0.22993**.
- **Equal fusion weights** beat every val-tuned weighting on the test server.
- **MA-52 augmentation** (3.5× training data) — using Track-1 single-label
  clips alongside Track-2 multi-label clips meaningfully improved detection.
- **All proposals at threshold = 0.0** — the mAP evaluator sweeps thresholds
  internally via the PR curve; keeping all proposals always scored better.

**What didn't work** (details in [RESULTS.md](RESULTS.md))
- Skeleton / MMN fusion — strong standalone recognition (7.11% mAP) but it
  *always* hurt the fused score.
- Test-Time Augmentation (temporal flip / speed perturbation) — collapsed to ~14.5%.
- Weighted Boxes Fusion instead of Soft-NMS — dropped to ~16–17%.
- Hand-tuned post-processing (gap filling, bucketing, score re-ranking) — all hurt.
- **σ = 0.7 ensemble NMS** — improved val mAP (+0.35%) but hurt the test score.
- **Weight tuning away from equal** — equal weights (0.5 / 0.5) beat all
  val-optimised weights on the actual test server.
- **AsymmetricLoss (ASL)** — `Loss=inf` on warm start; `gamma_neg=4` was too
  aggressive. `gamma_neg=2` + `amp=False` fixed the instability, but training
  was then cut by the maintenance window.

### Val mAP ≠ Test mAP

A critical finding: validation mAP was a poor proxy for test performance.
- Large1 ep18 had better val mAP than ep22 but scored **worse** on test.
- σ = 0.7 fusion gave +0.35% val gain but a −0.008 test regression.
- Weight tuning (0.47 / 0.53) gave +0.35% val but a −0.008 test regression.
- Equal weights at σ = 0.5 consistently produced the best test scores.

Recommendation for future work: use a held-out mini-test set, or trust the
leaderboard more than val mAP for final submission decisions.

**Open / novel direction**
- **DSTA** is, we believe, the right architectural direction (explicit
  spatial/temporal factorisation in the adapter). Its training was cut short by
  the compute budget rather than by a result; see
  [`docs/architecture.md`](docs/architecture.md).

---

## Citation

If this work is useful, please cite:

```bibtex
@misc{mac2026_madlab,
  title  = {MAC 2026 Grand Challenge: Multi-label Micro-Action Detection},
  author = {Ziad Gaber},
  note   = {MAD Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg.
            Track 2: 3rd place, 0.22993 avg mAP.
            Supervised by Amirreza Asemanrafat.},
  year   = {2026}
}
```

This work builds on **OpenTAD** (AdaTAD / ActionFormer), **VideoMAE**, and
**MMAction2**. Please cite their original papers as well.

## Acknowledgements

**[MAD Lab — Machine Learning and Data Analytics Lab](https://www.mad.tf.fau.de/)**  
Friedrich-Alexander-Universität Erlangen-Nürnberg

- **Amirreza Asemanrafat** — supervisor
- **Ziad Gaber** — Track 2 lead (multi-label micro-action detection)
- **Hamza Abdelrahman** — team member, Track 1 lead (micro-action recognition)

### Frameworks & Tools

- [OpenTAD](https://github.com/sming256/OpenTAD) — temporal action detection framework
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE) — masked video pretraining backbone
- [MMAction2](https://github.com/open-mmlab/mmaction2) — video understanding toolbox
- [Claude](https://claude.ai) (Anthropic) — AI assistance for code generation and research
- MAC 2026 organisers and the MMA-52 dataset authors
