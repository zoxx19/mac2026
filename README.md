# MA-52 Multi-Track Action Recognition

A multi-track action recognition pipeline for the MA-52 micro-action dataset, developed for the **Micro-Action Analysis Grand Challenge (MAC 2026)**.

## Overview

MA-52 contains 22,422 videos across 52 fine-grained micro-action classes from 7 body-part categories, collected from 205 subjects in professional psychological interviews. This pipeline builds separate recognition models per body-part track and fuses their outputs for final classification.

**Competition metric:**
```
F1-mean = (F1-macro + F1-micro) / 2
```

---

## Dataset

| Split | Videos | Classes |
|---|---|---|
| Train | 11,250 | 52 fine / 7 coarse |
| Val | 5,586 | 52 fine / 7 coarse |
| Test | 5,586 | hidden |

Videos: 900×1080, 30fps, ~2.8s each.

### Fine → Coarse mapping
| Coarse | Body part | Fine labels |
|---|---|---|
| 0 | Body (A) | A1–A5 (fine 0–4) |
| 1 | Head (B) | B1–B6 (fine 5–10) |
| 2 | Upper limb (C) | C1–C13 (fine 11–23) |
| 3 | Lower limb (D) | D1–D8 (fine 24–31) |
| 4 | Body-hand (E) | E1–E6 (fine 32–37) |
| 5 | Head-hand (F) | F1–F10 (fine 38–47) |
| 6 | Leg-hand (G) | G1–G4 (fine 48–51) |

---

## Tracks

| Track | Input | Model | Classes | Best F1-mean |
|---|---|---|---|---|
| Head (B) | RGB head crop 224×224 | VideoMAE-base SSv2 | B1–B6 + B7 | ✅ **0.7769** |
| Hand (C/E/F/G) | Skeleton keypoints | MMN ensemble J+B | C, E, F, G + no-hand (5) | ✅ **0.7533** |
| Leg (D) | Skeleton keypoints | MMN joint aug | D1–D8 + D9 no-leg (9) | ✅ **0.5733** |
| Body (A) | Skeleton keypoints | MMN joint aug | A1–A5 + A6 no-body (6) | ✅ **0.6234** |

---

## Results

### Track 1 — Head Recognition (B1–B7)

| Run | Model | Notes | F1-mean |
|---|---|---|---|
| **Run 1 ✅ Best** | VideoMAE-base SSv2 | Baseline | **0.7769** |
| Run 2A | VideoMAE-base SSv2 | + Weighted loss | 0.7570 |
| Run 2B | VideoMAE-base Kinetics | + Weighted loss | 0.7676 |
| Run 3A | VideoMAE-base SSv2 | + Dropout + augmentation | 0.6408 |
| Run 3B | TimeSformer-base K400 | + Dropout + augmentation | 0.5167 |
| Run 4A | VideoMAE-base Kinetics | Same as Run 1 | 0.6893 |
| Run 4C | VideoMAE-base Kinetics | + Dynamic crop | 0.7098 |

**Best model:** `outputs/head_best_run1/best_model.pt`

#### Per-class performance (val set)
| Class | F1 | Support |
|---|---|---|
| B1 nodding | 0.66 | 168 |
| B2 shaking head | 0.59 | 121 |
| B3 turning head | 0.50 | 228 |
| B4 tilting head | 0.69 | 480 |
| B5 bowing head | 0.84 | 641 |
| B6 head up | 0.88 | 595 |
| B7 no movement | 0.89 | 3,353 |

Val Accuracy: 83.35% | F1-macro: 0.7202 | F1-micro: 0.8335 | **F1-mean: 0.7769**

---

### Track 2 — Hand Recognition (C/E/F/G)

All runs use MMN model, 5 classes, aug=STCA, lr=1e-4, batch=32, 80 epochs.

| Run | Joints | Modality | Aug | F1-mean |
|---|---|---|---|---|
| v1 full | 17 | joint | ❌ | 0.7238 |
| v1 arm | 6 | joint | ❌ | 0.7145 |
| v2 full | 17 | joint | ✅ | 0.6969 |
| v2 arm | 6 | joint | ✅ | 0.7084 |
| bone full | 17 | bone | ✅ | **0.7337** |
| bone arm | 6 | bone | ✅ | 0.7038 |
| bone full 120ep | 17 | bone | ✅ | 0.7221 |
| joint full 120ep | 17 | joint | ✅ | 0.7197 |
| **Ensemble J+B** | 17 | joint+bone | — | **0.7533** |

**Key findings:**
- Bone modality beats joint (+0.01)
- Ensemble J+B gives free +0.02 boost
- 120 epochs hurts — model overfits after 80
- Best single model: `outputs/hand_full_bone_aug/best_model.pt`
- Best ensemble: joint(0.65) + bone(0.35)

---

### Track 3 — Leg Recognition (D1–D8)

All runs use MMN model, 9 classes, lr=1e-4, batch=32, 80 epochs.

| Run | Joints | Modality | Aug | F1-mean |
|---|---|---|---|---|
| leg joints | 6 | joint | ✅ | **0.5733** |
| leg joints | 6 | joint | ❌ | 0.5138 |
| full body | 17 | joint | ✅ | 0.5076 |
| full body | 17 | joint | ❌ | 0.4092 |
| leg joints | 6 | bone | ✅ | 0.5019 |
| full body | 17 | bone | ✅ | 0.5261 |

**Key findings:**
- Leg-only joints (hips+knees+ankles) beats full body (+0.07)
- Augmentation always helps
- Bone modality hurts for leg (unlike hand)
- Skeleton ceiling ~0.57 — RGB model needed to push further
- Best model: `outputs/leg_leg_joint/best_model.pt`

---

### Track 4 — Body Recognition (A1–A5)

All runs use MMN model, 6 classes, lr=1e-4, batch=32, 80 epochs.

| Run | Joints | Modality | Aug | F1-mean |
|---|---|---|---|---|
| full 17 | 17 | joint | ✅ | **0.6234** |
| full 17 | 17 | joint | ❌ | 0.5722 |
| torso 4 | 4 | joint | ✅ | 0.6013 |
| torso 4 | 4 | joint | ❌ | 0.5557 |
| full 17 | 17 | bone | ✅ | 0.5544 |
| torso 4 | 4 | bone | ✅ | 0.5163 |

**Key findings:**
- Full 17 joints beats torso-only
- Augmentation always helps (+0.05)
- Bone modality hurts for body (unlike hand)
- A3 (20 samples) and A4 (9 samples) always F1=0.0 — data scarcity
- Skeleton ceiling ~0.62 — RGB model needed to push further
- Best model: `outputs/body_full_joint/best_model.pt`

---

## Architecture

### Head track — VideoMAE
- Pretrained: VideoMAE-base fine-tuned on Something-Something-v2
- Input: 224×224 head+neck crop, 16 frames uniformly sampled
- Head: linear classifier → 7 classes
- Training: AdamW, cosine LR, weighted cross-entropy

### Skeleton tracks — MMN (Motion-guided Modulation Network)
- Paper: Gu et al., ACM MM 2025
- Input: 17-joint COCO keypoints from YOLOv8x-pose
- Two modalities:
  - **Joint (J)**: x,y coordinates → shape `(2, T, V, 1)`
  - **Bone (B)**: dx,dy,distance between connected joints → shape `(3, T, E, 1)`
- Ensemble: average logits of J and B models (2S strategy from paper)
- Training: AdamW lr=1e-4, weight_decay=0.1, batch=32, 80 epochs
- LR schedule: 20-epoch linear warmup + cosine annealing (3 cycles)
- Augmentation: STCA (skeletal rotation ±15°, scale 0.9-1.1, translate ±0.1, temporal jitter ±3)

### Joint subsets per track
| Mode | Joints | Count |
|---|---|---|
| full | all COCO joints | 17 |
| arm | shoulders+elbows+wrists (5,6,7,8,9,10) | 6 |
| leg | hips+knees+ankles (11,12,13,14,15,16) | 6 |
| torso | shoulders+hips (5,6,11,12) | 4 |

---

## Key Findings Summary

| Finding | Details |
|---|---|
| Augmentation always helps | STCA gives +0.05 avg across all tracks |
| Focused joints > full body for leg | 6 leg joints >> 17 joints (+0.07) |
| Full body > focused for body | 17 joints > 4 torso joints |
| Bone modality only helps hand | Hurts leg and body tracks |
| Ensemble J+B free boost | +0.02 for hand without retraining |
| 80 epochs optimal | 120 epochs causes overfitting |
| Skeleton ceiling exists | Leg ~0.57, Body ~0.62 — need RGB model |

---

## Pipeline

```
Raw RGB videos (MA-52)
        │
        ▼
YOLOv8x-pose (per frame)
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
Skeleton keypoints (.json)        Bounding boxes
        │                          (head, hands, feet)
        │                                 │
        ▼                                 ▼
MMN model (3 tracks)            Head crop 224×224
  ├── Hand → C/E/F/G/no-hand           │
  ├── Leg  → D1-D8/no-leg             ▼
  └── Body → A1-A5/no-body    VideoMAE fine-tuning
        │                              │
        │                              ▼
        │                       B1–B7 prediction
        │                              │
        └──────────── Label Fusion ────┘
                            │
                            ▼
              Final 52-class prediction
```

---

## Project Structure

```
ma52/
├── data/                          # gitignored
│   ├── annotations/
│   ├── videos/
│   ├── keypoints/
│   │   ├── train/                 # 11,250 JSON files
│   │   └── val/                   # 5,586 JSON files
│   ├── head_crops/
│   ├── head_crops_dynamic/
│   ├── head_dataset/
│   └── skeleton_dataset/
│       ├── hand_train.csv / hand_val.csv
│       ├── leg_train.csv  / leg_val.csv
│       └── body_train.csv / body_val.csv
├── src/
│   ├── extraction/
│   │   └── extract_pose.py
│   ├── head/
│   │   ├── prepare_dataset.py
│   │   ├── crop_clips.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── animate_wrong.py
│   └── skeleton/
│       ├── prepare_dataset.py       # Hand 5-class
│       ├── prepare_leg_dataset.py   # Leg 9-class
│       ├── prepare_body_dataset.py  # Body 6-class
│       ├── features.py              # Joint modality loader
│       ├── features_modality.py     # Joint + Bone modality loader
│       ├── train_mmn.py             # Hand v1
│       ├── train_mmn_v2.py          # Hand v2 (paper hyperparams)
│       ├── train_mmn_generic.py     # Generic v1
│       ├── train_mmn_generic_v2.py  # Generic v2 (bone + aug flag)
│       ├── evaluate_skeleton.py     # Ensemble + threshold tuning
│       └── MMN/                     # MMN repo
├── jobs/                          # SLURM batch scripts
├── models/                        # gitignored
│   ├── videomae-ssv2/
│   ├── videomae-kinetics/
│   ├── timesformer/
│   └── yolov8x-pose.pt
├── outputs/                       # gitignored except evaluation
│   ├── head_best_run1/
│   ├── hand_full_bone_aug/        # Best hand single model
│   ├── hand_skeleton_full/        # Best hand joint model
│   ├── leg_leg_joint/             # Best leg model
│   └── body_full_joint/           # Best body model
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Setup

### Requirements
```bash
conda create -n ma52 python=3.12
conda activate ma52
pip install torch torchvision
pip install ultralytics opencv-python-headless
pip install transformers accelerate
pip install timm
pip install scikit-learn pandas numpy tqdm
pip install matplotlib seaborn
pip install decord av
pip install huggingface_hub
```

### Pose extraction
```bash
sbatch.tinygpu jobs/job_extract_pose.sh
```

### Head track
```bash
python src/head/prepare_dataset.py
sbatch.tinygpu jobs/job_train_head.sh
sbatch.tinygpu jobs/job_evaluate_head.sh
```

### Skeleton tracks
```bash
python src/skeleton/prepare_dataset.py
python src/skeleton/prepare_leg_dataset.py
python src/skeleton/prepare_body_dataset.py

# Hand
sbatch.tinygpu jobs/job_hand_skeleton_full.sh
sbatch.tinygpu jobs/job_hand_skeleton_full_v2.sh

# Leg
sbatch.tinygpu jobs/job_leg_full.sh
sbatch.tinygpu jobs/job_leg_leg_noaug.sh

# Body
sbatch.tinygpu jobs/job_body_full.sh
sbatch.tinygpu jobs/job_body_full_noaug.sh
```

### Ensemble evaluation
```bash
PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python

$PYTHON src/skeleton/evaluate_skeleton.py \
    --track hand --mode full --num_classes 5 \
    --ensemble \
    --joint_model outputs/hand_skeleton_full/best_model.pt \
    --bone_model  outputs/hand_full_bone_aug/best_model.pt \
    --tune_threshold
```

---

## References

- **MA-52**: Guo et al., *Benchmarking Micro-action Recognition*, IEEE TCSVT 2024
- **MAC 2026**: ACM MM 2026 baseline F1-mean 65.64 (Video Swin Transformer)
- **MMN**: Gu et al., *Motion Matters: Motion-guided Modulation Network*, ACM MM 2025
- **VideoMAE**: Wang et al., *VideoMAE V2*, CVPR 2023
- **YOLOv8-pose**: Ultralytics, 2023
- **BlockGCN**: Zhou et al., *Redefine Topology Awareness*, CVPR 2024 (planned)