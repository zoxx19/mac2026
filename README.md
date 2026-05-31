# MA-52 Multi-Track Micro-Action Recognition

A multi-track action recognition pipeline for the **MA-52** micro-action dataset, developed for the [Micro-Action Analysis Grand Challenge (MAC 2026)](https://sites.google.com/view/micro-action).

## Overview

MA-52 contains 22,422 videos across 52 fine-grained micro-action classes from 7 body-part categories, collected from 205 subjects in professional psychological interviews. This pipeline builds **separate recognition models per body-part track**.

**Competition metric:**
```
F1-mean = (F1-macro + F1-micro) / 2
```

---

## Dataset

| Split | Videos | Duration |
|---|---|---|
| Train | 11,250 | 6.19h |
| Val | 5,586 | 3.05h |
| Test | 1,138 | hidden |

Videos: 900×1080px, 30fps, ~2.8s average.

### Fine → Coarse Mapping
| Coarse | Body Part | Fine Labels | Classes |
|---|---|---|---|
| 0 | Body (A) | fine 0–4 | A1–A5 |
| 1 | Head (B) | fine 5–10 | B1–B6 |
| 2 | Upper limb (C) | fine 11–23 | C1–C13 |
| 3 | Lower limb (D) | fine 24–31 | D1–D8 |
| 4 | Body-hand (E) | fine 32–37 | E1–E6 |
| 5 | Head-hand (F) | fine 38–47 | F1–F10 |
| 6 | Leg-hand (G) | fine 48–51 | G1–G4 |

---

## Results Summary

| Track | Input | Model | Classes | F1-mean |
|---|---|---|---|---|
| Head (B) | RGB head crop 224×224 | VideoMAE-base SSv2 | B1–B7 | **0.7769** |
| Hand (C/E/F/G) | Skeleton keypoints | MMN ensemble (joint+bone) | 5 coarse | **0.7533** |
| Hand fine-grained | RGB upper body + Skeleton | VideoMAE + MMN fusion | 34 fine | **0.3066** |
| Body (A) | Skeleton keypoints | MMN joint aug | A1–A5 + no-body | **0.6154** |
| Leg (D) | Skeleton keypoints | MMN joint aug | D1–D8 + no-leg | **0.4778** |

---

## Track 1 — Head Recognition (B1–B7)

**Input:** 224×224 head crop, 16 frames uniformly sampled  
**Model:** VideoMAE-base pretrained on Something-Something-v2, fine-tuned  
**Best model:** `outputs/head_best_run1/best_model.pt`

| Run | Model | Notes | F1-mean |
|---|---|---|---|
| **Run 1 ✅ Best** | VideoMAE-base SSv2 | Baseline | **0.7769** |
| Run 2A | VideoMAE-base SSv2 | + Weighted loss | 0.7570 |
| Run 2B | VideoMAE-base Kinetics | + Weighted loss | 0.7676 |
| Run 3A | VideoMAE-base SSv2 | + Dropout + aug | 0.6408 |
| Run 3B | TimeSformer K400 | + Dropout + aug | 0.5167 |
| Run 4A | VideoMAE-base Kinetics | Same as Run 1 | 0.6893 |
| Run 4C | VideoMAE-base Kinetics | + Dynamic crop | 0.7098 |

**Per-class (val):**
| Class | Description | F1 |
|---|---|---|
| B1 | Nodding | 0.66 |
| B2 | Shaking head | 0.59 |
| B3 | Turning head | 0.50 |
| B4 | Tilting head | 0.69 |
| B5 | Bowing head | 0.84 |
| B6 | Head up | 0.88 |
| B7 | No movement | 0.89 |

---

## Track 2 — Hand Recognition (C/E/F/G)

**Input:** COCO 17-joint skeleton keypoints (x,y)  
**Model:** MMN (Motion-guided Modulation Network), 5 coarse classes  
**Best ensemble:** `outputs/hand_skeleton_full/` (joint) + `outputs/hand_full_bone_aug/` (bone)

| Run | Joints | Modality | Aug | F1-mean |
|---|---|---|---|---|
| v1 full | 17 | joint | ❌ | 0.7238 |
| v1 arm | 6 | joint | ❌ | 0.7145 |
| v2 full | 17 | joint | ✅ | 0.6969 |
| bone full | 17 | bone | ✅ | **0.7337** |
| bone arm | 6 | bone | ✅ | 0.7038 |
| **Ensemble J+B** | 17 | joint+bone | — | **0.7533** |

**Ensemble weights:** joint=0.65, bone=0.35  
**Per-class ensemble:** C=0.68, E=0.55, F=0.78, G=0.63, no-hand=0.89

### Track 2b — Fine-grained Hand Recognition (34 classes)

**Input:** Full-frame RGB (16 frames) + Skeleton keypoints  
**Model:** Late fusion — frozen VideoMAE-SSv2 + MMN skeleton branch  
**Classes:** C1–C13, E1–E6, F1–F10, G1–G4, no-hand  
**Best model:** `outputs/hand_fine_fusion/best_model.pt`

| Experiment | Input | F1-mean |
|---|---|---|
| Full frame | 900×1080 → 224×224 | **0.3066** |
| Upper body crop | nose-to-hips → 224×224 | ⏳ running |

---

## Track 3 — Leg Recognition (D1–D8)

**Input:** 6 leg joints (hips+knees+ankles), COCO indices [11,12,13,14,15,16]  
**Model:** MMN joint modality  
**Best model:** `outputs/leg_skeleton_leg/best_model.pt`

| Run | Joints | Modality | Aug | F1-mean |
|---|---|---|---|---|
| leg joints | 6 | joint | ✅ | **0.5733** |
| leg joints | 6 | joint | ❌ | 0.5138 |
| full body | 17 | joint | ✅ | 0.5076 |
| full body | 17 | bone | ✅ | 0.5261 |
| leg joints | 6 | bone | ✅ | 0.5019 |
| **1× cap retrain** | 6 | joint | ✅ | ⏳ running |

**Note:** Skeleton ceiling ~0.57. Val set heavily imbalanced (87% no-leg).

---

## Track 4 — Body Recognition (A1–A5)

**Input:** All 17 COCO joints  
**Model:** MMN joint modality  
**Best model:** `outputs/body_skeleton_full/best_model.pt`

| Run | Joints | Modality | Aug | F1-mean |
|---|---|---|---|---|
| full 17 | 17 | joint | ✅ | **0.6234** |
| full 17 | 17 | joint | ❌ | 0.5722 |
| torso 4 | 4 | joint | ✅ | 0.6013 |
| full 17 | 17 | bone | ✅ | 0.5544 |
| **1× cap retrain** | 17 | joint | ✅ | ⏳ running |

**Note:** A3 (20 samples) and A4 (9 samples) always F1=0.0 due to data scarcity.  
Val set heavily imbalanced (94% no-body).

---

## Architecture

### Head Track — VideoMAE
- Pretrained: VideoMAE-base fine-tuned on Something-Something-v2
- Input: 224×224 head crop, 16 frames
- Head: Linear classifier → 7 classes
- Training: AdamW, cosine LR schedule, weighted cross-entropy

### Skeleton Tracks — MMN
- Paper: [Gu et al., Motion Matters, ACM MM 2025](https://arxiv.org/abs/2507.21977)
- Pose extraction: YOLOv8x-pose (COCO 17 joints)
- Two modalities:
  - **Joint (J):** x,y coordinates → `(2, T, V, 1)`
  - **Bone (B):** dx,dy,length between connected joints → `(3, T, E, 1)`
- Ensemble: weighted average of J and B logits
- Training: AdamW lr=1e-4, weight_decay=0.1, batch=32, 80 epochs
- LR schedule: 20-epoch linear warmup + cosine annealing (3 cycles)
- Augmentation (STCA): rotation ±15°, scale 0.9–1.1, translate ±0.1, temporal jitter ±3

### Hand Fine-grained — Late Fusion
- RGB branch: VideoMAE-SSv2 (frozen) → mean-pool patches → Linear(768→512)
- Skeleton branch: MMN (17 joints) → replace head with Identity → Linear(96→256)
- Fusion: concat(512+256) → Linear(768→256) → Linear(256→34)
- Only 2.1% of parameters trainable (skeleton branch + fusion MLP)
- VRAM: ~1.2GB at batch=16

---

## Key Findings

| Finding | Detail |
|---|---|
| Augmentation always helps | STCA gives +0.05 avg across all tracks |
| Focused joints beat full body for leg | 6 leg joints > 17 joints (+0.07) |
| Full body beats focused for body | 17 joints > 4 torso joints |
| Bone modality only helps hand | Hurts leg and body tracks |
| Ensemble J+B free boost | +0.02 for hand without retraining |
| 80 epochs optimal | 120 epochs causes overfitting |
| Val imbalance limits skeleton models | 94% no-body, 87% no-leg in val set |
| VideoMAE dominates head track | RGB crop outperforms skeleton significantly |

---

## Project Structure

```
ma52/
├── src/
│   ├── extraction/
│   │   └── extract_pose.py          # YOLOv8x-pose keypoint extraction
│   ├── head/
│   │   ├── prepare_dataset.py       # Build head crop CSVs
│   │   ├── crop_clips.py            # Extract head crop videos
│   │   ├── train.py                 # VideoMAE fine-tuning
│   │   ├── evaluate.py              # Confusion matrix + wrong clips
│   │   └── animate_wrong.py        # Annotated wrong prediction videos
│   ├── skeleton/
│   │   ├── prepare_dataset.py       # Hand 5-class CSV
│   │   ├── prepare_leg_dataset.py   # Leg 9-class CSV
│   │   ├── prepare_body_dataset.py  # Body 6-class CSV
│   │   ├── features_modality.py     # Joint + Bone modality loader
│   │   ├── train_mmn_generic_v2.py  # Main MMN training script
│   │   ├── evaluate_skeleton.py     # Ensemble + threshold tuning
│   │   ├── evaluate_body_leg.py     # Confusion matrix for body/leg
│   │   ├── animate_wrong_body_leg.py # Skeleton overlay wrong videos
│   │   └── MMN/                     # MMN model (Gu et al. 2025)
│   └── hand_fine/
│       ├── prepare_hand_fine_dataset.py  # 34-class hand CSV
│       ├── hand_fine_dataset.py          # Dual-modal dataset loader
│       ├── hand_fine_model.py            # VideoMAE + MMN fusion model
│       ├── train_hand_fine.py            # Full frame training
│       ├── train_hand_fine_crop.py       # Upper body crop training
│       └── crop_upperbody.py             # Upper body crop extraction
├── jobs/                            # SLURM batch scripts
├── outputs/                         # Model checkpoints + evaluation
│   ├── head_best_run1/              # Head VideoMAE (F1=0.7769)
│   ├── hand_skeleton_full/          # Hand MMN joint (F1=0.7238)
│   ├── hand_full_bone_aug/          # Hand MMN bone (F1=0.7337)
│   ├── leg_skeleton_leg/            # Leg MMN (F1=0.5733)
│   ├── body_skeleton_full/          # Body MMN (F1=0.6234)
│   ├── hand_fine_fusion/            # Hand fine fusion (F1=0.3066)
│   ├── eval_body/                   # Body confusion matrix + wrong clips
│   └── eval_leg/                    # Leg confusion matrix + wrong clips
├── data/                            # gitignored — download separately
├── models/                          # gitignored — download separately
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
conda create -n ma52 python=3.12
conda activate ma52
pip install torch torchvision
pip install ultralytics opencv-python-headless
pip install transformers accelerate timm
pip install scikit-learn pandas numpy tqdm
pip install matplotlib seaborn
pip install decord av
```

## Reproduce Results

### 1. Pose Extraction
```bash
sbatch.tinygpu jobs/job_extract_pose.sh
```

### 2. Head Track
```bash
python src/head/prepare_dataset.py
sbatch.tinygpu jobs/job_crop_head.sh
sbatch.tinygpu jobs/job_train_head.sh
```

### 3. Hand Track (Skeleton)
```bash
python src/skeleton/prepare_dataset.py
sbatch.tinygpu jobs/job_hand_skeleton_full.sh   # joint
sbatch.tinygpu jobs/job_hand_full_bone.sh        # bone
```

### 4. Leg Track
```bash
python src/skeleton/prepare_leg_dataset.py
sbatch.tinygpu jobs/job_leg_leg.sh
```

### 5. Body Track
```bash
python src/skeleton/prepare_body_dataset.py
sbatch.tinygpu jobs/job_body_full.sh
```

### 6. Hand Ensemble Evaluation
```bash
PYTHON=/home/woody/iwso/iwso226h/conda/envs/ma52/bin/python
$PYTHON src/skeleton/evaluate_skeleton.py \
    --track hand --mode full --num_classes 5 \
    --ensemble \
    --joint_model outputs/hand_skeleton_full/best_model.pt \
    --bone_model  outputs/hand_full_bone_aug/best_model.pt \
    --tune_threshold
```

### 7. Fine-grained Hand (34 classes)
```bash
python src/hand_fine/prepare_hand_fine_dataset.py
sbatch.tinygpu jobs/job_hand_fine_fusion.sh       # full frame
sbatch.tinygpu jobs/job_hand_fine_upperbody.sh    # upper body crop
```

---

## References

- **MA-52 Dataset:** Guo et al., *Benchmarking Micro-action Recognition*, IEEE TCSVT 2024
- **MAC 2026:** ACM MM 2026 Grand Challenge
- **MMN:** Gu et al., *Motion Matters: Motion-guided Modulation Network for Skeleton-based Micro-action Recognition*, ACM MM 2025
- **VideoMAE:** Wang et al., *VideoMAE V2: Scaling Video Masked Autoencoders*, CVPR 2023
- **YOLOv8-pose:** Ultralytics, 2023