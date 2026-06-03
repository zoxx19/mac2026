# MA-52 Multi-Track Micro-Action Recognition
## MAC 2026 Grand Challenge — Track 1: Micro-Action Recognition

Multi-track action recognition pipeline for the **MA-52** dataset, developed for the [Micro-Action Analysis Grand Challenge (MAC 2026)](https://sites.google.com/view/micro-action) held at ACM Multimedia 2026.



---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Competition Metric](#competition-metric)
4. [Current Best Results](#current-best-results)
5. [Head Track (B)](#head-track-b)
6. [Hand Track — Coarse (C/E/F/G)](#hand-track--coarse-cefg)
7. [Hand Track — Fine-grained (34 classes)](#hand-track--fine-grained-34-classes)
8. [Body Track (A)](#body-track-a)
9. [Leg Track (D)](#leg-track-d)
10. [Experiments](#experiments)
11. [Architecture Details](#architecture-details)
12. [Key Findings](#key-findings)
13. [Project Structure](#project-structure)
14. [Setup & Reproduce](#setup--reproduce)
15. [References](#references)

---

## Overview

MA-52 contains 22,422 short video clips (~2.8s, 900×1080, 30fps) annotated with 52 fine-grained micro-action categories across 7 coarse body-part groups. Each video has exactly one label.

Our approach builds **separate specialized models per track** rather than one monolithic 52-class model:

```
Video
  ├── Head model      → B1-B6 / B7-no-head     (7 classes)
  ├── Body model      → A1-A5 / no-body          (6 classes)
  ├── Leg model       → D1-D8 / no-leg           (9 classes)
  └── Hand two-stage pipeline
        Stage 1       → C / E / F / G / no-hand  (5 coarse classes)
        Stage 2-C     → C1-C13                   (13 fine classes)
        Stage 2-E     → E1-E6                    (6 fine classes)
        Stage 2-F     → F1-F10                   (10 fine classes)
        Stage 2-G     → G1-G4                    (4 fine classes)
            ↓
    Label Fusion → 1 of 52 fine-grained classes
            ↓
    Official F1_mean metric
```

---

## Dataset

| Split | Videos | Duration | Avg Length |
|---|---|---|---|
| Train | 11,250 | 6.19h | 1.98s |
| Val | 5,586 | 3.05h | 1.96s |
| Test | 1,138 | 0.71h | 2.23s (hidden) |

### Fine → Coarse Mapping
| Coarse ID | Body Part | Fine Labels | Classes | Train Count |
|---|---|---|---|---|
| 0 | Body (A) | fine 0–4 | A1–A5 | 644 active + 618 no-body |
| 1 | Head (B) | fine 5–10 | B1–B6 | 4,424 active + 2,566 no-head |
| 2 | Upper limb (C) | fine 11–23 | C1–C13 | 2,625 |
| 3 | Lower limb (D) | fine 24–31 | D1–D8 | 1,441 active + 796 no-leg |
| 4 | Body-hand (E) | fine 32–37 | E1–E6 | 317 |
| 5 | Head-hand (F) | fine 38–47 | F1–F10 | 719 |
| 6 | Leg-hand (G) | fine 48–51 | G1–G4 | 1,030 |

### Pose Extraction
- Model: YOLOv8x-pose (Ultralytics 2023)
- Format: COCO 17 keypoints per frame → `[x, y, confidence]`
- Output: one JSON per video at `data/keypoints/{split}/{video_id}.json`
- ~56 frames per video average

### COCO 17 Joint Mapping
```
0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow
9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip
13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
```

---

## Competition Metric

**Official F1_mean** (Track 1):
```
F1_mean = (F1_body_macro + F1_body_micro + F1_action_macro + F1_action_micro) / 4
```
- **body** = 7 coarse body-part categories (which of A/B/C/D/E/F/G)
- **action** = 52 fine-grained action categories (which specific action)
- All 4 track predictions must be fused into one 52-class prediction per video

**Top team at MAC 2025:** USTC-IAT-United achieved F1_mean = 77.75 using VideoMAE-v2 + InternVideo2 + LLM ensemble on A100 GPUs.

---

## Current Best Results

| Track | Classes | Best Model | F1_mean | Output |
|---|---|---|---|---|
| Head (B) | 7 | VideoMAE-base SSv2 | **0.7769** | outputs/head_best_run1 |
| Hand coarse | 5 | MMN joint+bone ensemble | **0.7533** | outputs/hand_skeleton_full + outputs/hand_full_bone_aug |
| Hand fine (34-class) | 34 | VideoMAE+MMN fusion, upper body crop | **0.3137** | outputs/hand_fine_upperbody |
| Body (A) | 6 | MMN full joint 1× cap | **0.6176** | outputs/body_full_joint_aug |
| Leg (D) | 9 | MMN leg joints 1× cap | **0.5802** | outputs/leg_leg_joint_aug |

---

## Head Track (B)

**Task:** Classify 7 head movement categories: B1-nodding, B2-shaking, B3-turning, B4-tilting, B5-bowing, B6-up, B7-no movement.

### Input Preparation
- Extract head+neck bounding box from COCO keypoints (nose, shoulders, hips)
- Box size derived from torso length L = |neck - mid_hip|
- Box width = L/1.4 centered on nose, with temporal smoothing (5-frame moving avg)
- Resize to 224×224, 16 frames sampled uniformly
- B7 (no head movement) capped at 2× largest B class = 2,566 samples

### Label Distribution (Train)
| Class | Count |
|---|---|
| B1 | 337 |
| B2 | 242 |
| B3 | 452 |
| B4 | 953 |
| B5 | 1,263 |
| B6 | 1,177 |
| B7 | 2,566 |

### All Experiments
| Run | Model | Notes | F1_mean |
|---|---|---|---|
| **Run 1 ✅ BEST** | VideoMAE-base SSv2 | 2-stage fine-tune, no extras | **0.7769** |
| Run 2A | VideoMAE-base SSv2 | + Weighted cross-entropy loss | 0.7570 |
| Run 2B | VideoMAE-base Kinetics | + Weighted loss | 0.7676 |
| Run 3A | VideoMAE-base SSv2 | + Dropout(0.3) + strong aug | 0.6408 |
| Run 3B | TimeSformer K400 | + Dropout + aug | 0.5167 |
| Run 4A | VideoMAE-base Kinetics | Same settings as Run 1 | 0.6893 |
| Run 4C | VideoMAE-base Kinetics | + Dynamic per-frame crop | 0.7098 |
| head_large | VideoMAE-large Kinetics | + MixUp + label smoothing | 🔄 running |

### Per-Class Results (Best Model)
| Class | Description | F1 | Val count |
|---|---|---|---|
| B1 | Nodding | 0.66 | 168 |
| B2 | Shaking head | 0.59 | 121 |
| B3 | Turning head | **0.50** (worst) | 228 |
| B4 | Tilting head | 0.69 | 480 |
| B5 | Bowing head | 0.84 | 641 |
| B6 | Head up | 0.88 | 595 |
| B7 | No movement | 0.89 | 3,353 |

Val Accuracy=83.35%, F1-macro=0.7202, F1-micro=0.8335

### Key Lessons
- SSv2 pretraining >> Kinetics for head movements (hand-object interactions in SSv2 transfer well)
- Simpler baseline wins — adding dropout/weighted loss/strong aug all hurt
- B3 (turning) is the hardest — visually similar to B2 (shaking) from fixed camera angle
- Dynamic crop (follows head frame-by-frame) does NOT help over static crop

---

## Hand Track — Coarse (C/E/F/G)

**Task:** Classify 5 coarse hand groups: C (upper limb), E (body-hand), F (head-hand), G (leg-hand), no-hand.

### Input Preparation
- COCO 17-joint skeleton keypoints (x,y coordinates normalized to [-1,1])
- Two modalities:
  - **Joint:** raw (x,y) per joint → shape `(2, 64, 17, 1)`
  - **Bone:** (dx,dy,length) between connected joints → shape `(3, 64, E, 1)`
- No-hand capped at 1× largest hand class = 497 samples (for Stage 1)
- Original coarse 5-class dataset: no-hand capped at 2× = 5,250 (MMN experiments)

### All Experiments
| Run | Joints | Modality | Aug | Epochs | LR | F1_mean |
|---|---|---|---|---|---|---|
| v1 full | 17 | joint | ❌ | 80 | 1e-3 | 0.7238 |
| v1 arm | 6 | joint | ❌ | 80 | 1e-3 | 0.7145 |
| v2 full | 17 | joint | ✅ | 80 | 1e-4 | 0.6969 |
| v2 arm | 6 | joint | ✅ | 80 | 1e-4 | 0.7084 |
| bone full | 17 | bone | ✅ | 80 | 1e-4 | **0.7337** |
| bone arm | 6 | bone | ✅ | 80 | 1e-4 | 0.7038 |
| bone full 120ep | 17 | bone | ✅ | 120 | 1e-4 | 0.7221 |
| joint full 120ep | 17 | joint | ✅ | 120 | 1e-4 | 0.7197 |
| **Ensemble J+B ✅ BEST** | 17 | joint+bone | — | — | — | **0.7533** |
| Stage1 VideoMAE-base SSv2 | — | RGB upper body | — | 30ep | — | 0.6568 |

**Ensemble details:** joint_weight=0.65, bone_weight=0.35  
**Per-class ensemble:** C=0.6839, E=0.5472, F=0.7831, G=0.6307, no-hand=0.8867

### Key Lessons
- Bone modality uniquely helps hand (unlike body/leg where it hurts)
- Ensemble J+B gives free +0.02 boost with no retraining
- 80 epochs optimal — 120 causes overfitting
- Arm-only joints (6) slightly worse than full body (17) for hand

---

## Hand Track — Fine-grained (34 classes)

**Task:** Classify 34 fine-grained hand action classes: C1-C13, E1-E6, F1-F10, G1-G4.

### Two-Stage Pipeline (Main Approach)

#### Stage 1 — Coarse Router (5 classes)
- **Model:** VideoMAE-base SSv2, fully fine-tuned (same as head track)
- **Input:** Upper body crop (top-of-head to hips, PAD_TOP=120px above nose, 224×224)
- **Labels:** 0=C, 1=E, 2=F, 3=G, 4=no-hand
- **No-hand cap:** 1× largest class = 497 samples
- **Training:** 2-stage (freeze backbone 10ep → full fine-tune 20ep), early stop patience=5
- **Best F1_mean:** 0.6568
- **Output:** `outputs/hand_stage1/best_model.pt`
- **CSV:** `data/hand_dataset/hand_stage1_train.csv`

#### Stage 2 — Group-specific Fine Classifier
Four separate VideoMAE models, one per group:

| Group | Classes | Train | Crop | Approach A F1 | Approach B F1 |
|---|---|---|---|---|---|
| C (upper limb) | 13 (C1-C13) | 2,625 | upper body | 0.4859 | 🔄 running |
| E (body-hand) | 6 (E1-E6) | 317 | upper body | 0.7457 | **0.8810** 🎉 |
| F (head-hand) | 10 (F1-F10) | 719 | head crop | 0.5105 | 🔄 running |
| G (leg-hand) | 4 (G1-G4) | 1,030 | upper body | 0.6881 | 🔄 running |

**Approach A:** Initialize Stage 2 from VideoMAE-SSv2 (independent of Stage 1)  
**Approach B:** Initialize Stage 2 backbone from Stage 1 weights (MA-52 adapted features)

Group E Approach B achieved **0.8810** — +0.135 gain over Approach A. This confirms that Stage 1 backbone transfer is highly effective, especially for small groups (E has only 317 samples).

#### Label Mapping
```
C group: local labels 0-12  → global labels 0-12  (fine labels 11-23)
E group: local labels 0-5   → global labels 13-18 (fine labels 32-37)
F group: local labels 0-9   → global labels 19-28 (fine labels 38-47)
G group: local labels 0-3   → global labels 29-32 (fine labels 48-51)
no-hand: global label 33
```

#### Inference
```
Video
  ↓ Stage 1 (VideoMAE or ensemble with MMN)
  → predict coarse group (C/E/F/G/no-hand)
  ↓ if not no-hand → Route to Stage 2 model for that group
  → predict local fine class (0-based within group)
  ↓ Convert to global 34-class label
  → Final: one of 34 fine-grained hand classes
```

### Flat 34-class Baseline (Old Approach)
| Experiment | Model | Input | F1_mean |
|---|---|---|---|
| Fusion full frame | Frozen VideoMAE + MMN late fusion | 900×1080 | 0.3066 |
| Fusion upper body | Frozen VideoMAE + MMN late fusion | nose-to-hips crop | 0.3137 |

The frozen VideoMAE approach only trains 2.1% of parameters — insufficient for 34-class fine-grained task.

### Training Details
- Stage 2 epochs: 10 (classifier only) + 30 (full fine-tune)
- Loss: weighted cross-entropy (handles class imbalance)
- LR: stage1=1e-3, stage2=1e-5
- Early stopping: patience=7
- Augmentation: flip, random crop, color jitter, temporal reverse

---

## Body Track (A)

**Task:** Classify 6 body movement categories: A1-leaning forward, A2-leaning backward, A3-leaning left, A4-leaning right, A5-swaying, no-body.

### Label Distribution (Train — severe imbalance)
| Class | Count | Notes |
|---|---|---|
| A1 | 309 | Main class |
| A2 | 150 | OK |
| A3 | 20 | Very rare — often F1=0.0 |
| A4 | 9 | Extremely rare — always F1=0.0 |
| A5 | 156 | OK |
| no-body | 618 (capped 2×) → 309 (1× cap) | Dominant |

### All Experiments
| Run | Model | Joints | Notes | F1_mean |
|---|---|---|---|---|
| MMN full joint no-aug | MMN | 17 | No augmentation | 0.5722 |
| MMN full joint aug | MMN | 17 | STCA augmentation | **0.6234** |
| MMN torso joint no-aug | MMN | 4 | Torso joints only | 0.5557 |
| MMN torso joint aug | MMN | 4 | + STCA | 0.6013 |
| MMN full bone aug | MMN | 17 | Bone modality | 0.5544 |
| MMN torso bone aug | MMN | 4 | Torso bone | 0.5163 |
| MMN 1× cap | MMN | 17 | No-body cap 1× | **0.6176** |
| VideoMAE-base body crop | VideoMAE SSv2 | — | Nose-to-ankles 224×224 | 0.5278 |
| body_large | VideoMAE-large K400 | — | + MixUp + label smooth | 🔄 running |
| body_oversample | VideoMAE-large K400 | — | A3×10, A4×22 + focal loss | 🔄 running |

### Oversampling Strategy
To address A3 (20 samples) and A4 (9 samples):
- A3: repeated 10× → 200 samples
- A4: repeated 22× → 198 samples
- Combined with heavy augmentation (rotation ±20°, speed perturbation, strong color jitter, temporal reverse)
- Focal loss (γ=2.0) instead of weighted cross-entropy — focuses gradient on hard/rare examples
- CSV: `data/skeleton_dataset/body_train_oversample.csv`

### VideoMAE Crop Details
- Top: nose (joint 0) - 40px
- Bottom: ankles (joints 15,16) + 60px
- Left/Right: shoulder width + 60px
- Resize to 224×224

### Key Lessons
- Skeleton ceiling ~0.62 — bone modality hurts body
- 17 joints > 4 torso joints (full body motion context needed)
- Data scarcity for A3/A4 is the fundamental bottleneck
- VideoMAE on full body crops (0.5278) weaker than MMN skeleton (0.6176)

---

## Leg Track (D)

**Task:** Classify 9 leg movement categories: D1-D8 (specific leg movements), no-leg.

### Label Distribution (Train)
| Class | Count | Notes |
|---|---|---|
| D1 | 238 | OK |
| D2 | 66 | Rare |
| D3 | 92 | OK |
| D4 | 97 | OK |
| D5 | 60 | Very rare |
| D6 | 231 | OK |
| D7 | 259 | OK |
| D8 | 398 | OK |
| no-leg | 796 (2× cap) → 398 (1× cap) | Dominant |

### All Experiments
| Run | Model | Joints | Notes | F1_mean |
|---|---|---|---|---|
| MMN full joint no-aug | MMN | 17 | No augmentation | 0.4092 |
| MMN full joint aug | MMN | 17 | STCA augmentation | 0.5076 |
| MMN leg joints no-aug | MMN | 6 | Leg joints only | 0.5138 |
| MMN leg joints aug | MMN | 6 | + STCA | **0.5733** |
| MMN full bone aug | MMN | 17 | Bone modality | 0.5261 |
| MMN leg joints bone | MMN | 6 | Leg joints bone | 0.5019 |
| MMN 1× cap | MMN | 6 | No-leg cap 1× | **0.5802** |
| VideoMAE-base leg crop | VideoMAE SSv2 | — | Hips-to-ankles 224×224 | 0.5776 |
| leg_large | VideoMAE-large K400 | — | + MixUp + label smooth | 🔄 running |
| leg_oversample | VideoMAE-large K400 | — | D2×3, D5×3 + focal loss | 🔄 running |
| leg_flow | VideoMAE-large K400 | — | Optical flow (Farneback) | 🔄 running |

### Oversampling Strategy
- D2: repeated 3× → 198 samples
- D3: repeated 2× → 184 samples
- D4: repeated 2× → 194 samples
- D5: repeated 3× → 180 samples
- CSV: `data/skeleton_dataset/leg_train_oversample.csv`

### VideoMAE Crop Details
- Top: hips (joints 11,12) - 60px
- Bottom: ankles (joints 15,16) + 60px
- Full width
- Resize to 224×224

### Optical Flow Details
- Algorithm: Farneback dense optical flow (OpenCV)
- Encoding: HSV — hue=direction, value=magnitude
- Saved as color MP4 at 224×224
- CSV: `data/skeleton_dataset/leg_train_flow.csv`

### Key Lessons
- Leg joints (6) >> full body (17) for leg classification (+0.07)
- Bone modality hurts leg track
- 1× no-leg cap significantly better than 2× (+0.10)
- VideoMAE leg crop ≈ MMN 1× cap (both ~0.58)
- Optical flow expected to capture subtle motion better than RGB

---

## Experiments

All experimental scripts are in `src/experiments/` to keep proven pipeline clean.

### Universal VideoMAE-large Training (`train_videomae_large.py`)
Supports all 4 tracks via `--track` argument. Key improvements over base:
- **VideoMAE-large** (307M params, 24 transformer layers vs 12 for base)
- **MixUp augmentation** (α=0.4) — blends two videos + labels, reduces overfitting
- **Label smoothing** (ε=0.1) — soft labels instead of hard 0/1
- **Cosine annealing with warm restarts** (T_0=7) — better than simple cosine
- **Gradient accumulation** (2 steps) — effective batch=16 without OOM
- **Lower Stage 2 LR** (5e-6 vs 1e-5) — prevents destroying pretrained features

### Running Experiments Summary
| Job | Description | Status |
|---|---|---|
| head_large | Head VideoMAE-large + MixUp | 🔄 running |
| body_large | Body VideoMAE-large + MixUp | 🔄 running |
| leg_large | Leg VideoMAE-large + MixUp | 🔄 running |
| body_oversample | Body oversample + focal loss | 🔄 running |
| leg_oversample | Leg oversample + focal loss | 🔄 running |
| leg_flow | Leg optical flow + VideoMAE-large | 🔄 running |
| hs2_C_B | Hand Stage 2 Group C Approach B | 🔄 running |
| hs2_F_B | Hand Stage 2 Group F Approach B | 🔄 running |
| hs2_G_B | Hand Stage 2 Group G Approach B | 🔄 running |

---

## Architecture Details

### VideoMAE Fine-tuning (Head/Body/Leg/Hand)
```
Input: 16 frames × 224×224 × 3 channels
  ↓ VideoMAE patch embedding (16×16 patches)
  ↓ 12 (base) or 24 (large) transformer encoder layers
  ↓ Mean pool all patch tokens
  ↓ Linear classifier → N classes
```
- **Stage 1 (10 epochs):** Freeze backbone, train classifier only (LR=1e-3)
- **Stage 2 (20-30 epochs):** Full fine-tune all parameters (LR=1e-5 or 5e-6)
- Loss: Weighted cross-entropy (weights = total / (n_classes × count))
- Optimizer: AdamW, weight_decay=0.01
- Augmentation: flip, random crop (0.85-1.0 scale), color jitter, temporal jitter, grayscale

### MMN (Motion-guided Modulation Network)
- Paper: Gu et al., Motion Matters, ACM MM 2025
- Input shape: `(B, C, T, V, M)` where C=2 (joint) or 3 (bone), T=64, V=joints, M=1
- embed_dim=96 (hardcoded — do NOT pass to constructor)
- Training: AdamW lr=1e-4, weight_decay=0.1, batch=32, 80 epochs
- Scheduler: 20-epoch linear warmup + cosine annealing 3 cycles
- Augmentation (STCA): rotation ±15°, scale 0.9-1.1, translate ±0.1, temporal jitter ±3
- Loss: Weighted cross-entropy

### Hand Two-Stage Architecture
```
Stage 1: VideoMAE-base-SSv2 (87M) → 5-class head
Stage 2: VideoMAE-base-SSv2 (87M) × 4 models (one per group)
  Approach A: Init from SSv2 pretrain
  Approach B: Init backbone from Stage 1 weights, reinit classifier
```

### Optical Flow (Leg)
```
Leg crop video (224×224 RGB)
  ↓ Farneback dense optical flow between consecutive frames
  ↓ Encode: hue=direction, value=magnitude (HSV→BGR)
  ↓ Save as color MP4 (224×224)
  ↓ VideoMAE-large fine-tuned on flow videos
```

---

## Key Findings

| # | Finding | Impact |
|---|---|---|
| 1 | STCA augmentation always helps skeleton models | +0.05 avg across tracks |
| 2 | Focused leg joints (6) beat full body (17) | +0.07 for leg |
| 3 | Full body (17) beats focused for body track | +0.02 vs torso-4 |
| 4 | Bone modality only helps hand | Hurts body and leg |
| 5 | MMN ensemble J+B free boost | +0.02 no retraining |
| 6 | 80 epochs optimal for MMN | 120 causes overfitting |
| 7 | 1× no-movement cap > 2× cap | +0.10 for leg, +0.002 for body |
| 8 | SSv2 pretraining > Kinetics for head | SSv2 has hand-object interactions |
| 9 | Two-stage hand >> flat 34-class | Group E: 0.74 vs 0.35 area |
| 10 | Approach B (Stage1 transfer) massive boost | +0.135 for E (317 samples) |
| 11 | VideoMAE-large fits RTX 3080 at batch=8 | 1.38GB inference VRAM |
| 12 | VideoMAE-base SSv2 > VideoMAE-base Kinetics | For head/hand movements |
| 13 | Simpler baseline wins for head | Dropout/weighted loss hurt |
| 14 | Dynamic crop does not help head | Static crop with smoothing better |

---

## Project Structure

```
ma52/
├── data/                           # gitignored
│   ├── annotations/                # fine2coarse.txt, train/val lists
│   ├── videos/train/ val/          # raw MA-52 videos
│   ├── keypoints/train/ val/       # YOLOv8 COCO-17 JSON files
│   ├── head_crops/train/ val/      # 224×224 head crop videos
│   ├── body_crops/train/ val/      # 224×224 body crop videos
│   ├── leg_crops/train/ val/       # 224×224 leg crop videos
│   ├── upperbody_crops/train/ val/ # 224×224 upper body crop videos
│   ├── leg_flow/train/ val/        # 224×224 optical flow videos
│   ├── head_dataset/               # head track CSVs
│   ├── skeleton_dataset/           # body/leg MMN CSVs + videomae CSVs
│   └── hand_dataset/               # hand stage1/stage2/group CSVs
├── src/
│   ├── extraction/
│   │   └── extract_pose.py         # YOLOv8x-pose extraction
│   ├── head/
│   │   ├── prepare_dataset.py      # build head CSVs
│   │   ├── crop_clips.py           # extract head crops (dynamic smoothing)
│   │   ├── train.py                # VideoMAE-base fine-tuning
│   │   ├── evaluate.py             # confusion matrix + F1
│   │   └── animate_wrong.py        # annotated wrong prediction videos
│   ├── hand/
│   │   ├── prepare_twostage.py     # generate all hand CSVs
│   │   ├── train_stage1.py         # 5-class coarse VideoMAE
│   │   ├── train_stage2.py         # per-group VideoMAE (A and B)
│   │   ├── evaluate_twostage.py    # full two-stage pipeline eval
│   │   ├── crop_upperbody.py       # PAD_TOP=120 upper body crop
│   │   └── legacy/                 # old fusion model scripts
│   ├── body_leg/
│   │   ├── crop_body.py            # nose-to-ankles crop
│   │   ├── crop_leg.py             # hips-to-ankles crop
│   │   ├── train_body_videomae.py  # VideoMAE-base body
│   │   └── train_leg_videomae.py   # VideoMAE-base leg
│   ├── skeleton/
│   │   ├── train_mmn_generic_v2.py # main MMN training (bone + augment)
│   │   ├── features_modality.py    # joint + bone feature loader
│   │   ├── evaluate_skeleton.py    # ensemble + threshold tuning
│   │   ├── evaluate_body_leg.py    # confusion matrix
│   │   ├── animate_wrong_body_leg.py # skeleton overlay videos
│   │   ├── prepare_dataset.py      # hand 5-class (old)
│   │   ├── prepare_leg_dataset_1x.py # leg 1× cap
│   │   ├── prepare_body_dataset_1x.py # body 1× cap
│   │   └── MMN/                    # MMN model repo
│   └── experiments/                # experimental scripts (separate)
│       ├── train_videomae_large.py # universal VideoMAE-large script
│       ├── oversample_body.py      # A3×10, A4×22 oversampling
│       ├── oversample_leg.py       # D2×3, D5×3 oversampling
│       ├── train_body_oversample.py # focal loss + heavy aug
│       ├── train_leg_oversample.py  # focal loss + heavy aug
│       ├── extract_optical_flow.py  # Farneback flow extraction
│       └── train_leg_flow.py        # VideoMAE on flow videos
├── jobs/
│   ├── head/                       # head SLURM scripts
│   ├── hand/                       # hand SLURM scripts
│   ├── skeleton/                   # MMN body/leg SLURM scripts
│   ├── body_leg/                   # VideoMAE body/leg SLURM scripts
│   └── experiments/                # experimental SLURM scripts
├── models/                         # gitignored
│   ├── videomae-ssv2/              # VideoMAE-base SSv2 (87M)
│   ├── videomae-large-kinetics/    # VideoMAE-large Kinetics (307M)
│   └── yolov8x-pose.pt             # pose extraction
├── outputs/                        # gitignored (models + eval)
│   ├── head_best_run1/             # Head F1=0.7769
│   ├── hand_skeleton_full/         # Hand MMN joint F1=0.7238
│   ├── hand_full_bone_aug/         # Hand MMN bone F1=0.7337
│   ├── hand_stage1/                # Stage 1 router F1=0.6568
│   ├── hand_stage2_{C,E,F,G}/      # Stage 2 Approach A
│   ├── hand_stage2_{C,E,F,G}_B/    # Stage 2 Approach B
│   ├── body_full_joint_aug/        # Body MMN F1=0.6176
│   ├── leg_leg_joint_aug/          # Leg MMN F1=0.5802
│   ├── body_videomae/              # Body VideoMAE-base F1=0.5278
│   └── leg_videomae/               # Leg VideoMAE-base F1=0.5776
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Setup & Reproduce

### Requirements
```bash
conda create -n ma52 python=3.12
conda activate ma52
pip install torch torchvision
pip install ultralytics opencv-python-headless
pip install transformers accelerate timm einops
pip install scikit-learn pandas numpy tqdm
pip install matplotlib seaborn decord av
```

### Download Models
```python
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# VideoMAE-base SSv2
VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2").save_pretrained("models/videomae-ssv2")
VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2").save_pretrained("models/videomae-ssv2")

# VideoMAE-large Kinetics (for experiments)
VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics").save_pretrained("models/videomae-large-kinetics")
VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics").save_pretrained("models/videomae-large-kinetics")
```

### Data Setup
1. Download MA-52 dataset → `data/videos/`
2. Extract data_csvs.zip and keypoints.zip (from FAUbox)
3. Extract pose: `sbatch.tinygpu jobs/head/job_extract_pose.sh`

### Reproduce Head Track
```bash
python src/head/prepare_dataset.py
sbatch.tinygpu jobs/head/job_crop_head.sh
sbatch.tinygpu jobs/head/job_train_head.sh
```

### Reproduce Hand Two-Stage
```bash
python src/hand/prepare_twostage.py
sbatch.tinygpu jobs/hand/job_hand_stage1.sh           # Stage 1
sbatch.tinygpu jobs/hand/job_hand_stage2_C.sh          # Stage 2 Approach A
sbatch.tinygpu jobs/hand/job_hand_stage2_C_B.sh        # Stage 2 Approach B (after Stage 1)
# ... repeat for E, F, G
sbatch.tinygpu jobs/hand/job_hand_eval_twostage.sh     # Full pipeline eval
```

### Reproduce Body/Leg
```bash
python src/skeleton/prepare_body_dataset_1x.py
python src/skeleton/prepare_leg_dataset_1x.py
sbatch.tinygpu jobs/skeleton/job_body_1x.sh
sbatch.tinygpu jobs/skeleton/job_leg_1x.sh
python src/body_leg/crop_body.py --split train && --split val
python src/body_leg/crop_leg.py --split train && --split val
sbatch.tinygpu jobs/body_leg/job_body_videomae.sh
sbatch.tinygpu jobs/body_leg/job_leg_videomae.sh
```

### SLURM Commands
```bash
sbatch.tinygpu jobs/FOLDER/SCRIPT.sh   # submit
squeue.tinygpu -u iwso226h              # check queue
scancel.tinygpu JOBID                   # cancel
sinfo.tinygpu                           # node status
```

### Check All Results
```bash
python3 - << 'EOF'
import os
log_dir = '/home/woody/iwso/iwso226h/ma52/logs/'
for f in sorted(os.listdir(log_dir)):
    if not f.endswith('.log'): continue
    path = os.path.join(log_dir, f)
    try:
        lines = open(path).readlines()
        best = [l.strip() for l in lines if 'Best F1' in l or 'Done.' in l]
        if best: print(f"{f}: {best[-1]}")
    except: pass
EOF
```

---

## Pending / TODO

1. Wait for 9 running experiments to finish
2. Compare Approach A vs B per hand group → pick best
3. Submit hand two-stage evaluation (`job_hand_eval_twostage.sh`)
4. Submit hand coarse VideoMAE-large job (after two-stage eval)
5. Ensemble body/leg: VideoMAE + MMN for free boost
6. Write label fusion script (combine all tracks → 52-class → official metric)
7. Final competition submission

---

## References

- **MA-52 Dataset:** Guo et al., *Benchmarking Micro-action Recognition: Dataset, Methods, and Applications*, IEEE TCSVT 2024
- **MAC 2025 Paper:** Li et al., *MAC 2025: The 2nd Micro-Action Analysis Grand Challenge*, ACM MM 2025
- **MMN:** Gu et al., *Motion Matters: Motion-guided Modulation Network for Skeleton-based Micro-action Recognition*, ACM MM 2025
- **VideoMAE:** Tong et al., *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*, NeurIPS 2022
- **YOLOv8-pose:** Ultralytics, 2023
