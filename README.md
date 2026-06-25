# MA-52 Micro-Action Recognition

Multi-track video action recognition pipeline for the **MA-52** dataset, built for the [Micro-Action Analysis Grand Challenge (MAC 2026)](https://sites.google.com/view/micro-action) at ACM Multimedia 2026.

The model classifies 52 fine-grained human micro-actions across 7 coarse body-part groups (head, body, upper limb, lower limb, body-hand, head-hand, leg-hand) from short RGB video clips, by training a specialized model per body part and fusing their predictions.

## Results

| Track | Approach | F1-mean (val) |
|---|---|---|
| Head | VideoMAE (SSv2 pretrained) | 0.777 |
| Hand, coarse | Skeleton GCN ensemble | 0.753 |
| Hand, fine-grained | Two-stage classifier | 0.484 |
| Body | Optical flow + skeleton ensemble | 0.647 |
| Leg | Optical flow + skeleton ensemble | 0.691 |
| **Combined (all 52 classes)** | **Fusion of all tracks** | **0.578** |

Metric: `F1_mean = (F1_body_macro + F1_body_micro + F1_action_macro + F1_action_micro) / 4`

## Approach

```
Video
  ├── Head model      → which of 6 head movements (or none)
  ├── Body model      → which of 5 body movements (or none)
  ├── Leg model       → which of 8 leg movements (or none)
  └── Hand pipeline (two stages)
        1. Coarse router: which limb is involved (arm / torso-hand / head-hand / leg-hand / none)
        2. Fine classifier: which specific action, conditioned on the limb
            ↓
    Fusion → single prediction across all 52 classes
```

Each track uses whichever model worked best for that signal:
- **Head & hand fine-grained:** VideoMAE video transformer, fine-tuned on cropped clips
- **Body & leg:** an ensemble of optical-flow-based VideoMAE and a skeleton graph network (MMN)
- **Hand coarse routing:** an ensemble of a VideoMAE classifier and two skeleton-based classifiers, which more than doubles routing accuracy over any single model

## Project Structure

```
src/
  extraction/      # pose extraction (YOLOv8-pose)
  head/            # head-movement classifier
  hand/            # two-stage hand classifier (router + fine-grained heads)
  body_leg/        # body/leg RGB-crop classifiers
  skeleton/        # skeleton-based GCN model (body, leg, hand-coarse)
  experiments/     # optical flow extraction + ensembling utilities
  fusion/          # combines all tracks into the final 52-class prediction
jobs/              # SLURM job scripts, mirrors src/ layout
models/            # pretrained weights (gitignored)
outputs/           # trained checkpoints + evaluation results (weights gitignored)
data/              # dataset, pose, and crop cache (gitignored)
```

## Setup

```bash
conda create -n ma52 python=3.12
conda activate ma52
pip install -r requirements.txt
```

Download pretrained weights:
```python
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2").save_pretrained("models/videomae-ssv2")
VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2").save_pretrained("models/videomae-ssv2")
```
Also place a YOLOv8-pose checkpoint (Ultralytics) at `models/yolov8x-pose.pt`.

Place the MA-52 dataset under `data/videos/train/` and `data/videos/val/`, then extract pose keypoints:
```bash
sbatch jobs/head/job_extract_pose.sh
```

## Training

Each track is trained independently. See `jobs/` for ready-to-run SLURM scripts, or call the underlying scripts in `src/` directly.

```bash
# Head
python src/head/prepare_dataset.py
sbatch jobs/head/job_crop_head.sh
sbatch jobs/head/job_train_head.sh

# Hand (two-stage)
python src/hand/prepare_twostage.py
sbatch jobs/hand/job_hand_stage1.sh
sbatch jobs/hand/job_hand_stage2_C_ssv2_os.sh   # repeat per limb group: C, E, F, G
sbatch jobs/hand/job_hand_skeleton_full.sh      # coarse-routing skeleton models
sbatch jobs/hand/job_hand_full_bone.sh
sbatch jobs/hand/job_hand_eval_ensemble2.sh     # evaluate the full two-stage pipeline

# Body / Leg
python src/skeleton/prepare_body_dataset_1x.py
python src/skeleton/prepare_leg_dataset_1x.py
sbatch jobs/skeleton/job_body_1x.sh
sbatch jobs/skeleton/job_leg_leg.sh
sbatch jobs/experiments/job_body_flow.sh
sbatch jobs/experiments/job_leg_flow_ssv2.sh
sbatch jobs/experiments/job_ensemble_body.sh
sbatch jobs/experiments/job_ensemble_leg.sh

# Final fusion across all tracks
sbatch jobs/fusion/job_label_fusion.sh
```

## Key Findings

- Pretraining domain matters more than model size: a base model pretrained on Something-Something-v2 consistently beat a larger model pretrained on Kinetics.
- A skeleton-based model and an RGB model make different mistakes; ensembling the two gave the largest single improvement on body and leg.
- Hierarchical (coarse-then-fine) classification clearly outperformed a single flat 34-class classifier for the hand track.
- Initializing the fine-grained classifier from the coarse router's weights, rather than from the base pretrained checkpoint, gave a large boost for low-data classes.
- Ensembling multiple models for the routing decision (rather than relying on one) substantially reduced downstream errors in the two-stage pipeline.

## References

- Guo et al., *Benchmarking Micro-action Recognition: Dataset, Methods, and Applications*, IEEE TCSVT 2024
- Gu et al., *Motion Matters: Motion-guided Modulation Network for Skeleton-based Micro-action Recognition*, ACM MM 2025
- Tong et al., *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*, NeurIPS 2022
- Ultralytics YOLOv8