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


## Dataset

[MA-52](https://huggingface.co/datasets/kunli-cs/MA-52) contains 22,422 short RGB clips (~2.8s, 30fps) of spontaneous micro-actions recorded during interviews: 11,250 train / 5,586 val / 1,138 test. Each clip is labeled with one of 52 fine-grained actions, which map onto 7 coarse body-part groups:

| Coarse group | Fine labels |
|---|---|
| A — body | 0–4 |
| B — head | 5–10 |
| C — upper limb | 11–23 |
| D — lower limb | 24–31 |
| E — body-hand | 32–37 |
| F — head-hand | 38–47 |
| G — leg-hand | 48–51 |

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
- **Hand coarse routing:** an ensemble of a VideoMAE classifier and two skeleton-based classifiers

## What We Tried

We explored a wide range of architectures and training strategies before settling on the final per-track ensembles above.

**Backbones evaluated:** VideoMAE-base (SSv2 and Kinetics pretraining), VideoMAE-large, TimeSformer (base and high-resolution variants), and two skeleton-graph architectures (MMN; CTR-GCN and BlockGCN were prototyped but not integrated due to graph-shape incompatibilities with our joint layout).

**Input representations evaluated:** raw RGB crops at multiple framings (full-body, upper-body, head-only, limb-focused), COCO-17 joint coordinates, bone/limb-vector features, and Farneback optical flow encoded as HSV video.

**Training strategies evaluated:** class-weighted loss vs. focal loss, oversampling of rare classes, MixUp and label smoothing, dropout regularization, dynamic per-frame cropping, single-stage flat classification vs. coarse-to-fine hierarchical classification, and transferring a router model's backbone into the downstream fine-grained classifiers.

**Fusion strategies evaluated:** single best-model per track vs. weighted ensembles of 2–3 models per track, with ensemble weights swept on a held-out validation grid.

We also attempted to train a single end-to-end model over all 52 classes directly (skipping the per-track split entirely), to see whether a unified model could outperform the fusion approach; this was not completed in the time available and is noted as future work below.

## Key Findings

- **Pretraining domain dominated model size.** A base model pretrained on Something-Something-v2 (an action dataset with fine hand/body motion) consistently beat a 3.5x larger model pretrained on Kinetics, across every track. TimeSformer, despite a different attention mechanism, underperformed VideoMAE on every track tried.
- **RGB and skeleton models fail differently — ensembling them was our single biggest win.** A skeleton-based classifier and an optical-flow RGB classifier disagree on different videos; averaging their predicted probabilities gave the largest improvement of any single change, lifting the leg track by +0.11 F1 and the body track by +0.03 F1 over the better individual model.
- **Hierarchical classification beat flat classification by a wide margin.** Splitting the 34-class fine-grained hand problem into a coarse router (which limb?) followed by 4 small per-limb classifiers outperformed a single 34-way classifier by roughly +0.17 F1.
- **Backbone transfer between stages matters for low-data classes.** Initializing a fine-grained classifier from the coarse router's fine-tuned backbone (rather than from the generic pretrained checkpoint) gave a +0.135 F1 jump for the smallest, most data-scarce class group.
- **Ensembling the routing decision reduced cascading errors.** In the two-stage hand pipeline, an error at the routing step guarantees an error downstream. Replacing a single router with a 3-way ensemble (RGB + two skeleton modalities) raised routing accuracy from 70% to 81%, which alone moved the final hand-track score from 0.41 to 0.48.
- **More augmentation and more capacity were not free wins.** Stronger augmentation, longer training schedules, and larger models each helped in some configurations but hurt in others (e.g. extending head-track training degraded its F1 from 0.777 to 0.697); every change was validated rather than assumed beneficial.
- **Validation-set tuning does not guarantee held-out generalization.** Our held-out validation score (0.578) was noticeably higher than what we observed on a separate, unseen test split (0.45). We attribute this gap to ensemble weights and class-balancing ratios being tuned against the same validation set used for final evaluation — a useful reminder that tuning and final evaluation splits should ideally be kept separate.

## Limitations & Future Work

- The fine-grained hand track (34 classes) remains the weakest link (F1 ≈ 0.48); more training data per class or a stronger backbone (e.g. a larger video transformer than fits on a single 10GB GPU) would likely help most here.
- A single unified 52-class end-to-end model was prototyped but not completed; comparing it properly against the per-track fusion approach is the natural next experiment.
- Tuning ensemble weights via cross-validation rather than a single validation split would likely close some of the validation/test generalization gap noted above.

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

Each track is trained independently. See `jobs/` for ready-to-run SLURM scripts (update the `PYTHON=` and `cd` lines at the top of each script to match your environment), or call the underlying scripts in `src/` directly.

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
python src/fusion/label_fusion.py
```

## References

- Guo et al., *Benchmarking Micro-action Recognition: Dataset, Methods, and Applications*, IEEE TCSVT 2024
- Gu et al., *Motion Matters: Motion-guided Modulation Network for Skeleton-based Micro-action Recognition*, ACM MM 2025
- Tong et al., *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*, NeurIPS 2022
- Ultralytics YOLOv8- Ultralytics, *YOLOv8*, 2023
