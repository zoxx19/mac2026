# Results — MAC 2026 Track 2

Honest, complete record of every experiment, including the many that did not
work. Metric is **average mAP at tIoU = {0.2, 0.5, 0.7}** unless noted.
"Test (leaderboard)" is the official test-server avg mAP for the submitted file.

## Competition Result

- **Final leaderboard position: 3rd place**
- **Best test score: 0.22993 avg mAP**
- Submitted entry: **Large1 (ep22) + Large2 (ep9), equal-weight Soft-NMS fusion (σ=0.5)**
- Solo student entry (one person: modelling, pipeline, and HPC management).

---

## Main Models

| Model | Backbone | Cls loss | Epochs | Val avg mAP | Test (leaderboard) |
|-------|----------|----------|:------:|:-----------:|:------------------:|
| Large1 | VideoMAE-L + bottleneck adapter | weighted FocalLoss | 22 | 21.67% (ep22 alone: 20.56%) | 0.22691 |
| Large2 | VideoMAE-L + bottleneck adapter | vanilla FocalLoss | 9 | 21.31% (ep9 alone: 21.29%) | 0.21591 |
| Large Restart | warm restart from Large1 ep18 | weighted FocalLoss | 6 | 21.08% | — |
| DSTA | VideoMAE-L + **dual-path adapter** | vanilla FocalLoss | ep0 (700/1972 iters) | — | — |
| ASL | VideoMAE-L + bottleneck adapter | AsymmetricLoss | 0 (Loss=inf) | — | — |
| TG modality | VideoMAE-L on temporal-gradient video | FocalLoss | 2 | — | — |
| MMN skeleton | GCN (MMN) recogniser | ASL | 28 | 7.11% (detection: 6.02%) | — |

Note the inversion: **Large2 had higher val mAP than Large1 but a lower test
score.** This is the first sign that val mAP is an unreliable proxy here.

---

## Ablation Studies — Fusion & Ensembling

| Experiment | Val avg mAP | Test | Notes |
|-----------|:-----------:|:----:|-------|
| Large1 ep22 alone | 20.56% | 0.22691 | Baseline single model |
| Large2 ep9 alone | 21.29% | 0.21591 | Higher val, **lower** test |
| **Large1 + Large2, equal, σ=0.5** | 21.76% | **0.22993** | **BEST — submitted** |
| Large1 + Large2, tuned 0.47/0.53, σ=0.7, top-1000 | 22.10% | (−0.008 vs best) | Best **val**, worse **test** |
| 3-way + Large ep18 | 21.70% | 0.22956 | ep18 hurts the test score |
| Weighted Boxes Fusion (WBF) | 16–17% | — | Hurt badly vs Soft-NMS |
| TTA — 4 variants (flip + speed) | 14.53% | — | Hurt badly |

### Fusion takeaways
- **Equal weights win on test.** Every val-optimised weighting (0.47/0.53, etc.)
  improved val mAP but regressed the test score.
- **σ = 0.5 beats σ = 0.7 on test.** σ=0.7 gave +0.35% val but −0.008 test.
- **Soft-NMS ≫ WBF.** WBF's segment averaging blurs already-tight boundaries.
- **More models is not always better.** Adding Large ep18 (3-way) slightly hurt.

---

## Ablation Studies — Post-processing (all hurt)

The OpenTAD mAP evaluator performs an internal threshold sweep over the PR
curve, so any external thresholding / pruning can only remove useful recall.

| Post-processing | Val avg mAP | Verdict |
|-----------------|:-----------:|---------|
| Baseline (no post-proc, all proposals, thresh 0.0) | ~21.8% | best |
| Score thresholding (> 0) | 20.19–20.70% | hurt |
| Temporal gap-filling / bucketing | 20.19–20.70% | hurt |
| Body-part group re-ranking | 20.19–20.70% | hurt |
| Skeleton re-ranking (`rerank_with_skeleton.py`, α sweep) | ≤ baseline | hurt |

**Conclusion: threshold = 0.0 + keep all proposals is optimal.**

---

## Auxiliary Modalities (none helped the final score)

| Stream | Standalone | In fusion | Status |
|--------|:----------:|:---------:|--------|
| RGB VideoMAE-Base | 2.12% | hurts | too weak |
| Body-crop VideoMAE-Base | 2.08% | hurts | too weak |
| Optical flow VideoMAE-Base | — | — | incomplete |
| TG (temporal gradient) | — | — | 2 epochs, no eval, cut by maintenance |
| MMN skeleton (GCN) | 7.11% recog / 6.02% det | **always hurts** | abandoned for fusion |

The skeleton model recognised micro-actions reasonably well on its own, but
fusing it with the RGB detector consistently *lowered* the score — its temporal
localisation is too coarse to help the detector's already-sharp boundaries.

---

## DSTA — Dual-path Spatial-Temporal Adapter (novel)

Our main architectural contribution; see [`docs/architecture.md`](docs/architecture.md).

- Implemented in the final hours after concluding the gap to #1 was
  architectural, not tunable.
- Extends the MMAD (ICCV 2025) dual-path adapter idea from VideoMAE-**Base** to
  VideoMAE-**Large**, accounting for this backbone's quirks (no CLS token,
  10×10 grid, fully frozen backbone, identity-init for warm start).
- **Training log (`train_dsta_1692722.log`):**
  - 4×A100-40GB, 98% GPU util, ~19.3 GB/GPU, fp32 (`amp=False`).
  - 700 / 1972 iterations of epoch 0 (~1.5 h wall-clock).
  - Loss **0.6878 → 0.5238** — stable, finite, monotonically decreasing.
  - LR still warming up (1.1e-05 → 2.6e-05); 0 mAP evaluations reached.
  - Ended 06:23, cut by the **06:30 maintenance window** — not a model failure.

**Status: implementation complete and correct; needs more training time to
evaluate.**

---

## Key Learnings

1. **Val mAP ≠ test mAP.** The single most important and most expensive lesson.
   Higher val (Large2 > Large1; σ=0.7; tuned weights) repeatedly meant *lower*
   test. Trust the leaderboard, or hold out a mini-test set.
2. **Equal weights beat tuned weights** for this dataset's ensemble.
3. **Scale the backbone first.** VideoMAE-Large was the dominant lever; nothing
   else came close.
4. **Keep all proposals (threshold 0.0).** The evaluator does the optimal
   threshold sweep internally; external pruning only removes recall.
5. **Post-processing and TTA hurt.** Every hand-tuned refinement reduced mAP.
6. **Skeleton/MMN fusion hurts** despite decent standalone recognition.
7. **DSTA is the right architectural direction** — explicit spatial/temporal
   factorisation in the adapter — and was cut short by compute, not by results.
8. **Numerical stability matters at Large scale:** `amp=False` for VideoMAE-L
   (fp16 → `Loss=inf`); ASL needed `gamma_neg=2` (not 4) to train at all.
