# Experiments — full log

A narrative companion to [RESULTS.md](../RESULTS.md). Numbers are average mAP at
tIoU = {0.2, 0.5, 0.7}; "test" = official test-server score. Where exact log
values exist they are quoted; otherwise the value is marked approximate.

## Timeline

| Phase | Run | Duration | Outcome |
|-------|-----|----------|---------|
| Week 1 | Large1 (VideoMAE-L, weighted FocalLoss) | 22 epochs | 21.67% val, 0.22691 test |
| Week 1 | Large2 (VideoMAE-L, vanilla FocalLoss) | 9 epochs | 21.31% val (still climbing) |
| Week 2 | Large Restart (warm restart from ep18) | 6 epochs | 21.08–21.17% val, no ensemble gain |
| Week 2 | MMN skeleton (GCN, ASL) | 28 epochs | 7.11% recog / 6.02% det, abandoned |
| Week 2 | TG (temporal-gradient modality) | 2 epochs | loss 0.447→0.404, no eval reached |
| Final night | ASL (AsymmetricLoss warm start) | 0 epochs | Loss=inf |
| Final night | DSTA (dual-path adapter, 4×A100) | 700/1972 iters (ep0) | loss 0.688→0.524, cut by maintenance |

**Submitted: Large1 ep22 + Large2 ep9, equal weights, σ=0.5 → 0.22993 (3rd).**

## Backbone scaling

The single biggest lever. Base-scale adapters (RGB / body / flow) plateaued at
~2% standalone; moving to VideoMAE-Large lifted single-model val mAP above 20%.
Everything in this project is downstream of that decision.

## The two ensemble members

We deliberately trained two Large models that make different mistakes:

- **Large1** — per-class **weighted** FocalLoss (inverse class frequency). Helps
  rare classes; ep22 → 20.56% alone, 0.22691 test.
- **Large2** — **vanilla** FocalLoss; ep9 → 21.29% alone, but only 0.21591 test.

Large2 has higher val but lower test than Large1 — the first concrete evidence
that **val mAP misranks models** on this dataset.

## Fusion sweep

| Config | Val | Test | Note |
|--------|----:|-----:|------|
| equal weights, σ=0.5 | 21.76% | **0.22993** | submitted |
| weights 0.47/0.53, σ=0.7, top-1000 | **22.10%** | −0.008 vs best | best val, worse test |
| + Large ep18 (3-way) | 21.70% | 0.22956 | ep18 drags test down |
| WBF instead of Soft-NMS | 16–17% | — | boundary averaging hurts |

Conclusions: **equal weights** and **σ=0.5** are best on test; tuning either
toward the val optimum regressed the leaderboard.

## Post-processing (everything hurt)

The OpenTAD mAP evaluator sweeps the score threshold internally over the PR
curve, so any external pruning removes useful recall:
- Score thresholding / gap-filling / bucketing: 20.19–20.70% (all below baseline).
- Body-part group re-ranking: ≤ baseline.
- Skeleton re-ranking (`rerank_with_skeleton.py`, α sweep): ≤ baseline.
- **Best policy: `threshold = 0.0`, keep all proposals.**

## Test-Time Augmentation (hurt badly)

Synthesising temporal-flip and speed-perturbation hypotheses and merging with
Soft-NMS (`tta_inference.py`) collapsed to **14.53%** — the augmented segments
are not faithful to the data and dilute the sharp originals.

## Auxiliary modalities (none helped fusion)

- **RGB / body / flow VideoMAE-Base**: ~2% standalone, too weak to contribute.
- **TG (temporal gradient)**: `extract_tg.py` builds a slow+fast frame-difference
  video; the Large run on it reached only 2 epochs (loss 0.447→0.404) before the
  maintenance window — no eval.
- **MMN skeleton (GCN)**: best non-RGB recogniser (7.11% recog / 6.02% det) but
  **fusing it always lowered** the combined score; its temporal localisation is
  too coarse for the sharp RGB detector.

## ASL instability

AsymmetricLoss with `gamma_neg=4` produced `Loss=inf` immediately on warm start.
`gamma_neg=2` + `amp=False` stabilised it, but the run was then cut by the
maintenance window before a usable checkpoint. See
[`src/models/losses.py`](../src/models/losses.py) and
[`configs/adatad/asl.py`](../configs/adatad/asl.py).

## DSTA (novel; cut short)

See [`docs/architecture.md`](architecture.md) for the design. Training was clean
and stable — 4×A100-40GB at 98% util, fp32, loss 0.688 → 0.524 over the first
700/1972 iterations of epoch 0 — and ended only because the SLURM job hit its
01:30 limit at the 06:30 maintenance reservation. No validation evaluation was
reached, so DSTA has no mAP number yet; the implementation is complete and the
loss trajectory is healthy.

## What we'd do next

1. Finish DSTA training (multiple full epochs) and evaluate.
2. Use a held-out mini-test set to make submission decisions, since val mAP is an
   unreliable proxy here.
3. Explore stronger temporal modelling in the adapter (the DSTA direction) rather
   than more post-processing — post-processing consistently hurt.
