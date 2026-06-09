# Technical Notes — MAC 2026 Track 2

Hard-won, mostly painful lessons from running this challenge solo on an HPC
cluster. Intended to save the next team (or future me) days of debugging.

## Environment Setup Issues

- **The project directory previously had a trailing space — this has been fixed.**
  A `git clone` now gives you the clean name `micro_challenge` (no space).
  Historically the trailing space broke bash heredocs and naive shell operations;
  if you ever encounter a path with a trailing space, quote it
  (`cd "…/micro_challenge "`) or rename it. Writing files with Python `open()`
  instead of shell redirection sidesteps the problem entirely.
- Use the project's conda Python for everything (`conda activate opentad`); the
  OpenMMLab stack (`mmcv` / `mmaction2`) is version-sensitive and must match the
  torch/CUDA build.

## SLURM Configuration (FAU NHR HPC)

- Always set `--account=iwso`.
- **Never use `--mem`** — it causes errors on these partitions.
- Don't pin a GPU type in `--gres` on partitions that don't expose one.
- **Maintenance windows block *new* jobs.** A job is killed only if its
  `--time` overruns into the reservation; check `--time` fits before the window.
  (`IGNORE_JOBS` means already-running jobs survive the reservation.)
- The a100 partition was usable until ~06:30 on maintenance days; one VideoMAE-L
  epoch is ~3.8 h, so plan checkpoints accordingly.

## Training Stability

- **`amp=False` is required for VideoMAE-Large** — fp16 produced `Loss=inf`.
  (The smaller adapters tolerate amp, but the Large e2e run does not.)
- **`warmup_epoch` must be ≥ 2** — `LinearWarmupCosineAnnealingLR` divides by
  `warmup_epoch − 1`, so `warmup_epoch=1` is a divide-by-zero.
- **Resume with the `--resume` CLI flag, NOT `--cfg-options resume=...`** — the
  latter silently starts a fresh run.
- **Select checkpoints by modification time, not `sort -V`** — version sort
  picks the wrong epoch (e.g. `epoch_9` after `epoch_18`). Use
  `find ... -printf '%T@ %p\n' | sort -n | tail -1`.
- **Do not use `set -euo pipefail` in the SLURM scripts** — unbound variables
  and benign non-zero exits abort the job mid-pipeline.

## DSTA Implementation Notes

- **VideoMAE here has NO CLS token** (unusual). Tokens are `(B, N, C)` with
  `N == T·h·w`, ordered `[T, h, w]` (T outermost).
- **Spatial grid is 10×10**, not 14×14 — input is 160 px ÷ 16 px patches. Never
  hard-code H/W/num_patches; derive `T = N // (h·w)` at runtime.
- **Backbone is fully frozen** — only the adapters (and the TAD head outside the
  backbone) train. `_freeze_layers()` runs every forward.
- **Identity init for stable warm start:** zero-init the up-projections (`up_s`,
  `up_t`) so the adapter is a no-op at step 0 — exactly like the proven
  bottleneck adapter. Unlike "scale = 0", zero-init keeps gradient flowing into
  the up-projections from step 1. Scales (`s_scale`, `t_scale`) start at 1.0.
- **Load the warm-start checkpoint with `strict=False`** — the fresh DSTA
  adapter keys won't exist in the source checkpoint.
- Build the warm-start file with `tools/strip_checkpoint.py` (weights-only,
  `epoch=-1`) so `train.py` starts a fresh optimizer/scheduler/EMA.

## What Worked vs What Didn't

| Tried | Result |
|-------|--------|
| VideoMAE-Large backbone | ✅ biggest single win |
| Two-model equal-weight Soft-NMS fusion | ✅ 0.22691 → **0.22993** |
| MA-52 augmentation (~3.5× data) | ✅ helped |
| threshold = 0.0 (keep all proposals) | ✅ best |
| Tuned fusion weights (0.47/0.53) | ❌ +val, −test |
| σ = 0.7 Soft-NMS | ❌ +val, −test |
| Weighted Boxes Fusion | ❌ 16–17% |
| TTA (flip / speed) | ❌ 14.5% |
| Post-processing (gap-fill, buckets, rerank) | ❌ all hurt |
| Skeleton / MMN fusion | ❌ always hurt |
| AsymmetricLoss (gamma_neg=4) | ❌ Loss=inf |

Full numbers in [RESULTS.md](RESULTS.md).

## Storage Management

- Woody storage quota: **1000 GB** soft / 1500 GB hard. We ran at ~960 GB —
  keep an eye on it.
- Each VideoMAE-Large checkpoint is **~3 GB**; keep only the best 2–3 per model.
- Prediction JSONs are **200–400 MB** each.
- Strip checkpoints to weights-only (`tools/strip_checkpoint.py`) for archival /
  warm-starting — it drops the optimizer/EMA bloat.

## Competition-Specific Insights

- **`threshold = 0.0` is always best** — more proposals → better mAP, because the
  evaluator sweeps thresholds via the PR curve internally.
- **Equal fusion weights beat tuned weights** on the test server.
- **Val mAP is an unreliable proxy for test mAP** (the recurring theme): higher
  val frequently meant lower test (ep18 vs ep22; σ=0.7; tuned weights).
- **Post-processing hurts** — the optimal threshold sweep is already internal.
- **Skeleton/MMN fusion always hurts** despite good standalone recognition; its
  temporal boundaries are too coarse to help a sharp RGB detector.
- **Decide submissions by leaderboard, not val** — or hold out a mini-test set.

This document, with [RESULTS.md](RESULTS.md) and
[`docs/architecture.md`](docs/architecture.md), should let a future team skip
most of the dead ends.
