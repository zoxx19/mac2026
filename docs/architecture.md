# Architecture — DSTA and the full detection pipeline

This document explains the model architecture in detail, with emphasis on
**DSTA (Dual-path Spatial-Temporal Adapter)**, our novel contribution.

## 1. The detection stack (AdaTAD on VideoMAE-Large)

We use **end-to-end AdaTAD**: the video backbone is trained jointly with the
ActionFormer-style temporal detection head, rather than extracting frozen
features offline.

```
mp4 clip (160×160, 512-frame window)
   │  reshape into 32 chunks of 16 frames        (b n c (t1 t) h w -> (b t1) n c t h w)
   ▼
VideoMAE-Large ViT  (depth=24, dim=1024, heads=16, patch=16, tubelet=2, NO CLS token)
   │  per block: Attn → FFN → Adapter            (adapter on all 24 blocks)
   │  backbone FROZEN; only adapters require grad
   ▼  feature map (B, 1024, T, 10, 10)
mean-pool over the 10×10 spatial grid → (B, 1024, T) → interpolate to 512
   ▼
AdaTAD projection (in_channels=1024, max_seq_len=512, full attention window)
   ▼
ActionFormer head: classification (FocalLoss) + regression (DIoULoss), 52 classes
   ▼
sliding-window inference → Soft-NMS (σ=0.7, multiclass, voting 0.7)
   ▼
{video_id: [{segment:[t_s,t_e], label, score}]}
```

Key configuration (see [`configs/adatad/large1.py`](../configs/adatad/large1.py)):
- `window_size = 512`, `scale_factor = 1`, `chunk_num = 32`.
- Backbone LR = 0 (frozen); adapter LR = 1e-4, weight decay 0.05; AdamW.
- `LinearWarmupCosineAnnealingLR`, `warmup_epoch=5` (DSTA uses 2), `max_epoch=30`.
- `amp=True` for the bottleneck-adapter Large models; **`amp=False` for DSTA**
  (fp16 caused `Loss=inf`).
- Large1 uses a **per-class weighted FocalLoss** (inverse class frequency);
  Large2 uses **vanilla FocalLoss**. This difference is what makes the two
  models complementary in the ensemble.

## 2. The baseline adapter (Large1 / Large2)

The proven `VisionTransformerAdapter` inserts a **bottleneck adapter** after the
FFN of every ViT block:

```
x ──► down_proj (1024→256) ──► GELU ──► depth-wise conv ──► up_proj (256→1024, zero-init) ──► + x
```

The up-projection is zero-initialised, so the adapter starts as identity and the
frozen backbone's behaviour is preserved at step 0.

## 3. DSTA — Dual-path Spatial-Temporal Adapter (novel)

Implemented in [`src/models/dsta_adapter.py`](../src/models/dsta_adapter.py) as
`VisionTransformerDSTA` (a drop-in replacement for `VisionTransformerAdapter`).
Everything except the adapter — Attention, Block, ViT, freezing — is identical
to the proven backbone, so training behaviour is unchanged apart from the
adapter itself.

### Motivation

Micro-actions are defined by **short temporal dynamics on top of fine spatial
detail** (e.g. "rubbing eyes" vs "touching nose"). A single depth-wise conv path
mixes both. DSTA factorises them into two explicit paths, inspired by the MMAD
(ICCV 2025) dual-path adapter — which only reported VideoMAE-**Base**. We
extended it to VideoMAE-**Large**.

### Design

```
                       x  (B, N, C),  N = T·h·w,  no CLS token
                       │
                  down_proj (shared)  C → C·0.25
                       │  GELU
        ┌──────────────┴───────────────┐
        ▼                               ▼
   S-PATH (spatial)               T-PATH (temporal)
   reshape → (B·T, mid, h, w)     reshape → (B·h·w, mid, T)
   1×1 dwconv → GELU              1 dwconv → GELU
   3×3 dwconv → GELU              3 dwconv → GELU
   1×1 dwconv                     1 dwconv
   reshape back → (B, N, mid)     reshape back → (B, N, mid)
        │                               │
   up_s (zero-init)                up_t (zero-init)
        │                               │
        └────────►  x + s_scale·up_s(zs) + t_scale·up_t(zt)  ◄────────┘
```

- **S-path**: processes each of the T frames independently with 2-D depth-wise
  convs over the 10×10 grid — refines spatial structure.
- **T-path**: processes each of the h·w spatial locations independently with
  1-D depth-wise convs over T — captures the temporal signature.
- Two **separate up-projections**, two **learnable scales** (`s_scale`,
  `t_scale`), added back to the residual.

### Critical implementation details (this backbone's quirks)

1. **No CLS token.** Tokens are `(B, N, C)` with `N == T·h·w`. Do not slice off a
   leading CLS — there isn't one.
2. **Token order `[T, h, w]`** (T outermost) — the reshapes in `forward` assume
   this; they must match the backbone's own temporal reshape.
3. **10×10 grid, not 14×14.** Input is 160 px ÷ 16 px = 10. `T` is derived as
   `N // (h·w)` at runtime; nothing is hard-coded. Positional embeddings are
   bicubically interpolated from the pretrained grid to (h, w).
4. **Frozen backbone.** `_freeze_layers()` puts every non-adapter module in
   `eval()` and `requires_grad=False` on every forward.
5. **Identity initialisation:** `up_s` / `up_t` are zero-initialised so the
   adapter output equals its input at step 0 (matching the bottleneck adapter).
   Unlike "scale = 0", zero-init keeps gradient flowing into the up-projections
   from the first step, so the adapter starts learning immediately. `s_scale` /
   `t_scale` start at 1.0; conv init mirrors the proven dwconv init.

### Warm-starting DSTA

DSTA was warm-started (weights only) from the trained Large1 ep22 checkpoint:

```bash
# 1. strip the trained checkpoint to weights-only, epoch=-1
python tools/strip_checkpoint.py work_dirs/large1/checkpoint/epoch_22.pth warmstart.pth
# 2. resume from it — the fresh DSTA adapter keys load with strict=False
torchrun ... tools/train.py configs/adatad/mma52/e2e_mma52_videomae_l_dsta.py --resume warmstart.pth
```

`epoch=-1` forces `train.py` to build a fresh optimizer/scheduler/EMA while
loading the proven backbone+head weights; the identity-init DSTA adapters simply
stay fresh and begin contributing from step 1.

### Training observed (cut short)

4×A100-40GB, 98% util, fp32 (`amp=False`). Epoch 0 reached iteration 700/1972
(~1.5 h); `Loss=0.6878 → 0.5238`, stable and decreasing; LR still in warmup. No
validation eval reached before the 06:30 maintenance window. See
[RESULTS.md](../RESULTS.md).

## 4. Registering the custom components in OpenTAD

`setup.sh` copies the files into OpenTAD, but two registrations are manual:

```python
# opentad/models/backbones/__init__.py
from .videomae_dsta import VisionTransformerDSTA
# (and add "VisionTransformerDSTA" to __all__)

# opentad/models/losses/__init__.py — register AsymmetricLoss (see src/models/losses.py)
```

The configs reference these by `type="VisionTransformerDSTA"` and
`type="AsymmetricLoss"`, so the registry must know them before training.
