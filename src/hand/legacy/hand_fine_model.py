"""
hand_fine_model.py  (v3 — correct MMN_ kwargs + self.head not self.fc)

MMN_ signature: MMN_(**kwargs) → passes to MMN(...)
MMN class uses: self.head = nn.Linear(channels[-1], num_classes)
                NOT self.fc
MMN embed_dim hardcoded to 96; channels[-1] = 96 (default channels=(96,96,96,96))
So self.head.in_features = 96.
"""

import sys
import os
import torch
import torch.nn as nn
from transformers import VideoMAEModel

MMN_ROOT = os.path.join(os.path.dirname(__file__), '..', 'skeleton', 'MMN')
sys.path.insert(0, MMN_ROOT)
from model.MMN import MMN_  # noqa: E402

MMN_FEAT_DIM = 96   # channels[-1] default, always 96


def _build_mmn(in_channels, num_classes, num_joints, num_frames):
    """
    Call MMN_ with keyword args only (it's a **kwargs factory).
    Matches exactly what train_mmn_generic_v2.py does.
    DO NOT pass embed_dim — it is hardcoded to 96 inside MMN_.
    """
    return MMN_(
        in_channels=in_channels,
        num_classes=num_classes,
        num_people=1,
        num_frames=num_frames,
        num_points=num_joints,
        kernel_size=3,
        num_heads=3,
        head_drop=0.0,
        drop=0.0,
        drop_path=0.1,
    )


# ── RGB branch ────────────────────────────────────────────────────────────────

class RGBBranch(nn.Module):
    """VideoMAE backbone (frozen) + trainable projection head."""

    def __init__(self, videomae_path: str, out_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.videomae = VideoMAEModel.from_pretrained(videomae_path)
        for p in self.videomae.parameters():
            p.requires_grad = False

        hidden = self.videomae.config.hidden_size   # 768 for base
        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, pixel_values):
        """
        pixel_values: (B, T, C, H, W) — T=16, C=3, H=W=224.
        Training script must permute (B,3,T,H,W) → (B,T,3,H,W) before calling.
        """
        with torch.no_grad():
            out = self.videomae(pixel_values=pixel_values)
        feat = out.last_hidden_state.mean(dim=1)   # (B, 768)
        return self.proj(feat)


# ── Skeleton branch ───────────────────────────────────────────────────────────

class SkeletonBranch(nn.Module):
    """
    MMN backbone with classification head replaced by Identity.
    MMN uses self.head (not self.fc) — Linear(96, num_classes).
    We replace it with Identity and add our own projection.
    """

    def __init__(self, num_joints: int, num_frames: int,
                 in_channels: int = 2,
                 out_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        # Build MMN with num_classes=MMN_FEAT_DIM so head is Linear(96,96)
        # then we replace head with Identity — clean 96-dim passthrough
        self.mmn = _build_mmn(in_channels, MMN_FEAT_DIM, num_joints, num_frames)

        # Confirm it's .head, not .fc
        assert hasattr(self.mmn, 'head'), \
            "MMN has no attribute 'head' — check MMN source"
        assert self.mmn.head.in_features == MMN_FEAT_DIM, \
            f"Expected head.in_features={MMN_FEAT_DIM}, got {self.mmn.head.in_features}"

        self.mmn.head = nn.Identity()

        self.proj = nn.Sequential(
            nn.Linear(MMN_FEAT_DIM, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, C, T, V, M) → (B, out_dim)"""
        feat = self.mmn(x, torch.arange(x.shape[2], device=x.device))  # (B, 96)
        return self.proj(feat)


# ── Fusion model ──────────────────────────────────────────────────────────────

class HandFusionModel(nn.Module):
    def __init__(self,
                 videomae_path: str,
                 num_classes:   int   = 34,
                 num_joints:    int   = 17,
                 num_frames:    int   = 64,
                 in_channels:   int   = 2,
                 rgb_dim:       int   = 512,
                 skel_dim:      int   = 256,
                 fusion_dim:    int   = 256,
                 dropout:       float = 0.3):
        super().__init__()
        self.rgb_branch  = RGBBranch(videomae_path, out_dim=rgb_dim, dropout=dropout)
        self.skel_branch = SkeletonBranch(num_joints, num_frames,
                                          in_channels=in_channels,
                                          out_dim=skel_dim, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(rgb_dim + skel_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, rgb, skel):
        """
        rgb:  (B, T, C, H, W)     — 8 frames, 224×224
        skel: (B, C, T, V, M)     — joint tensor
        """
        f_rgb  = self.rgb_branch(rgb)
        f_skel = self.skel_branch(skel)
        return self.fusion(torch.cat([f_rgb, f_skel], dim=1))


def build_hand_fusion_model(videomae_path, num_classes=34,
                             num_joints=17, num_frames=64):
    return HandFusionModel(
        videomae_path=videomae_path,
        num_classes=num_classes,
        num_joints=num_joints,
        num_frames=num_frames,
        in_channels=2,
        rgb_dim=512,
        skel_dim=256,
        fusion_dim=256,
        dropout=0.3,
    )
