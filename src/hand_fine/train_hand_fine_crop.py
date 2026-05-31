"""
train_hand_fine.py
Training script for fine-grained hand fusion model (34 classes).
RGB (VideoMAE frozen backbone) + Skeleton (MMN, joint modality).

Usage:
    python src/hand_fine/train_hand_fine.py \
        --videomae_path models/videomae-ssv2 \
        --output_dir outputs/hand_fine_fusion \
        --augment

Notes:
  - VideoMAE backbone is FROZEN throughout training.
  - Only skeleton branch + fusion head are trained.
  - Batch 8 to stay within 10 GB VRAM.
  - RGB frames: 8 frames, 224x224, VideoMAE normalisation.
  - Skeleton: all 17 joints, joint modality, T=64.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score

# Add src to path for local imports
SRC = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, SRC)

from src.hand_fine.hand_fine_dataset import HandFineDataset
from src.hand_fine.hand_fine_model   import build_hand_fusion_model

NUM_CLASSES = 34
LABEL_NAMES = {
    **{i:   f'C{i+1}'   for i in range(0,  13)},
    **{i:   f'E{i-12}'  for i in range(13, 19)},
    **{i:   f'F{i-18}'  for i in range(19, 29)},
    **{i:   f'G{i-28}'  for i in range(29, 33)},
    33: 'no-hand',
}


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    kp_train = 'data/keypoints/train'
    kp_val   = 'data/keypoints/val'

    train_ds = HandFineDataset(
        csv_path      = args.train_csv,
        kp_dir        = kp_train,
        joint_indices = list(range(17)),
        augment       = args.augment,
        rgb_augment   = args.augment,
    )
    val_ds = HandFineDataset(
        csv_path      = args.val_csv,
        kp_dir        = kp_val,
        joint_indices = list(range(17)),
        augment       = False,
        rgb_augment   = False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Class weights ─────────────────────────────────────────────────────────
    labels  = pd.read_csv(args.train_csv)['local_label'].values
    counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = torch.tensor(
        len(labels) / (NUM_CLASSES * np.maximum(counts, 1)),
        dtype=torch.float32,
    ).to(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_hand_fusion_model(
        videomae_path = args.videomae_path,
        num_classes   = NUM_CLASSES,
        num_joints    = 17,
        num_frames    = 64,
    ).to(device)

    # Only train skeleton branch + fusion (RGB backbone is frozen)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_total   = sum(p.numel() for p in model.parameters())
    n_train   = sum(p.numel() for p in trainable)
    print(f"Params — total: {n_total:,}  trainable: {n_train:,}  "
          f"({100*n_train/n_total:.1f}%)")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)

    # 20-epoch linear warmup + cosine annealing (3 cycles)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=20)
    cosine = CosineAnnealingLR(optimizer,
                               T_max=(args.epochs - 20) // 3,
                               eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[20])

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1   = 0.0
    best_path = os.path.join(args.output_dir, 'best_model.pt')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for rgb, skel, labels_b in train_loader:
            # VideoMAE expects (B, T, C, H, W) — our loader gives (B, C, T, H, W)
            # Permute: (B,3,T,H,W) → (B,T,3,H,W)
            rgb    = rgb.permute(0, 2, 1, 3, 4).to(device)
            skel   = skel.to(device)
            labels_b = labels_b.to(device)

            optimizer.zero_grad()
            out  = model(rgb, skel)
            loss = criterion(out, labels_b)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for rgb, skel, labels_b in val_loader:
                rgb  = rgb.permute(0, 2, 1, 3, 4).to(device)
                skel = skel.to(device)
                preds = model(rgb, skel).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(labels_b.numpy())

        f1      = f1_score(all_true, all_preds, average=None, labels=list(range(NUM_CLASSES)),
                           zero_division=0)
        f1_mean = float(np.mean(f1))
        lr_now  = scheduler.get_last_lr()[0]

        # Per-group F1
        f1_C = np.mean(f1[0:13]);  f1_E = np.mean(f1[13:19])
        f1_F = np.mean(f1[19:29]); f1_G = np.mean(f1[29:33])
        f1_N = f1[33]

        print(f"Ep {epoch:3d}/{args.epochs} | "
              f"loss={total_loss/len(train_loader):.4f} | "
              f"F1-mean={f1_mean:.4f} | "
              f"C={f1_C:.4f} E={f1_E:.4f} F={f1_F:.4f} G={f1_G:.4f} N={f1_N:.4f} | "
              f"lr={lr_now:.6f}")

        if f1_mean > best_f1:
            best_f1 = f1_mean
            torch.save(model.state_dict(), best_path)
            print(f"  → Best F1-mean: {best_f1:.4f}  (saved)")

    print(f"\nBest F1-mean: {best_f1:.4f}")
    print(f"Model: {best_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--videomae_path', default='models/videomae-ssv2')
    p.add_argument('--output_dir',    required=True)
    p.add_argument('--train_csv', default='data/hand_dataset/hand_fine_upperbody_train.csv')
    p.add_argument('--val_csv',   default='data/hand_dataset/hand_fine_upperbody_val.csv')
    p.add_argument('--augment',       action='store_true')
    p.add_argument('--epochs',        type=int,   default=80)
    p.add_argument('--batch_size',    type=int,   default=8)   # 10 GB VRAM limit
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=0.1)
    args = p.parse_args()
    train(args)
