"""
src/skeleton/train_mmn_v2.py

Trains MMN on 5-class hand-movement labels.
Hyperparameters corrected from MMN paper:
  - LR: 1e-4 (not 1e-3)
  - Weight decay: 0.1 (not 1e-4)
  - Batch size: 256
  - Scheduler: cosine annealing with warmup (20 epochs linear warmup)
  - STCA augmentation: skeletal (rotation/scale/translate) + temporal jitter

Usage:
    python src/skeleton/train_mmn_v2.py --mode full   # Run C: all 17 joints
    python src/skeleton/train_mmn_v2.py --mode arm    # Run D: 6 arm joints only
"""

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent / "MMN"))
from model.MMN import MMN_
from features import extract_features

# ── Config ────────────────────────────────────────────────────────────────────

BASE     = Path("/home/woody/iwso/iwso226h/ma52")
DATA_DIR = BASE / "data/skeleton_dataset"
OUT_DIR  = BASE / "outputs"

NUM_CLASSES  = 5
NUM_FRAMES   = 64
BATCH_SIZE   = 32
NUM_WORKERS  = 8
EPOCHS       = 80          # more epochs to let warmup + cosine work properly
LR           = 1e-4        # paper value
LR_MIN       = 1e-6        # paper value
WEIGHT_DECAY = 0.1         # paper value
WARMUP_EPOCHS = 20         # paper value
COSINE_CYCLES = 3          # paper value
EMBED_DIM    = 96

LABEL_NAMES = {
    0: "C upper limb",
    1: "E body-hand",
    2: "F head-hand",
    3: "G leg-hand",
    4: "no hand move",
}

# ── STCA Augmentation ─────────────────────────────────────────────────────────

def stca_augment(kps: np.ndarray, training: bool = True) -> np.ndarray:
    """
    Skeletal-Temporal Context-aware Augmentation (STCA) from MMN paper.
    kps: (T, V, 2)  — x,y only (no conf at this stage)
    Returns: (T, V, 2)
    """
    if not training:
        return kps

    kps = kps.copy()
    T = kps.shape[0]

    # ── Skeletal augmentation: random rotation, scale, translate ──
    theta = np.random.uniform(-15, 15) * np.pi / 180
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    scale = np.random.uniform(0.9, 1.1)
    trans = np.random.uniform(-0.1, 0.1, size=(2,)).astype(np.float32)

    # Apply same transform to all frames (broadcast over joints)
    xy = kps.reshape(-1, 2)           # (T*V, 2)
    xy = (xy @ R.T) * scale + trans
    kps = xy.reshape(kps.shape)

    # ── Temporal augmentation: jitter frame indices ──
    delta = np.random.randint(-3, 4, size=(T,))   # U(-3,3)
    new_idx = np.clip(np.arange(T) + delta, 0, T - 1)
    kps = kps[new_idx]

    return kps


# ── Dataset ───────────────────────────────────────────────────────────────────

class HandDataset(Dataset):
    def __init__(self, csv_path: str, mode: str = "full",
                 num_frames: int = NUM_FRAMES, training: bool = True):
        self.df         = pd.read_csv(csv_path)
        self.mode       = mode
        self.num_frames = num_frames
        self.training   = training
        self.num_joints = 17 if mode == "full" else 6

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["hand_label"])

        # Get raw (2, T, V, 1) tensor from features.py
        tensor = extract_features(row["kp_path"], mode=self.mode,
                                  num_frames=self.num_frames)  # (2, T, V, 1)

        # Apply STCA on x,y array before returning
        if self.training:
            xy = tensor[:, :, :, 0].transpose(1, 2, 0)   # (T, V, 2)
            xy = stca_augment(xy, training=True)
            tensor = xy.transpose(2, 0, 1)[:, :, :, None]  # (2, T, V, 1)

        index_t = torch.arange(self.num_frames, dtype=torch.long)
        return torch.from_numpy(tensor.astype(np.float32)), index_t, label


# ── LR Scheduler: linear warmup + cosine annealing with restarts ─────────────

def build_scheduler(optimizer, warmup_epochs, total_epochs, cycles, lr_min, lr_base):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from 1e-7 to lr_base
            return (1e-7 + (lr_base - 1e-7) * epoch / warmup_epochs) / lr_base
        # Cosine annealing with `cycles` restarts
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cycle_progress = (progress * cycles) % 1.0
        cosine = 0.5 * (1 + math.cos(math.pi * cycle_progress))
        ratio  = lr_min / lr_base
        return ratio + (1 - ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(csv_path, num_classes, device):
    df     = pd.read_csv(csv_path)
    counts = Counter(df["hand_label"].tolist())
    total  = sum(counts.values())
    w = torch.zeros(num_classes)
    for c in range(num_classes):
        w[c] = total / (num_classes * max(counts.get(c, 1), 1))
    return w.to(device)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds):
    f1_macro = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    f1_mean  = (f1_macro + f1_micro) / 2
    acc      = (np.array(all_preds) == np.array(all_labels)).mean()
    return acc, f1_macro, f1_micro, f1_mean

def print_per_class_f1(all_labels, all_preds):
    per_class = f1_score(all_labels, all_preds, average=None,
                         labels=list(range(NUM_CLASSES)), zero_division=0)
    print("  Per-class F1:")
    for c, f1 in enumerate(per_class):
        count = all_labels.count(c)
        print(f"    {c} {LABEL_NAMES[c]:<20}  F1={f1:.4f}  ({count} samples)")


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for tensor, index_t, labels in loader:
        tensor  = tensor.to(device)
        index_t = index_t.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        logits = model(tensor, index_t)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for tensor, index_t, labels in loader:
        tensor  = tensor.to(device)
        index_t = index_t.to(device)
        labels  = labels.to(device)

        logits = model(tensor, index_t)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds  += logits.argmax(dim=1).cpu().tolist()
        all_labels += labels.cpu().tolist()

    avg_loss = total_loss / len(loader.dataset)
    acc, f1_macro, f1_micro, f1_mean = compute_metrics(all_labels, all_preds)
    return avg_loss, acc, f1_macro, f1_micro, f1_mean, all_labels, all_preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["full", "arm"], default="full")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--num_frames", type=int,   default=NUM_FRAMES)
    args = parser.parse_args()

    run_name = f"hand_skeleton_{args.mode}_v2"
    out_dir  = OUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Mode       : {args.mode}")
    print(f"Run        : {run_name}")
    print(f"LR         : {args.lr}  |  WD: {WEIGHT_DECAY}  |  BS: {args.batch_size}")

    train_csv = DATA_DIR / "hand_train.csv"
    val_csv   = DATA_DIR / "hand_val.csv"

    train_ds = HandDataset(train_csv, mode=args.mode,
                           num_frames=args.num_frames, training=True)
    val_ds   = HandDataset(val_csv,   mode=args.mode,
                           num_frames=args.num_frames, training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    print(f"Train      : {len(train_ds)} samples")
    print(f"Val        : {len(val_ds)} samples")

    num_points = 17 if args.mode == "full" else 6

    model = MMN_(
        in_channels=2,
        num_classes=NUM_CLASSES,
        num_people=1,
        num_frames=args.num_frames,
        num_points=num_points,
        kernel_size=3,
        num_heads=4,
        head_drop=0.2,
        drop=0.1,
        drop_path=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params     : {n_params/1e6:.2f}M")

    class_weights = compute_class_weights(train_csv, NUM_CLASSES, device)
    print(f"CLS weights: {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, args.epochs,
                                COSINE_CYCLES, LR_MIN, args.lr)

    best_f1_mean = 0.0
    log = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, acc, f1_macro, f1_micro, f1_mean, all_labels, all_preds = \
            eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        improved   = f1_mean > best_f1_mean
        if improved:
            best_f1_mean = f1_mean
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        marker = " ★" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"lr={current_lr:.2e}  loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  acc={acc:.4f}  "
            f"f1-macro={f1_macro:.4f}  f1-micro={f1_micro:.4f}  "
            f"f1-mean={f1_mean:.4f}{marker}"
        )
        if improved:
            print_per_class_f1(all_labels, all_preds)

        log.append({
            "epoch": epoch, "lr": current_lr,
            "train_loss": train_loss, "val_loss": val_loss,
            "acc": acc, "f1_macro": f1_macro,
            "f1_micro": f1_micro, "f1_mean": f1_mean,
        })

    pd.DataFrame(log).to_csv(out_dir / "training_log.csv", index=False)
    print(f"\nBest F1-mean : {best_f1_mean:.4f}")
    print(f"Model saved  : {out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
