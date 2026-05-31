"""
src/skeleton/train_mmn.py

Trains MMN on 5-class hand-movement labels using 17-joint COCO keypoints.

Classes:
    0  C  upper limb
    1  E  body-hand
    2  F  head-hand
    3  G  leg-hand
    4  no hand movement

Usage:
    python src/skeleton/train_mmn.py --mode full   # Run A: all 17 joints
    python src/skeleton/train_mmn.py --mode arm    # Run B: 6 arm joints only
"""

import os
import sys
import argparse
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

# Add MMN to path
sys.path.insert(0, str(Path(__file__).parent / "MMN"))
from model.MMN import MMN_

from features import extract_features

# ── Config ────────────────────────────────────────────────────────────────────

BASE       = Path("/home/woody/iwso/iwso226h/ma52")
DATA_DIR   = BASE / "data/skeleton_dataset"
OUTPUT_DIR = BASE / "outputs"

NUM_CLASSES  = 5
NUM_FRAMES   = 64
BATCH_SIZE   = 32
NUM_WORKERS  = 4
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EMBED_DIM    = 96

LABEL_NAMES = {
    0: "C upper limb",
    1: "E body-hand",
    2: "F head-hand",
    3: "G leg-hand",
    4: "no hand move",
}

# ── Dataset ───────────────────────────────────────────────────────────────────

class HandDataset(Dataset):
    def __init__(self, csv_path: str, mode: str = "full", num_frames: int = NUM_FRAMES):
        self.df        = pd.read_csv(csv_path)
        self.mode      = mode
        self.num_frames = num_frames
        self.num_joints = 17 if mode == "full" else 6

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["hand_label"])

        tensor = extract_features(row["kp_path"], mode=self.mode,
                                  num_frames=self.num_frames)  # (2, T, V, 1)

        # index_t: frame position indices for sinusoidal temporal encoding
        index_t = torch.arange(self.num_frames, dtype=torch.long)

        return torch.from_numpy(tensor), index_t, label


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(csv_path: str, num_classes: int, device: torch.device):
    df = pd.read_csv(csv_path)
    counts = Counter(df["hand_label"].tolist())
    total  = sum(counts.values())
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        weights[c] = total / (num_classes * max(counts.get(c, 1), 1))
    return weights.to(device)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds):
    f1_macro = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    f1_mean  = (f1_macro + f1_micro) / 2
    acc      = (np.array(all_preds) == np.array(all_labels)).mean()
    return acc, f1_macro, f1_micro, f1_mean

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

        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds  += preds
        all_labels += labels.cpu().tolist()

    avg_loss = total_loss / len(loader.dataset)
    acc, f1_macro, f1_micro, f1_mean = compute_metrics(all_labels, all_preds)
    return avg_loss, acc, f1_macro, f1_micro, f1_mean, all_labels, all_preds

# ── Per-class F1 report ───────────────────────────────────────────────────────

def print_per_class_f1(all_labels, all_preds):
    per_class = f1_score(all_labels, all_preds, average=None,
                         labels=list(range(NUM_CLASSES)), zero_division=0)
    print("  Per-class F1:")
    for c, f1 in enumerate(per_class):
        count = all_labels.count(c)
        print(f"    {c} {LABEL_NAMES[c]:<20}  F1={f1:.4f}  ({count} samples)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "arm"], default="full",
                        help="full=17 joints, arm=6 joints")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--num_frames", type=int,   default=NUM_FRAMES)
    args = parser.parse_args()

    run_name  = f"hand_skeleton_{args.mode}"
    out_dir   = OUTPUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Mode   : {args.mode}")
    print(f"Run    : {run_name}")

    # ── Data ──────────────────────────────────────────────────────
    train_csv = DATA_DIR / "hand_train.csv"
    val_csv   = DATA_DIR / "hand_val.csv"

    train_ds = HandDataset(train_csv, mode=args.mode, num_frames=args.num_frames)
    val_ds   = HandDataset(val_csv,   mode=args.mode, num_frames=args.num_frames)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    print(f"Train  : {len(train_ds)} samples")
    print(f"Val    : {len(val_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────
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
    print(f"Params : {n_params/1e6:.2f}M")

    # ── Loss, optimiser, scheduler ────────────────────────────────
    class_weights = compute_class_weights(train_csv, NUM_CLASSES, device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────
    best_f1_mean = 0.0
    log = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, acc, f1_macro, f1_micro, f1_mean, all_labels, all_preds = \
            eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        improved = f1_mean > best_f1_mean
        if improved:
            best_f1_mean = f1_mean
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        marker = " ★" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={acc:.4f}  f1-macro={f1_macro:.4f}  "
            f"f1-micro={f1_micro:.4f}  f1-mean={f1_mean:.4f}{marker}"
        )

        if improved:
            print_per_class_f1(all_labels, all_preds)

        log.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "acc": acc, "f1_macro": f1_macro, "f1_micro": f1_micro,
            "f1_mean": f1_mean,
        })

    # ── Save log ──────────────────────────────────────────────────
    pd.DataFrame(log).to_csv(out_dir / "training_log.csv", index=False)
    print(f"\nBest F1-mean : {best_f1_mean:.4f}")
    print(f"Model saved  : {out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
