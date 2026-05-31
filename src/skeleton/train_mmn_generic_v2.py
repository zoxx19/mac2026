"""
src/skeleton/train_mmn_generic.py

Generic MMN trainer — works for hand (5 classes), leg (9 classes), body (6 classes).
Supports joint and bone modalities, with or without STCA augmentation.

Usage:
    # Joint modality with STCA (paper setup)
    python train_mmn_generic.py --track hand --mode full  --num_classes 5 --modality joint --augment

    # Joint modality no augmentation (baseline)
    python train_mmn_generic.py --track leg  --mode full  --num_classes 9 --modality joint

    # Bone modality
    python train_mmn_generic.py --track hand --mode full  --num_classes 5 --modality bone --augment

Joint modes:
    full  = all 17 COCO joints
    arm   = shoulders+elbows+wrists      (6 joints)
    leg   = hips+knees+ankles            (6 joints)
    torso = shoulders+hips               (4 joints)
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
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent / "MMN"))
from model.MMN import MMN_
from features_modality import extract_features, MODE_JOINTS, MODE_EDGES

BASE     = Path("/home/woody/iwso/iwso226h/ma52")
DATA_DIR = BASE / "data/skeleton_dataset"
OUT_DIR  = BASE / "outputs"

NUM_FRAMES    = 64
BATCH_SIZE    = 32
NUM_WORKERS   = 8
EPOCHS        = 80
LR            = 1e-4
LR_MIN        = 1e-6
WEIGHT_DECAY  = 0.1
WARMUP_EPOCHS = 20
COSINE_CYCLES = 3

LABEL_COL = {
    "hand": "hand_label",
    "leg":  "leg_label",
    "body": "body_label",
}

# ── STCA Augmentation ─────────────────────────────────────────────────────────

def stca_augment(kps: np.ndarray) -> np.ndarray:
    """kps: (T, V, 2) -> (T, V, 2)"""
    kps   = kps.copy()
    T     = kps.shape[0]
    theta = np.random.uniform(-15, 15) * np.pi / 180
    c, s  = np.cos(theta), np.sin(theta)
    R     = np.array([[c, -s], [s, c]], dtype=np.float32)
    scale = np.random.uniform(0.9, 1.1)
    trans = np.random.uniform(-0.1, 0.1, size=(2,)).astype(np.float32)
    xy    = kps.reshape(-1, 2)
    xy    = (xy @ R.T) * scale + trans
    kps   = xy.reshape(kps.shape)
    delta = np.random.randint(-3, 4, size=(T,))
    return kps[np.clip(np.arange(T) + delta, 0, T - 1)]


# ── Dataset ───────────────────────────────────────────────────────────────────

class SkeletonDataset(Dataset):
    def __init__(self, csv_path, track, mode="full", modality="joint",
                 num_frames=NUM_FRAMES, training=True, augment=False):
        self.df         = pd.read_csv(csv_path)
        self.track      = track
        self.mode       = mode
        self.modality   = modality
        self.label_col  = LABEL_COL[track]
        self.num_frames = num_frames
        self.training   = training
        self.augment    = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row[self.label_col])

        tensor = extract_features(
            row["kp_path"],
            mode=self.mode,
            modality=self.modality,
            num_frames=self.num_frames
        )  # (2,T,V,1) or (3,T,E,1)

        # Apply STCA only on joint modality (x,y)
        if self.training and self.augment and self.modality == "joint":
            xy     = tensor[:, :, :, 0].transpose(1, 2, 0)   # (T, V, 2)
            xy     = stca_augment(xy)
            tensor = xy.transpose(2, 0, 1)[:, :, :, None]

        index_t = torch.arange(self.num_frames, dtype=torch.long)
        return torch.from_numpy(tensor.astype(np.float32)), index_t, label


# ── LR scheduler ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, warmup_epochs, total_epochs, cycles,
                    lr_min, lr_base, use_warmup=True):
    if not use_warmup:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=lr_min)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (1e-7 + (lr_base - 1e-7) * epoch / warmup_epochs) / lr_base
        progress       = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cycle_progress = (progress * cycles) % 1.0
        cosine         = 0.5 * (1 + math.cos(math.pi * cycle_progress))
        ratio          = lr_min / lr_base
        return ratio + (1 - ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(csv_path, label_col, num_classes, device):
    df     = pd.read_csv(csv_path)
    counts = Counter(df[label_col].tolist())
    total  = sum(counts.values())
    w = torch.zeros(num_classes)
    for c in range(num_classes):
        w[c] = total / (num_classes * max(counts.get(c, 1), 1))
    return w.to(device)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds, num_classes):
    f1_macro = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    f1_mean  = (f1_macro + f1_micro) / 2
    acc      = (np.array(all_preds) == np.array(all_labels)).mean()
    per_cls  = f1_score(all_labels, all_preds, average=None,
                        labels=list(range(num_classes)), zero_division=0)
    return acc, f1_macro, f1_micro, f1_mean, per_cls


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for tensor, index_t, labels in loader:
        tensor  = tensor.to(device)
        index_t = index_t.to(device)
        labels  = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(tensor, index_t), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []
    for tensor, index_t, labels in loader:
        tensor  = tensor.to(device)
        index_t = index_t.to(device)
        labels  = labels.to(device)
        logits  = model(tensor, index_t)
        total_loss += criterion(logits, labels).item() * len(labels)
        all_preds  += logits.argmax(dim=1).cpu().tolist()
        all_labels += labels.cpu().tolist()
    avg_loss = total_loss / len(loader.dataset)
    acc, f1_macro, f1_micro, f1_mean, per_cls = \
        compute_metrics(all_labels, all_preds, num_classes)
    return avg_loss, acc, f1_macro, f1_micro, f1_mean, per_cls, all_labels, all_preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track",       choices=["hand","leg","body"], required=True)
    parser.add_argument("--mode",        choices=["full","arm","leg","torso"], default="full")
    parser.add_argument("--modality",    choices=["joint","bone"], default="joint")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--augment",     action="store_true", help="Enable STCA augmentation")
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--num_frames",  type=int,   default=NUM_FRAMES)
    args = parser.parse_args()

    # in_channels: 2 for joint, 3 for bone
    in_channels = 2 if args.modality == "joint" else 3

    # num_points: joints for joint modality, edges for bone modality
    if args.modality == "joint":
        num_points = len(MODE_JOINTS[args.mode])
    else:
        num_points = len(MODE_EDGES[args.mode])

    aug_tag  = "_aug" if args.augment else ""
    run_name = f"{args.track}_{args.mode}_{args.modality}{aug_tag}"
    out_dir  = OUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Track      : {args.track}  |  Mode: {args.mode}  |  "
          f"Modality: {args.modality}  |  Augment: {args.augment}")
    print(f"Run        : {run_name}")
    print(f"in_channels: {in_channels}  |  num_points: {num_points}")

    train_csv = DATA_DIR / f"{args.track}_train.csv"
    val_csv   = DATA_DIR / f"{args.track}_val.csv"
    label_col = LABEL_COL[args.track]

    train_ds = SkeletonDataset(train_csv, args.track, args.mode, args.modality,
                               args.num_frames, training=True,  augment=args.augment)
    val_ds   = SkeletonDataset(val_csv,   args.track, args.mode, args.modality,
                               args.num_frames, training=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    print(f"Train      : {len(train_ds)} | Val: {len(val_ds)} | "
          f"Joints/Bones: {num_points}")

    model = MMN_(
        in_channels=in_channels,
        num_classes=args.num_classes,
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

    class_weights = compute_class_weights(train_csv, label_col,
                                          args.num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, args.epochs,
                                COSINE_CYCLES, LR_MIN, args.lr,
                                use_warmup=args.augment)

    best_f1_mean = 0.0
    log = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, acc, f1_macro, f1_micro, f1_mean, per_cls, all_labels, all_preds = \
            eval_epoch(model, val_loader, criterion, device, args.num_classes)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        improved   = f1_mean > best_f1_mean
        if improved:
            best_f1_mean = f1_mean
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        marker = " ★" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs}  lr={current_lr:.2e}  "
            f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={acc:.4f}  f1-macro={f1_macro:.4f}  "
            f"f1-micro={f1_micro:.4f}  f1-mean={f1_mean:.4f}{marker}"
        )
        if improved:
            print(f"  Per-class F1: {np.round(per_cls, 4).tolist()}")

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