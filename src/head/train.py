"""
train.py v3
Fine-tunes VideoMAE or TimeSformer on head+neck crops for B1-B7 recognition.

Changes from v2:
    - Dropout on classification head to reduce overfitting
    - Stronger augmentation: random crop, temporal jitter, stronger color jitter
    - Early stopping: stops if val F1 doesn't improve for patience epochs
    - No weighted loss (Run 1 without it was best)
    - Fewer Stage 2 epochs (10) to prevent overfitting

Labels:
    B1: nodding
    B2: shaking head
    B3: turning head
    B4: tilting head
    B5: bowing head
    B6: head up
    B7: no head movement
"""

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    VideoMAEForVideoClassification,
    TimesformerForVideoClassification,
    VideoMAEImageProcessor,
    AutoImageProcessor,
)
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL2ID = {"B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4, "B6": 5, "B7": 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_FRAMES  = 16
IMG_SIZE    = 224
NUM_CLASSES = 7


# ── Dataset ───────────────────────────────────────────────────────────────────
class HeadClipDataset(Dataset):
    def __init__(self, csv_path, processor, is_train=True):
        self.df        = pd.read_csv(csv_path)
        self.processor = processor
        self.is_train  = is_train

        self.df = self.df[self.df["crop_path"].apply(
            lambda p: Path(p).exists()
        )].reset_index(drop=True)

        print(f"  Loaded {len(self.df)} samples from {csv_path}")
        print(f"  Class distribution:")
        print(self.df["head_label"].value_counts().sort_index().to_string())

    def __len__(self):
        return len(self.df)

    def _load_frames(self, video_path):
        cap   = cv2.VideoCapture(str(video_path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

        # temporal jitter: randomly shift the sampling window
        if self.is_train and total > NUM_FRAMES:
            max_offset = total - NUM_FRAMES
            offset = np.random.randint(0, max_offset + 1)
            indices = np.linspace(offset, offset + NUM_FRAMES - 1, NUM_FRAMES, dtype=int)
            indices = np.clip(indices, 0, total - 1)
        else:
            indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frames.append(frames[-1].copy() if frames else
                               np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].copy())
        return frames[:NUM_FRAMES]

    def _augment(self, frames):
        h, w = frames[0].shape[:2]

        # random horizontal flip
        if np.random.random() > 0.5:
            frames = [np.fliplr(f) for f in frames]

        # random crop (crop to 90-100% of size then resize back)
        if np.random.random() > 0.3:
            scale  = np.random.uniform(0.85, 1.0)
            new_h  = int(h * scale)
            new_w  = int(w * scale)
            top    = np.random.randint(0, h - new_h + 1)
            left   = np.random.randint(0, w - new_w + 1)
            frames = [cv2.resize(f[top:top+new_h, left:left+new_w], (w, h))
                      for f in frames]

        # stronger color jitter
        alpha = np.random.uniform(0.7, 1.3)   # contrast
        beta  = np.random.randint(-30, 30)     # brightness
        gamma = np.random.uniform(0.8, 1.2)   # per-channel scale
        frames = [
            np.clip(f.astype(np.float32) * alpha * gamma + beta, 0, 255).astype(np.uint8)
            for f in frames
        ]

        # random grayscale (helps model not rely on color)
        if np.random.random() > 0.8:
            frames = [
                np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)] * 3, axis=-1)
                for f in frames
            ]

        return frames

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        video_path = row["crop_path"]
        label      = LABEL2ID[row["head_label"]]

        frames = self._load_frames(video_path)
        if self.is_train:
            frames = self._augment(frames)

        inputs       = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return pixel_values, torch.tensor(label, dtype=torch.long)


# ── Model with dropout ────────────────────────────────────────────────────────
def add_dropout_to_classifier(model, dropout_rate=0.3, model_type="videomae"):
    """Replace the classifier head with one that includes dropout."""
    if model_type == "videomae":
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, NUM_CLASSES)
        )
    elif model_type == "timesformer":
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, NUM_CLASSES)
        )
    return model


# ── Training functions ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, epoch, criterion, model_type):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for pixel_values, labels in pbar:
        pixel_values = pixel_values.to(device)
        labels       = labels.to(device)

        optimizer.zero_grad()

        if model_type == "timesformer":
            outputs = model(pixel_values=pixel_values)
        else:
            outputs = model(pixel_values=pixel_values)

        loss = criterion(outputs.logits, labels)
        loss.backward()

        # gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        preds       = outputs.logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        pbar.set_postfix({
            "loss": f"{total_loss / (total / loader.batch_size):.4f}",
            "acc":  f"{correct / total:.4f}"
        })

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device, epoch, criterion):
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Val   Epoch {epoch}")
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device)
            labels       = labels.to(device)

            outputs     = model(pixel_values=pixel_values)
            loss        = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc     = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_mac  = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    f1_mic  = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    f1_mean = (f1_mac + f1_mic) / 2

    print(f"\n  Val Loss: {total_loss / len(loader):.4f} | "
          f"Acc: {acc:.4f} | F1-macro: {f1_mac:.4f} | "
          f"F1-micro: {f1_mic:.4f} | F1-mean: {f1_mean:.4f}")

    print("\n  Per-class report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)],
        zero_division=0
    ))

    return total_loss / len(loader), acc, f1_mean


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",     required=True)
    parser.add_argument("--val_csv",       required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--model_name",    default="MCG-NJU/videomae-base-finetuned-ssv2")
    parser.add_argument("--model_type",    default="videomae", choices=["videomae", "timesformer"])
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--stage1_epochs", type=int,   default=10)
    parser.add_argument("--stage2_epochs", type=int,   default=10)
    parser.add_argument("--lr_stage1",     type=float, default=1e-3)
    parser.add_argument("--lr_stage2",     type=float, default=1e-5)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--patience",      type=int,   default=5)
    parser.add_argument("--num_workers",   type=int,   default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Model: {args.model_name} ({args.model_type})")
    print(f"Dropout: {args.dropout} | Patience: {args.patience}")

    # ── Load processor and model ──────────────────────────────────────────────
    print(f"\nLoading model...")
    processor = AutoImageProcessor.from_pretrained(args.model_name)

    if args.model_type == "timesformer":
        model = TimesformerForVideoClassification.from_pretrained(
            args.model_name,
            num_labels=NUM_CLASSES,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    else:
        model = VideoMAEForVideoClassification.from_pretrained(
            args.model_name,
            num_labels=NUM_CLASSES,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

    # add dropout to classifier head
    model = add_dropout_to_classifier(model, args.dropout, args.model_type)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded ✅ ({total_params:.1f}M parameters)")

    criterion = nn.CrossEntropyLoss()

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = HeadClipDataset(args.train_csv, processor, is_train=True)
    val_dataset   = HeadClipDataset(args.val_csv,   processor, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    best_f1       = 0.0
    best_epoch    = 0
    no_improve    = 0

    # ── Stage 1: freeze backbone ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 1: Head only ({args.stage1_epochs} epochs) | LR={args.lr_stage1}")
    print(f"{'='*60}")

    if args.model_type == "timesformer":
        for param in model.timesformer.parameters():
            param.requires_grad = False
    else:
        for param in model.videomae.parameters():
            param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_stage1, weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.stage1_epochs)

    for epoch in range(1, args.stage1_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch,
                        criterion, args.model_type)
        _, _, val_f1 = evaluate(model, val_loader, device, epoch, criterion)
        scheduler.step()

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✅ Best model saved (F1-mean: {best_f1:.4f})")
        else:
            no_improve += 1

    # ── Stage 2: full fine-tuning ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 2: Full fine-tuning ({args.stage2_epochs} epochs) | LR={args.lr_stage2}")
    print(f"{'='*60}")

    if args.model_type == "timesformer":
        for param in model.timesformer.parameters():
            param.requires_grad = True
    else:
        for param in model.videomae.parameters():
            param.requires_grad = True

    optimizer = AdamW(
        model.parameters(), lr=args.lr_stage2, weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.stage2_epochs)
    no_improve = 0

    for epoch in range(args.stage1_epochs + 1,
                       args.stage1_epochs + args.stage2_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch,
                        criterion, args.model_type)
        _, _, val_f1 = evaluate(model, val_loader, device, epoch, criterion)
        scheduler.step()

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✅ Best model saved (F1-mean: {best_f1:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve} epochs")
            if no_improve >= args.patience:
                print(f"  Early stopping triggered at epoch {epoch}")
                break

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best epoch: {best_epoch} | Best F1-mean: {best_f1:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()