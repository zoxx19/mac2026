"""
train_head_detector.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/train_head_detector.py --help` for options where applicable.
"""
import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight


class HeadCropDataset(Dataset):
    def __init__(self, csv_path, video_dir, num_frames=16, size=224):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.size = size
        self.processor = VideoMAEImageProcessor.from_pretrained(
            "data/videomae-base"
        )

        df = pd.read_csv(csv_path)
        csv_ids = set(str(v) for v in df["video_id"].unique())

        # Every video_id in the CSV → label=1; absent from CSV → label=0
        video_labels = {}
        for fname in os.listdir(video_dir):
            if fname.endswith("_head.mp4"):
                vid_id = fname.replace("_head.mp4", "")
                video_labels[vid_id] = 1 if vid_id in csv_ids else 0

        self.samples = list(video_labels.items())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_id, label = self.samples[idx]
        video_path = os.path.join(self.video_dir, f"{vid_id}_head.mp4")
        frames = self._load_frames(video_path)
        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (16, 3, 224, 224)
        return pixel_values, torch.tensor(label, dtype=torch.long)

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return [np.zeros((self.size, self.size, 3), dtype=np.uint8)] * self.num_frames

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.size, self.size))
            frames.append(frame)
        cap.release()
        return frames


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--train_video_dir", required=True)
    parser.add_argument("--val_video_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = HeadCropDataset(args.train_csv, args.train_video_dir)
    val_dataset = HeadCropDataset(args.val_csv, args.val_video_dir)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_labels = np.array([label for _, label in train_dataset.samples])
    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=train_labels
    )
    print(f"Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch_idx, (pixel_values, labels) in enumerate(train_loader):
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Step {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {running_loss/total:.4f} | "
                    f"Acc: {correct/total:.4f}"
                )

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} — "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, "best_model")
            model.save_pretrained(ckpt_path)
            print(f"  Saved best model (val_acc={best_val_acc:.4f}) → {ckpt_path}")

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
