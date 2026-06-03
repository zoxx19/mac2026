"""
train_leg_oversample.py
VideoMAE fine-tuning on oversampled leg crops with heavier augmentation
for rare classes (D2, D5).

Key differences from train_body_videomae.py:
  - Uses oversampled CSV (body_train_oversample.csv)
  - Heavier augmentation: rotation, speed perturbation, stronger color jitter
  - Focal loss instead of weighted cross-entropy

Usage:
  python src/experiments/train_leg_oversample.py \
      --train_csv data/skeleton_dataset/leg_train_oversample.csv \
      --val_csv   data/skeleton_dataset/leg_val_videomae.csv \
      --output_dir outputs/leg_oversample \
      --model_path models/videomae-ssv2
"""

import cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

NUM_FRAMES  = 16
IMG_SIZE    = 224
NUM_CLASSES = 9
ID2LABEL    = {0:'D1', 1:'D2', 2:'D3', 3:'D4', 4:'D5', 5:'D6', 6:'D7', 7:'D8', 8:'no-leg'}
LABEL2ID    = {v:k for k,v in ID2LABEL.items()}


class FocalLoss(nn.Module):
    """Focal loss — focuses training on hard/rare examples."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets,
                                         weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class LegOversampleDataset(Dataset):
    def __init__(self, csv_path, processor, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['crop_path'].apply(
            lambda p: Path(p).exists())].reset_index(drop=True)
        self.processor = processor
        self.is_train  = is_train
        print(f"  Loaded {len(self.df)} samples from {csv_path}")
        print(self.df['leg_label'].value_counts().sort_index().to_string())

    def __len__(self): return len(self.df)

    def _load_frames(self, path, speed_factor=1.0):
        cap   = cv2.VideoCapture(str(path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

        # speed perturbation: sample more/fewer frames
        effective_total = max(int(total * speed_factor), NUM_FRAMES)
        if self.is_train and effective_total > NUM_FRAMES:
            offset  = np.random.randint(0, effective_total - NUM_FRAMES + 1)
            indices = np.linspace(offset, offset+NUM_FRAMES-1,
                                  NUM_FRAMES, dtype=int)
            indices = np.clip(indices, 0, total-1)
        else:
            indices = np.linspace(0, total-1, NUM_FRAMES, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frames.append(frames[-1].copy() if frames else
                               np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        while len(frames) < NUM_FRAMES: frames.append(frames[-1].copy())
        return frames[:NUM_FRAMES]

    def _augment_heavy(self, frames, label):
        h, w = frames[0].shape[:2]

        # Always flip for rare classes
        if np.random.random() > 0.5 or label in [1, 4]:
            frames = [np.fliplr(f) for f in frames]

        # Random crop
        if np.random.random() > 0.3:
            scale = np.random.uniform(0.8, 1.0)
            nh, nw = int(h*scale), int(w*scale)
            top  = np.random.randint(0, h-nh+1)
            left = np.random.randint(0, w-nw+1)
            frames = [cv2.resize(f[top:top+nh, left:left+nw], (w,h))
                      for f in frames]

        # Rotation — especially for leaning classes
        if np.random.random() > 0.4 or label in [1, 4]:
            angle = np.random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            frames = [cv2.warpAffine(f, M, (w,h)) for f in frames]

        # Stronger color jitter for rare classes
        jitter = 0.5 if label in [1, 4] else 0.3
        alpha = np.random.uniform(1-jitter, 1+jitter)
        beta  = np.random.randint(int(-40*jitter), int(40*jitter))
        frames = [np.clip(f.astype(np.float32)*alpha+beta, 0, 255
                          ).astype(np.uint8) for f in frames]

        # Temporal reverse
        if np.random.random() > 0.7:
            frames = frames[::-1]

        # Grayscale
        if np.random.random() > 0.8:
            frames = [np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)]*3,
                               axis=-1) for f in frames]

        return frames

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row['leg_label'])

        # Speed perturbation for rare classes
        speed = 1.0
        if self.is_train and label in [1, 4]:
            speed = np.random.uniform(0.8, 1.2)

        frames = self._load_frames(row['crop_path'], speed)
        if self.is_train:
            frames = self._augment_heavy(frames, label)

        inputs = self.processor(images=frames, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0), \
               torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device, epoch, criterion):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    with torch.no_grad():
        for pv, labels in tqdm(loader, desc=f'Val {epoch}'):
            out  = model(pixel_values=pv.to(device))
            loss = criterion(out.logits, labels.to(device))
            total_loss += loss.item()
            all_preds.extend(out.logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.numpy())
    f1_mac  = f1_score(all_labels, all_preds, average='macro',  zero_division=0)
    f1_mic  = f1_score(all_labels, all_preds, average='micro',  zero_division=0)
    f1_mean = (f1_mac + f1_mic) / 2
    print(f"  F1_macro={f1_mac:.4f} | F1_micro={f1_mic:.4f} | "
          f"F1_mean={f1_mean:.4f} | loss={total_loss/len(loader):.4f}")
    print(classification_report(all_labels, all_preds,
          target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)],
          zero_division=0))
    return f1_mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv',     required=True)
    p.add_argument('--val_csv',       required=True)
    p.add_argument('--output_dir',    required=True)
    p.add_argument('--model_path',    default='models/videomae-ssv2')
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--stage1_epochs', type=int,   default=10)
    p.add_argument('--stage2_epochs', type=int,   default=25)
    p.add_argument('--lr_stage1',     type=float, default=1e-3)
    p.add_argument('--lr_stage2',     type=float, default=1e-5)
    p.add_argument('--patience',      type=int,   default=7)
    p.add_argument('--focal_gamma',   type=float, default=2.0)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    processor = VideoMAEImageProcessor.from_pretrained(args.model_path)
    model     = VideoMAEForVideoClassification.from_pretrained(
        args.model_path, num_labels=NUM_CLASSES,
        id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True).to(device)

    # Focal loss with class weights
    train_df = pd.read_csv(args.train_csv)
    counts   = train_df['leg_label'].value_counts().sort_index()
    weights  = torch.tensor(
        [len(train_df)/(NUM_CLASSES*counts.get(i,1))
         for i in range(NUM_CLASSES)], dtype=torch.float).to(device)
    criterion = FocalLoss(gamma=args.focal_gamma, weight=weights)

    train_ds = LegOversampleDataset(args.train_csv, processor, is_train=True)
    val_ds   = LegOversampleDataset(args.val_csv,   processor, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    best_f1, no_improve = 0.0, 0

    print(f"\n=== Stage 1: classifier only ({args.stage1_epochs} epochs) ===")
    for param in model.videomae.parameters(): param.requires_grad = False
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr_stage1, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.stage1_epochs)

    for epoch in range(1, args.stage1_epochs+1):
        model.train()
        for pv, labels in tqdm(train_loader, desc=f'Train {epoch}'):
            optimizer.zero_grad()
            out  = model(pixel_values=pv.to(device))
            loss = criterion(out.logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        f1 = evaluate(model, val_loader, device, epoch, criterion)
        scheduler.step()
        if f1 > best_f1:
            best_f1 = f1; no_improve = 0
            torch.save(model.state_dict(), out_dir/'best_model.pt')
            print(f"  ✅ Best F1_mean: {best_f1:.4f}")
        else:
            no_improve += 1

    print(f"\n=== Stage 2: full fine-tune ({args.stage2_epochs} epochs) ===")
    for param in model.videomae.parameters(): param.requires_grad = True
    optimizer  = AdamW(model.parameters(), lr=args.lr_stage2, weight_decay=0.01)
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.stage2_epochs)
    no_improve = 0

    for epoch in range(args.stage1_epochs+1,
                       args.stage1_epochs+args.stage2_epochs+1):
        model.train()
        for pv, labels in tqdm(train_loader, desc=f'Train {epoch}'):
            optimizer.zero_grad()
            out  = model(pixel_values=pv.to(device))
            loss = criterion(out.logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        f1 = evaluate(model, val_loader, device, epoch, criterion)
        scheduler.step()
        if f1 > best_f1:
            best_f1 = f1; no_improve = 0
            torch.save(model.state_dict(), out_dir/'best_model.pt')
            print(f"  ✅ Best F1_mean: {best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nDone. Best F1_mean: {best_f1:.4f}")
    print(f"Model: {out_dir/'best_model.pt'}")

if __name__ == '__main__':
    main()
