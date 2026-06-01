"""
train_leg_videomae.py
Fine-tunes VideoMAE on full body crops for leg track (D1-D8 + no-leg).
Same architecture as head track train.py — 2-stage fine-tuning.

Labels:
  0=A1, 1=A2, 2=A3, 3=A4, 4=A5, 5=no-body

Usage:
  python src/body_leg/train_leg_videomae.py \
      --train_csv data/skeleton_dataset/leg_train_videomae.csv \
      --val_csv   data/skeleton_dataset/leg_val_videomae.csv \
      --output_dir outputs/leg_videomae \
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
ID2LABEL    = {0:'D1',1:'D2',2:'D3',3:'D4',4:'D5',5:'D6',6:'D7',7:'D8',8:'no-leg'}
LABEL2ID    = {v:k for k,v in ID2LABEL.items()}


class LegCropDataset(Dataset):
    def __init__(self, csv_path, processor, is_train=True):
        self.df        = pd.read_csv(csv_path)
        self.df        = self.df[self.df['crop_path'].apply(
                             lambda p: Path(p).exists())].reset_index(drop=True)
        self.processor = processor
        self.is_train  = is_train
        print(f"  Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self): return len(self.df)

    def _load_frames(self, path):
        cap   = cv2.VideoCapture(str(path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        if self.is_train and total > NUM_FRAMES:
            offset  = np.random.randint(0, total - NUM_FRAMES + 1)
            indices = np.linspace(offset, offset+NUM_FRAMES-1, NUM_FRAMES, dtype=int)
        else:
            indices = np.linspace(0, total-1, NUM_FRAMES, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frames.append(frames[-1].copy() if frames else
                               np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        while len(frames) < NUM_FRAMES: frames.append(frames[-1].copy())
        return frames[:NUM_FRAMES]

    def _augment(self, frames):
        h, w = frames[0].shape[:2]
        if np.random.random() > 0.5:
            frames = [np.fliplr(f) for f in frames]
        if np.random.random() > 0.3:
            scale = np.random.uniform(0.85, 1.0)
            nh, nw = int(h*scale), int(w*scale)
            top  = np.random.randint(0, h-nh+1)
            left = np.random.randint(0, w-nw+1)
            frames = [cv2.resize(f[top:top+nh,left:left+nw],(w,h)) for f in frames]
        alpha = np.random.uniform(0.7, 1.3)
        beta  = np.random.randint(-30, 30)
        frames = [np.clip(f.astype(np.float32)*alpha+beta,0,255).astype(np.uint8)
                  for f in frames]
        return frames

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row['leg_label'])
        frames = self._load_frames(row['crop_path'])
        if self.is_train: frames = self._augment(frames)
        inputs = self.processor(images=frames, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0), torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device, epoch):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for pv, labels in tqdm(loader, desc=f'Val Epoch {epoch}'):
            out = model(pixel_values=pv.to(device))
            all_preds.extend(out.logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.numpy())
    f1_mac  = f1_score(all_labels, all_preds, average='macro',  zero_division=0)
    f1_mic  = f1_score(all_labels, all_preds, average='micro',  zero_division=0)
    f1_mean = (f1_mac + f1_mic) / 2
    print(f"  F1_macro={f1_mac:.4f} | F1_micro={f1_mic:.4f} | F1_mean={f1_mean:.4f}")
    print(classification_report(all_labels, all_preds,
          target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)], zero_division=0))
    return f1_mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv',     required=True)
    p.add_argument('--val_csv',       required=True)
    p.add_argument('--output_dir',    required=True)
    p.add_argument('--model_path',    default='models/videomae-ssv2')
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--stage1_epochs', type=int,   default=10)
    p.add_argument('--stage2_epochs', type=int,   default=20)
    p.add_argument('--lr_stage1',     type=float, default=1e-3)
    p.add_argument('--lr_stage2',     type=float, default=1e-5)
    p.add_argument('--patience',      type=int,   default=5)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    processor = VideoMAEImageProcessor.from_pretrained(args.model_path)
    model     = VideoMAEForVideoClassification.from_pretrained(
        args.model_path, num_labels=NUM_CLASSES,
        id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True).to(device)

    # weighted loss for class imbalance
    train_df = pd.read_csv(args.train_csv)
    counts   = train_df['leg_label'].value_counts().sort_index()
    weights  = torch.tensor([len(train_df)/(NUM_CLASSES*counts.get(i,1))
                              for i in range(NUM_CLASSES)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_ds = LegCropDataset(args.train_csv, processor, is_train=True)
    val_ds   = LegCropDataset(args.val_csv,   processor, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    best_f1, no_improve = 0.0, 0

    # Stage 1 — head only
    print(f"\n=== Stage 1: head only ({args.stage1_epochs} epochs) ===")
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
        f1 = evaluate(model, val_loader, device, epoch)
        scheduler.step()
        if f1 > best_f1:
            best_f1 = f1; no_improve = 0
            torch.save(model.state_dict(), out_dir/'best_model.pt')
            print(f"  ✅ Best F1_mean: {best_f1:.4f}")
        else:
            no_improve += 1

    # Stage 2 — full fine-tuning
    print(f"\n=== Stage 2: full fine-tune ({args.stage2_epochs} epochs) ===")
    for param in model.videomae.parameters(): param.requires_grad = True
    optimizer  = AdamW(model.parameters(), lr=args.lr_stage2, weight_decay=0.01)
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.stage2_epochs)
    no_improve = 0

    for epoch in range(args.stage1_epochs+1, args.stage1_epochs+args.stage2_epochs+1):
        model.train()
        for pv, labels in tqdm(train_loader, desc=f'Train {epoch}'):
            optimizer.zero_grad()
            out  = model(pixel_values=pv.to(device))
            loss = criterion(out.logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        f1 = evaluate(model, val_loader, device, epoch)
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
