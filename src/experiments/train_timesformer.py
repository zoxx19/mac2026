"""
train_timesformer.py
Universal TimeSformer fine-tuning script for any track.
Supports both timesformer-base (224x224) and timesformer-hr (448x448).

Key difference from VideoMAE:
  - Divided space-time attention (separate spatial + temporal attention)
  - Different temporal modeling than MAE
  - HR version uses 448x448 — captures more spatial detail

Usage:
  # TimeSformer-base SSv2
  python src/experiments/train_timesformer.py \
      --track head \
      --train_csv data/head_dataset/train_crops.csv \
      --val_csv   data/head_dataset/val_crops.csv \
      --video_col crop_path \
      --label_col head_label \
      --num_classes 7 \
      --output_dir outputs/head_timesformer \
      --model_path models/timesformer-ssv2

  # TimeSformer-HR SSv2
  python src/experiments/train_timesformer.py \
      --track head \
      --train_csv data/head_dataset/train_crops.csv \
      --val_csv   data/head_dataset/val_crops.csv \
      --video_col crop_path \
      --label_col head_label \
      --num_classes 7 \
      --output_dir outputs/head_timesformer_hr \
      --model_path models/timesformer-hr-ssv2 \
      --img_size 448
"""

import cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

NUM_FRAMES = 16   # TimeSformer uses 8 frames by default
TRACK_LABELS = {
    'head':        {0:'B1',1:'B2',2:'B3',3:'B4',4:'B5',5:'B6',6:'B7'},
    'body':        {0:'A1',1:'A2',2:'A3',3:'A4',4:'A5',5:'no-body'},
    'leg':         {0:'D1',1:'D2',2:'D3',3:'D4',4:'D5',5:'D6',6:'D7',7:'D8',8:'no-leg'},
    'hand_coarse': {0:'C',1:'E',2:'F',3:'G',4:'no-hand'},
}


class VideoDataset(Dataset):
    def __init__(self, csv_path, processor, video_col, label_col,
                 img_size=224, is_train=True):
        self.df        = pd.read_csv(csv_path)
        self.df        = self.df[self.df[video_col].apply(
                             lambda p: Path(p).exists())].reset_index(drop=True)
        self.processor = processor
        self.video_col = video_col
        self.label_col = label_col
        self.img_size  = img_size
        self.is_train  = is_train

        # Handle string labels
        sample = self.df[label_col].iloc[0]
        if isinstance(sample, str):
            unique = sorted(self.df[label_col].unique())
            self.label_map = {l: i for i, l in enumerate(unique)}
        else:
            self.label_map = None

        print(f"  Loaded {len(self.df)} samples from {csv_path}")
        print(self.df[label_col].value_counts().sort_index().to_string())

    def __len__(self): return len(self.df)

    def _load_frames(self, path):
        cap   = cv2.VideoCapture(str(path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        if self.is_train and total > NUM_FRAMES:
            offset  = np.random.randint(0, total - NUM_FRAMES + 1)
            indices = np.linspace(offset, offset+NUM_FRAMES-1,
                                  NUM_FRAMES, dtype=int)
        else:
            indices = np.linspace(0, total-1, NUM_FRAMES, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frames.append(frames[-1].copy() if frames else
                               np.zeros((self.img_size, self.img_size, 3),
                                        dtype=np.uint8))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.img_size != 224:
                    frame = cv2.resize(frame, (self.img_size, self.img_size))
                frames.append(frame)
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
            frames = [cv2.resize(f[top:top+nh, left:left+nw], (w,h))
                      for f in frames]
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            frames = [cv2.warpAffine(f, M, (w,h)) for f in frames]
        alpha = np.random.uniform(0.7, 1.3)
        beta  = np.random.randint(-30, 30)
        frames = [np.clip(f.astype(np.float32)*alpha+beta, 0, 255
                          ).astype(np.uint8) for f in frames]
        return frames

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        raw   = row[self.label_col]
        label = self.label_map[raw] if self.label_map else int(raw)
        frames = self._load_frames(row[self.video_col])
        if self.is_train: frames = self._augment(frames)
        inputs = self.processor(images=frames, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0), \
               torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device, epoch, criterion, num_classes, id2label):
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
          target_names=[id2label[i] for i in range(num_classes)],
          zero_division=0))
    return f1_mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--track',         required=True,
                   choices=['head','body','leg','hand_coarse'])
    p.add_argument('--train_csv',     required=True)
    p.add_argument('--val_csv',       required=True)
    p.add_argument('--video_col',     required=True)
    p.add_argument('--label_col',     required=True)
    p.add_argument('--num_classes',   type=int, required=True)
    p.add_argument('--output_dir',    required=True)
    p.add_argument('--model_path',    default='models/timesformer-ssv2')
    p.add_argument('--img_size',      type=int, default=224)
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--stage1_epochs', type=int,   default=10)
    p.add_argument('--stage2_epochs', type=int,   default=20)
    p.add_argument('--lr_stage1',     type=float, default=1e-3)
    p.add_argument('--lr_stage2',     type=float, default=1e-5)
    p.add_argument('--no_weighted_loss', action='store_true', help='Disable weighted loss')
    p.add_argument('--patience',      type=int,   default=7)
    args = p.parse_args()

    id2label = TRACK_LABELS.get(args.track,
               {i: str(i) for i in range(args.num_classes)})
    label2id = {v:k for k,v in id2label.items()}

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Track: {args.track} | img_size: {args.img_size}")
    print(f"Model: {args.model_path}")

    processor = AutoImageProcessor.from_pretrained(args.model_path)
    model     = TimesformerForVideoClassification.from_pretrained(
        args.model_path, num_labels=args.num_classes,
        id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.1f}M")

    # Weighted loss
    train_df  = pd.read_csv(args.train_csv)
    counts    = train_df[args.label_col].value_counts().sort_index()
    weights   = torch.tensor(
        [len(train_df)/(args.num_classes*counts.get(i,1))
         for i in range(args.num_classes)],
        dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss() if args.no_weighted_loss else nn.CrossEntropyLoss(weight=weights)

    train_ds = VideoDataset(args.train_csv, processor, args.video_col,
                            args.label_col, args.img_size, is_train=True)
    val_ds   = VideoDataset(args.val_csv,   processor, args.video_col,
                            args.label_col, args.img_size, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    best_f1, no_improve = 0.0, 0

    # Stage 1 — classifier only
    print(f"\n=== Stage 1: classifier only ({args.stage1_epochs} epochs) ===")
    for param in model.timesformer.parameters(): param.requires_grad = False
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
        f1 = evaluate(model, val_loader, device, epoch,
                      criterion, args.num_classes, id2label)
        scheduler.step()
        if f1 > best_f1:
            best_f1 = f1; no_improve = 0
            torch.save(model.state_dict(), out_dir/'best_model.pt')
            print(f"  ✅ Best F1_mean: {best_f1:.4f}")
        else:
            no_improve += 1

    # Stage 2 — full fine-tune
    print(f"\n=== Stage 2: full fine-tune ({args.stage2_epochs} epochs) ===")
    for param in model.timesformer.parameters(): param.requires_grad = True
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
        f1 = evaluate(model, val_loader, device, epoch,
                      criterion, args.num_classes, id2label)
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

    print(f"\nDone. Track={args.track} | Best F1_mean: {best_f1:.4f}")
    print(f"Model: {out_dir/'best_model.pt'}")

if __name__ == '__main__':
    main()
