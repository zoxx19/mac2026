"""
train_videomae_large.py
Universal VideoMAE-large fine-tuning script for any track.
Supports: head (7), body (6), leg (9), hand_coarse (5)

Improvements over base model:
  - VideoMAE-large (307M params, 24 transformer layers vs 12)
  - Cosine annealing with warm restarts (3 cycles)
  - MixUp augmentation
  - Label smoothing
  - Stronger spatial + temporal augmentation
  - Gradient accumulation for effective larger batch

Usage:
  python src/experiments/train_videomae_large.py \
      --track head \
      --train_csv data/head_dataset/train_crops.csv \
      --val_csv   data/head_dataset/val_crops.csv \
      --video_col crop_path \
      --label_col head_label \
      --num_classes 7 \
      --output_dir outputs/head_large
"""

import cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

NUM_FRAMES = 16
IMG_SIZE   = 224

# Track-specific label mappings
TRACK_LABELS = {
    'head':       {0:'B1',1:'B2',2:'B3',3:'B4',4:'B5',5:'B6',6:'B7'},
    'body':       {0:'A1',1:'A2',2:'A3',3:'A4',4:'A5',5:'no-body'},
    'leg':        {0:'D1',1:'D2',2:'D3',3:'D4',4:'D5',5:'D6',6:'D7',7:'D8',8:'no-leg'},
    'hand_coarse':{0:'C',1:'E',2:'F',3:'G',4:'no-hand'},
}


class MixUp:
    """MixUp augmentation for video classification."""
    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def __call__(self, x, y, num_classes):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        bs = x.size(0)
        idx = torch.randperm(bs)
        mixed_x = lam * x + (1 - lam) * x[idx]
        # soft labels
        y_onehot = torch.zeros(bs, num_classes, device=x.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        y_onehot_b = y_onehot[idx]
        mixed_y = lam * y_onehot + (1 - lam) * y_onehot_b
        return mixed_x, mixed_y


class VideoDataset(Dataset):
    def __init__(self, csv_path, processor, video_col, label_col, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df[video_col].apply(
            lambda p: Path(p).exists())].reset_index(drop=True)
        self.processor  = processor
        self.video_col  = video_col
        self.label_col  = label_col
        self.is_train   = is_train

        # Build label map if labels are strings
        sample_label = self.df[label_col].iloc[0]
        if isinstance(sample_label, str):
            unique_labels = sorted(self.df[label_col].unique())
            self.label_map = {l: i for i, l in enumerate(unique_labels)}
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
                               np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        while len(frames) < NUM_FRAMES: frames.append(frames[-1].copy())
        return frames[:NUM_FRAMES]

    def _augment(self, frames):
        h, w = frames[0].shape[:2]

        # Horizontal flip
        if np.random.random() > 0.5:
            frames = [np.fliplr(f) for f in frames]

        # Random crop
        if np.random.random() > 0.3:
            scale = np.random.uniform(0.8, 1.0)
            nh, nw = int(h*scale), int(w*scale)
            top  = np.random.randint(0, h-nh+1)
            left = np.random.randint(0, w-nw+1)
            frames = [cv2.resize(f[top:top+nh, left:left+nw], (w,h))
                      for f in frames]

        # Rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            frames = [cv2.warpAffine(f, M, (w,h)) for f in frames]

        # Color jitter
        alpha = np.random.uniform(0.7, 1.3)
        beta  = np.random.randint(-30, 30)
        frames = [np.clip(f.astype(np.float32)*alpha+beta, 0, 255
                          ).astype(np.uint8) for f in frames]

        # Temporal reverse
        if np.random.random() > 0.8:
            frames = frames[::-1]

        # Grayscale
        if np.random.random() > 0.85:
            frames = [np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)]*3,
                               axis=-1) for f in frames]

        return frames

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        raw_label = row[self.label_col]
        if self.label_map is not None:
            label = self.label_map[raw_label]
        else:
            label = int(raw_label)
        frames = self._load_frames(row[self.video_col])
        if self.is_train: frames = self._augment(frames)
        inputs = self.processor(images=frames, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0), \
               torch.tensor(label, dtype=torch.long)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes
        self.weight      = weight

    def forward(self, pred, target):
        # target can be hard labels or soft (MixUp)
        if target.dim() == 1:
            # hard labels → convert to soft
            soft = torch.zeros_like(pred).scatter_(
                1, target.unsqueeze(1), 1.0)
        else:
            soft = target
        # apply label smoothing
        soft = soft * (1 - self.smoothing) + \
               self.smoothing / self.num_classes
        log_prob = torch.nn.functional.log_softmax(pred, dim=-1)
        if self.weight is not None:
            loss = -(soft * log_prob * self.weight.unsqueeze(0)).sum(-1).mean()
        else:
            loss = -(soft * log_prob).sum(-1).mean()
        return loss


def evaluate(model, loader, device, epoch, num_classes, id2label):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for pv, labels in tqdm(loader, desc=f'Val {epoch}'):
            out = model(pixel_values=pv.to(device))
            all_preds.extend(out.logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.numpy())
    f1_mac  = f1_score(all_labels, all_preds, average='macro',  zero_division=0)
    f1_mic  = f1_score(all_labels, all_preds, average='micro',  zero_division=0)
    f1_mean = (f1_mac + f1_mic) / 2
    print(f"  F1_macro={f1_mac:.4f} | F1_micro={f1_mic:.4f} | F1_mean={f1_mean:.4f}")
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
    p.add_argument('--model_path',    default='models/videomae-large-kinetics')
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--accum_steps',   type=int,   default=2,
                   help='Gradient accumulation steps (effective batch = batch*accum)')
    p.add_argument('--stage1_epochs', type=int,   default=10)
    p.add_argument('--stage2_epochs', type=int,   default=20)
    p.add_argument('--lr_stage1',     type=float, default=1e-3)
    p.add_argument('--lr_stage2',     type=float, default=5e-6)
    p.add_argument('--patience',      type=int,   default=7)
    p.add_argument('--mixup_alpha',   type=float, default=0.4)
    p.add_argument('--label_smooth',  type=float, default=0.1)
    args = p.parse_args()

    id2label = TRACK_LABELS.get(args.track,
               {i: str(i) for i in range(args.num_classes)})
    label2id = {v:k for k,v in id2label.items()}

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Track: {args.track} | Classes: {args.num_classes}")
    print(f"Model: {args.model_path}")
    print(f"MixUp alpha: {args.mixup_alpha} | Label smoothing: {args.label_smooth}")
    print(f"Effective batch: {args.batch_size * args.accum_steps}")

    processor = VideoMAEImageProcessor.from_pretrained(args.model_path)
    model     = VideoMAEForVideoClassification.from_pretrained(
        args.model_path, num_labels=args.num_classes,
        id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.1f}M")

    # Weighted loss + label smoothing
    train_df = pd.read_csv(args.train_csv)
    counts   = train_df[args.label_col].value_counts().sort_index()
    weights  = torch.tensor(
        [len(train_df)/(args.num_classes * counts.get(i,1))
         for i in range(args.num_classes)],
        dtype=torch.float).to(device)
    criterion = LabelSmoothingLoss(args.num_classes, args.label_smooth, weights)
    mixup     = MixUp(alpha=args.mixup_alpha)

    train_ds = VideoDataset(args.train_csv, processor,
                            args.video_col, args.label_col, is_train=True)
    val_ds   = VideoDataset(args.val_csv,   processor,
                            args.video_col, args.label_col, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    best_f1, no_improve = 0.0, 0

    # Stage 1 — classifier only
    print(f"\n=== Stage 1: classifier only ({args.stage1_epochs} epochs) ===")
    for param in model.videomae.parameters(): param.requires_grad = False
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr_stage1, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

    for epoch in range(1, args.stage1_epochs+1):
        model.train()
        optimizer.zero_grad()
        for step, (pv, labels) in enumerate(tqdm(train_loader,
                                                   desc=f'Train {epoch}')):
            pv = pv.to(device); labels = labels.to(device)
            pv, soft_labels = mixup(pv, labels, args.num_classes)
            out  = model(pixel_values=pv)
            loss = criterion(out.logits, soft_labels) / args.accum_steps
            loss.backward()
            if (step+1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        f1 = evaluate(model, val_loader, device, epoch,
                      args.num_classes, id2label)
        scheduler.step()
        if f1 > best_f1:
            best_f1 = f1; no_improve = 0
            torch.save(model.state_dict(), out_dir/'best_model.pt')
            print(f"  ✅ Best F1_mean: {best_f1:.4f}")
        else:
            no_improve += 1

    # Stage 2 — full fine-tune with cosine warm restarts
    print(f"\n=== Stage 2: full fine-tune ({args.stage2_epochs} epochs) ===")
    for param in model.videomae.parameters(): param.requires_grad = True
    optimizer  = AdamW(model.parameters(), lr=args.lr_stage2,
                       weight_decay=0.01)
    scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=1)
    no_improve = 0

    for epoch in range(args.stage1_epochs+1,
                       args.stage1_epochs+args.stage2_epochs+1):
        model.train()
        optimizer.zero_grad()
        for step, (pv, labels) in enumerate(tqdm(train_loader,
                                                   desc=f'Train {epoch}')):
            pv = pv.to(device); labels = labels.to(device)
            pv, soft_labels = mixup(pv, labels, args.num_classes)
            out  = model(pixel_values=pv)
            loss = criterion(out.logits, soft_labels) / args.accum_steps
            loss.backward()
            if (step+1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        f1 = evaluate(model, val_loader, device, epoch,
                      args.num_classes, id2label)
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
