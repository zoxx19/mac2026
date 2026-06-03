"""
train_stage2.py
Stage 2: Fine-tunes VideoMAE on per-group fine-grained hand classes.

Two initialization approaches:
  Approach A: init from VideoMAE-SSv2 (--model_path only)
  Approach B: init from Stage 1 weights (--init_model)

Usage:
  # Approach A
  python src/hand/train_stage2.py --group C \
      --train_csv  data/hand_dataset/hand_C_train.csv \
      --val_csv    data/hand_dataset/hand_C_val.csv \
      --output_dir outputs/hand_stage2_C_A \
      --model_path models/videomae-ssv2

  # Approach B
  python src/hand/train_stage2.py --group C \
      --train_csv  data/hand_dataset/hand_C_train.csv \
      --val_csv    data/hand_dataset/hand_C_val.csv \
      --output_dir outputs/hand_stage2_C_B \
      --model_path models/videomae-ssv2 \
      --init_model outputs/hand_stage1/best_model.pt
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

NUM_FRAMES = 16
IMG_SIZE   = 224

GROUP_CONFIG = {
    'C': {'n_classes': 13, 'labels': {i: f'C{i+1}' for i in range(13)}},
    'E': {'n_classes': 6,  'labels': {i: f'E{i+1}' for i in range(6)}},
    'F': {'n_classes': 10, 'labels': {i: f'F{i+1}' for i in range(10)}},
    'G': {'n_classes': 4,  'labels': {i: f'G{i+1}' for i in range(4)}},
}


class HandStage2Dataset(Dataset):
    def __init__(self, csv_path, processor, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['crop_path'].apply(
            lambda p: Path(p).exists())].reset_index(drop=True)
        self.processor = processor
        self.is_train  = is_train
        print(f"  Loaded {len(self.df)} samples from {csv_path}")
        print(self.df['label'].value_counts().sort_index().to_string())

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
                               np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8))
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
            frames = [cv2.resize(f[top:top+nh, left:left+nw], (w,h)) for f in frames]
        alpha = np.random.uniform(0.7, 1.3)
        beta  = np.random.randint(-30, 30)
        frames = [np.clip(f.astype(np.float32)*alpha+beta, 0, 255).astype(np.uint8)
                  for f in frames]
        return frames

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        label  = int(row['label'])
        frames = self._load_frames(row['crop_path'])
        if self.is_train: frames = self._augment(frames)
        inputs = self.processor(images=frames, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0), \
               torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device, epoch, criterion, id2label):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    n_classes = len(id2label)
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
          target_names=[id2label[i] for i in range(n_classes)], zero_division=0))
    return f1_mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group',         required=True, choices=['C','E','F','G'])
    p.add_argument('--train_csv',     required=True)
    p.add_argument('--val_csv',       required=True)
    p.add_argument('--output_dir',    required=True)
    p.add_argument('--model_path',    default='models/videomae-ssv2')
    p.add_argument('--init_model',    default=None,
                   help='Stage 1 model path for transfer learning (Approach B)')
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--stage1_epochs', type=int,   default=10)
    p.add_argument('--stage2_epochs', type=int,   default=30)
    p.add_argument('--lr_stage1',     type=float, default=1e-3)
    p.add_argument('--lr_stage2',     type=float, default=1e-5)
    p.add_argument('--patience',      type=int,   default=7)
    args = p.parse_args()

    cfg    = GROUP_CONFIG[args.group]
    n_cls  = cfg['n_classes']
    id2lab = cfg['labels']
    lab2id = {v:k for k,v in id2lab.items()}

    out_dir  = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    approach = 'B (Stage1 transfer)' if args.init_model else 'A (SSv2 pretrain)'
    print(f"Device: {device} | Group: {args.group} | Classes: {n_cls} | Approach: {approach}")

    processor = VideoMAEImageProcessor.from_pretrained(args.model_path)
    model     = VideoMAEForVideoClassification.from_pretrained(
        args.model_path, num_labels=n_cls,
        id2label=id2lab, label2id=lab2id,
        ignore_mismatched_sizes=True)

    # Approach B: load Stage 1 backbone weights
    if args.init_model:
        print(f"Loading Stage 1 weights from {args.init_model}...")
        state       = torch.load(args.init_model, map_location='cpu')
        model_state = model.state_dict()
        pretrained  = {k: v for k, v in state.items()
                       if k in model_state and 'classifier' not in k
                       and model_state[k].shape == v.shape}
        model_state.update(pretrained)
        model.load_state_dict(model_state)
        print(f"  Loaded {len(pretrained)}/{len(state)} layers ✅")

    model = model.to(device)

    train_df  = pd.read_csv(args.train_csv)
    counts    = train_df['label'].value_counts().sort_index()
    weights   = torch.tensor(
        [len(train_df)/(n_cls*counts.get(i,1)) for i in range(n_cls)],
        dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_ds     = HandStage2Dataset(args.train_csv, processor, is_train=True)
    val_ds       = HandStage2Dataset(args.val_csv,   processor, is_train=False)
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
        f1 = evaluate(model, val_loader, device, epoch, criterion, id2lab)
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
        f1 = evaluate(model, val_loader, device, epoch, criterion, id2lab)
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

    print(f"\nDone. Group {args.group} | {approach} | Best F1_mean: {best_f1:.4f}")
    print(f"Model: {out_dir/'best_model.pt'}")

if __name__ == '__main__':
    main()
