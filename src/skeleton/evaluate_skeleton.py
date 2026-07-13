"""
src/skeleton/evaluate_skeleton.py

Evaluates skeleton models with three modes:
1. Single model evaluation
2. Ensemble (joint + bone) — averages logits
3. Per-class threshold tuning on val set

Usage:
    # Single model
    python evaluate_skeleton.py \
        --track hand --mode full --modality joint \
        --num_classes 5 \
        --model_path outputs/hand_skeleton_full/best_model.pt

    # Ensemble joint + bone
    python evaluate_skeleton.py \
        --track hand --mode full \
        --num_classes 5 \
        --ensemble \
        --joint_model outputs/hand_skeleton_full/best_model.pt \
        --bone_model  outputs/hand_full_bone_aug/best_model.pt

    # Ensemble + threshold tuning
    python evaluate_skeleton.py \
        --track hand --mode full \
        --num_classes 5 \
        --ensemble \
        --joint_model outputs/hand_skeleton_full/best_model.pt \
        --bone_model  outputs/hand_full_bone_aug/best_model.pt \
        --tune_threshold
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent / "MMN"))
from model.MMN import MMN_
from features_modality import extract_features, MODE_JOINTS, MODE_EDGES

BASE     = Path(".")
DATA_DIR = BASE / "data/skeleton_dataset"

NUM_FRAMES  = 64
BATCH_SIZE  = 64
NUM_WORKERS = 8

LABEL_COL = {
    "hand": "hand_label",
    "leg":  "leg_label",
    "body": "body_label",
}

LABEL_NAMES = {
    "hand": {0:"C upper limb", 1:"E body-hand", 2:"F head-hand",
             3:"G leg-hand",   4:"no hand move"},
    "leg":  {0:"D1", 1:"D2", 2:"D3", 3:"D4", 4:"D5",
             5:"D6", 6:"D7", 7:"D8", 8:"D9 no leg"},
    "body": {0:"A1", 1:"A2", 2:"A3", 3:"A4", 4:"A5", 5:"A6 no body"},
}

# ── Dataset ───────────────────────────────────────────────────────────────────

class SkeletonDataset(Dataset):
    def __init__(self, csv_path, track, mode, modality, num_frames=NUM_FRAMES):
        self.df        = pd.read_csv(csv_path)
        self.track     = track
        self.mode      = mode
        self.modality  = modality
        self.label_col = LABEL_COL[track]
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row[self.label_col])
        tensor = extract_features(row["kp_path"], mode=self.mode,
                                  modality=self.modality,
                                  num_frames=self.num_frames)
        index_t = torch.arange(self.num_frames, dtype=torch.long)
        return torch.from_numpy(tensor.astype(np.float32)), index_t, label


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_path, in_channels, num_classes, num_points, device):
    model = MMN_(
        in_channels=in_channels,
        num_classes=num_classes,
        num_people=1,
        num_frames=NUM_FRAMES,
        num_points=num_points,
        kernel_size=3,
        num_heads=4,
        head_drop=0.2,
        drop=0.1,
        drop_path=0.1,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_logits(model, loader, device):
    all_logits, all_labels = [], []
    for tensor, index_t, labels in loader:
        tensor  = tensor.to(device)
        index_t = index_t.to(device)
        logits  = model(tensor, index_t)
        all_logits.append(logits.cpu())
        all_labels.extend(labels.tolist())
    return torch.cat(all_logits, dim=0), all_labels


# ── Metrics ───────────────────────────────────────────────────────────────────

def report(all_labels, all_preds, num_classes, label_names, title=""):
    f1_macro = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    f1_mean  = (f1_macro + f1_micro) / 2
    acc      = (np.array(all_preds) == np.array(all_labels)).mean()
    per_cls  = f1_score(all_labels, all_preds, average=None,
                        labels=list(range(num_classes)), zero_division=0)

    print(f"\n{'='*55}")
    if title:
        print(f"  {title}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-macro : {f1_macro:.4f}")
    print(f"  F1-micro : {f1_micro:.4f}")
    print(f"  F1-mean  : {f1_mean:.4f}")
    print(f"\n  Per-class F1:")
    counts = Counter(all_labels)
    for c, f1 in enumerate(per_cls):
        name = label_names.get(c, str(c))
        print(f"    {c} {name:<22}  F1={f1:.4f}  ({counts.get(c,0)} samples)")
    return f1_mean


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_thresholds(logits, all_labels, num_classes, label_names):
    """
    Find per-class threshold that maximizes F1-mean on val set.
    Uses softmax probabilities and tunes threshold for each class.
    """
    probs = torch.softmax(logits, dim=1).numpy()  # (N, C)

    print(f"\n{'='*55}")
    print("  Threshold Tuning")
    print(f"{'='*55}")

    best_thresholds = np.ones(num_classes) * 0.5
    thresholds_to_try = np.arange(0.1, 0.95, 0.05)

    # Tune each class threshold independently
    for c in range(num_classes):
        best_f1 = 0
        best_t  = 0.5
        for t in thresholds_to_try:
            # Predict class c if prob > t, otherwise predict argmax of rest
            preds = []
            for i in range(len(all_labels)):
                if probs[i, c] > t:
                    preds.append(c)
                else:
                    masked = probs[i].copy()
                    masked[c] = 0
                    preds.append(masked.argmax())
            f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t
        best_thresholds[c] = best_t
        name = label_names.get(c, str(c))
        print(f"    {c} {name:<22}  threshold={best_t:.2f}  F1-macro={best_f1:.4f}")

    # Apply best thresholds
    print(f"\n  Applying tuned thresholds...")
    preds = []
    for i in range(len(all_labels)):
        # Find class with highest prob above its threshold
        above = [(probs[i, c], c) for c in range(num_classes)
                 if probs[i, c] > best_thresholds[c]]
        if above:
            preds.append(max(above)[1])
        else:
            preds.append(probs[i].argmax())

    return preds, best_thresholds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track",       choices=["hand","leg","body"], required=True)
    parser.add_argument("--mode",        choices=["full","arm","leg","torso"], default="full")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--num_frames",  type=int, default=NUM_FRAMES)

    # Single model
    parser.add_argument("--modality",   choices=["joint","bone"], default="joint")
    parser.add_argument("--model_path", type=str, default=None)

    # Ensemble
    parser.add_argument("--ensemble",    action="store_true")
    parser.add_argument("--joint_model", type=str, default=None)
    parser.add_argument("--bone_model",  type=str, default=None)
    parser.add_argument("--joint_weight", type=float, default=0.5)
    parser.add_argument("--bone_weight",  type=float, default=0.5)

    # Threshold tuning
    parser.add_argument("--tune_threshold", action="store_true")

    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_csv    = DATA_DIR / f"{args.track}_val.csv"
    label_col  = LABEL_COL[args.track]
    label_names = LABEL_NAMES[args.track]

    joint_points = len(MODE_JOINTS[args.mode])
    bone_points  = len(MODE_EDGES[args.mode])

    if args.ensemble:
        # Load both modalities
        print("Loading joint model...")
        j_ds     = SkeletonDataset(val_csv, args.track, args.mode, "joint", args.num_frames)
        j_loader = DataLoader(j_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
        j_model  = load_model(args.joint_model, 2, args.num_classes,
                               joint_points, device)
        j_logits, all_labels = get_logits(j_model, j_loader, device)

        print("Loading bone model...")
        b_ds     = SkeletonDataset(val_csv, args.track, args.mode, "bone", args.num_frames)
        b_loader = DataLoader(b_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
        b_model  = load_model(args.bone_model, 3, args.num_classes,
                               bone_points, device)
        b_logits, _ = get_logits(b_model, b_loader, device)

        # Evaluate individually first
        j_preds = j_logits.argmax(dim=1).tolist()
        b_preds = b_logits.argmax(dim=1).tolist()
        report(all_labels, j_preds, args.num_classes, label_names, "Joint only")
        report(all_labels, b_preds, args.num_classes, label_names, "Bone only")

        # Try different ensemble weights
        print(f"\n{'='*55}")
        print("  Ensemble weight search")
        print(f"{'='*55}")
        best_f1   = 0
        best_w    = 0.5
        for w in np.arange(0.3, 0.8, 0.05):
            ens_logits = w * j_logits + (1 - w) * b_logits
            ens_preds  = ens_logits.argmax(dim=1).tolist()
            f1 = f1_score(all_labels, ens_preds, average="macro", zero_division=0)
            f1_micro = f1_score(all_labels, ens_preds, average="micro", zero_division=0)
            f1_mean = (f1 + f1_micro) / 2
            print(f"  w_joint={w:.2f}  f1-mean={f1_mean:.4f}")
            if f1_mean > best_f1:
                best_f1 = f1_mean
                best_w  = w

        print(f"\n  Best weight: joint={best_w:.2f}  bone={1-best_w:.2f}")
        ens_logits = best_w * j_logits + (1 - best_w) * b_logits
        ens_preds  = ens_logits.argmax(dim=1).tolist()
        report(all_labels, ens_preds, args.num_classes, label_names,
               f"Ensemble (joint={best_w:.2f} bone={1-best_w:.2f})")

        if args.tune_threshold:
            tuned_preds, thresholds = tune_thresholds(
                ens_logits, all_labels, args.num_classes, label_names)
            report(all_labels, tuned_preds, args.num_classes, label_names,
                   "Ensemble + Threshold tuning")

    else:
        # Single model
        in_ch   = 2 if args.modality == "joint" else 3
        n_pts   = joint_points if args.modality == "joint" else bone_points
        ds      = SkeletonDataset(val_csv, args.track, args.mode,
                                  args.modality, args.num_frames)
        loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
        model   = load_model(args.model_path, in_ch, args.num_classes, n_pts, device)
        logits, all_labels = get_logits(model, loader, device)
        preds   = logits.argmax(dim=1).tolist()
        report(all_labels, preds, args.num_classes, label_names, "Single model")

        if args.tune_threshold:
            tuned_preds, _ = tune_thresholds(
                logits, all_labels, args.num_classes, label_names)
            report(all_labels, tuned_preds, args.num_classes, label_names,
                   "Single model + Threshold tuning")


if __name__ == "__main__":
    main()