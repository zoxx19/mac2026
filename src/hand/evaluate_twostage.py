"""
evaluate_twostage.py
Evaluates the full two-stage hand pipeline on the validation set.

Stage 1: predicts coarse group (C/E/F/G/no-hand)
Stage 2: predicts fine class within group

Computes F1_mean over 34 fine-grained hand classes.

Usage:
  python src/hand/evaluate_twostage.py \
      --stage1_model  outputs/hand_stage1/best_model.pt \
      --stage2_C      outputs/hand_stage2_C/best_model.pt \
      --stage2_E      outputs/hand_stage2_E/best_model.pt \
      --stage2_F      outputs/hand_stage2_F/best_model.pt \
      --stage2_G      outputs/hand_stage2_G/best_model.pt \
      --val_csv       data/hand_dataset/hand_stage1_val.csv \
      --output_dir    outputs/eval_hand_twostage
"""

import cv2, numpy as np, pandas as pd, torch, torch.nn as nn, os
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings("ignore")

NUM_FRAMES  = 16
IMG_SIZE    = 224

# Mapping from (group, local_label) -> global fine label (0-33)
GROUP_TO_GLOBAL = {
    'C': lambda l: l,        # C1-C13: 0-12
    'E': lambda l: l + 13,   # E1-E6:  13-18
    'F': lambda l: l + 19,   # F1-F10: 19-28
    'G': lambda l: l + 29,   # G1-G4:  29-32
}
NO_HAND_GLOBAL = 33

GLOBAL_NAMES = {
    **{i: f'C{i+1}'    for i in range(0,  13)},
    **{i: f'E{i-12}'   for i in range(13, 19)},
    **{i: f'F{i-18}'   for i in range(19, 29)},
    **{i: f'G{i-28}'   for i in range(29, 33)},
    33: 'no-hand'
}

GROUP_CONFIG = {
    'C': {'n_classes': 13, 'stage1_label': 0},
    'E': {'n_classes': 6,  'stage1_label': 1},
    'F': {'n_classes': 10, 'stage1_label': 2},
    'G': {'n_classes': 4,  'stage1_label': 3},
}

def load_frames(path, is_train=False):
    cap   = cv2.VideoCapture(str(path))
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
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


def load_model(model_path, n_classes, processor_path, device):
    model = VideoMAEForVideoClassification.from_pretrained(
        processor_path, num_labels=n_classes,
        ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def get_true_global_label(row):
    """Convert val CSV row to global 34-class label."""
    fine = int(row['fine_label'])
    if 11 <= fine <= 23: return fine - 11        # C: 0-12
    if 32 <= fine <= 37: return fine - 32 + 13   # E: 13-18
    if 38 <= fine <= 47: return fine - 38 + 19   # F: 19-28
    if 48 <= fine <= 51: return fine - 48 + 29   # G: 29-32
    return NO_HAND_GLOBAL                         # no-hand: 33


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1_model', required=True)
    p.add_argument('--stage2_C',     required=True)
    p.add_argument('--stage2_E',     required=True)
    p.add_argument('--stage2_F',     required=True)
    p.add_argument('--stage2_G',     required=True)
    p.add_argument('--val_csv',      required=True)
    p.add_argument('--output_dir',   default='outputs/eval_hand_twostage')
    p.add_argument('--model_path',   default='models/videomae-ssv2')
    p.add_argument('--batch_size',   type=int, default=16)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    processor = VideoMAEImageProcessor.from_pretrained(args.model_path)

    # Load all models
    print("Loading models...")
    stage1 = load_model(args.stage1_model, 5, args.model_path, device)
    stage2 = {
        'C': load_model(args.stage2_C, 13, args.model_path, device),
        'E': load_model(args.stage2_E,  6, args.model_path, device),
        'F': load_model(args.stage2_F, 10, args.model_path, device),
        'G': load_model(args.stage2_G,  4, args.model_path, device),
    }
    print("All models loaded ✅")

    # Load val data
    val_df = pd.read_csv(args.val_csv)
    val_df = val_df[val_df['crop_path'].apply(
        lambda p: Path(p).exists())].reset_index(drop=True)
    print(f"Val samples: {len(val_df)}")

    all_true, all_pred = [], []
    stage1_correct = 0

    for _, row in tqdm(val_df.iterrows(), total=len(val_df),
                       desc='Evaluating'):
        true_global = get_true_global_label(row)
        all_true.append(true_global)

        # Load frames
        frames = load_frames(row['crop_path'])
        inputs = processor(images=frames, return_tensors='pt')
        pv = inputs['pixel_values'].to(device)

        # Stage 1 — predict coarse group
        with torch.no_grad():
            s1_logits = stage1(pixel_values=pv).logits
        s1_pred = int(s1_logits.argmax(-1).cpu())

        # Map stage1 prediction to group
        group_map = {0:'C', 1:'E', 2:'F', 3:'G', 4:None}
        group = group_map[s1_pred]

        if group is None:
            # Predicted no-hand
            all_pred.append(NO_HAND_GLOBAL)
        else:
            # Stage 2 — predict fine class within group
            with torch.no_grad():
                s2_logits = stage2[group](pixel_values=pv).logits
            local_pred = int(s2_logits.argmax(-1).cpu())
            global_pred = GROUP_TO_GLOBAL[group](local_pred)
            all_pred.append(global_pred)

        # Track stage1 accuracy
        true_group_label = {
            range(0,13): 0, range(13,19): 1,
            range(19,29): 2, range(29,33): 3
        }
        true_s1 = 4  # no-hand default
        for rng, lbl in true_group_label.items():
            if true_global in rng:
                true_s1 = lbl; break
        if s1_pred == true_s1:
            stage1_correct += 1

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Metrics
    f1_mac  = f1_score(all_true, all_pred, average='macro',  zero_division=0)
    f1_mic  = f1_score(all_true, all_pred, average='micro',  zero_division=0)
    f1_mean = (f1_mac + f1_mic) / 2
    s1_acc  = stage1_correct / len(val_df)

    print(f"\n{'='*60}")
    print(f"Stage 1 accuracy: {s1_acc:.4f}")
    print(f"F1_macro={f1_mac:.4f} | F1_micro={f1_mic:.4f} | "
          f"F1_mean={f1_mean:.4f}")
    print(f"{'='*60}")

    class_names = [GLOBAL_NAMES[i] for i in range(34)]
    print(classification_report(all_true, all_pred,
          target_names=class_names, zero_division=0))

    # Confusion matrix
    cm      = confusion_matrix(all_true, all_pred, labels=list(range(34)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(32, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.3, annot_kws={'size':6})
    axes[0].set_title('Raw Counts')
    axes[0].tick_params(axis='x', rotation=90, labelsize=7)
    axes[0].tick_params(axis='y', labelsize=7)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.3, vmin=0, vmax=1,
                annot_kws={'size':6})
    axes[1].set_title('Normalised')
    axes[1].tick_params(axis='x', rotation=90, labelsize=7)
    axes[1].tick_params(axis='y', labelsize=7)

    plt.suptitle(f'Hand Two-Stage (34 classes) | F1_mean={f1_mean:.4f}',
                 fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
