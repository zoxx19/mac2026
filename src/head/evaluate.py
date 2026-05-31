"""
evaluate.py
Generates confusion matrix for the best trained head recognition model.
Works with VideoMAE or TimeSformer.
Saves:
    - confusion_matrix.png (raw counts + normalized)
    - wrong_predictions.csv (all misclassified samples)
    - wrong_predictions/ folder with sample wrong clips per class

Usage:
    python src/head/evaluate.py \
        --val_csv    data/head_dataset/val_crops.csv \
        --model_path outputs/head_vXX/best_model.pt \
        --model_name /path/to/local/model \
        --model_type videomae \
        --output_dir outputs/head_vXX/evaluation
"""

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    VideoMAEForVideoClassification,
    TimesformerForVideoClassification,
    AutoImageProcessor,
)
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL2ID = {"B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4, "B6": 5, "B7": 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = {
    "B1": "B1: nodding",
    "B2": "B2: shaking head",
    "B3": "B3: turning head",
    "B4": "B4: tilting head",
    "B5": "B5: bowing head",
    "B6": "B6: head up",
    "B7": "B7: no movement",
}
NUM_FRAMES  = 16
NUM_CLASSES = 7


# ── Dataset ───────────────────────────────────────────────────────────────────
class HeadClipDataset(Dataset):
    def __init__(self, csv_path, processor):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.df = self.df[self.df["crop_path"].apply(
            lambda p: Path(p).exists()
        )].reset_index(drop=True)
        print(f"Loaded {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def _load_frames(self, video_path):
        cap     = cv2.VideoCapture(str(video_path))
        total   = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
        frames  = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frames.append(frames[-1].copy() if frames else
                               np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].copy())
        return frames[:NUM_FRAMES]

    def __getitem__(self, idx):
        row          = self.df.iloc[idx]
        frames       = self._load_frames(row["crop_path"])
        inputs       = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label        = LABEL2ID[row["head_label"]]
        return pixel_values, torch.tensor(label, dtype=torch.long), str(row["crop_path"])


def collate_fn(batch):
    pixels = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    paths  = [b[2] for b in batch]
    return pixels, labels, paths


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(model_path, model_name, model_type, device):
    processor = AutoImageProcessor.from_pretrained(model_name)

    if model_type == "timesformer":
        model = TimesformerForVideoClassification.from_pretrained(
            model_name,
            num_labels=NUM_CLASSES,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    else:
        model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=NUM_CLASSES,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

    # load saved weights — handle both plain and dropout classifier
    state_dict = torch.load(model_path, map_location=device)
    new_state  = {}
    for k, v in state_dict.items():
        if k == "classifier.weight":
            new_state["classifier.weight"] = v
        elif k == "classifier.bias":
            new_state["classifier.bias"] = v
        elif k == "classifier.1.weight":
            new_state["classifier.weight"] = v
        elif k == "classifier.1.bias":
            new_state["classifier.bias"] = v
        else:
            new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"Missing keys: {missing}")

    model = model.to(device)
    model.eval()
    return model, processor


# ── Confusion matrix plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(cm, output_dir):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = [LABEL_NAMES[ID2LABEL[i]] for i in range(NUM_CLASSES)]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Confusion Matrix — Raw Counts", fontsize=14, pad=12)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].tick_params(axis="y", rotation=0)

    # normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix — Normalized (per true class)", fontsize=14, pad=12)
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].tick_params(axis="y", rotation=0)

    plt.suptitle("Head Recognition — Val Set Evaluation", fontsize=16, y=1.02)
    plt.tight_layout()
    out_path = output_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv",    required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_type", default="videomae",
                        choices=["videomae", "timesformer"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_wrong",    type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers",type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name} ({args.model_type})")

    model, processor = load_model(
        args.model_path, args.model_name, args.model_type, device
    )

    dataset = HeadClipDataset(args.val_csv, processor)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    all_preds  = []
    all_labels = []
    all_paths  = []

    with torch.no_grad():
        for pixel_values, labels, paths in tqdm(loader, desc="Evaluating"):
            pixel_values = pixel_values.to(device)
            outputs      = model(pixel_values=pixel_values)
            preds        = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # metrics
    acc     = np.mean(all_preds == all_labels)
    f1_mac  = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    f1_mic  = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    f1_mean = (f1_mac + f1_mic) / 2

    print(f"\nVal Acc: {acc:.4f} | F1-macro: {f1_mac:.4f} | "
          f"F1-micro: {f1_mic:.4f} | F1-mean: {f1_mean:.4f}")
    print("\nPer-class report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=[LABEL_NAMES[ID2LABEL[i]] for i in range(NUM_CLASSES)],
        zero_division=0
    ))

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, output_dir)

    # wrong predictions
    df_results = pd.DataFrame({
        "path":      all_paths,
        "true":      [ID2LABEL[l] for l in all_labels],
        "predicted": [ID2LABEL[p] for p in all_preds],
    })
    df_wrong = df_results[df_results["true"] != df_results["predicted"]]
    df_wrong.to_csv(output_dir / "wrong_predictions.csv", index=False)
    print(f"\nTotal wrong: {len(df_wrong)} / {len(df_results)} "
          f"({len(df_wrong)/len(df_results)*100:.1f}%)")

    # copy sample wrong clips per class
    wrong_dir = output_dir / "wrong_clips"
    wrong_dir.mkdir(exist_ok=True)
    for true_class in [ID2LABEL[i] for i in range(NUM_CLASSES)]:
        class_wrong = df_wrong[df_wrong["true"] == true_class].head(args.n_wrong)
        if len(class_wrong) == 0:
            print(f"  {true_class}: no wrong predictions ✅")
            continue
        class_dir = wrong_dir / true_class
        class_dir.mkdir(exist_ok=True)
        for _, row in class_wrong.iterrows():
            src = Path(row["path"])
            dst = class_dir / f"true_{true_class}_pred_{row['predicted']}_{src.name}"
            shutil.copy(src, dst)
        print(f"  {true_class}: {len(class_wrong)} wrong clips saved")

    print(f"\nWrong clips saved → {wrong_dir}")
    print("Done ✅")


if __name__ == "__main__":
    main()
