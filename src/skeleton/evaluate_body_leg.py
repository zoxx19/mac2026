"""
evaluate_body_leg.py
Evaluates body or leg MMN model on the validation set.

Metric: F1_mean = (F1_macro + F1_micro) / 2
        Same formula as the MAC paper and your training logs.

Outputs:
    confusion_matrix.png
    wrong_predictions.csv  (columns: video_id, true, predicted)

Usage:
    python src/skeleton/evaluate_body_leg.py \
        --track body \
        --model_path outputs/body_skeleton_full/best_model.pt \
        --output_dir outputs/eval_body

    python src/skeleton/evaluate_body_leg.py \
        --track leg \
        --model_path outputs/leg_skeleton_leg/best_model.pt \
        --output_dir outputs/eval_leg
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (confusion_matrix, f1_score,
                              classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# MMN import
MMN_ROOT = os.path.join(os.path.dirname(__file__), 'MMN')
sys.path.insert(0, MMN_ROOT)
from model.MMN import MMN_

# ── Track configs ─────────────────────────────────────────────────────────────
TRACK_CFG = {
    'body': {
        'csv':           'data/skeleton_dataset/body_val.csv',
        'kp_dir':        'data/keypoints/val',
        'label_col':     'body_label',
        'video_col':     'video',
        'num_classes':   6,
        'joint_indices': list(range(17)),          # all 17 joints
        'class_names':   ['A1','A2','A3','A4','A5','no-body'],
    },
    'leg': {
        'csv':           'data/skeleton_dataset/leg_val.csv',
        'kp_dir':        'data/keypoints/val',
        'label_col':     'leg_label',
        'video_col':     'video',
        'num_classes':   9,
        'joint_indices': [11,12,13,14,15,16],      # 6 leg joints
        'class_names':   ['D1','D2','D3','D4','D5','D6','D7','D8','no-leg'],
    },
}


def load_skeleton(kp_path, joint_indices, num_frames=64):
    """Load keypoint JSON → skeleton tensor (1, 2, 64, V, 1)."""
    with open(kp_path) as f:
        data = json.load(f)
    # x,y only (in_channels=2, same as training)
    frames = [np.array(fr['keypoints'], dtype=np.float32)[:, :2]
              for fr in data]
    skel = np.stack(frames, axis=0)              # (T_raw, 17, 2)
    # normalise
    mn = skel.min(axis=(0,1), keepdims=True)
    mx = skel.max(axis=(0,1), keepdims=True)
    skel = (skel - mn) / np.where(mx-mn > 1e-6, mx-mn, 1.0) * 2 - 1
    # select joints
    skel = skel[:, joint_indices, :]             # (T_raw, V, 2)
    # resample to fixed length
    T = skel.shape[0]
    if T >= num_frames:
        idx = np.linspace(0, T-1, num_frames, dtype=int)
        skel = skel[idx]
    else:
        pad  = num_frames - T
        skel = np.concatenate([skel,
               np.tile(skel[-1:], (pad, 1, 1))], axis=0)
    # (2, 64, V, 1) → (1, 2, 64, V, 1)
    skel = skel.transpose(2, 0, 1)[:, :, :, np.newaxis]
    return torch.from_numpy(skel.astype(np.float32)).unsqueeze(0)


def load_model(model_path, num_classes, num_joints, device):
    """Load MMN from checkpoint. Reads num_heads from checkpoint shape."""
    state     = torch.load(model_path, map_location=device)
    num_heads = state['MFM.0.blocks.0.gconv'].shape[0]
    model = MMN_(
        in_channels=2,
        num_classes=num_classes,
        num_people=1,
        num_frames=64,
        num_points=num_joints,
        kernel_size=3,
        num_heads=num_heads,
        head_drop=0.0,
        drop=0.0,
        drop_path=0.1,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def plot_confusion_matrix(cm, class_names, track, f1mean, out_path):
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Raw Counts', fontsize=12)
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
    axes[0].tick_params(axis='x', rotation=45)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Normalised', fontsize=12)
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
    axes[1].tick_params(axis='x', rotation=45)
    plt.suptitle(f'{track.upper()} track  |  F1_mean = {f1mean:.4f}',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--track',      choices=['body','leg'], required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    cfg    = TRACK_CFG[args.track]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Track: {args.track} | Device: {device}")

    model = load_model(args.model_path, cfg['num_classes'],
                       len(cfg['joint_indices']), device)

    df = pd.read_csv(cfg['csv'])
    print(f"Val samples: {len(df)}")

    all_true, all_pred, all_ids = [], [], []
    index_t = torch.arange(64, dtype=torch.long).to(device)

    for _, row in df.iterrows():
        vid_id  = str(row[cfg['video_col']]).replace('.mp4', '')
        label   = int(row[cfg['label_col']])
        kp_path = os.path.join(cfg['kp_dir'], f"{vid_id}.json")
        if not os.path.exists(kp_path):
            continue
        x = load_skeleton(kp_path, cfg['joint_indices']).to(device)
        with torch.no_grad():
            pred = model(x, index_t).argmax(1).item()
        all_true.append(label)
        all_pred.append(pred)
        all_ids.append(vid_id)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Metric: F1_mean = (F1_macro + F1_micro) / 2
    f1_macro = float(f1_score(all_true, all_pred,
                              average='macro', zero_division=0))
    f1_micro = float(f1_score(all_true, all_pred,
                              average='micro', zero_division=0))
    f1mean   = (f1_macro + f1_micro) / 2

    print(f"\nF1_macro: {f1_macro:.4f}")
    print(f"F1_micro: {f1_micro:.4f}")
    print(f"F1_mean:  {f1mean:.4f}")
    print(f"\n{classification_report(all_true, all_pred,
          target_names=cfg['class_names'], zero_division=0)}")

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred,
                          labels=list(range(cfg['num_classes'])))
    plot_confusion_matrix(cm, cfg['class_names'], args.track,
                          f1mean,
                          os.path.join(args.output_dir,
                                       'confusion_matrix.png'))

    # Wrong predictions CSV
    names    = cfg['class_names']
    df_wrong = pd.DataFrame({
        'video_id':  [i for i, (t,p) in zip(all_ids,
                      zip(all_true,all_pred)) if t != p],
        'true':      [names[t] for t,p in zip(all_true,all_pred) if t!=p],
        'predicted': [names[p] for t,p in zip(all_true,all_pred) if t!=p],
    })
    csv_path = os.path.join(args.output_dir, 'wrong_predictions.csv')
    df_wrong.to_csv(csv_path, index=False)
    print(f"Wrong: {len(df_wrong)} / {len(all_true)}")
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
