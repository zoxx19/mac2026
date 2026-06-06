"""
ensemble_leg.py
Ensembles leg optical flow (VideoMAE) + MMN skeleton predictions.
Loads saved softmax probabilities from both models and averages them.

Step 1: Get flow model probs on val set
Step 2: Get MMN model probs on val set  
Step 3: Weighted average → final prediction
Step 4: Compute F1_mean

Usage:
  python src/experiments/ensemble_leg.py
"""

import sys, os, json, torch, numpy as np, pandas as pd
sys.path.insert(0, 'src/skeleton/MMN')
sys.path.insert(0, 'src/experiments')

from train_body_flow import BodyFlowDataset
from sklearn.metrics import f1_score, classification_report
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

NUM_CLASSES = 6
ID2LABEL    = {0:'A1',1:'A2',2:'A3',3:'A4',4:'A5',5:'no-body'}

def get_flow_probs(flow_model_path, val_csv, model_path, device):
    processor  = VideoMAEImageProcessor.from_pretrained(model_path)
    model      = VideoMAEForVideoClassification.from_pretrained(
        model_path, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(flow_model_path, map_location=device))
    model.to(device).eval()

    val_ds     = BodyFlowDataset(val_csv, processor, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    probs, labels = [], []
    with torch.no_grad():
        for pv, lbl in tqdm(val_loader, desc='Flow inference'):
            out = model(pixel_values=pv.to(device))
            probs.extend(torch.softmax(out.logits, dim=-1).cpu().numpy())
            labels.extend(lbl.numpy())
    return np.array(probs), np.array(labels)


def get_mmn_probs(mmn_model_path, val_csv, device):
    from model.MMN import MMN_
    sys.path.insert(0, 'src/skeleton')
    from features_modality import load_keypoints, resample_frames, normalize_keypoints, build_joint

    model = MMN_(num_classes=NUM_CLASSES, num_points=17)
    model.load_state_dict(torch.load(mmn_model_path, map_location=device))
    model.to(device).eval()

    df = pd.read_csv(val_csv)
    # Use leg joints (indices 11-16)
    BODY_JOINTS = list(range(17))

    probs, labels = [], []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc='MMN inference'):
            kp_path = f"data/keypoints/val/{str(row['video']).replace('.mp4','')}.json"
            if not os.path.exists(kp_path):
                continue
            kps = load_keypoints(kp_path)
            kps = resample_frames(kps)
            kps = normalize_keypoints(kps)
            kps = kps[:, BODY_JOINTS, :]
            feat = build_joint(kps)
            feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            index_t = torch.arange(64, dtype=torch.long).unsqueeze(0).to(device)
            out  = model(feat, index_t)
            probs.append(torch.softmax(out, dim=-1).cpu().numpy()[0])
            labels.append(int(row['body_label']))

    return np.array(probs), np.array(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--flow_model', default='outputs/body_flow/best_model.pt')
    p.add_argument('--mmn_model',  default='outputs/body_full_joint_aug/best_model.pt')
    p.add_argument('--flow_csv',   default='data/skeleton_dataset/body_val_flow.csv')
    p.add_argument('--mmn_csv',    default='data/skeleton_dataset/body_val.csv')
    p.add_argument('--model_path', default='models/videomae-ssv2')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nGetting flow model probabilities...")
    flow_probs, flow_labels = get_flow_probs(
        args.flow_model, args.flow_csv, args.model_path, device)

    print("\nGetting MMN probabilities...")
    mmn_probs, mmn_labels = get_mmn_probs(
        args.mmn_model, args.mmn_csv, device)

    # Align labels (both should be same val set)
    assert len(flow_probs) == len(mmn_probs), \
        f"Size mismatch: flow={len(flow_probs)}, mmn={len(mmn_probs)}"
    labels = flow_labels

    # Flow only
    f1_flow = (f1_score(labels, flow_probs.argmax(1), average='macro', zero_division=0) +
               f1_score(labels, flow_probs.argmax(1), average='micro', zero_division=0)) / 2

    # MMN only
    f1_mmn = (f1_score(labels, mmn_probs.argmax(1), average='macro', zero_division=0) +
              f1_score(labels, mmn_probs.argmax(1), average='micro', zero_division=0)) / 2

    print(f"\nFlow only:  F1_mean={f1_flow:.4f}")
    print(f"MMN only:   F1_mean={f1_mmn:.4f}")

    # Search best ensemble weights
    print("\nSearching best ensemble weights...")
    best_f1, best_w = 0.0, 0.5
    for w in np.arange(0.1, 1.0, 0.1):
        ensemble = w * flow_probs + (1-w) * mmn_probs
        preds    = ensemble.argmax(1)
        f1 = (f1_score(labels, preds, average='macro', zero_division=0) +
              f1_score(labels, preds, average='micro', zero_division=0)) / 2
        print(f"  flow_w={w:.1f} mmn_w={1-w:.1f} → F1_mean={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_w = f1, w

    print(f"\nBest ensemble: flow_w={best_w:.1f} mmn_w={1-best_w:.1f} → F1_mean={best_f1:.4f}")

    # Print classification report for best
    best_ensemble = best_w * flow_probs + (1-best_w) * mmn_probs
    best_preds    = best_ensemble.argmax(1)
    print(classification_report(labels, best_preds,
          target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)], zero_division=0))

if __name__ == '__main__':
    main()
