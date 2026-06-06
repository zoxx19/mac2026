"""
evaluate_twostage_ensemble.py
Evaluates the full two-stage hand pipeline with ensembled Stage 1.

Stage 1: Ensemble of VideoMAE + MMN joint + MMN bone (better routing)
Stage 2: Per-group VideoMAE models (best approach per group)

Searches best ensemble weights for Stage 1 models.

Usage:
  python src/hand/evaluate_twostage_ensemble.py \
      --stage1_videomae outputs/hand_stage1/best_model.pt \
      --stage1_mmn_joint outputs/hand_skeleton_full/best_model.pt \
      --stage1_mmn_bone  outputs/hand_full_bone_aug/best_model.pt \
      --stage2_C     outputs/hand_stage2_C_B/best_model.pt \
      --stage2_E     outputs/hand_stage2_E_B/best_model.pt \
      --stage2_F     outputs/hand_stage2_F_B/best_model.pt \
      --stage2_G     outputs/hand_stage2_G/best_model.pt \
      --val_csv      data/hand_dataset/hand_stage1_val.csv
"""

import sys, os, torch, numpy as np, pandas as pd
sys.path.insert(0, 'src/skeleton/MMN')
sys.path.insert(0, 'src/skeleton')
sys.path.insert(0, 'src/hand')

from evaluate_twostage import load_frames, get_true_global_label, \
    GROUP_TO_GLOBAL, NO_HAND_GLOBAL, GLOBAL_NAMES
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings("ignore")

NUM_CLASSES_S1 = 5
NUM_FRAMES     = 16
IMG_SIZE       = 224

# MMN config for hand coarse (5 classes, 17 joints)
HAND_JOINTS = list(range(17))


def get_videomae_probs(model_path, val_df, processor_path, device):
    """Get VideoMAE Stage 1 softmax probs for all val samples."""
    model = VideoMAEForVideoClassification.from_pretrained(
        processor_path, num_labels=NUM_CLASSES_S1,
        ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    processor = VideoMAEImageProcessor.from_pretrained(processor_path)

    all_probs = []
    with torch.no_grad():
        for _, row in tqdm(val_df.iterrows(), total=len(val_df),
                           desc='VideoMAE Stage1'):
            if not Path(row['crop_path']).exists():
                all_probs.append(np.ones(NUM_CLASSES_S1)/NUM_CLASSES_S1)
                continue
            frames = load_frames(row['crop_path'])
            inputs = processor(images=frames, return_tensors='pt')
            pv     = inputs['pixel_values'].to(device)
            out    = model(pixel_values=pv)
            probs  = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)
    return np.array(all_probs)


def get_mmn_probs(model_path, val_df, modality, device):
    """Get MMN Stage 1 softmax probs for all val samples."""
    from model.MMN import MMN_
    from features_modality import (load_keypoints, resample_frames,
                                    normalize_keypoints, build_joint, build_bone)
    from animate_wrong import COCO_EDGES

    if modality == "bone":
        model = MMN_(num_classes=NUM_CLASSES_S1, num_points=18, in_channels=3)
    else:
        model = MMN_(num_classes=NUM_CLASSES_S1, num_points=17)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    all_probs = []
    with torch.no_grad():
        for _, row in tqdm(val_df.iterrows(), total=len(val_df),
                           desc=f'MMN {modality}'):
            vid_id  = str(row['video']).replace('.mp4','')
            kp_path = f"data/keypoints/val/{vid_id}.json"
            if not os.path.exists(kp_path):
                all_probs.append(np.ones(NUM_CLASSES_S1)/NUM_CLASSES_S1)
                continue
            kps = load_keypoints(kp_path)
            kps = resample_frames(kps)
            kps = normalize_keypoints(kps)
            if modality == 'joint':
                feat = build_joint(kps)
            else:
                feat = build_bone(kps, COCO_EDGES)
            feat    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            index_t = torch.arange(64, dtype=torch.long).unsqueeze(0).to(device)
            out     = model(feat, index_t)
            probs   = torch.softmax(out, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)
    return np.array(all_probs)


def load_stage2_model(model_path, n_classes, processor_path, device):
    model = VideoMAEForVideoClassification.from_pretrained(
        processor_path, num_labels=n_classes, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def evaluate_twostage(stage1_probs, val_df, stage2_models,
                      processor, device):
    """Run Stage 2 with given Stage 1 probs and compute 34-class F1."""
    group_map  = {0:'C', 1:'E', 2:'F', 3:'G', 4:None}
    all_true, all_pred = [], []
    stage1_correct = 0

    for i, (_, row) in enumerate(val_df.iterrows()):
        true_global = get_true_global_label(row)
        all_true.append(true_global)

        # Stage 1 prediction from ensemble probs
        s1_pred = int(stage1_probs[i].argmax())
        group   = group_map[s1_pred]

        if group is None:
            all_pred.append(NO_HAND_GLOBAL)
        else:
            if not Path(row['crop_path']).exists():
                all_pred.append(NO_HAND_GLOBAL)
                continue
            frames = load_frames(row['crop_path'])
            inputs = processor(images=frames, return_tensors='pt')
            pv     = inputs['pixel_values'].to(device)
            with torch.no_grad():
                s2_logits = stage2_models[group](pixel_values=pv).logits
            local_pred  = int(s2_logits.argmax(-1).cpu())
            global_pred = GROUP_TO_GLOBAL[group](local_pred)
            all_pred.append(global_pred)

        # Track Stage 1 accuracy
        true_s1 = 4
        for rng, lbl in {range(0,13):0, range(13,19):1,
                         range(19,29):2, range(29,33):3}.items():
            if true_global in rng:
                true_s1 = lbl; break
        if s1_pred == true_s1:
            stage1_correct += 1

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    f1_mac   = f1_score(all_true, all_pred, average='macro',  zero_division=0)
    f1_mic   = f1_score(all_true, all_pred, average='micro',  zero_division=0)
    f1_mean  = (f1_mac + f1_mic) / 2
    s1_acc   = stage1_correct / len(val_df)
    return f1_mean, f1_mac, f1_mic, s1_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1_videomae',  required=True)
    p.add_argument('--stage1_mmn_joint', required=True)
    p.add_argument('--stage1_mmn_bone',  required=True)
    p.add_argument('--stage2_C',         required=True)
    p.add_argument('--stage2_E',         required=True)
    p.add_argument('--stage2_F',         required=True)
    p.add_argument('--stage2_G',         required=True)
    p.add_argument('--val_csv',          required=True)
    p.add_argument('--model_path',       default='models/videomae-ssv2')
    p.add_argument('--output_dir',       default='outputs/eval_hand_twostage_ensemble')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    val_df = pd.read_csv(args.val_csv)
    print(f"Val samples: {len(val_df)}")

    # Get Stage 1 probs from all 3 models
    print("\n--- Getting Stage 1 probabilities ---")
    vmae_probs  = get_videomae_probs(args.stage1_videomae, val_df,
                                      args.model_path, device)
    joint_probs = get_mmn_probs(args.stage1_mmn_joint, val_df, 'joint', device)
    bone_probs  = get_mmn_probs(args.stage1_mmn_bone,  val_df, 'bone',  device)

    # Load Stage 2 models
    print("\n--- Loading Stage 2 models ---")
    processor = VideoMAEImageProcessor.from_pretrained(args.model_path)
    stage2 = {
        'C': load_stage2_model(args.stage2_C, 13, args.model_path, device),
        'E': load_stage2_model(args.stage2_E,  6, args.model_path, device),
        'F': load_stage2_model(args.stage2_F, 10, args.model_path, device),
        'G': load_stage2_model(args.stage2_G,  4, args.model_path, device),
    }
    print("All Stage 2 models loaded ✅")

    # Evaluate single models first
    print("\n--- Single model Stage 1 results ---")
    for name, probs in [('VideoMAE', vmae_probs),
                        ('MMN joint', joint_probs),
                        ('MMN bone',  bone_probs)]:
        f1, mac, mic, s1_acc = evaluate_twostage(
            probs, val_df, stage2, processor, device)
        print(f"{name}: Stage1_acc={s1_acc:.4f} | F1_mean={f1:.4f}")

    # Search best ensemble weights
    print("\n--- Searching best ensemble weights ---")
    best_f1, best_weights = 0.0, (0.33, 0.33, 0.34)

    for vw in np.arange(0.1, 0.8, 0.1):
        for jw in np.arange(0.1, 0.8, 0.1):
            bw = round(1.0 - vw - jw, 2)
            if bw < 0.05 or bw > 0.8:
                continue
            ensemble = vw*vmae_probs + jw*joint_probs + bw*bone_probs
            f1, mac, mic, s1_acc = evaluate_twostage(
                ensemble, val_df, stage2, processor, device)
            if f1 > best_f1:
                best_f1      = f1
                best_weights = (vw, jw, bw)
                print(f"  New best: vmae={vw:.1f} joint={jw:.1f} bone={bw:.2f} "
                      f"→ Stage1_acc={s1_acc:.4f} F1_mean={f1:.4f}")

    print(f"\n{'='*60}")
    print(f"Best weights: vmae={best_weights[0]:.1f} joint={best_weights[1]:.1f} "
          f"bone={best_weights[2]:.2f}")
    print(f"Best F1_mean: {best_f1:.4f}")
    print(f"{'='*60}")

    # Final evaluation with best weights
    vw, jw, bw  = best_weights
    best_ensemble = vw*vmae_probs + jw*joint_probs + bw*bone_probs
    f1, mac, mic, s1_acc = evaluate_twostage(
        best_ensemble, val_df, stage2, processor, device)
    print(f"Final: Stage1_acc={s1_acc:.4f} | F1_macro={mac:.4f} | "
          f"F1_micro={mic:.4f} | F1_mean={f1:.4f}")

    # Save classification report
    group_map = {0:'C', 1:'E', 2:'F', 3:'G', 4:None}
    all_true, all_pred = [], []
    for i, (_, row) in enumerate(val_df.iterrows()):
        true_global = get_true_global_label(row)
        all_true.append(true_global)
        s1_pred = int(best_ensemble[i].argmax())
        group   = group_map[s1_pred]
        if group is None:
            all_pred.append(NO_HAND_GLOBAL)
        else:
            if not Path(row['crop_path']).exists():
                all_pred.append(NO_HAND_GLOBAL); continue
            frames  = load_frames(row['crop_path'])
            inputs  = processor(images=frames, return_tensors='pt')
            pv      = inputs['pixel_values'].to(device)
            with torch.no_grad():
                s2_logits = stage2[group](pixel_values=pv).logits
            local_pred  = int(s2_logits.argmax(-1).cpu())
            global_pred = GROUP_TO_GLOBAL[group](local_pred)
            all_pred.append(global_pred)

    class_names = [GLOBAL_NAMES[i] for i in range(34)]
    print(classification_report(np.array(all_true), np.array(all_pred),
          target_names=class_names, zero_division=0))

    print(f"Done. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
