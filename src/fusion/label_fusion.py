"""
label_fusion.py
Combines predictions from all 4 tracks into a single 52-class prediction
and computes the official competition metric.

Official metric:
  F1_mean = (F1_body_macro + F1_body_micro + F1_action_macro + F1_action_micro) / 4

Fine2Coarse mapping:
  0-4   → 0 (body A)
  5-10  → 1 (head B)
  11-23 → 2 (upper limb C)
  24-31 → 3 (lower limb D)
  32-37 → 4 (body-hand E)
  38-47 → 5 (head-hand F)
  48-51 → 6 (leg-hand G)

Strategy:
  Each track outputs softmax probabilities over its classes.
  No-movement classes are excluded from final prediction.
  The track with highest max non-no-movement probability wins.
  Final prediction = fine label of winning track prediction.

Usage:
  python src/fusion/label_fusion.py
"""

import sys, os, torch, numpy as np, pandas as pd
sys.path.insert(0, 'src/skeleton/MMN')
sys.path.insert(0, 'src/skeleton')
sys.path.insert(0, 'src/hand')

from sklearn.metrics import f1_score, classification_report
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings; warnings.filterwarnings("ignore")

# ── Fine2Coarse mapping ────────────────────────────────────────────────────────
def fine_to_coarse(fine):
    if 0  <= fine <= 4:  return 0
    if 5  <= fine <= 10: return 1
    if 11 <= fine <= 23: return 2
    if 24 <= fine <= 31: return 3
    if 32 <= fine <= 37: return 4
    if 38 <= fine <= 47: return 5
    if 48 <= fine <= 51: return 6
    return -1

# ── Label mappings per track ──────────────────────────────────────────────────
HEAD_TO_FINE   = {0:5, 1:6, 2:7, 3:8, 4:9, 5:10}
HEAD_NO_MOVE   = 6
BODY_TO_FINE   = {0:0, 1:1, 2:2, 3:3, 4:4}
BODY_NO_MOVE   = 5
LEG_TO_FINE    = {0:24, 1:25, 2:26, 3:27, 4:28, 5:29, 6:30, 7:31}
LEG_NO_MOVE    = 8
HAND_C_TO_FINE = {i: i+11 for i in range(13)}
HAND_E_TO_FINE = {i: i+32 for i in range(6)}
HAND_F_TO_FINE = {i: i+38 for i in range(10)}
HAND_G_TO_FINE = {i: i+48 for i in range(4)}

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_videomae(model_path, n_classes, processor_path, device):
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    model = VideoMAEForVideoClassification.from_pretrained(
        processor_path, num_labels=n_classes, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    processor = VideoMAEImageProcessor.from_pretrained(processor_path)
    return model, processor

def load_mmn(model_path, n_classes, n_points, device, in_channels=2):
    from model.MMN import MMN_
    model = MMN_(num_classes=n_classes, num_points=n_points,
                 in_channels=in_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

def get_videomae_probs(model, processor, video_path, device):
    import cv2
    cap   = cv2.VideoCapture(str(video_path))
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = np.linspace(0, total-1, 16, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frames.append(frames[-1].copy() if frames else
                           np.zeros((224,224,3), dtype=np.uint8))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    while len(frames) < 16: frames.append(frames[-1].copy())
    inputs = processor(images=frames[:16], return_tensors='pt')
    with torch.no_grad():
        out = model(pixel_values=inputs['pixel_values'].to(device))
    return torch.softmax(out.logits, dim=-1).cpu().numpy()[0]

def get_mmn_probs(model, kp_path, joint_indices, modality, device, edges=None):
    from features_modality import (load_keypoints, resample_frames,
                                    normalize_keypoints, build_joint, build_bone)
    kps = load_keypoints(kp_path)
    kps = resample_frames(kps)
    kps = normalize_keypoints(kps)
    kps = kps[:, joint_indices, :]
    feat = build_joint(kps) if modality == 'joint' else build_bone(kps, edges)
    feat    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
    index_t = torch.arange(64, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(feat, index_t)
    return torch.softmax(out, dim=-1).cpu().numpy()[0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--val_csv',     default='data/annotations/val_list_videos.txt')
    p.add_argument('--model_path',  default='models/videomae-ssv2')
    p.add_argument('--output_dir',  default='outputs/label_fusion')
    p.add_argument('--head_model',  default='outputs/head_best_run1/best_model.pt')
    p.add_argument('--body_flow',   default='outputs/body_flow/best_model.pt')
    p.add_argument('--body_mmn',    default='outputs/body_full_joint_aug/best_model.pt')
    p.add_argument('--leg_flow',    default='outputs/leg_flow_ssv2/best_model.pt')
    p.add_argument('--leg_mmn',     default='outputs/leg_leg_joint_aug/best_model.pt')
    p.add_argument('--hand_stage1',   default='outputs/hand_stage1/best_model.pt')
    p.add_argument('--hand_mmn_joint', default='outputs/hand_skeleton_full/best_model.pt')
    p.add_argument('--hand_mmn_bone',  default='outputs/hand_full_bone_aug/best_model.pt')
    p.add_argument('--hand_vmae_w',   type=float, default=0.4)
    p.add_argument('--hand_joint_w',  type=float, default=0.3)
    p.add_argument('--hand_bone_w',   type=float, default=0.3)
    p.add_argument('--hand_s2_C',   default='outputs/hand_stage2_C_B/best_model.pt')
    p.add_argument('--hand_s2_E',   default='outputs/hand_stage2_E_B/best_model.pt')
    p.add_argument('--hand_s2_F',   default='outputs/hand_stage2_F_B/best_model.pt')
    p.add_argument('--hand_s2_G',   default='outputs/hand_stage2_G/best_model.pt')
    p.add_argument('--body_flow_w', type=float, default=0.7)
    p.add_argument('--body_mmn_w',  type=float, default=0.3)
    p.add_argument('--leg_flow_w',  type=float, default=0.6)
    p.add_argument('--leg_mmn_w',   type=float, default=0.4)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load val annotations
    val_data = []
    with open(args.val_csv) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                val_data.append({'video': parts[0], 'fine_label': int(parts[1])})
    print(f"Val videos: {len(val_data)}")

    # Load all models
    print("\nLoading models...")
    head_model, head_proc = load_videomae(args.head_model, 7, args.model_path, device)
    body_flow_model, body_flow_proc = load_videomae(args.body_flow, 6, args.model_path, device)
    body_mmn_model = load_mmn(args.body_mmn, 6, 17, device)
    leg_flow_model, leg_flow_proc = load_videomae(args.leg_flow, 9, args.model_path, device)
    leg_mmn_model = load_mmn(args.leg_mmn, 9, 6, device)
    hand_s1, hand_s1_proc = load_videomae(args.hand_stage1, 5, args.model_path, device)
    hand_mmn_joint = load_mmn(args.hand_mmn_joint, 5, 17, device)
    hand_mmn_bone  = load_mmn(args.hand_mmn_bone,  5, 18, device, in_channels=3)
    hand_s2 = {
        'C': load_videomae(args.hand_s2_C, 13, args.model_path, device)[0],
        'E': load_videomae(args.hand_s2_E,  6, args.model_path, device)[0],
        'F': load_videomae(args.hand_s2_F, 10, args.model_path, device)[0],
        'G': load_videomae(args.hand_s2_G,  4, args.model_path, device)[0],
    }
    print("All models loaded ✅")

    from animate_wrong import COCO_EDGES
    LEG_JOINTS  = [11, 12, 13, 14, 15, 16]
    BODY_JOINTS = list(range(17))

    all_true_fine, all_pred_fine = [], []
    all_true_coarse, all_pred_coarse = [], []

    for item in tqdm(val_data, desc='Fusion inference'):
        vid_id      = item['video'].replace('.mp4','')
        true_fine   = item['fine_label']
        true_coarse = fine_to_coarse(true_fine)
        all_true_fine.append(true_fine)
        all_true_coarse.append(true_coarse)

        kp_path    = f"data/keypoints/val/{vid_id}.json"
        head_crop  = f"data/head_crops/val/{vid_id}.mp4"
        body_flow  = f"data/body_flow/val/{vid_id}.mp4"
        leg_flow   = f"data/leg_flow/val/{vid_id}.mp4"
        upper_crop = f"data/upperbody_crops/val/{vid_id}.mp4"

        track_probs = {}

        # Head
        if Path(head_crop).exists():
            probs = get_videomae_probs(head_model, head_proc, head_crop, device)
            active = probs[:HEAD_NO_MOVE]
            track_probs['head'] = (float(active.max()), HEAD_TO_FINE[int(active.argmax())])

        # Body ensemble
        if Path(body_flow).exists():
            fp = get_videomae_probs(body_flow_model, body_flow_proc, body_flow, device)
            mp = get_mmn_probs(body_mmn_model, kp_path, BODY_JOINTS, 'joint', device) \
                 if Path(kp_path).exists() else np.ones(6)/6
            ens = args.body_flow_w*fp + args.body_mmn_w*mp
            active = ens[:BODY_NO_MOVE]
            track_probs['body'] = (float(active.max()), BODY_TO_FINE[int(active.argmax())])

        # Leg ensemble
        if Path(leg_flow).exists():
            fp = get_videomae_probs(leg_flow_model, leg_flow_proc, leg_flow, device)
            mp = get_mmn_probs(leg_mmn_model, kp_path, LEG_JOINTS, 'joint', device) \
                 if Path(kp_path).exists() else np.ones(9)/9
            ens = args.leg_flow_w*fp + args.leg_mmn_w*mp
            active = ens[:LEG_NO_MOVE]
            track_probs['leg'] = (float(active.max()), LEG_TO_FINE[int(active.argmax())])

        # Hand two-stage with ensemble Stage 1
        if Path(upper_crop).exists():
            vmae_p  = get_videomae_probs(hand_s1, hand_s1_proc, upper_crop, device)
            joint_p = get_mmn_probs(hand_mmn_joint, kp_path, list(range(17)),
                                     'joint', device) if Path(kp_path).exists()                        else np.ones(5)/5
            bone_p  = get_mmn_probs(hand_mmn_bone, kp_path, list(range(17)),
                                     'bone', device, edges=COCO_EDGES)                        if Path(kp_path).exists() else np.ones(5)/5
            s1_p    = (args.hand_vmae_w * vmae_p +
                       args.hand_joint_w * joint_p +
                       args.hand_bone_w  * bone_p)
            s1_pred = int(s1_p.argmax())
            gmap = {0:'C', 1:'E', 2:'F', 3:'G', 4:None}
            grp  = gmap[s1_pred]
            if grp is not None:
                s2_p  = get_videomae_probs(hand_s2[grp], hand_s1_proc, upper_crop, device)
                fmaps = {'C':HAND_C_TO_FINE,'E':HAND_E_TO_FINE,
                         'F':HAND_F_TO_FINE,'G':HAND_G_TO_FINE}
                track_probs['hand'] = (float(s2_p.max())*float(s1_p.max()),
                                       fmaps[grp][int(s2_p.argmax())])

        if not track_probs:
            all_pred_fine.append(0); all_pred_coarse.append(0); continue

        best   = max(track_probs, key=lambda k: track_probs[k][0])
        pf     = track_probs[best][1]
        all_pred_fine.append(pf)
        all_pred_coarse.append(fine_to_coarse(pf))

    # Compute official metric
    atf = np.array(all_true_fine);   apf = np.array(all_pred_fine)
    atc = np.array(all_true_coarse); apc = np.array(all_pred_coarse)

    f1_bm = f1_score(atc, apc, average='macro',  zero_division=0)
    f1_bmi= f1_score(atc, apc, average='micro',  zero_division=0)
    f1_am = f1_score(atf, apf, average='macro',  zero_division=0)
    f1_ami= f1_score(atf, apf, average='micro',  zero_division=0)
    f1    = (f1_bm + f1_bmi + f1_am + f1_ami) / 4

    print(f"\n{'='*60}")
    print(f"OFFICIAL COMPETITION METRIC")
    print(f"F1_body_macro:   {f1_bm:.4f}")
    print(f"F1_body_micro:   {f1_bmi:.4f}")
    print(f"F1_action_macro: {f1_am:.4f}")
    print(f"F1_action_micro: {f1_ami:.4f}")
    print(f"{'='*60}")
    print(f"F1_MEAN (official): {f1:.4f}")
    print(f"{'='*60}")

    COARSE_NAMES = ['A-body','B-head','C-upper','D-lower',
                    'E-body-hand','F-head-hand','G-leg-hand']
    print(classification_report(atc, apc, target_names=COARSE_NAMES, zero_division=0))

    pd.DataFrame({'video':[d['video'] for d in val_data],
                  'true_fine':atf,'pred_fine':apf,
                  'true_coarse':atc,'pred_coarse':apc}
                 ).to_csv(f"{args.output_dir}/predictions.csv", index=False)
    print(f"Saved: {args.output_dir}/predictions.csv")

if __name__ == '__main__':
    main()
