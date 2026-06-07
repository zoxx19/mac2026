"""
generate_submission.py
Generates Top-5 Kaggle submission CSV for MAC2026 Track 1.

IMPORTANT: Run preprocess_test.py FIRST to extract crops and flows.

For each test video:
  1. Runs all track models on preprocessed crops/flows
  2. Builds a 52-class probability distribution
  3. Takes Top-5 action predictions (0-51)
  4. Derives Top-5 body predictions (0-6) from Top-5 action predictions

Output format:
  vid,action_pred_1,...,action_pred_5,body_pred_1,...,body_pred_5

Usage:
  python src/fusion/generate_submission.py \
      --test_dir  data/test/videos \
      --output    outputs/submission.csv
"""

import sys, os, torch, numpy as np, pandas as pd
sys.path.insert(0, 'src/skeleton/MMN')
sys.path.insert(0, 'src/skeleton')
sys.path.insert(0, 'src/hand')
sys.path.insert(0, 'src/fusion')

from label_fusion import (load_videomae, load_mmn, fine_to_coarse,
                           get_videomae_probs, get_mmn_probs,
                           HEAD_TO_FINE, HEAD_NO_MOVE,
                           BODY_TO_FINE, BODY_NO_MOVE,
                           LEG_TO_FINE, LEG_NO_MOVE,
                           HAND_C_TO_FINE, HAND_E_TO_FINE,
                           HAND_F_TO_FINE, HAND_G_TO_FINE)
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings; warnings.filterwarnings("ignore")

NUM_FINE   = 52
NUM_COARSE = 7


def build_52d_probs(track_raw):
    """Combine per-track probabilities into a 52-class distribution."""
    probs52 = np.zeros(NUM_FINE, dtype=np.float32)

    if 'head' in track_raw:
        p = track_raw['head']
        for local, fine in HEAD_TO_FINE.items():
            if local < len(p): probs52[fine] += p[local]

    if 'body' in track_raw:
        p = track_raw['body']
        for local, fine in BODY_TO_FINE.items():
            if local < len(p): probs52[fine] += p[local]

    if 'leg' in track_raw:
        p = track_raw['leg']
        for local, fine in LEG_TO_FINE.items():
            if local < len(p): probs52[fine] += p[local]

    if 'hand' in track_raw:
        p, grp = track_raw['hand']
        fmaps = {'C': HAND_C_TO_FINE, 'E': HAND_E_TO_FINE,
                 'F': HAND_F_TO_FINE, 'G': HAND_G_TO_FINE}
        if grp in fmaps:
            for local, fine in fmaps[grp].items():
                if local < len(p): probs52[fine] += p[local]

    total = probs52.sum()
    if total > 0: probs52 /= total
    else: probs52 = np.ones(NUM_FINE) / NUM_FINE
    return probs52


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_dir',      default='data/test/videos')
    p.add_argument('--output',        default='outputs/submission.csv')
    p.add_argument('--model_path',    default='models/videomae-ssv2')
    p.add_argument('--head_model',    default='outputs/head_best_run1/best_model.pt')
    p.add_argument('--body_flow',     default='outputs/body_flow/best_model.pt')
    p.add_argument('--body_mmn',      default='outputs/body_full_joint_aug/best_model.pt')
    p.add_argument('--leg_flow',      default='outputs/leg_flow_ssv2/best_model.pt')
    p.add_argument('--leg_mmn',       default='outputs/leg_leg_joint_aug/best_model.pt')
    p.add_argument('--hand_stage1',   default='outputs/hand_stage1/best_model.pt')
    p.add_argument('--hand_mmn_joint',default='outputs/hand_skeleton_full/best_model.pt')
    p.add_argument('--hand_mmn_bone', default='outputs/hand_full_bone_aug/best_model.pt')
    p.add_argument('--hand_s2_C',     default='outputs/hand_stage2_C_B/best_model.pt')
    p.add_argument('--hand_s2_E',     default='outputs/hand_stage2_E_B/best_model.pt')
    p.add_argument('--hand_s2_F',     default='outputs/hand_stage2_F_B/best_model.pt')
    p.add_argument('--hand_s2_G',     default='outputs/hand_stage2_G/best_model.pt')
    p.add_argument('--body_flow_w',   type=float, default=0.7)
    p.add_argument('--body_mmn_w',    type=float, default=0.3)
    p.add_argument('--leg_flow_w',    type=float, default=0.6)
    p.add_argument('--leg_mmn_w',     type=float, default=0.4)
    p.add_argument('--hand_vmae_w',   type=float, default=0.4)
    p.add_argument('--hand_joint_w',  type=float, default=0.3)
    p.add_argument('--hand_bone_w',   type=float, default=0.3)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_videos = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.mp4')])
    print(f"Test videos: {len(test_videos)}")

    # Load all models
    print("\nLoading models...")
    head_model, head_proc         = load_videomae(args.head_model,    7,  args.model_path, device)
    body_flow_model, body_flow_proc = load_videomae(args.body_flow,   6,  args.model_path, device)
    body_mmn_model                = load_mmn(args.body_mmn,           6,  17, device)
    leg_flow_model, leg_flow_proc = load_videomae(args.leg_flow,      9,  args.model_path, device)
    leg_mmn_model                 = load_mmn(args.leg_mmn,            9,  6,  device)
    hand_s1, hand_s1_proc         = load_videomae(args.hand_stage1,   5,  args.model_path, device)
    hand_mmn_joint                = load_mmn(args.hand_mmn_joint,     5,  17, device)
    hand_mmn_bone                 = load_mmn(args.hand_mmn_bone,      5,  18, device, in_channels=3)
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

    rows = []

    for vid_name in tqdm(test_videos, desc='Test inference'):
        vid_id = vid_name.replace('.mp4', '')

        head_crop  = f"data/test/head_crops/{vid_id}.mp4"
        body_flow  = f"data/test/body_flow/{vid_id}.mp4"
        leg_flow   = f"data/test/leg_flow/{vid_id}.mp4"
        upper_crop = f"data/test/upperbody_crops/{vid_id}.mp4"
        kp_path    = f"data/test/keypoints/{vid_id}.json"

        track_raw = {}

        # Head
        if Path(head_crop).exists():
            probs = get_videomae_probs(head_model, head_proc, head_crop, device)
            track_raw['head'] = probs[:HEAD_NO_MOVE]

        # Body ensemble
        if Path(body_flow).exists():
            fp  = get_videomae_probs(body_flow_model, body_flow_proc, body_flow, device)
            mp  = get_mmn_probs(body_mmn_model, kp_path, BODY_JOINTS, 'joint', device) \
                  if Path(kp_path).exists() else np.ones(6)/6
            ens = args.body_flow_w*fp + args.body_mmn_w*mp
            track_raw['body'] = ens[:BODY_NO_MOVE]

        # Leg ensemble
        if Path(leg_flow).exists():
            fp  = get_videomae_probs(leg_flow_model, leg_flow_proc, leg_flow, device)
            mp  = get_mmn_probs(leg_mmn_model, kp_path, LEG_JOINTS, 'joint', device) \
                  if Path(kp_path).exists() else np.ones(9)/9
            ens = args.leg_flow_w*fp + args.leg_mmn_w*mp
            track_raw['leg'] = ens[:LEG_NO_MOVE]

        # Hand ensemble Stage1 + Stage2
        if Path(upper_crop).exists():
            vmae_p  = get_videomae_probs(hand_s1, hand_s1_proc, upper_crop, device)
            joint_p = get_mmn_probs(hand_mmn_joint, kp_path, list(range(17)),
                                     'joint', device) if Path(kp_path).exists() \
                       else np.ones(5)/5
            bone_p  = get_mmn_probs(hand_mmn_bone, kp_path, list(range(17)),
                                     'bone', device, edges=COCO_EDGES) \
                       if Path(kp_path).exists() else np.ones(5)/5
            s1_p    = (args.hand_vmae_w*vmae_p +
                       args.hand_joint_w*joint_p +
                       args.hand_bone_w*bone_p)
            s1_pred = int(s1_p.argmax())
            gmap    = {0:'C', 1:'E', 2:'F', 3:'G', 4:None}
            grp     = gmap[s1_pred]
            if grp is not None:
                s2_p = get_videomae_probs(hand_s2[grp], hand_s1_proc,
                                           upper_crop, device)
                track_raw['hand'] = (s2_p, grp)

        # Build 52-class distribution and get Top-5
        probs52     = build_52d_probs(track_raw)
        top5_action = np.argsort(probs52)[::-1][:5].tolist()
        top5_body   = [fine_to_coarse(a) for a in top5_action]

        rows.append({
            'vid':           vid_name,
            'action_pred_1': top5_action[0],
            'action_pred_2': top5_action[1],
            'action_pred_3': top5_action[2],
            'action_pred_4': top5_action[3],
            'action_pred_5': top5_action[4],
            'body_pred_1':   top5_body[0],
            'body_pred_2':   top5_body[1],
            'body_pred_3':   top5_body[2],
            'body_pred_4':   top5_body[3],
            'body_pred_5':   top5_body[4],
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nSubmission saved: {args.output}")
    print(f"Total videos: {len(df)}")
    print(f"\nSample:")
    print(df.head(3).to_string())

if __name__ == '__main__':
    main()
