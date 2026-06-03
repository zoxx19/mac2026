"""
hand_fine_dataset.py  (v2 — 16 frames for VideoMAE-SSv2)

VideoMAE-SSv2 was pretrained with 16 frames → position embeddings = 1568 tokens.
Passing 8 frames gives 784 tokens → size mismatch error.
Fix: always sample exactly 16 frames.

Returns:
  rgb:  (3, 16, 224, 224)  float32  — permute to (16, 3, H, W) before VideoMAE
  skel: (2, 64, V, 1)      float32
  label: int
"""

import os
import json
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

try:
    import decord
    decord.bridge.set_bridge('torch')
    USE_DECORD = True
except ImportError:
    USE_DECORD = False
    import av

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_FRAMES_RGB  = 16     # VideoMAE-SSv2 requires exactly 16 frames
FRAME_SIZE      = 224
NUM_FRAMES_SKEL = 64


def augment_skeleton(skeleton, rot=15, scale=(0.9, 1.1), trans=0.1, jitter=3):
    T, V, _ = skeleton.shape
    angle = math.radians(random.uniform(-rot, rot))
    ca, sa = math.cos(angle), math.sin(angle)
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
    skeleton = skeleton @ R.T
    skeleton *= random.uniform(*scale)
    skeleton += np.array([random.uniform(-trans, trans),
                           random.uniform(-trans, trans)], dtype=np.float32)
    shift = random.randint(-jitter, jitter)
    if shift > 0:
        skeleton = np.concatenate([skeleton[shift:],  skeleton[-shift:]], axis=0)
    elif shift < 0:
        skeleton = np.concatenate([skeleton[:shift],  skeleton[:-shift]], axis=0)
    return skeleton


def load_video_frames(video_path, num_frames=NUM_FRAMES_RGB, size=FRAME_SIZE):
    """Returns (T, H, W, 3) uint8 numpy array with exactly num_frames frames."""
    if USE_DECORD:
        vr     = decord.VideoReader(video_path, width=size, height=size)
        total  = len(vr)
        idxs   = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(idxs).numpy()
    else:
        container = av.open(video_path)
        stream    = container.streams.video[0]
        all_f     = [f.to_ndarray(format='rgb24')
                     for f in container.decode(stream)]
        container.close()
        total  = len(all_f)
        idxs   = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = np.stack([all_f[i] for i in idxs])
        if frames.shape[1] != size or frames.shape[2] != size:
            import cv2
            frames = np.stack([cv2.resize(f, (size, size)) for f in frames])
    return frames   # (T, H, W, 3)


def preprocess_frames(frames, hflip=False):
    """(T, H, W, 3) uint8 → (3, T, H, W) float32 normalised."""
    if hflip:
        frames = frames[:, :, ::-1, :].copy()
    frames = frames.astype(np.float32) / 255.0
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD   # (T, H, W, 3)
    frames = frames.transpose(3, 0, 1, 2)              # (3, T, H, W)
    return frames


def load_skeleton(kp_path, joint_indices, num_frames=NUM_FRAMES_SKEL, augment=False):
    """Returns (2, T, V, 1) float32."""
    with open(kp_path) as f:
        data = json.load(f)
    all_kps = []
    for frame in data:
        kps = np.array(frame['keypoints'], dtype=np.float32)[:, :2]
        all_kps.append(kps)
    skel = np.stack(all_kps, axis=0)   # (T_raw, 17, 2)

    mins = skel.min(axis=(0, 1), keepdims=True)
    maxs = skel.max(axis=(0, 1), keepdims=True)
    rng  = np.where(maxs - mins > 1e-6, maxs - mins, 1.0)
    skel = (skel - mins) / rng * 2 - 1

    if augment:
        skel = augment_skeleton(skel)

    skel = skel[:, joint_indices, :]   # (T_raw, V, 2)

    T_raw = skel.shape[0]
    if T_raw >= num_frames:
        idxs = np.linspace(0, T_raw - 1, num_frames, dtype=int)
        skel = skel[idxs]
    else:
        pad  = num_frames - T_raw
        skel = np.concatenate([skel, np.tile(skel[-1:], (pad, 1, 1))], axis=0)

    skel = skel.transpose(2, 0, 1)[:, :, :, np.newaxis]   # (2, T, V, 1)
    return skel.astype(np.float32)


class HandFineDataset(Dataset):
    def __init__(self, csv_path, kp_dir, joint_indices=None,
                 augment=False, rgb_augment=False):
        self.df            = pd.read_csv(csv_path)
        self.kp_dir        = kp_dir
        self.joint_indices = joint_indices if joint_indices else list(range(17))
        self.augment       = augment
        self.rgb_augment   = rgb_augment
        self.V             = len(self.joint_indices)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        video_id   = row['video_id']
        label      = int(row['local_label'])
        video_path = row['video_path']
        has_kp     = bool(row['has_keypoints'])

        # ── RGB: (3, 16, 224, 224) ────────────────────────────────────────────
        try:
            frames = load_video_frames(video_path, num_frames=NUM_FRAMES_RGB)
            hflip  = self.rgb_augment and random.random() < 0.5
            rgb    = preprocess_frames(frames, hflip=hflip)
        except Exception:
            rgb = np.zeros((3, NUM_FRAMES_RGB, FRAME_SIZE, FRAME_SIZE),
                           dtype=np.float32)

        # ── Skeleton: (2, 64, V, 1) ───────────────────────────────────────────
        kp_path = os.path.join(self.kp_dir, f"{video_id}.json")
        if has_kp and os.path.exists(kp_path):
            skel = load_skeleton(kp_path, self.joint_indices,
                                 augment=self.augment)
        else:
            skel = np.zeros((2, NUM_FRAMES_SKEL, self.V, 1), dtype=np.float32)

        return (torch.from_numpy(rgb),
                torch.from_numpy(skel),
                label)
