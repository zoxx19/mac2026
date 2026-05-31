"""
src/skeleton/features.py

Loads 17-joint COCO keypoints from YOLOv8-pose JSON files and builds
MMN-compatible input tensors.

JSON format (per file):
    list of frames, each frame:
        {
          "keypoints": [[x,y,conf], ...],   # 17 joints
          "boxes": { "head":..., ... }
        }

MMN input shape: (B, C, T, V, M)
    C = 2  (x, y  — confidence dropped, matching MMN default in_channels=2)
    T = num_frames (resampled to fixed length)
    V = num joints (17 full-body OR 6 arm-only)
    M = 1  (single person)

COCO joint indices:
    0  nose
    1  left_eye      2  right_eye
    3  left_ear      4  right_ear
    5  left_shoulder 6  right_shoulder
    7  left_elbow    8  right_elbow
    9  left_wrist   10  right_wrist
   11  left_hip     12  right_hip
   13  left_knee    14  right_knee
   15  left_ankle   16  right_ankle
"""

import json
import numpy as np
from typing import Literal

# ── Graph definitions ────────────────────────────────────────────────────────

FULL_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# Arm-only: shoulders(5,6) + elbows(7,8) + wrists(9,10)
ARM_JOINT_IDX = [5, 6, 7, 8, 9, 10]   # original COCO indices to keep
ARM_EDGES = [                           # remapped to 0-5
    (0, 1),  # left_shoulder  - right_shoulder
    (0, 2),  # left_shoulder  - left_elbow
    (2, 4),  # left_elbow     - left_wrist
    (1, 3),  # right_shoulder - right_elbow
    (3, 5),  # right_elbow    - right_wrist
]

# ── JSON loading ─────────────────────────────────────────────────────────────

def load_keypoints(kp_path: str) -> np.ndarray:
    """
    Load JSON file -> numpy array (T, 17, 3) with [x, y, conf].
    Frames with no detection are filled with zeros.
    """
    with open(kp_path) as f:
        data = json.load(f)   # list of frame dicts

    T = len(data)
    kps = np.zeros((T, 17, 3), dtype=np.float32)

    for t, frame in enumerate(data):
        if frame and "keypoints" in frame:
            raw = np.array(frame["keypoints"], dtype=np.float32)  # (17, 3)
            if raw.shape == (17, 3):
                kps[t] = raw

    return kps  # (T, 17, 3)

# ── Temporal resampling ───────────────────────────────────────────────────────

def resample_frames(kps: np.ndarray, num_frames: int = 64) -> np.ndarray:
    """Uniformly sample or pad to exactly num_frames. (T,V,3) -> (num_frames,V,3)"""
    T = kps.shape[0]
    if T == 0:
        return np.zeros((num_frames, kps.shape[1], 3), dtype=np.float32)
    if T == num_frames:
        return kps
    idx = np.round(np.linspace(0, T - 1, num_frames)).astype(int)
    return kps[idx]

# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_keypoints(kps: np.ndarray) -> np.ndarray:
    """
    Centre on hip midpoint and scale by torso height.
    Falls back to mean of all visible joints if hips not visible.
    Operates on x,y only; conf is preserved.
    kps: (T, 17, 3) -> (T, 17, 3)
    """
    kps = kps.copy()
    xy = kps[:, :, :2]  # (T, 17, 2)

    hip_conf = kps[:, 11, 2] * kps[:, 12, 2]
    valid = hip_conf > 0

    if valid.sum() == 0:
        vis = kps[:, :, 2] > 0
        center = (xy * vis[:, :, None]).sum(axis=(0, 1)) / (vis.sum() + 1e-6)
        xy -= center
        scale = np.abs(xy[vis]).max() + 1e-6
        xy /= scale
    else:
        hip_mid = (xy[valid, 11] + xy[valid, 12]) / 2
        center = hip_mid.mean(axis=0)
        xy -= center
        sho_mid  = (xy[valid, 5]  + xy[valid, 6])  / 2
        hip_mid2 = (xy[valid, 11] + xy[valid, 12]) / 2
        scale = np.linalg.norm(sho_mid - hip_mid2, axis=1).mean() + 1e-6
        xy /= scale

    kps[:, :, :2] = xy
    return kps

# ── Build MMN input tensor ────────────────────────────────────────────────────

def build_tensor(kps: np.ndarray) -> np.ndarray:
    """
    Drop confidence, keep only x and y.
    (T, V, 3) -> (2, T, V, 1)  i.e. (C, T, V, M)
    """
    xy = kps[:, :, :2]              # (T, V, 2)
    out = xy.transpose(2, 0, 1)     # (2, T, V)
    return out[:, :, :, None]       # (2, T, V, 1)

# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(
    kp_path: str,
    mode: Literal["full", "arm"] = "full",
    num_frames: int = 64,
) -> np.ndarray:
    """
    Load keypoints and return MMN input tensor.

    Returns:
        tensor: (2, T, V, 1)
            T = num_frames
            V = 17 if mode=="full", 6 if mode=="arm"
    """
    kps = load_keypoints(kp_path)           # (T, 17, 3)
    kps = normalize_keypoints(kps)          # (T, 17, 3)
    kps = resample_frames(kps, num_frames)  # (num_frames, 17, 3)

    if mode == "arm":
        kps = kps[:, ARM_JOINT_IDX, :]     # (T, 6, 3)

    return build_tensor(kps)               # (2, T, V, 1)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    for mode in ("full", "arm"):
        t = extract_features(path, mode=mode)
        V = 17 if mode == "full" else 6
        assert t.shape == (2, 64, V, 1), f"unexpected shape {t.shape}"
        print(f"[{mode:4s}]  shape={t.shape}  min={t[0].min():.3f}  max={t[0].max():.3f}")
    print("OK")