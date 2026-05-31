
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
    C = 2  (x, y  — joint modality)
    C = 3  (dx, dy, dist — bone modality)
    T = num_frames
    V = num joints / num bones
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
 
# ── Graph definitions ─────────────────────────────────────────────────────────
 
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
 
# Joint subsets
ARM_JOINT_IDX   = [5, 6, 7, 8, 9, 10]       # shoulders+elbows+wrists
LEG_JOINT_IDX   = [11, 12, 13, 14, 15, 16]  # hips+knees+ankles
TORSO_JOINT_IDX = [5, 6, 11, 12]            # shoulders+hips
 
# Bone edges for each mode (remapped to local indices)
ARM_EDGES = [
    (0, 1),  # left_shoulder  - right_shoulder
    (0, 2),  # left_shoulder  - left_elbow
    (2, 4),  # left_elbow     - left_wrist
    (1, 3),  # right_shoulder - right_elbow
    (3, 5),  # right_elbow    - right_wrist
]
 
LEG_EDGES = [
    (0, 1),  # left_hip   - right_hip
    (0, 2),  # left_hip   - left_knee
    (2, 4),  # left_knee  - left_ankle
    (1, 3),  # right_hip  - right_knee
    (3, 5),  # right_knee - right_ankle
]
 
TORSO_EDGES = [
    (0, 1),  # left_shoulder  - right_shoulder
    (0, 2),  # left_shoulder  - left_hip
    (1, 3),  # right_shoulder - right_hip
    (2, 3),  # left_hip       - right_hip
]
 
MODE_JOINTS = {
    "full":  list(range(17)),
    "arm":   ARM_JOINT_IDX,
    "leg":   LEG_JOINT_IDX,
    "torso": TORSO_JOINT_IDX,
}
 
MODE_EDGES = {
    "full":  FULL_EDGES,
    "arm":   ARM_EDGES,
    "leg":   LEG_EDGES,
    "torso": TORSO_EDGES,
}
 
# ── JSON loading ──────────────────────────────────────────────────────────────
 
def load_keypoints(kp_path: str) -> np.ndarray:
    """
    Load JSON file -> numpy array (T, 17, 3) with [x, y, conf].
    Frames with no detection are filled with zeros.
    """
    with open(kp_path) as f:
        data = json.load(f)
 
    T   = len(data)
    kps = np.zeros((T, 17, 3), dtype=np.float32)
 
    for t, frame in enumerate(data):
        if frame and "keypoints" in frame:
            raw = np.array(frame["keypoints"], dtype=np.float32)
            if raw.shape == (17, 3):
                kps[t] = raw
 
    return kps  # (T, 17, 3)
 
# ── Temporal resampling ───────────────────────────────────────────────────────
 
def resample_frames(kps: np.ndarray, num_frames: int = 64) -> np.ndarray:
    """Uniformly sample or pad to exactly num_frames. (T,V,C) -> (num_frames,V,C)"""
    T = kps.shape[0]
    if T == 0:
        return np.zeros((num_frames, kps.shape[1], kps.shape[2]), dtype=np.float32)
    if T == num_frames:
        return kps
    idx = np.round(np.linspace(0, T - 1, num_frames)).astype(int)
    return kps[idx]
 
# ── Normalisation ─────────────────────────────────────────────────────────────
 
def normalize_keypoints(kps: np.ndarray) -> np.ndarray:
    """
    Centre on hip midpoint and scale by torso height.
    Falls back to mean of all visible joints if hips not visible.
    kps: (T, 17, 3) -> (T, 17, 3)
    """
    kps  = kps.copy()
    xy   = kps[:, :, :2]
 
    hip_conf = kps[:, 11, 2] * kps[:, 12, 2]
    valid    = hip_conf > 0
 
    if valid.sum() == 0:
        vis    = kps[:, :, 2] > 0
        center = (xy * vis[:, :, None]).sum(axis=(0, 1)) / (vis.sum() + 1e-6)
        xy    -= center
        scale  = np.abs(xy[vis]).max() + 1e-6
        xy    /= scale
    else:
        hip_mid = (xy[valid, 11] + xy[valid, 12]) / 2
        center  = hip_mid.mean(axis=0)
        xy     -= center
        sho_mid  = (xy[valid, 5]  + xy[valid, 6])  / 2
        hip_mid2 = (xy[valid, 11] + xy[valid, 12]) / 2
        scale    = np.linalg.norm(sho_mid - hip_mid2, axis=1).mean() + 1e-6
        xy      /= scale
 
    kps[:, :, :2] = xy
    return kps
 
# ── Joint modality ────────────────────────────────────────────────────────────
 
def build_joint(kps: np.ndarray) -> np.ndarray:
    """
    Joint modality: x, y only.
    (T, V, 3) -> (2, T, V, 1)
    """
    xy  = kps[:, :, :2]             # (T, V, 2)
    out = xy.transpose(2, 0, 1)     # (2, T, V)
    return out[:, :, :, None]       # (2, T, V, 1)
 
# ── Bone modality ─────────────────────────────────────────────────────────────
 
def build_bone(kps: np.ndarray, edges: list) -> np.ndarray:
    """
    Bone modality: for each edge (src->dst) compute [dx, dy, dist].
    Uses confidence of both endpoints as weight.
    (T, V, 3) -> (3, T, E, 1)  where E = len(edges)
    """
    T = kps.shape[0]
    E = len(edges)
    bone = np.zeros((T, E, 3), dtype=np.float32)
 
    for e_idx, (src, dst) in enumerate(edges):
        diff = kps[:, dst, :2] - kps[:, src, :2]          # (T, 2)
        dist = np.linalg.norm(diff, axis=1)                # (T,)
        conf = kps[:, src, 2] * kps[:, dst, 2]            # (T,)
        bone[:, e_idx, 0] = diff[:, 0] * conf
        bone[:, e_idx, 1] = diff[:, 1] * conf
        bone[:, e_idx, 2] = dist * conf
 
    out = bone.transpose(2, 0, 1)   # (3, T, E)
    return out[:, :, :, None]       # (3, T, E, 1)
 
# ── Public API ────────────────────────────────────────────────────────────────
 
def extract_features(
    kp_path: str,
    mode: Literal["full", "arm", "leg", "torso"] = "full",
    modality: Literal["joint", "bone"] = "joint",
    num_frames: int = 64,
) -> np.ndarray:
    """
    Load keypoints and return MMN input tensor.
 
    Args:
        kp_path:    path to keypoint JSON file
        mode:       joint subset — full/arm/leg/torso
        modality:   joint (x,y) or bone (dx,dy,dist)
        num_frames: temporal length
 
    Returns:
        joint: (2, T, V, 1)
        bone:  (3, T, E, 1)
    """
    kps = load_keypoints(kp_path)           # (T, 17, 3)
    kps = normalize_keypoints(kps)          # (T, 17, 3)
    kps = resample_frames(kps, num_frames)  # (num_frames, 17, 3)
 
    joint_idx = MODE_JOINTS[mode]
    edges     = MODE_EDGES[mode]
    kps       = kps[:, joint_idx, :]        # (T, V, 3)
 
    if modality == "bone":
        return build_bone(kps, edges)       # (3, T, E, 1)
    else:
        return build_joint(kps)             # (2, T, V, 1)
 
 
# ── Smoke test ────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import sys
    path = sys.argv[1]
 
    expected = {
        ("full",  "joint"): (2, 64, 17, 1),
        ("full",  "bone"):  (3, 64, 18, 1),
        ("arm",   "joint"): (2, 64,  6, 1),
        ("arm",   "bone"):  (3, 64,  5, 1),
        ("leg",   "joint"): (2, 64,  6, 1),
        ("leg",   "bone"):  (3, 64,  5, 1),
        ("torso", "joint"): (2, 64,  4, 1),
        ("torso", "bone"):  (3, 64,  4, 1),
    }
 
    all_ok = True
    for (mode, mod), exp_shape in expected.items():
        t = extract_features(path, mode=mode, modality=mod)
        status = "OK" if t.shape == exp_shape else f"FAIL got {t.shape}"
        print(f"  [{mode:5s} {mod:5s}]  shape={t.shape}  {status}")
        if t.shape != exp_shape:
            all_ok = False
 
    print("\nAll OK" if all_ok else "\nSome shapes wrong!")