"""
feeder_mmad_52class.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/feeder_mmad_52class.py --help` for options where applicable.
"""
import os
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset

# MediaPipe 33-joint indices to keep
JOINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 3, 4]
# new index → MediaPipe joint
# 0→nose, 1→L.ear, 2→R.ear, 3→L.shoulder, 4→R.shoulder,
# 5→L.elbow, 6→R.elbow, 7→L.wrist, 8→R.wrist,
# 9→L.hip, 10→R.hip, 11→L.knee, 12→R.knee,
# 13→L.ankle, 14→R.ankle, 15→L.eye_outer, 16→R.eye_inner

NUM_CLASSES = 52


class Feeder(Dataset):
    def __init__(self, split, keypoint_dir, ann_csv,
                 window_size=32, stride=8, overlap_threshold=0.5,
                 data_type='j', p=0.5):
        self.split = split
        self.time_steps = window_size
        self.window_size = window_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
        self.data_type = data_type
        self.p = p

        self.bone = [
            (3, 5), (5, 7),    # L shoulder → elbow → wrist
            (4, 6), (6, 8),    # R shoulder → elbow → wrist
            (3, 4), (5, 6),    # shoulder pair, elbow pair
            (9, 11), (11, 13), # L hip → knee → ankle
            (10, 12), (12, 14),# R hip → knee → ankle
            (9, 10), (11, 12), # hip pair, knee pair
            (0, 15), (0, 16),  # nose → eye_outer / eye_inner
        ]

        self.load_data(keypoint_dir, ann_csv)

    def load_data(self, keypoint_dir, ann_csv):
        df = pd.read_csv(ann_csv)
        ann_lookup = {}
        for _, row in df.iterrows():
            vid = str(row['video_id'])
            if vid not in ann_lookup:
                ann_lookup[vid] = []
            ann_lookup[vid].append((int(row['start_frame']), int(row['end_frame']), int(row['class'])))

        all_windows = []

        for npy_file in sorted(os.listdir(keypoint_dir)):
            if not npy_file.endswith('_keypoints.npy'):
                continue
            video_id = npy_file.replace('_keypoints.npy', '')
            kp = np.load(os.path.join(keypoint_dir, npy_file))  # (N, 33, 4)
            kp = kp[:, JOINTS, :2].astype(np.float32)           # (N, 17, 2)
            N = kp.shape[0]
            annotations = ann_lookup.get(video_id, [])

            for t_start in range(0, max(1, N - self.window_size + 1), self.stride):
                t_end = t_start + self.window_size
                if t_end > N:
                    break
                window = kp[t_start:t_end]  # (window_size, 17, 2)

                label = np.zeros(NUM_CLASSES, dtype=np.float32)
                for start, end, class_id in annotations:
                    overlap = max(0, min(t_end, end) - max(t_start, start))
                    if overlap / self.window_size >= self.overlap_threshold:
                        label[class_id] = 1.0

                all_windows.append((window, label))

        if self.split == 'train':
            random.shuffle(all_windows)

        self.data = [w for w, _ in all_windows]
        self.label = [lbl for _, lbl in all_windows]

        n_pos = sum(1 for lbl in self.label if lbl.any())
        n_neg = len(self.label) - n_pos
        print(f"[{self.split}] windows_with_action={n_pos}, background={n_neg}, total={len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        channel = 2
        label = self.label[index]
        value = self.data[index].copy()  # (window_size, 17, 2)

        # center on hip midpoint (new idx 9=L.hip, 10=R.hip)
        center = (value[0, 9, :] + value[0, 10, :]) / 2.0

        if self.split == 'train':

            def affine_transform(v):
                angle = np.random.uniform(-15, 15) * np.pi / 180
                cos_val, sin_val = np.cos(angle), np.sin(angle)
                rotation = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
                scale = np.random.uniform(0.9, 1.1)
                scale_matrix = np.eye(2) * scale
                translation = np.random.uniform(-0.1, 0.1, size=(1, 1, 2))
                v = np.matmul(v, rotation.T)
                v = np.matmul(v, scale_matrix)
                v += translation
                return v

            def temporal_jitter(v, max_jitter=3):
                T = v.shape[0]
                jittered = np.zeros_like(v)
                for t in range(T):
                    offset = np.random.randint(-max_jitter, max_jitter + 1)
                    new_t = min(max(t + offset, 0), T - 1)
                    jittered[t] = v[new_t]
                return jittered

            value = value - center

            scalerValue = np.reshape(value, (-1, channel))
            epsilon = 1e-6
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (
                np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0) + epsilon)
            scalerValue = scalerValue * 2 - 1
            value = np.reshape(scalerValue, (-1, value.shape[1], channel))

            length = value.shape[0]
            random_idx = random.sample(list(np.arange(length)) * self.time_steps, self.time_steps)
            random_idx.sort()
            data = np.zeros((self.time_steps, value.shape[1], channel))
            data[:, :, :] = value[random_idx, :, :]
            index_t = 2 * np.array(random_idx).astype(np.float32) / length - 1

            if random.random() < self.p:
                value = affine_transform(value)

            if random.random() < self.p:
                value = temporal_jitter(value)

            if random.random() < self.p:
                value = value[::-1]

            if random.random() < self.p:
                axis_next = random.randint(0, 1)
                data[:, :, axis_next] = 0

            if random.random() < self.p:
                T, V, C = data.shape
                frame_count = random.randint(1, 16)
                frames_to_drop = random.sample(range(T), frame_count)
                data[frames_to_drop, :, :] = 0

            if random.random() < self.p:
                block_size = random.randint(4, 16)
                start = random.randint(0, self.time_steps - block_size)
                data[start:start + block_size, :, :] = 0

            # Gaussian noise on joint coordinates (σ=0.02 in normalised [-1,1] space)
            if random.random() < self.p:
                data = data + np.random.normal(0, 0.02, data.shape).astype(np.float32)

            # Random joint masking — zero out 1-4 joints entirely
            if random.random() < self.p:
                n_mask = random.randint(1, 4)
                joints_to_mask = random.sample(range(data.shape[1]), n_mask)
                data[:, joints_to_mask, :] = 0

        else:
            value = value - center

            scalerValue = np.reshape(value, (-1, channel))
            epsilon = 1e-6
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (
                np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0) + epsilon)
            scalerValue = scalerValue * 2 - 1
            value = np.reshape(scalerValue, (-1, value.shape[1], channel))

            length = value.shape[0]
            idx = np.linspace(0, length - 1, self.time_steps).astype(int)
            data = np.zeros((self.time_steps, value.shape[1], channel))
            data[:, :, :] = value[idx, :, :]
            index_t = 2 * idx.astype(np.float32) / length - 1

        if 'b' in self.data_type:
            data_bone = np.zeros_like(data)
            for v1, v2 in self.bone:
                data_bone[:, v1, :] = data[:, v1, :] - data[:, v2, :]
            data = data_bone

        if 'm' in self.data_type:
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion

        # (T, V, C) → (C, T, V, 1)
        data = np.transpose(data, (2, 0, 1))
        C, T, V = data.shape
        data = np.reshape(data, (C, T, V, 1))

        return data, index_t, label, index


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
