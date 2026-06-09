"""
feeder_hand_detection.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/feeder_hand_detection.py --help` for options where applicable.
"""
import os
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, split, keypoint_dir, ann_csv,
                 window_size=30, stride=8, overlap_threshold=0.5,
                 b7_ratio=2.0, data_type='j', p=0.5):
        self.split = split
        self.time_steps = window_size
        self.window_size = window_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
        self.b7_ratio = b7_ratio
        self.data_type = data_type
        self.p = p
        self.partition = False

        # 6-joint arm connections: 0=L.shoulder, 1=R.shoulder, 2=L.elbow, 3=R.elbow, 4=L.wrist, 5=R.wrist
        self.bone = [
            (2, 4), (4, 0),   # left shoulder -> left elbow -> left wrist
            (3, 5), (5, 1),   # right shoulder -> right elbow -> right wrist
            (0, 1),           # left shoulder -> right shoulder
            (2, 3),           # left elbow -> right elbow (cross connection)
        ]

        self.load_data(keypoint_dir, ann_csv)

    def load_data(self, keypoint_dir, ann_csv):
        df = pd.read_csv(ann_csv)
        ann_lookup = {}
        for _, row in df.iterrows():
            vid = str(row['video_id'])
            if vid not in ann_lookup:
                ann_lookup[vid] = []
            ann_lookup[vid].append((int(row['start_frame']), int(row['end_frame'])))

        pos_windows, neg_windows = [], []

        for npy_file in sorted(os.listdir(keypoint_dir)):
            if not npy_file.endswith('_keypoints.npy'):
                continue
            video_id = npy_file.replace('_keypoints.npy', '')
            kp = np.load(os.path.join(keypoint_dir, npy_file))  # (N, 33, 4)
            HAND_JOINTS = [11, 12, 13, 14, 15, 16]  # left/right shoulder, elbow, wrist
            kp = kp[:, HAND_JOINTS, :2].astype(np.float32)      # (N, 6, 2)
            N = kp.shape[0]
            annotations = ann_lookup.get(video_id, [])

            for t_start in range(0, max(1, N - self.window_size + 1), self.stride):
                t_end = t_start + self.window_size
                if t_end > N:
                    break
                window = kp[t_start:t_end]  # (window_size, 33, 2)

                label = 0
                for start, end in annotations:
                    overlap = max(0, min(t_end, end) - max(t_start, start))
                    if overlap / self.window_size >= self.overlap_threshold:
                        label = 1
                        break

                (pos_windows if label == 1 else neg_windows).append((window, label))

        if self.split == 'train' and len(pos_windows) > 0:
            max_neg = int(len(pos_windows) * self.b7_ratio)
            if len(neg_windows) > max_neg:
                random.shuffle(neg_windows)
                neg_windows = neg_windows[:max_neg]

        all_windows = pos_windows + neg_windows
        if self.split == 'train':
            random.shuffle(all_windows)

        self.data = [w for w, _ in all_windows]
        self.label = [lbl for _, lbl in all_windows]

        print(f"[{self.split}] pos={len(pos_windows)}, "
              f"neg={len(neg_windows)}, total={len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        channel = 2
        label = self.label[index]
        value = self.data[index].copy()  # (window_size, 33, 2)

        if self.split == 'train':

            def affine_transform(value):
                T, V, C = value.shape
                assert C == 2
                angle = np.random.uniform(-15, 15) * np.pi / 180
                cos_val, sin_val = np.cos(angle), np.sin(angle)
                rotation = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
                scale = np.random.uniform(0.9, 1.1)
                scale_matrix = np.eye(2) * scale
                translation = np.random.uniform(-0.1, 0.1, size=(1, 1, 2))
                value = np.matmul(value, rotation.T)
                value = np.matmul(value, scale_matrix)
                value += translation
                return value

            def temporal_jitter(value, max_jitter=3):
                T = value.shape[0]
                jittered = np.zeros_like(value)
                for t in range(T):
                    offset = np.random.randint(-max_jitter, max_jitter + 1)
                    new_t = min(max(t + offset, 0), T - 1)
                    jittered[t] = value[new_t]
                return jittered

            random.random()

            # Center on midpoint of hips (joints 23, 24) — torso center for MediaPipe 33-joint
            center = (value[0, 0, :] + value[0, 1, :]) / 2.0
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
                joint_count = 0
                joints_to_drop = random.sample(range(V), joint_count)
                frame_count = random.randint(1, 16)
                frames_to_drop = random.sample(range(T), frame_count)
                data[np.ix_(frames_to_drop, joints_to_drop)] = 0

            if random.random() < self.p:
                block_size = random.randint(4, 16)
                start = random.randint(0, self.time_steps - block_size)
                data[start:start + block_size, :, :] = 0

        else:
            random.random()

            center = (value[0, 0, :] + value[0, 1, :]) / 2.0
            value = value - center

            scalerValue = np.reshape(value, (-1, channel))
            epsilon = 1e-6
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (
                np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0) + epsilon)
            scalerValue = scalerValue * 2 - 1
            scalerValue = np.reshape(scalerValue, (-1, value.shape[1], channel))

            data = np.zeros((self.time_steps, value.shape[1], channel))
            value = scalerValue
            length = value.shape[0]
            idx = np.linspace(0, length - 1, self.time_steps).astype(int)
            data[:, :, :] = value[idx, :, :]
            index_t = 2 * idx.astype(np.float32) / length - 1

        if 'b' in self.data_type:
            data_bone = np.zeros_like(data)
            for bone_idx in range(len(self.bone)):
                data_bone[:, self.bone[bone_idx][0], :] = (
                    data[:, self.bone[bone_idx][0], :] - data[:, self.bone[bone_idx][1], :])
            data = data_bone

        if 'm' in self.data_type:
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion

        # (T, V, C) → (C, T, V) → (C, T, V, 1)
        data = np.transpose(data, (2, 0, 1))
        C, T, V = data.shape
        data = np.reshape(data, (C, T, V, 1))

        return data, index_t, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
