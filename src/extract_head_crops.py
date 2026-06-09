"""
Runs MediaPipe Holistic on one MMA-52 video, crops the head region per frame
using the eye-midpoint anchor, and saves a 112x112 head clip + bboxes.npy.
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


# ── landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE       = 2
RIGHT_EYE      = 5
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24

# ── crop parameters ───────────────────────────────────────────────────────────
HEAD_SCALE = 0.80
CROP_SIZE  = (112, 112)


# ── geometry helpers ──────────────────────────────────────────────────────────
def extract_pose_keypoints(results, frame_w, frame_h):
    if results.pose_landmarks is None:
        return None
    lms = results.pose_landmarks.landmark
    kps = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(lms):
        kps[i, 0] = lm.x * frame_w
        kps[i, 1] = lm.y * frame_h
        kps[i, 2] = lm.z
        kps[i, 3] = lm.visibility
    return kps


def point(kps, idx):
    return kps[idx, 0], kps[idx, 1]


def midpoint(p1, p2):
    return (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def torso_length(kps):
    neck_mid = midpoint(point(kps, LEFT_SHOULDER), point(kps, RIGHT_SHOULDER))
    hip_mid  = midpoint(point(kps, LEFT_HIP),      point(kps, RIGHT_HIP))
    return distance(neck_mid, hip_mid)


def compute_head_bbox(kps, frame_w, frame_h):
    S = torso_length(kps)
    if S < 5:
        return None

    eye_mid = midpoint(point(kps, LEFT_EYE), point(kps, RIGHT_EYE))
    cx, cy  = eye_mid
    half    = (S * HEAD_SCALE) / 2.0

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))

    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w - 1, x2))
    y2 = max(0, min(frame_h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


# ── main processing ───────────────────────────────────────────────────────────
def process_video(input_path, output_dir, skip_if_exists=False):
    input_path  = Path(input_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id    = input_path.stem
    head_path   = output_dir / f"{video_id}_head.mp4"
    bboxes_path = output_dir / f"{video_id}_bboxes.npy"

    if skip_if_exists and head_path.exists() and bboxes_path.exists():
        print(f"[SKIP] {video_id} already exists")
        return

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {input_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(head_path), fourcc, fps, CROP_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {head_path}")

    black_frame = np.zeros((CROP_SIZE[1], CROP_SIZE[0], 3), dtype=np.uint8)
    all_bboxes  = []

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            kps  = extract_pose_keypoints(results, frame_w, frame_h)
            bbox = compute_head_bbox(kps, frame_w, frame_h) if kps is not None else None

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, CROP_SIZE) if crop.size > 0 else black_frame
            else:
                crop = black_frame

            writer.write(crop)
            all_bboxes.append({"head": bbox})

    cap.release()
    writer.release()

    np.save(bboxes_path, np.array(all_bboxes, dtype=object), allow_pickle=True)
    print(f"[DONE] {video_id}: {len(all_bboxes)} frames -> {head_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract head crops from a single MMA-52 video.")
    parser.add_argument("--input",          required=True,               help="Path to input .mp4")
    parser.add_argument("--output_dir",     required=True,               help="Directory to write outputs")
    parser.add_argument("--skip_if_exists", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args.input, args.output_dir, skip_if_exists=args.skip_if_exists)
