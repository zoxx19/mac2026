"""
Runs MediaPipe Holistic on one MMA-52 video, crops the full-body region per
frame using a shoulder-to-ankle bounding box, and saves a 224x224 body clip.

Fallback: frames where no pose is detected are saved as the full frame
resized to 224x224.
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


# ── landmark indices ──────────────────────────────────────────────────────────
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28

# ── crop parameters ───────────────────────────────────────────────────────────
PADDING_RATIO = 0.20
CROP_SIZE     = (224, 224)


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


def compute_body_bbox(kps, frame_w, frame_h):
    ls = point(kps, LEFT_SHOULDER)
    rs = point(kps, RIGHT_SHOULDER)
    lh = point(kps, LEFT_HIP)
    rh = point(kps, RIGHT_HIP)
    la = point(kps, LEFT_ANKLE)
    ra = point(kps, RIGHT_ANKLE)

    shoulder_y = min(ls[1], rs[1])
    torso_height = max(la[1], ra[1]) - shoulder_y
    top_y = shoulder_y - torso_height * 0.35
    bottom_y = max(la[1], ra[1])
    body_height = bottom_y - top_y
    if body_height < 5:
        return None

    padding = PADDING_RATIO * body_height

    x1 = int(round(max(0,           min(ls[0], lh[0]) - padding)))
    y1 = int(round(max(0,           top_y              - padding)))
    x2 = int(round(min(frame_w - 1, max(rs[0], rh[0]) + padding)))
    y2 = int(round(min(frame_h - 1, bottom_y           + padding)))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


# ── main processing ───────────────────────────────────────────────────────────
def letterbox_resize(img, target_size):
    """Resize maintaining aspect ratio, pad with black."""
    th, tw = target_size[1], target_size[0]
    h, w = img.shape[:2]
    scale = min(tw/w, th/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x_off = (tw - new_w) // 2
    y_off = (th - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas


def process_video(video_path, output_dir, skip_if_exists=False):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id   = video_path.stem
    body_path  = output_dir / f"{video_id}_body.mp4"

    if skip_if_exists and body_path.exists():
        print(f"[SKIP] {video_id} already exists")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(body_path), fourcc, fps, CROP_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {body_path}")

    n_frames = 0
    n_fallback = 0

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
            bbox = compute_body_bbox(kps, frame_w, frame_h) if kps is not None else None

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                crop = letterbox_resize(crop, CROP_SIZE) if crop.size > 0 else letterbox_resize(frame, CROP_SIZE)
            else:
                crop = letterbox_resize(frame, CROP_SIZE)
                n_fallback += 1

            writer.write(crop)
            n_frames += 1

    cap.release()
    writer.release()

    print(f"[DONE] {video_id}: {n_frames} frames ({n_fallback} fallback) -> {body_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract body crops from a single MMA-52 video.")
    parser.add_argument("--video_path",     required=True,  help="Path to input .mp4")
    parser.add_argument("--output_dir",     required=True,  help="Base output directory")
    parser.add_argument("--split",          required=True,  help="Split name (train/val/test), appended to output_dir")
    parser.add_argument("--skip_if_exists", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = Path(args.output_dir) / args.split
    process_video(args.video_path, out, skip_if_exists=args.skip_if_exists)
