"""
pose_extractor.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/pose_extractor.py --help` for options where applicable.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


# -----------------------------
# MediaPipe Pose landmark indices
# -----------------------------
NOSE = 0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# -----------------------------
# Bbox scale factors
# -----------------------------
HEAD_SCALE = 0.80
HAND_SCALE = 0.50


# -----------------------------
# OpenCV BGR colors
# -----------------------------
COLORS = {
    "head": (0, 255, 255),        # yellow
    "left_hand": (0, 255, 0),     # green
    "right_hand": (0, 165, 255),  # orange
}

SKELETON_COLOR = (0, 255, 0)


# -----------------------------
# Simple skeleton connections
# -----------------------------
POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]


def make_bbox(cx, cy, size, frame_w, frame_h):
    """
    Create square bbox centered at (cx, cy), clipped to image boundaries.
    Returns [x1, y1, x2, y2].
    """
    if cx is None or cy is None or size is None:
        return None

    half = size / 2.0

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


def bbox_from_points(points, frame_w, frame_h, pad=0):
    """
    Create bbox around multiple points with optional padding.
    points: list of (x, y)
    """
    if points is None or len(points) == 0:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = int(round(min(xs) - pad))
    y1 = int(round(min(ys) - pad))
    x2 = int(round(max(xs) + pad))
    y2 = int(round(max(ys) + pad))

    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w - 1, x2))
    y2 = max(0, min(frame_h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def hand_landmarks_to_points(hand_landmarks, frame_w, frame_h):
    """
    Convert MediaPipe hand landmarks to pixel points.
    """
    if hand_landmarks is None:
        return None

    points = []

    for lm in hand_landmarks.landmark:
        x = lm.x * frame_w
        y = lm.y * frame_h
        points.append((x, y))

    return points


def point(kps, idx):
    """
    Return x, y from keypoint array.

    kps shape:
        [33, 4]

    columns:
        x_pixel, y_pixel, z_relative, visibility
    """
    return kps[idx, 0], kps[idx, 1]


def visibility(kps, idx):
    return kps[idx, 3]


def valid_point(kps, idx, min_visibility=0.3):
    """
    Check whether landmark visibility is good enough.
    """
    return visibility(kps, idx) >= min_visibility


def midpoint(p1, p2):
    return (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0


def distance(p1, p2):
    return float(
        np.linalg.norm(
            np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
        )
    )


def torso_length(kps):
    """
    Torso length S = distance(neck midpoint, hip midpoint).

    MediaPipe has no direct neck landmark.
    So we approximate:

        neck_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid  = midpoint(left_hip, right_hip)

    This S is used as the scale anchor for all bboxes.
    """
    left_shoulder = point(kps, LEFT_SHOULDER)
    right_shoulder = point(kps, RIGHT_SHOULDER)

    left_hip = point(kps, LEFT_HIP)
    right_hip = point(kps, RIGHT_HIP)

    neck_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)

    return distance(neck_mid, hip_mid)


def shifted_hand_center(wrist, elbow, shift_ratio=0.30):
    """
    Shift bbox center from wrist slightly outward toward the hand direction.

    Direction:
        elbow -> wrist -> hand

    So:
        center = wrist + shift_ratio * (wrist - elbow)
    """
    wx, wy = wrist
    ex, ey = elbow

    cx = wx + shift_ratio * (wx - ex)
    cy = wy + shift_ratio * (wy - ey)

    return cx, cy


def compute_bboxes(kps, frame_w, frame_h, left_hand_landmarks=None, right_hand_landmarks=None):
    """
    Compute stable head and hand bboxes.

    Hands use pose landmarks only:
        center = wrist shifted away from elbow

    This is more stable than MediaPipe hand landmarks when hands overlap.
    """
    S = torso_length(kps)

    if S < 5:
        return {
            "head": None,
            "left_hand": None,
            "right_hand": None,
        }

    # -----------------------------
    # Head bbox
    # -----------------------------
    left_eye = point(kps, 2)
    right_eye = point(kps, 5)
    eye_mid = midpoint(left_eye, right_eye)

    head_bbox = make_bbox(
        cx=eye_mid[0],
        cy=eye_mid[1],
        size=S * HEAD_SCALE,
        frame_w=frame_w,
        frame_h=frame_h,
    )

    # -----------------------------
    # Hand bboxes
    # -----------------------------
    left_wrist = point(kps, LEFT_WRIST)
    right_wrist = point(kps, RIGHT_WRIST)

    left_elbow = point(kps, LEFT_ELBOW)
    right_elbow = point(kps, RIGHT_ELBOW)

    left_hand_center = shifted_hand_center(
        wrist=left_wrist,
        elbow=left_elbow,
        shift_ratio=0.30,
    )

    right_hand_center = shifted_hand_center(
        wrist=right_wrist,
        elbow=right_elbow,
        shift_ratio=0.30,
    )

    left_hand_bbox = make_bbox(
        cx=left_hand_center[0],
        cy=left_hand_center[1],
        size=S * HAND_SCALE,
        frame_w=frame_w,
        frame_h=frame_h,
    )

    right_hand_bbox = make_bbox(
        cx=right_hand_center[0],
        cy=right_hand_center[1],
        size=S * HAND_SCALE,
        frame_w=frame_w,
        frame_h=frame_h,
    )

    return {
        "head": head_bbox,
        "left_hand": left_hand_bbox,
        "right_hand": right_hand_bbox,
    }


def extract_pose_keypoints(results, frame_w, frame_h):
    """
    Convert MediaPipe normalized pose landmarks to pixel coordinates.

    Output shape:
        [33, 4]

    Columns:
        x_pixel, y_pixel, z_relative, visibility

    If no pose is detected:
        return None
    """
    if results.pose_landmarks is None:
        return None

    landmarks = results.pose_landmarks.landmark
    kps = np.zeros((33, 4), dtype=np.float32)

    for i, lm in enumerate(landmarks):
        kps[i, 0] = lm.x * frame_w
        kps[i, 1] = lm.y * frame_h
        kps[i, 2] = lm.z
        kps[i, 3] = lm.visibility

    return kps


def draw_skeleton(frame, kps, min_visibility=0.3):
    """
    Draw a simple pose skeleton.
    """
    for a, b in POSE_CONNECTIONS:
        if not valid_point(kps, a, min_visibility):
            continue
        if not valid_point(kps, b, min_visibility):
            continue

        ax, ay = point(kps, a)
        bx, by = point(kps, b)

        cv2.line(
            frame,
            (int(ax), int(ay)),
            (int(bx), int(by)),
            SKELETON_COLOR,
            2,
            cv2.LINE_AA,
        )

    important_points = [
        NOSE,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE,
    ]

    for idx in important_points:
        if not valid_point(kps, idx, min_visibility):
            continue

        x, y = point(kps, idx)

        cv2.circle(
            frame,
            (int(x), int(y)),
            3,
            SKELETON_COLOR,
            -1,
            cv2.LINE_AA,
        )

    return frame


def draw_bboxes(frame, bboxes):
    """
    Draw head and hand bounding boxes and labels.
    """
    display_names = {
        "head": "Head",
        "left_hand": "L-Hand",
        "right_hand": "R-Hand",
    }

    for name, bbox in bboxes.items():
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        color = COLORS[name]
        label = display_names.get(name, name)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame


def process_video(input_video, output_dir, save_video=False, model_complexity=1):
    input_video = Path(input_video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    video_stem = input_video.stem

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    print(f"[INFO] Input: {input_video}")
    print(f"[INFO] Resolution: {frame_w}x{frame_h}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Frames: {total_frames}")

    writer = None
    annotated_path = output_dir / f"{video_stem}_annotated.mp4"

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(annotated_path),
            fourcc,
            fps,
            (frame_w, frame_h),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {annotated_path}")

    all_keypoints = []
    all_bboxes = []

    missing_pose_frames = 0
    bad_torso_frames = 0
    missing_left_hand_frames = 0
    missing_right_hand_frames = 0
    frame_idx = 0

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
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

            rgb.flags.writeable = True

            kps = extract_pose_keypoints(results, frame_w, frame_h)

            if kps is None:
                missing_pose_frames += 1

                all_keypoints.append(
                    np.full((33, 4), np.nan, dtype=np.float32)
                )

                bboxes = {
                    "head": None,
                    "left_hand": None,
                    "right_hand": None,
                }

                all_bboxes.append(bboxes)

            else:
                S = torso_length(kps)

                if S < 5:
                    bad_torso_frames += 1

                if results.left_hand_landmarks is None:
                    missing_left_hand_frames += 1

                if results.right_hand_landmarks is None:
                    missing_right_hand_frames += 1

                bboxes = compute_bboxes(
                    kps,
                    frame_w,
                    frame_h,
                    left_hand_landmarks=results.left_hand_landmarks,
                    right_hand_landmarks=results.right_hand_landmarks,
                )

                all_keypoints.append(kps)
                all_bboxes.append(bboxes)

                if save_video:
                    frame = draw_skeleton(frame, kps)
                    frame = draw_bboxes(frame, bboxes)

            if save_video:
                writer.write(frame)

            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"[INFO] Processed {frame_idx}/{total_frames} frames")

    cap.release()

    if writer is not None:
        writer.release()

    keypoints_arr = np.stack(all_keypoints, axis=0)

    keypoints_path = output_dir / f"{video_stem}_keypoints.npy"
    bboxes_path = output_dir / f"{video_stem}_bboxes.npy"

    np.save(keypoints_path, keypoints_arr)
    np.save(bboxes_path, np.array(all_bboxes, dtype=object), allow_pickle=True)

    print("[DONE] Saved:")
    print(f"  keypoints: {keypoints_path}")
    print(f"  bboxes:    {bboxes_path}")

    if save_video:
        print(f"  video:     {annotated_path}")

    print("[SUMMARY]")
    print(f"  total frames:              {frame_idx}")
    print(f"  missing pose frames:       {missing_pose_frames}")
    print(f"  bad torso frames:          {bad_torso_frames}")
    print(f"  missing left hand frames:  {missing_left_hand_frames}")
    print(f"  missing right hand frames: {missing_right_hand_frames}")

    if frame_idx > 0:
        print(f"  missing pose ratio:        {missing_pose_frames / frame_idx:.3f}")
        print(f"  bad torso ratio:           {bad_torso_frames / frame_idx:.3f}")
        print(f"  missing left hand ratio:   {missing_left_hand_frames / frame_idx:.3f}")
        print(f"  missing right hand ratio:  {missing_right_hand_frames / frame_idx:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic pose keypoints and head/hand bboxes."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input mp4 video.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/pose_head_hands",
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save annotated mp4 video.",
    )

    parser.add_argument(
        "--model_complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe Holistic pose model complexity. Use 1 first.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_video(
        input_video=args.input,
        output_dir=args.output_dir,
        save_video=args.save_video,
        model_complexity=args.model_complexity,
    )