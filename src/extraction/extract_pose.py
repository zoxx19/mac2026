import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import argparse
import random
from tqdm import tqdm

# COCO 17 keypoint indices
NOSE        = 0
LEFT_EYE    = 1
RIGHT_EYE   = 2
LEFT_EAR    = 3
RIGHT_EAR   = 4
LEFT_SHLDR  = 5
RIGHT_SHLDR = 6
LEFT_ELBOW  = 7
RIGHT_ELBOW = 8
LEFT_WRIST  = 9
RIGHT_WRIST = 10
LEFT_HIP    = 11
RIGHT_HIP   = 12
LEFT_KNEE   = 13
RIGHT_KNEE  = 14
LEFT_ANKLE  = 15
RIGHT_ANKLE = 16

# Standard COCO bones
SKELETON = [
    (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHLDR, RIGHT_SHLDR),
    (LEFT_SHLDR, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHLDR, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHLDR, LEFT_HIP), (RIGHT_SHLDR, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]


def get_keypoint(kps, idx):
    return int(kps[idx][0]), int(kps[idx][1])


def compute_hand_center(wrist, elbow, offset_pixels):
    """Shift box center from wrist toward fingertips."""
    wrist     = np.array(wrist)
    elbow     = np.array(elbow)
    direction = wrist - elbow
    norm      = np.linalg.norm(direction)
    if norm < 1e-3:
        return wrist
    return wrist + (direction / norm) * offset_pixels


def compute_foot_center(ankle, knee, offset_pixels):
    """Shift box center from ankle downward away from knee."""
    ankle     = np.array(ankle)
    knee      = np.array(knee)
    direction = ankle - knee
    norm      = np.linalg.norm(direction)
    if norm < 1e-3:
        return ankle
    return ankle + (direction / norm) * offset_pixels


def compute_boxes(kps):
    l_shldr  = np.array(kps[LEFT_SHLDR][:2])
    r_shldr  = np.array(kps[RIGHT_SHLDR][:2])
    l_hip    = np.array(kps[LEFT_HIP][:2])
    r_hip    = np.array(kps[RIGHT_HIP][:2])
    nose     = np.array(kps[NOSE][:2])
    l_wrist  = np.array(kps[LEFT_WRIST][:2])
    r_wrist  = np.array(kps[RIGHT_WRIST][:2])
    l_elbow  = np.array(kps[LEFT_ELBOW][:2])
    r_elbow  = np.array(kps[RIGHT_ELBOW][:2])
    l_ankle  = np.array(kps[LEFT_ANKLE][:2])
    r_ankle  = np.array(kps[RIGHT_ANKLE][:2])
    l_knee   = np.array(kps[LEFT_KNEE][:2])
    r_knee   = np.array(kps[RIGHT_KNEE][:2])

    neck     = (l_shldr + r_shldr) / 2
    mid_hip  = (l_hip + r_hip) / 2
    L        = np.linalg.norm(neck - mid_hip)

    if L < 10:
        return None

    def make_box(center, w, h):
        cx, cy = int(center[0]), int(center[1])
        hw, hh = int(w / 2), int(h / 2)
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    head_size  = L / 1.4
    hand_size  = L / 2.5
    foot_w     = L / 2.5   # wider
    foot_h     = L / 3.0   # taller than before

    # shift hand centers from wrist toward fingertips
    offset_hand   = hand_size * 0.55
    l_hand_ctr    = compute_hand_center(l_wrist, l_elbow, offset_hand)
    r_hand_ctr    = compute_hand_center(r_wrist, r_elbow, offset_hand)

    # shift foot centers from ankle downward away from knee
    offset_foot   = foot_h * 0.5
    l_foot_ctr    = compute_foot_center(l_ankle, l_knee, offset_foot)
    r_foot_ctr    = compute_foot_center(r_ankle, r_knee, offset_foot)

    return {
        "L":          float(L),
        "head":       make_box(nose,       head_size, head_size),
        "right_hand": make_box(r_hand_ctr, hand_size, hand_size),
        "left_hand":  make_box(l_hand_ctr, hand_size, hand_size),
        "right_foot": make_box(r_foot_ctr, foot_w,    foot_h),
        "left_foot":  make_box(l_foot_ctr, foot_w,    foot_h),
    }


def draw_skeleton(frame, kps):
    # Draw standard COCO bones
    for i, j in SKELETON:
        pt1 = get_keypoint(kps, i)
        pt2 = get_keypoint(kps, j)
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Draw virtual neck (midpoint of shoulders) -> nose connection
    l_shldr = get_keypoint(kps, LEFT_SHLDR)
    r_shldr = get_keypoint(kps, RIGHT_SHLDR)
    neck_pt = ((l_shldr[0] + r_shldr[0]) // 2,
               (l_shldr[1] + r_shldr[1]) // 2)
    nose_pt = get_keypoint(kps, NOSE)
    cv2.line(frame, neck_pt, nose_pt, (0, 255, 0), 2)
    cv2.circle(frame, neck_pt, 4, (0, 200, 255), -1)  # orange dot for virtual neck

    # Draw all keypoint joints
    for idx in range(len(kps)):
        pt = get_keypoint(kps, idx)
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def draw_boxes(frame, boxes):
    colors = {
        "head":       (0, 165, 255),  # orange
        "right_hand": (255, 0, 0),    # blue
        "left_hand":  (0, 0, 255),    # red
        "right_foot": (255, 255, 0),  # cyan
        "left_foot":  (0, 255, 255),  # yellow
    }
    labels = {
        "head":       "Head",
        "right_hand": "R-Hand",
        "left_hand":  "L-Hand",
        "right_foot": "R-Foot",
        "left_foot":  "L-Foot",
    }
    for key, color in colors.items():
        x1, y1, x2, y2 = boxes[key]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, labels[key], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def process_video(model, video_path, out_kp_path, out_vis_path=None):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_vis_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_vis_path), fourcc, fps, (W, H))

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        frame_data = {"keypoints": None, "boxes": None}

        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            kps   = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
            boxes = compute_boxes(kps)
            frame_data["keypoints"] = kps.tolist()
            frame_data["boxes"]     = boxes

            if writer:
                draw_skeleton(frame, kps)
                if boxes:
                    draw_boxes(frame, boxes)

        if writer:
            writer.write(frame)

        all_frames.append(frame_data)

    cap.release()
    if writer:
        writer.release()

    with open(out_kp_path, "w") as f:
        json.dump(all_frames, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",  required=True)
    parser.add_argument("--kp_dir",     required=True)
    parser.add_argument("--vis_dir",    default=None)
    parser.add_argument("--n_vis",      type=int, default=20,
                        help="Number of videos to save visualizations for")
    parser.add_argument("--max_videos", type=int, default=0,
                        help="0 = process all, >0 = limit to N videos")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--split",      default="train")
    args = parser.parse_args()

    random.seed(args.seed)

    model = YOLO("models/yolov8x-pose.pt")

    video_dir = Path(args.video_dir)
    kp_dir    = Path(args.kp_dir);  kp_dir.mkdir(parents=True, exist_ok=True)
    vis_dir   = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ALL videos in the folder — random sampling from full pool
    all_videos = sorted(video_dir.glob("*.mp4"))

    # limit total processing only if requested
    process_pool = all_videos if args.max_videos == 0 else random.sample(all_videos, min(args.max_videos, len(all_videos)))

    # pick visualization subset randomly from the process pool
    vis_set = set(v.stem for v in random.sample(process_pool, min(args.n_vis, len(process_pool)))) if vis_dir else set()

    print(f"Processing {len(process_pool)} videos | visualizing {len(vis_set)}")

    for vp in tqdm(process_pool, desc=f"Processing {args.split}"):
        kp_path = kp_dir / (vp.stem + ".json")
        if kp_path.exists():
            continue  # resumable

        vis_path = None
        if vis_dir and vp.stem in vis_set:
            vis_path = vis_dir / (vp.stem + "_vis.mp4")

        process_video(model, vp, kp_path, vis_path)


if __name__ == "__main__":
    main()