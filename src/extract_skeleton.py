"""
extract_skeleton.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/extract_skeleton.py --help` for options where applicable.
"""
import argparse
import os
import cv2
import numpy as np
import mediapipe as mp


def extract_skeleton(input_path, output_dir, skip_if_exists):
    video_id = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{video_id}_keypoints.npy")

    if skip_if_exists and os.path.exists(out_path):
        print(f"Skipping {video_id} (exists)")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    keypoints = np.zeros((n_frames, 33, 4), dtype=np.float32)
    frame_idx = 0

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.pose_landmarks:
                for j, lm in enumerate(results.pose_landmarks.landmark):
                    keypoints[frame_idx, j, 0] = lm.x * width
                    keypoints[frame_idx, j, 1] = lm.y * height
                    keypoints[frame_idx, j, 2] = lm.z
                    keypoints[frame_idx, j, 3] = lm.visibility

            frame_idx += 1

    cap.release()

    # Trim to actual frames read (guard against header mismatch)
    keypoints = keypoints[:frame_idx]

    os.makedirs(output_dir, exist_ok=True)
    np.save(out_path, keypoints)
    print(f"Saved {keypoints.shape} → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--skip_if_exists", action="store_true")
    args = parser.parse_args()

    extract_skeleton(args.input, args.output_dir, args.skip_if_exists)


if __name__ == "__main__":
    main()
