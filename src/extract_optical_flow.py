"""
extract_optical_flow.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/extract_optical_flow.py --help` for options where applicable.
"""
import argparse
import os
import cv2
import numpy as np

VIDEO_BASE = "data/MMA-52/extracted"
FLOW_BASE = "data/optical_flow"

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)
FRAME_SIZE = (224, 224)


def extract_flow(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] cannot open {video_path}")
        return False

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(cv2.resize(frame, FRAME_SIZE), cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    if len(frames) < 2:
        print(f"  [ERROR] too few frames: {video_path}")
        return False

    flows = np.empty((len(frames) - 1, FRAME_SIZE[1], FRAME_SIZE[0], 2), dtype=np.float16)
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(frames[i], frames[i + 1], None, **FARNEBACK_PARAMS)
        flows[i] = flow.astype(np.float16)

    np.save(out_path, flows)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    video_dir = os.path.join(VIDEO_BASE, args.split, args.split)
    out_dir = os.path.join(FLOW_BASE, args.split)
    os.makedirs(out_dir, exist_ok=True)

    videos = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
    videos = videos[args.start_idx : args.end_idx]

    print(f"Split: {args.split}  |  {len(videos)} videos  |  output: {out_dir}")

    ok = 0
    for idx, filename in enumerate(videos):
        video_path = os.path.join(video_dir, filename)
        stem = os.path.splitext(filename)[0]
        out_path = os.path.join(out_dir, stem + ".npy")

        if os.path.exists(out_path):
            ok += 1
            if (idx + 1) % 50 == 0:
                print(f"  [{idx + 1}/{len(videos)}] skipped (exists): {stem}")
            continue

        success = extract_flow(video_path, out_path)
        if success:
            ok += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(videos)}] done={ok}")

    print(f"\nFinished: {ok}/{len(videos)} succeeded")


if __name__ == "__main__":
    main()
