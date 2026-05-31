"""
animate_wrong.py
Creates annotated MP4 videos from wrong prediction clips.
Overlays: true label, predicted label, confidence score.
Helps visually verify if wrong predictions are labeling errors or model errors.

Usage:
    python src/head/animate_wrong.py \
        --wrong_csv    outputs/head_vXX/evaluation/wrong_predictions.csv \
        --output_dir   outputs/head_vXX/evaluation/animated_wrong \
        --n_per_class  5
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

LABEL2ID = {"B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4, "B6": 5, "B7": 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

LABEL_FULL = {
    "B1": "Nodding",
    "B2": "Shaking head",
    "B3": "Turning head",
    "B4": "Tilting head",
    "B5": "Bowing head",
    "B6": "Head up",
    "B7": "No movement",
}

# colors BGR
GREEN  = (0,   200,  0)
RED    = (0,   0,    220)
WHITE  = (255, 255,  255)
BLACK  = (0,   0,    0)
YELLOW = (0,   220,  220)


def draw_label_overlay(frame, true_label, pred_label, frame_idx, total_frames):
    """Draw true/predicted labels on frame with colored background."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # top banner background
    cv2.rectangle(overlay, (0, 0), (w, 80), BLACK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # true label (green)
    cv2.putText(frame,
                f"TRUE:  {true_label} — {LABEL_FULL.get(true_label, '')}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN, 2)

    # predicted label (red if wrong)
    color = RED if true_label != pred_label else GREEN
    cv2.putText(frame,
                f"PRED:  {pred_label} — {LABEL_FULL.get(pred_label, '')}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # frame counter bottom right
    cv2.putText(frame,
                f"Frame {frame_idx+1}/{total_frames}",
                (w - 140, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    # red border for wrong predictions
    if true_label != pred_label:
        cv2.rectangle(frame, (0, 0), (w-1, h-1), RED, 3)

    return frame


def animate_clip(src_path, dst_path, true_label, pred_label, slow_factor=3):
    """
    Read a crop clip, add label overlay, write annotated video.
    slow_factor: repeat each frame N times to slow down playback for inspection.
    """
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"  Cannot open {src_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (w, h))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    for idx, frame in enumerate(frames):
        annotated = draw_label_overlay(
            frame.copy(), true_label, pred_label, idx, len(frames)
        )
        # write each frame slow_factor times to slow down
        for _ in range(slow_factor):
            writer.write(annotated)

    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrong_csv",   required=True,
                        help="CSV from evaluate.py with wrong predictions")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--n_per_class", type=int, default=5,
                        help="Number of wrong clips to animate per true class")
    parser.add_argument("--slow_factor", type=int, default=3,
                        help="Repeat each frame N times to slow down video")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_wrong = pd.read_csv(args.wrong_csv)
    print(f"Total wrong predictions: {len(df_wrong)}")

    total_saved = 0

    for true_class in [ID2LABEL[i] for i in range(7)]:
        class_wrong = df_wrong[df_wrong["true"] == true_class]

        if len(class_wrong) == 0:
            print(f"\n{true_class}: no wrong predictions ✅")
            continue

        print(f"\n{true_class} ({LABEL_FULL[true_class]}): "
              f"{len(class_wrong)} wrong predictions total")

        # show breakdown of what it was confused with
        confused_with = class_wrong["predicted"].value_counts()
        for pred, count in confused_with.items():
            print(f"  → predicted as {pred} ({LABEL_FULL[pred]}): {count} times")

        # animate top N
        class_dir = output_dir / true_class
        class_dir.mkdir(exist_ok=True)

        samples = class_wrong.head(args.n_per_class)
        for i, (_, row) in enumerate(samples.iterrows()):
            src_path = Path(row["path"])
            if not src_path.exists():
                continue

            dst_name = (f"{i+1:02d}_true_{true_class}_"
                        f"pred_{row['predicted']}_{src_path.name}")
            dst_path = class_dir / dst_name

            success = animate_clip(
                src_path, dst_path,
                row["true"], row["predicted"],
                slow_factor=args.slow_factor
            )
            if success:
                print(f"  Saved: {dst_name}")
                total_saved += 1

    print(f"\nTotal animated clips saved: {total_saved}")
    print(f"Output directory: {output_dir}")
    print("\nHow to use:")
    print("  Download the output folder and watch the videos.")
    print("  RED border = wrong prediction")
    print("  GREEN = true label, RED text = predicted label")
    print("  Videos are slowed down for easier inspection.")


if __name__ == "__main__":
    main()
