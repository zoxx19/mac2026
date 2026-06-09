"""
Converts an AdaTAD result_detection.json to the flat list format expected
by fusion.py.

AdaTAD output format:
{
  "results": {
    "video_id": [
      {"segment": [start_sec, end_sec], "label": "class_name", "score": 0.87}
    ]
  }
}

Fusion input format:
[{"video_id": "train0000", "start": 23, "end": 71, "class": 9, "score": 0.87}, ...]

Segment seconds are converted to frames using per-video fps from ann_csv.
Videos not found in ann_csv fall back to --default_fps (default 30.0).

Usage:
python src/convert_adatad_predictions.py \
    --input  /path/to/result_detection.json \
    --output /path/to/rgb_preds.json \
    --ann_csv data/MMA-52/Annotations/val.csv \
    --category_idx data/MMA-52/Annotations/adatad/category_idx.txt
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import argparse
import json
import os

import pandas as pd


def load_class_map(category_idx_path):
    """Returns dict {label_string: class_index}."""
    with open(category_idx_path) as f:
        lines = [l.rstrip("\n") for l in f]
    return {name: idx for idx, name in enumerate(lines)}


def load_fps_map(ann_csv_path):
    """Returns dict {video_id: fps} from the ann_csv fps column."""
    df = pd.read_csv(ann_csv_path)
    fps_map = {}
    for vid, group in df.groupby("video_id"):
        fps_map[str(vid)] = float(group["fps"].iloc[0])
    return fps_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          required=True,  help="AdaTAD result_detection.json")
    parser.add_argument("--output",         required=True,  help="Output flat predictions JSON")
    parser.add_argument(
        "--ann_csv",
        default=(
            "."
            "/data/MMA-52/Annotations/val.csv"
        ),
        help="val.csv for fps lookup",
    )
    parser.add_argument(
        "--category_idx",
        default=(
            "."
            "/data/MMA-52/Annotations/adatad/category_idx.txt"
        ),
        help="category_idx.txt (line number = class index)",
    )
    parser.add_argument("--default_fps", type=float, default=30.0,
                        help="Fallback fps for videos not found in ann_csv")
    args = parser.parse_args()

    class_map = load_class_map(args.category_idx)
    fps_map   = load_fps_map(args.ann_csv)

    with open(args.input) as f:
        data = json.load(f)

    results = data["results"]
    output_preds = []
    n_fps_fallback = 0
    n_unknown_label = 0

    for video_id, segments in results.items():
        fps = fps_map.get(str(video_id))
        if fps is None:
            fps = args.default_fps
            n_fps_fallback += 1

        for seg in segments:
            start_sec, end_sec = seg["segment"]
            label = seg["label"]
            score = float(seg["score"])

            class_idx = class_map.get(label)
            if class_idx is None:
                n_unknown_label += 1
                continue

            output_preds.append({
                "video_id": str(video_id),
                "start":    start_sec * fps,
                "end":      end_sec   * fps,
                "class":    class_idx,
                "score":    score,
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_preds, f, indent=2)

    print(f"Converted {len(output_preds)} segments from {len(results)} videos")
    if n_fps_fallback:
        print(f"  fps fallback ({args.default_fps}): {n_fps_fallback} videos")
    if n_unknown_label:
        print(f"  unknown labels skipped: {n_unknown_label}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
