"""
Extract Temporal Gradient (TG) video with slow+fast blend and adaptive normalization.

  TG[t] = normalize(0.6 * |frame[t+1]-frame[t]| + 0.4 * |frame[t+2]-frame[t]|/2)

Output has N-1 frames at the same fps.  Uses a rolling 3-frame buffer (O(3) memory).

Modes
-----
Single-file:
    python extract_tg.py --input VIDEO --output_dir DIR
                         [--split {train,val,test}] [--use_body_mask] [--skip_if_exists]

SLURM array:
    python extract_tg.py --split {train,val,test}
                         --array_idx $SLURM_ARRAY_TASK_ID [--chunk_size 50]
                         [--use_body_mask]

Legacy flag --gain is accepted but ignored; normalization is now per-frame adaptive.
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

VIDEO_BASE = "data/MMA-52/extracted"
TG_BASE = "data/tg"
BODY_CROP_BASE = "data/body_crops"


def _read_src(cap):
    ret, frame = cap.read()
    return frame.astype(np.float32) if ret else None


def _read_body(cap_body):
    if cap_body is None:
        return None
    ret, frame = cap_body.read()
    return frame if ret else None


def process_video(src_path: Path, out_path: Path, skip_if_exists: bool,
                  use_body_mask: bool = False, body_crop_dir=None) -> bool:
    if skip_if_exists and out_path.exists():
        print(f"[SKIP] {src_path.name}")
        return True

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"[ERROR] cannot open {src_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional body crop for masking
    cap_body = None
    if use_body_mask and body_crop_dir is not None:
        body_path = Path(body_crop_dir) / f"{src_path.stem}_body.mp4"
        if body_path.exists():
            cap_body = cv2.VideoCapture(str(body_path))
            if not cap_body.isOpened():
                print(f"[WARN] cannot open body crop for {src_path.name}, masking disabled")
                cap_body = None
        else:
            print(f"[WARN] no body crop for {src_path.name}, masking disabled")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[ERROR] cannot create {out_path}")
        cap.release()
        if cap_body:
            cap_body.release()
        return False

    # Seed rolling buffer
    f0 = _read_src(cap)
    if f0 is None:
        print(f"[ERROR] empty video {src_path}")
        cap.release()
        return False
    b0 = _read_body(cap_body)

    f1 = _read_src(cap)
    if f1 is None:
        print(f"[ERROR] only 1 frame in {src_path}, cannot compute diff")
        cap.release()
        if cap_body:
            cap_body.release()
        return False
    b1 = _read_body(cap_body)

    n_written = 0

    while f1 is not None:
        f2 = _read_src(cap)
        b2 = _read_body(cap_body)

        # Slow + fast TG blend
        fast = np.abs(f1 - f0)
        slow = np.abs(f2 - f0) / 2.0 if f2 is not None else fast
        blended = 0.6 * fast + 0.4 * slow

        # Adaptive per-frame normalization: scale 95th percentile to 200
        p95 = np.percentile(blended, 95) + 1e-6
        tg = np.clip(blended / p95 * 200, 0, 255).astype(np.uint8)

        # Optional body masking: b0 aligns with f0 = frame[t]
        if cap_body is not None and b0 is not None:
            b0_resized = cv2.resize(b0, (w, h)) if b0.shape[:2] != (h, w) else b0
            mask = (b0_resized.max(axis=2) > 10).astype(np.uint8)[:, :, np.newaxis]
            tg = tg * mask

        writer.write(tg)
        n_written += 1

        # Advance buffer
        f0, b0 = f1, b1
        f1, b1 = f2, b2
        if f2 is None:
            break

    cap.release()
    if cap_body:
        cap_body.release()
    writer.release()

    if n_written == 0:
        print(f"[ERROR] no TG frames written for {src_path}")
        out_path.unlink(missing_ok=True)
        return False

    print(f"[DONE] {src_path.name} -> {n_written} TG frames")
    return True


def run_single(args):
    src = Path(args.input)
    out_dir = Path(args.output_dir)
    body_crop_dir = None
    if args.use_body_mask:
        if not args.split:
            print("[ERROR] --split is required when --use_body_mask is set in single-file mode")
            raise SystemExit(1)
        body_crop_dir = Path(BODY_CROP_BASE) / args.split
    ok = process_video(src, out_dir / src.name, args.skip_if_exists,
                       use_body_mask=args.use_body_mask, body_crop_dir=body_crop_dir)
    if not ok:
        raise SystemExit(1)


def run_batch(args):
    split = args.split
    src_dir = Path(VIDEO_BASE) / split / split
    out_dir = Path(TG_BASE) / split
    body_crop_dir = Path(BODY_CROP_BASE) / split if args.use_body_mask else None
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(f for f in src_dir.iterdir() if f.suffix == ".mp4")
    start = args.array_idx * args.chunk_size
    chunk = videos[start: start + args.chunk_size]

    print(f"Split: {split} | array_idx={args.array_idx} | "
          f"videos {start}-{start + len(chunk) - 1} of {len(videos)}"
          f"{' [body_mask ON]' if args.use_body_mask else ''}")

    ok = err = 0
    for vid in chunk:
        if process_video(vid, out_dir / vid.name, skip_if_exists=True,
                         use_body_mask=args.use_body_mask, body_crop_dir=body_crop_dir):
            ok += 1
        else:
            err += 1

    print(f"\nFinished chunk: ok={ok} err={err}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract temporal gradient video (slow+fast blend, adaptive norm).")
    parser.add_argument("--gain", type=float, default=None,
                        help="(legacy, ignored) Fixed gain from original v1 script")
    parser.add_argument("--use_body_mask", action="store_true",
                        help="Multiply TG by non-black body-crop mask (requires body crops dir)")

    # single-file mode
    parser.add_argument("--input", default=None, help="Path to source .mp4")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--skip_if_exists", action="store_true")

    # SLURM batch mode
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--array_idx", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=50)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input is not None:
        run_single(args)
    elif args.split is not None and args.array_idx is not None:
        run_batch(args)
    else:
        print("Usage:\n"
              "  Single: --input VIDEO --output_dir DIR [--split SPLIT] "
              "[--use_body_mask] [--skip_if_exists]\n"
              "  Batch:  --split {train,val,test} --array_idx N "
              "[--chunk_size 50] [--use_body_mask]")
        raise SystemExit(1)
