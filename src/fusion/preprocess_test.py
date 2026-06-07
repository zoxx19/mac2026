"""
preprocess_test.py
Preprocesses test set videos for all tracks:
  1. Extract YOLOv8 keypoints
  2. Extract head crops
  3. Extract upper body crops (hand)
  4. Extract body crops + optical flow
  5. Extract leg crops + optical flow

Run this BEFORE generate_submission.py

Usage:
  python src/fusion/preprocess_test.py --test_dir data/test/videos
"""

import sys, os, cv2, json, numpy as np
sys.path.insert(0, 'src/head')
sys.path.insert(0, 'src/hand')
sys.path.insert(0, 'src/body_leg')
sys.path.insert(0, 'src/experiments')

from pathlib import Path
from tqdm import tqdm
import argparse
import warnings; warnings.filterwarnings("ignore")


def extract_flow(video_path, out_path, size=224):
    """Extract Farneback optical flow from video, save as HSV color MP4."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, frame = cap.read()
    if not ret: cap.release(); return False
    prev_gray = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2GRAY)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (size, size))
    writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
               0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv = np.zeros((size, size, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = mag.astype(np.uint8)
        writer.write(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        prev_gray = gray
    cap.release(); writer.release()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_dir',    default='data/test/videos')
    p.add_argument('--model_path',  default='models/yolov8x-pose.pt')
    args = p.parse_args()

    test_dir = Path(args.test_dir)
    videos   = sorted(test_dir.glob('*.mp4'))
    print(f"Test videos: {len(videos)}")

    # Create output dirs
    dirs = {
        'kp':    Path('data/test/keypoints'),
        'head':  Path('data/test/head_crops'),
        'upper': Path('data/test/upperbody_crops'),
        'body':  Path('data/test/body_crops'),
        'leg':   Path('data/test/leg_crops'),
        'bflow': Path('data/test/body_flow'),
        'lflow': Path('data/test/leg_flow'),
    }
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract keypoints
    print("\n=== Step 1: Extract keypoints ===")
    from ultralytics import YOLO
    pose_model = YOLO(args.model_path)

    for vid in tqdm(videos, desc='Keypoints'):
        vid_id = vid.stem
        out_kp = dirs['kp'] / f"{vid_id}.json"
        if out_kp.exists(): continue

        cap = cv2.VideoCapture(str(vid))
        frames_kp = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            results = pose_model(frame, verbose=False)
            if results and results[0].keypoints is not None:
                kps = results[0].keypoints.data.cpu().numpy()
                if len(kps) > 0:
                    frames_kp.append({'keypoints': kps[0].tolist()})
                    continue
            frames_kp.append({'keypoints': [[0, 0, 0]] * 17})
        cap.release()

        with open(out_kp, 'w') as f:
            json.dump(frames_kp, f)

    print(f"Keypoints done: {len(list(dirs['kp'].glob('*.json')))}")

    # Step 2: Extract crops
    print("\n=== Step 2: Extract crops ===")
    import importlib
    head_crop_fn   = importlib.import_module('crop_clips').crop_video
    upper_crop_fn  = importlib.import_module('crop_upperbody').crop_video
    body_crop_fn   = importlib.import_module('crop_body').crop_video
    leg_crop_fn    = importlib.import_module('crop_leg').crop_video

    for vid in tqdm(videos, desc='Crops'):
        vid_id  = vid.stem
        kp_path = str(dirs['kp'] / f"{vid_id}.json")
        if not Path(kp_path).exists(): continue

        # Head crop
        out = str(dirs['head'] / f"{vid_id}.mp4")
        if not Path(out).exists():
            try: head_crop_fn(str(vid), kp_path, out)
            except Exception as e: print(f"Head crop failed {vid_id}: {e}")

        # Upper body crop
        out = str(dirs['upper'] / f"{vid_id}.mp4")
        if not Path(out).exists():
            try: upper_crop_fn(str(vid), kp_path, out)
            except Exception as e: print(f"Upper crop failed {vid_id}: {e}")

        # Body crop
        out = str(dirs['body'] / f"{vid_id}.mp4")
        if not Path(out).exists():
            try: body_crop_fn(str(vid), kp_path, out)
            except Exception as e: print(f"Body crop failed {vid_id}: {e}")

        # Leg crop
        out = str(dirs['leg'] / f"{vid_id}.mp4")
        if not Path(out).exists():
            try: leg_crop_fn(str(vid), kp_path, out)
            except Exception as e: print(f"Leg crop failed {vid_id}: {e}")

    print(f"Head crops: {len(list(dirs['head'].glob('*.mp4')))}")
    print(f"Upper crops: {len(list(dirs['upper'].glob('*.mp4')))}")
    print(f"Body crops: {len(list(dirs['body'].glob('*.mp4')))}")
    print(f"Leg crops: {len(list(dirs['leg'].glob('*.mp4')))}")

    # Step 3: Extract optical flows
    print("\n=== Step 3: Extract optical flows ===")
    for vid in tqdm(videos, desc='Flows'):
        vid_id = vid.stem

        # Body flow from body crop
        body_crop = str(dirs['body'] / f"{vid_id}.mp4")
        out = str(dirs['bflow'] / f"{vid_id}.mp4")
        if Path(body_crop).exists() and not Path(out).exists():
            extract_flow(body_crop, out)

        # Leg flow from leg crop
        leg_crop = str(dirs['leg'] / f"{vid_id}.mp4")
        out = str(dirs['lflow'] / f"{vid_id}.mp4")
        if Path(leg_crop).exists() and not Path(out).exists():
            extract_flow(leg_crop, out)

    print(f"Body flow: {len(list(dirs['bflow'].glob('*.mp4')))}")
    print(f"Leg flow: {len(list(dirs['lflow'].glob('*.mp4')))}")
    print("\nPreprocessing done ✅")


if __name__ == '__main__':
    main()
