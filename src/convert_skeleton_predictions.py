"""
convert_skeleton_predictions.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/convert_skeleton_predictions.py --help` for options where applicable.
"""
import os, json, pickle
import numpy as np
import pandas as pd

WINDOW  = 32
STRIDE  = 8
THRESH  = 0.02
MAX_GAP = 2
JOINTS  = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 3, 4]

PKL_VAL  = '../Micro-action-skeleton/MMN/work_dir/train/mmad_52class_J/epoch30_test_score.pkl'
SKE_VAL  = 'data/skeleton/val'
ANN_VAL  = './data/MMA-52/Annotations/val.csv'
CAT_FILE = './data/MMA-52/Annotations/adatad/category_idx.txt'
OUT_DIR  = 'predictions'

def load_categories():
    return [l.strip() for l in open(CAT_FILE) if l.strip()]

def merge_segments(window_scores, class_idx, threshold, max_gap):
    segments, seg_start, seg_end, seg_max, gap = [], None, None, 0.0, 0
    for t_start, t_end, scores in window_scores:
        score = float(scores[class_idx])
        if score >= threshold:
            if seg_start is None:
                seg_start, seg_end, seg_max, gap = t_start, t_end, score, 0
            else:
                seg_end = t_end
                seg_max = max(seg_max, score)
                gap = 0
        else:
            if seg_start is not None:
                gap += 1
                if gap > max_gap:
                    segments.append((seg_start, seg_end, seg_max))
                    seg_start = seg_end = None
                    seg_max = 0.0
                    gap = 0
    if seg_start is not None:
        segments.append((seg_start, seg_end, seg_max))
    return segments

def convert(pkl_path, ske_dir, ann_csv, out_path, cats):
    scores_all = pickle.load(open(pkl_path, 'rb'))
    df = pd.read_csv(ann_csv)
    
    # Build fps lookup
    fps_lookup = {}
    for _, row in df.iterrows():
        fps_lookup[str(row['video_id'])] = float(row['fps'])
    
    # Iterate npy files in SAME order as feeder: sorted(os.listdir)
    npy_files = sorted([f for f in os.listdir(ske_dir) if f.endswith('_keypoints.npy')])
    print(f"NPC files: {len(npy_files)}, PKL windows: {len(scores_all)}")
    
    results = {}
    win_idx = 0
    
    for npy_file in npy_files:
        video_id = npy_file.replace('_keypoints.npy', '')
        npy_path = os.path.join(ske_dir, npy_file)
        
        # Load keypoints to get frame count (same as feeder)
        kp = np.load(npy_path)
        kp = kp[:, JOINTS, :2].astype(np.float32)
        N = kp.shape[0]
        fps = fps_lookup.get(video_id, 30.0)
        
        # Count windows exactly as feeder does
        window_scores = []
        for t_start in range(0, max(1, N - WINDOW + 1), STRIDE):
            t_end = t_start + WINDOW
            if t_end > N:
                break
            if win_idx >= len(scores_all):
                break
            sig = 1.0 / (1.0 + np.exp(-scores_all[win_idx]))  # logits → probabilities
            window_scores.append((t_start, t_end, sig))
            win_idx += 1
        
        # Convert to detection proposals
        proposals = []
        for c_idx, c_name in enumerate(cats):
            segs = merge_segments(window_scores, c_idx, THRESH, MAX_GAP)
            for (fs, fe, score) in segs:
                t_s = round(fs / fps, 3)
                t_e = round(fe / fps, 3)
                if t_e > t_s:
                    proposals.append({
                        'segment': [t_s, t_e],
                        'label': c_name,
                        'score': round(float(score), 6)
                    })
        
        proposals.sort(key=lambda x: -x['score'])
        results[video_id] = proposals
    
    print(f"Converted: {len(results)} videos, {sum(len(v) for v in results.values())} proposals")
    print(f"Windows used: {win_idx} / {len(scores_all)}")
    
    # Sample
    for vid in list(results.keys())[:2]:
        print(f"  {vid}: {len(results[vid])} proposals | top: {results[vid][0] if results[vid] else 'none'}")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({'results': results}, open(out_path, 'w'))
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    cats = load_categories()
    convert(PKL_VAL, SKE_VAL, ANN_VAL,
            os.path.join(OUT_DIR, 'skeleton_val.json'), cats)

def convert_test():
    PKL_TEST = '../Micro-action-skeleton/MMN/work_dir/train/mmad_52class_J/epoch30_actual_test_score.pkl'
    SKE_TEST = 'data/skeleton/test'
    ANN_TEST = './data/MMA-52/Annotations/test.csv'
    OUT_TEST = 'predictions/skeleton_test.json'
    
    if not os.path.exists(PKL_TEST):
        print(f"Test pkl not ready yet: {PKL_TEST}")
        return
    
    cats = load_categories()
    convert(PKL_TEST, SKE_TEST, ANN_TEST, OUT_TEST, cats)

if __name__ == '__main__':
    cats = load_categories()
    # Val
    convert(PKL_VAL, SKE_VAL, ANN_VAL,
            os.path.join(OUT_DIR, 'skeleton_val.json'), cats)
    # Test (only if pkl exists)
    convert_test()
