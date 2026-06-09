#!/usr/bin/env python3
"""Modular post-processing pipeline for MAC 2026 Track 2 TAD predictions.
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""

import json
import math
import os
import sys
import tempfile
import argparse
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────
OPENTAD_DIR  = '../../OpenTAD'
ANNO_DEFAULT = (
    '.'
    '/data/MMA-52/Annotations/adatad/mma52_anno.json'
)
PRED_BASE   = 'predictions'
BUCKET_SIZE = 0.5  # seconds (Module A)

# ── Body-part groups (exact strings from category_idx.txt, all 52 classes) ───
BODY_GROUPS = {
    'A_Body': [
        'shaking body', 'sitting straightly', 'shrugging', 'turning around',
        'rising up', 'arms akimbo',
    ],
    'B_Head': [
        'bowing head', 'head up', 'tilting head', 'turning head', 'nodding', 'shaking head',
    ],
    'C_UpperLimb': [
        'scratching arms', 'playing objects', 'putting hands together', 'rubbing hands',
        'pointing oneself', 'clenching fist', 'stretching arms', 'retracting arms',
        'waving', 'spreading hands', 'hands touching fingers', 'other finger movements',
        'illustrative gestures', 'crossing arms', 'playing or tidying hair',
    ],
    'D_LowerLimb': [
        'shaking legs', 'curling legs', 'spread legs', 'closing legs', 'crossing legs',
        'stretching feet', 'retracting feet', 'tiptoe',
    ],
    'E_BodyHand': [
        'scratching or touching neck', 'scratching or touching chest',
        'scratching or touching back', 'scratching or touching shoulder',
        'scratching or touching hindbrain', 'touching nose', 'touching ears',
        'covering face', 'pushing glasses',
    ],
    'F_HeadHand': [
        'scratching or touching forehead', 'scratching or touching face',
        'rubbing eyes', 'covering mouth',
    ],
    'G_LegHand': [
        'patting legs', 'touching legs', 'scratching legs', 'scratching feet',
    ],
}

LABEL_TO_GROUP = {lbl: grp for grp, lbls in BODY_GROUPS.items() for lbl in lbls}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _iou(a, b):
    """Temporal IoU of two [t_s, t_e] segments."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0.0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0.0 else 0.0


def _soft_nms(proposals, sigma=0.5):
    """Gaussian soft-NMS on a list of proposals (same video + label)."""
    props = sorted(proposals, key=lambda x: x['score'], reverse=True)
    kept = []
    while props:
        best = props.pop(0)
        kept.append(best)
        for p in props:
            iou = _iou(best['segment'], p['segment'])
            if iou > 0.0:
                p['score'] *= math.exp(-(iou ** 2) / sigma)
        props.sort(key=lambda x: x['score'], reverse=True)
    return kept


def build_fps_dict(anno_path):
    """Return (fps_map: {video_id: fps}, median_fps) derived from annotation duration+frame."""
    db = json.load(open(anno_path))['database']
    fps_map = {}
    for vid, entry in db.items():
        dur = entry.get('duration', 0.0)
        frm = entry.get('frame', 0)
        if dur > 0.0:
            fps_map[vid] = frm / dur
    vals = sorted(fps_map.values())
    median_fps = vals[len(vals) // 2] if vals else 30.0
    return fps_map, median_fps


def build_cooc_matrix(anno_path):
    """
    P(B|A) from training annotations.
    Co-occurrence = at least one segment of A overlaps at least one segment of B
    in the same training video (video-level count, not annotation-pair count).
    """
    db = json.load(open(anno_path))['database']
    occur_count = defaultdict(int)
    cooc_count  = defaultdict(lambda: defaultdict(int))

    for entry in db.values():
        if entry.get('subset') != 'training':
            continue
        anns = entry.get('annotations', [])
        if not anns:
            continue
        label_segs = defaultdict(list)
        for a in anns:
            label_segs[a['label']].append(a['segment'])
        for lbl in label_segs:
            occur_count[lbl] += 1
        labels = list(label_segs.keys())
        for lA in labels:
            for lB in labels:
                if lA == lB:
                    continue
                if any(_iou(sA, sB) > 0.0
                       for sA in label_segs[lA]
                       for sB in label_segs[lB]):
                    cooc_count[lA][lB] += 1

    cooc_prob = {}
    for lA, cnts in cooc_count.items():
        n = occur_count[lA]
        if n > 0:
            cooc_prob[lA] = {lB: cnt / n for lB, cnt in cnts.items()}
    return cooc_prob


# ── Module A: Hierarchical Body-Part Suppression ─────────────────────────────
def module_a_hierarchical(results, group_threshold=0.5, suppression_factor=0.1,
                           score_thresh=0.1):
    """
    For each 0.5-second time bucket in a video, suppress individual proposals
    that are BOTH low-confidence (score < score_thresh) AND belong to a body-part
    group whose total energy in that bucket is < group_threshold.
    High-confidence proposals are never suppressed regardless of group energy.
    A proposal can belong to multiple buckets (overlap-based assignment).
    """
    out = {}
    for vid, proposals in results.items():
        props = [dict(p) for p in proposals]

        # Collect all bucket indices that have at least one overlapping proposal
        all_buckets = set()
        for p in props:
            t_s, t_e = p['segment']
            b_first = int(t_s / BUCKET_SIZE)
            b_last  = max(b_first, int((t_e - 1e-9) / BUCKET_SIZE))
            for b in range(b_first, b_last + 1):
                all_buckets.add(b)

        for b in all_buckets:
            b_lo = b * BUCKET_SIZE
            b_hi = (b + 1) * BUCKET_SIZE
            # Group all proposals that overlap this bucket by body-part group
            group_props = defaultdict(list)
            for p in props:
                t_s, t_e = p['segment']
                if t_e > b_lo and t_s < b_hi:
                    grp = LABEL_TO_GROUP.get(p['label'])
                    if grp:
                        group_props[grp].append(p)
            for grp, grp_list in group_props.items():
                if sum(p['score'] for p in grp_list) < group_threshold:
                    for p in grp_list:
                        if p['score'] < score_thresh:  # only suppress low-confidence proposals
                            p['score'] *= suppression_factor

        props.sort(key=lambda x: x['score'], reverse=True)
        out[vid] = props
    return out


# ── Module B: Temporal Boundary Jittering ─────────────────────────────────────
def module_b_boundary_jitter(results, fps_map, median_fps,
                              duration_thresh=2.0, variant_weight=0.7, nms_sigma=0.5,
                              score_thresh=0.05, top_k_per_label=100, top_k_out=200):
    """
    For short proposals (< duration_thresh), generate 4 boundary variants with
    score = original * variant_weight. Pipeline:
      1. Filter to score > score_thresh
      2. Limit to top_k_per_label per (video, label) before generating variants
      3. Soft-NMS per label on originals + variants
      4. Keep top top_k_out proposals per video
    """
    out = {}
    for vid, proposals in results.items():
        dt = 1.0 / fps_map.get(vid, median_fps)

        # Step 1: score filter
        filtered = [p for p in proposals if float(p['score']) > score_thresh]

        # Step 2: group by label, keep top_k_per_label each
        by_label = defaultdict(list)
        for p in filtered:
            by_label[p['label']].append(p)
        for lbl in by_label:
            by_label[lbl].sort(key=lambda x: x['score'], reverse=True)
            by_label[lbl] = by_label[lbl][:top_k_per_label]

        # Step 3: add variants then soft-NMS per label
        nms_out = []
        for lbl, lbl_props in by_label.items():
            expanded = []
            for p in lbl_props:
                t_s, t_e = p['segment']
                sc = float(p['score'])
                expanded.append({'segment': [t_s, t_e], 'label': lbl, 'score': sc})
                if (t_e - t_s) < duration_thresh:
                    for ns, ne in (
                        (t_s + dt, t_e),
                        (t_s,      t_e - dt),
                        (max(0.0, t_s - dt), t_e),
                        (t_s,      t_e + dt),
                    ):
                        if ns < ne:
                            expanded.append(
                                {'segment': [ns, ne], 'label': lbl, 'score': sc * variant_weight}
                            )
            nms_out.extend(_soft_nms(expanded, sigma=nms_sigma))

        # Step 4: keep top_k_out per video
        nms_out.sort(key=lambda x: x['score'], reverse=True)
        out[vid] = nms_out[:top_k_out]
    return out


# ── Module C: Co-occurrence Score Boosting ────────────────────────────────────
def module_c_cooccurrence(results, anno_path, cooc_threshold=0.3,
                           high_threshold=0.5, boost_factor=1.5):
    """
    For each high-confidence proposal (anchor), find co-occurring classes from
    training statistics and boost EXISTING proposals that overlap the anchor.
    No synthetic proposals are added.
    """
    cooc_prob = build_cooc_matrix(anno_path)
    out = {}
    for vid, proposals in results.items():
        props = [dict(p) for p in proposals]
        anchors = [p for p in props if p['score'] > high_threshold]

        for anchor in anchors:
            cond = cooc_prob.get(anchor['label'], {})
            for lB, p_ba in cond.items():
                if p_ba <= cooc_threshold:
                    continue
                for p in props:
                    if p['label'] == lB and _iou(p['segment'], anchor['segment']) > 0.3:
                        p['score'] *= boost_factor

        props.sort(key=lambda x: x['score'], reverse=True)
        out[vid] = props
    return out


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(results, anno_path, split, tiou_thresholds=None):
    """Evaluate using OpenTAD's mAP evaluator."""
    if tiou_thresholds is None:
        tiou_thresholds = [0.2, 0.5, 0.7]
    if OPENTAD_DIR not in sys.path:
        sys.path.insert(0, OPENTAD_DIR)
    from opentad.evaluations import mAP as mAPEval

    subset = {'val': 'validation', 'test': 'testing'}[split]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'results': results}, f)
        tmp = f.name
    try:
        ev = mAPEval(
            ground_truth_filename=anno_path,
            prediction_filename=tmp,
            subset=subset,
            tiou_thresholds=tiou_thresholds,
        )
        return ev.evaluate()
    finally:
        os.unlink(tmp)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='MAC 2026 Track 2 post-processing pipeline'
    )
    parser.add_argument('--input',  default=None,
                        help='Predictions JSON (default: large_{split}_original.json)')
    parser.add_argument('--output', default=None)
    parser.add_argument('--split',  default='val', choices=['val', 'test'])
    parser.add_argument('--module', default='all',
                        choices=['A', 'B', 'C', 'AB', 'ABC', 'all'])
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--anno',     default=ANNO_DEFAULT)
    parser.add_argument('--fps_file', default=None,
                        help='JSON {video_id: fps} overrides for Module B')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Max proposals per video before any module runs')
    # Module B
    parser.add_argument('--duration_thresh',    type=float, default=2.0)
    parser.add_argument('--variant_weight',     type=float, default=0.7)
    # Module A
    parser.add_argument('--group_threshold',    type=float, default=0.5)
    parser.add_argument('--suppression_factor', type=float, default=0.1)
    # Module C
    parser.add_argument('--cooc_threshold', type=float, default=0.3)
    parser.add_argument('--high_threshold', type=float, default=0.5)
    parser.add_argument('--boost_factor',   type=float, default=1.5)

    args = parser.parse_args()

    inp  = args.input  or os.path.join(PRED_BASE, f'large_{args.split}_original.json')
    outp = args.output or os.path.join(PRED_BASE, f'postprocessed_{args.split}.json')

    if not os.path.exists(inp):
        print(f'[postprocess] Input file not found: {inp}', file=sys.stderr)
        print('[postprocess] Use --input to specify a predictions file.')
        return

    with open(inp) as f:
        data = json.load(f)
    results = data['results'] if 'results' in data else data
    print(f'Loaded {sum(len(v) for v in results.values())} proposals '
          f'from {len(results)} videos')

    # Pre-filter: keep top_k proposals per video before any module runs
    if args.top_k and args.top_k > 0:
        for vid in results:
            results[vid] = sorted(results[vid], key=lambda x: x['score'], reverse=True)[:args.top_k]
        print(f'After top-{args.top_k} pre-filter: '
              f'{sum(len(v) for v in results.values())} proposals')

    run_a = args.module in ('A', 'AB', 'ABC', 'all')
    run_b = args.module in ('B', 'AB', 'ABC', 'all')
    run_c = args.module in ('C', 'ABC', 'all')

    fps_map, median_fps = build_fps_dict(args.anno)
    if args.fps_file:
        with open(args.fps_file) as f:
            fps_map.update(json.load(f))

    if run_a:
        print('Running Module A (hierarchical suppression)...')
        results = module_a_hierarchical(
            results,
            group_threshold=args.group_threshold,
            suppression_factor=args.suppression_factor,
        )
    if run_b:
        print('Running Module B (boundary jitter)...')
        results = module_b_boundary_jitter(
            results, fps_map, median_fps,
            duration_thresh=args.duration_thresh,
            variant_weight=args.variant_weight,
            top_k_per_label=args.top_k,
        )
    if run_c:
        print('Running Module C (co-occurrence boost)...')
        results = module_c_cooccurrence(
            results, args.anno,
            cooc_threshold=args.cooc_threshold,
            high_threshold=args.high_threshold,
            boost_factor=args.boost_factor,
        )

    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
    with open(outp, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f'Saved {sum(len(v) for v in results.values())} proposals → {outp}')

    if args.evaluate:
        print('Evaluating...')
        metrics = evaluate(results, args.anno, args.split)
        if metrics:
            for k, v in sorted(metrics.items()):
                print(f'  {k}: {v:.4f}')


if __name__ == '__main__':
    main()
