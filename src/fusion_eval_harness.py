"""
Efficient in-process sweep harness for MAC 2026 Track 2 fusion.

Loads each prediction file once, normalises once, then evaluates many
(weights, sigma, top_k) configs against val ground truth. Reuses the fast
vectorised soft_nms from multi_ensemble.py.
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""
import json, os, sys, tempfile, itertools
from collections import defaultdict

OPENTAD_DIR = "../../OpenTAD"
PROJ = "."
PRED_BASE = "predictions"
ANNO = os.path.join(PROJ, "data/MMA-52/Annotations/adatad/mma52_anno.json")

sys.path.insert(0, OPENTAD_DIR)
sys.path.insert(0, os.path.join(PROJ, "src"))
from multi_ensemble import soft_nms  # vectorised
from opentad.evaluations import mAP as mAPEvaluator

import numpy as np

_cache = {}
def load_norm(path):
    """Load results dict, normalise scores to [0,1]. Cached."""
    if path in _cache:
        return _cache[path]
    full = path if os.path.isabs(path) else os.path.join(PRED_BASE, path)
    with open(full) as f:
        data = json.load(f)
    results = data.get("results", data)
    all_scores = [p["score"] for props in results.values() for p in props]
    mx = max(all_scores) if all_scores else 1.0
    if mx > 0:
        for props in results.values():
            for p in props:
                p["score"] = p["score"] / mx
    _cache[path] = results
    return results

def fuse(inputs, weights, sigma=0.5, top_k=0):
    """inputs: list of results dicts. weights: list of floats."""
    groups = defaultdict(list)
    for results, w in zip(inputs, weights):
        for vid, props in results.items():
            for p in props:
                groups[(vid, p["label"])].append(
                    {"segment": p["segment"], "label": p["label"], "score": p["score"] * w})
    out = defaultdict(list)
    for (vid, _lbl), props in groups.items():
        out[vid].extend(soft_nms(props, sigma=sigma))
    fused = {}
    for vid, props in out.items():
        props.sort(key=lambda x: x["score"], reverse=True)
        fused[vid] = props[:top_k] if top_k and top_k > 0 else props
    return fused

def evaluate(fused, split="val"):
    subset = {"val": "validation", "test": "testing"}[split]
    tiou = [0.2, 0.5, 0.7]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"results": fused}, f); tmp = f.name
    try:
        ev = mAPEvaluator(ground_truth_filename=ANNO, prediction_filename=tmp,
                          subset=subset, tiou_thresholds=tiou)
        res = ev.evaluate()
    finally:
        os.unlink(tmp)
    return res

def run_config(label, paths, weights, sigma=0.5, top_k=0, split="val"):
    inputs = [load_norm(p) for p in paths]
    fused = fuse(inputs, weights, sigma=sigma, top_k=top_k)
    res = evaluate(fused, split=split)
    avg = res.get("average_mAP", 0.0) * 100
    m2 = res.get("mAP@0.2", 0.0) * 100
    m5 = res.get("mAP@0.5", 0.0) * 100
    m7 = res.get("mAP@0.7", 0.0) * 100
    print(f"{label:<46} avg={avg:6.3f}  @0.2={m2:6.3f} @0.5={m5:6.3f} @0.7={m7:6.3f}", flush=True)
    return avg

if __name__ == "__main__":
    pass
