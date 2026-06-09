"""
run_test_inference.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/run_test_inference.py --help` for options where applicable.
"""
import torch, json, os, sys, glob, tqdm
sys.path.insert(0, '../../OpenTAD')
from mmengine.config import Config
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader

cfg = Config.fromfile('../../OpenTAD/configs/adatad/mma52/e2e_mma52_videomae_l_adapter.py')
test_ds = build_dataset(cfg.dataset.test)
print("Test dataset:", len(test_ds), "videos")

# Load latest Large checkpoint
cks = sorted(glob.glob('work_dirs/e2e_mma52_videomae_l_adapter/gpu2_id0/checkpoint/epoch_*.pth'), key=os.path.getmtime)
cks = [c for c in cks if 'STALE' not in c and 'BACKUP' not in c]
ck = cks[-1]
print("Checkpoint:", ck)

model = build_detector(cfg.model).cuda().eval()
sd = torch.load(ck, map_location='cpu')
sd_use = sd.get('state_dict_ema', sd['state_dict'])
sd_use = {k.replace('module.',''):v for k,v in sd_use.items()}
model.load_state_dict(sd_use, strict=False)

loader = build_dataloader(test_ds, batch_size=4, rank=0, world_size=1, num_workers=2, shuffle=False, drop_last=False)
post_cfg = cfg.post_processing
post_cfg.sliding_window = True
ext_cls = test_ds.class_map

results = {}
with torch.no_grad():
    for data in tqdm.tqdm(loader):
        for k in list(data.keys()):
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].cuda()
        res = model(**data, return_loss=False, infer_cfg=cfg.inference, post_cfg=post_cfg, ext_cls=ext_cls)
        results.update(res)

save_path = 'data/test_predictions_large.json'
json.dump(dict(results=results), open(save_path, 'w'))
print(f"Saved {len(results)} predictions to {save_path}")
vid = list(results.keys())[0]
print("Sample:", vid, "->", sorted(results[vid], key=lambda x:-x['score'])[:2])

# Evaluate against local test annotations
from opentad.evaluations import build_evaluator
eval_cfg = dict(
    type='mAP',
    subset='testing',
    tiou_thresholds=[0.2, 0.5, 0.7],
    ground_truth_filename='./data/MMA-52/Annotations/adatad/mma52_anno.json',
    prediction_filename=dict(results=results),
)
print("\n=== TEST SET mAP ===")
evaluator = build_evaluator(eval_cfg)
evaluator.evaluate()
