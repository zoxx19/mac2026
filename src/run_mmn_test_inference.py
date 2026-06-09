"""
Run MMN skeleton inference on test split and save scores pkl.
~5 min on any GPU.
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""
import os, sys, pickle
import numpy as np
import torch

MMN_REPO = "../Micro-action-skeleton/MMN"
sys.path.insert(0, os.path.join(MMN_REPO, "torchlight"))
sys.path.insert(0, os.path.join(MMN_REPO, "torchpack"))
sys.path.insert(0, MMN_REPO)

from model.MMN import MMN
from torch.utils.data import DataLoader
sys.path.insert(0, './src')
from feeder_mmad_52class import Feeder

CKPT     = MMN_REPO + '/work_dir/train/mmad_52class_J/runs-30-148800.pt'
SKE_TEST = 'data/skeleton/test'
ANN_TEST = './data/MMA-52/Annotations/test.csv'
OUT_PKL  = MMN_REPO + '/work_dir/train/mmad_52class_J/epoch30_actual_test_score.pkl'

JOINTS   = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 3, 4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
model = MMN(in_channels=2, num_classes=52, num_people=1, num_frames=32,
            num_points=17, kernel_size=3, num_heads=4, drop=0.0,
            head_drop=0.1, drop_path=0.3, mlp_ratio=2.0, index_t=True)
ckpt = torch.load(CKPT, map_location='cpu')
sd = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
model.load_state_dict(sd, strict=True)
model.to(device).eval()
print("Model loaded")

# Load test dataset
dataset = Feeder(split='val',  # use val split mode (no shuffle)
                 keypoint_dir=SKE_TEST,
                 ann_csv=ANN_TEST,
                 window_size=32, stride=8,
                 overlap_threshold=0.5, data_type='j')
loader = DataLoader(dataset, batch_size=64, shuffle=False,
                    num_workers=4, pin_memory=True)
print(f"Test windows: {len(dataset)}")

# Run inference
all_scores = {}
idx = 0
with torch.no_grad():
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            data, label = batch[0], batch[1]
        else:
            data, label = batch, None
        
        # Handle index_t
        if isinstance(data, (list, tuple)):
            x, ix = data[0].to(device), data[1].to(device)
            out = torch.sigmoid(model(x, ix))
        else:
            T = data.shape[2] if len(data.shape) > 2 else 32
            ix = torch.linspace(-1, 1, T).unsqueeze(0).expand(data.shape[0], -1).to(device)
            out = torch.sigmoid(model(data.to(device), ix))
        
        scores = out.cpu().numpy()
        for s in scores:
            all_scores[idx] = s
            idx += 1

print(f"Inference done: {len(all_scores)} windows")
pickle.dump(all_scores, open(OUT_PKL, 'wb'))
print(f"Saved: {OUT_PKL}")
