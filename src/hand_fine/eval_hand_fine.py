import sys, os, torch, numpy as np, pandas as pd
sys.path.insert(0, 'src/skeleton/MMN')
sys.path.insert(0, 'src/hand_fine')
from hand_fine_model import build_hand_fusion_model
from hand_fine_dataset import HandFineDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

NAMES = {**{i: f'C{i+1}' for i in range(0,13)},
         **{i: f'E{i-12}' for i in range(13,19)},
         **{i: f'F{i-18}' for i in range(19,29)},
         **{i: f'G{i-28}' for i in range(29,33)},
         33: 'no-hand'}
class_names = [NAMES[i] for i in range(34)]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = build_hand_fusion_model('models/videomae-ssv2', num_classes=34)
state = torch.load('outputs/hand_fine_fusion/best_model.pt', map_location=device)
model.load_state_dict(state)
model.to(device).eval()
ds = HandFineDataset('data/hand_dataset/hand_fine_val.csv', 'data/keypoints/val', augment=False)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
all_true, all_pred = [], []
with torch.no_grad():
    for rgb, skel, labels in loader:
        rgb = rgb.permute(0,2,1,3,4).to(device)
        skel = skel.to(device)
        pred = model(rgb, skel).argmax(1).cpu().numpy()
        all_pred.extend(pred)
        all_true.extend(labels.numpy())
all_true = np.array(all_true)
all_pred = np.array(all_pred)
f1_macro = float(f1_score(all_true, all_pred, average='macro', zero_division=0))
f1_micro = float(f1_score(all_true, all_pred, average='micro', zero_division=0))
f1mean = (f1_macro + f1_micro) / 2
print(f"F1_macro={f1_macro:.4f} | F1_micro={f1_micro:.4f} | F1_mean={f1mean:.4f}")
cm = confusion_matrix(all_true, all_pred, labels=list(range(34)))
cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
fig, axes = plt.subplots(1, 2, figsize=(32, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0], linewidths=0.3, annot_kws={'size':6})
axes[0].set_title('Raw Counts'); axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
axes[0].tick_params(axis='x', rotation=90, labelsize=7); axes[0].tick_params(axis='y', labelsize=7)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1], linewidths=0.3, vmin=0, vmax=1, annot_kws={'size':6})
axes[1].set_title('Normalised'); axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
axes[1].tick_params(axis='x', rotation=90, labelsize=7); axes[1].tick_params(axis='y', labelsize=7)
plt.suptitle(f'Hand Fine Fusion (34 classes) | F1_mean={f1mean:.4f}', fontsize=14)
plt.tight_layout()
os.makedirs('outputs/eval_hand_fine', exist_ok=True)
plt.savefig('outputs/eval_hand_fine/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/eval_hand_fine/confusion_matrix.png")
