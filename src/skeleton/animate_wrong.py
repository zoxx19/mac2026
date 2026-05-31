"""
animate_wrong.py  (skeleton version)
Runs inference, saves wrong_predictions.csv, plots confusion matrix, and
renders annotated MP4 stick-figure videos for wrong prediction clips.

Usage:
    python src/skeleton/animate_wrong.py \
        --track        leg \
        --model_path   outputs/leg_leg_joint/best_model.pt \
        --output_dir   outputs/eval_leg_animated \
        --n_per_class  5

    python src/skeleton/animate_wrong.py \
        --track        body \
        --model_path   outputs/body_full_joint/best_model.pt \
        --output_dir   outputs/eval_body_animated \
        --n_per_class  5

Outputs in --output_dir/:
    confusion_matrix.png
    wrong_predictions.csv
    animated_wrong/
        <class>/  01_true_D1_pred_D3_<video_id>.mp4
                  ...
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score

MMN_ROOT = os.path.join(os.path.dirname(__file__), "MMN")
sys.path.insert(0, MMN_ROOT)
from model.MMN import MMN_

# ── Label maps ────────────────────────────────────────────────────────────────
BODY_LABELS = {
    "A1": "Shaking body",
    "A2": "Sitting straightly",
    "A3": "Shrugging",
    "A4": "Turning around",
    "A5": "Rising up",
    "A6": "No body movement",
}

LEG_LABELS = {
    "D1": "Shaking legs",
    "D2": "Curling legs",
    "D3": "Spread legs",
    "D4": "Closing legs",
    "D5": "Crossing legs",
    "D6": "Stretching feet",
    "D7": "Retracting feet",
    "D8": "Tiptoe",
    "D9": "No leg movement",
}

TRACK_CFG = {
    "body": {
        "csv_val":       "data/skeleton_dataset/body_val.csv",
        "kp_dir":        "data/keypoints/val",
        "num_classes":   6,
        "label_col":     "body_label",
        "video_col":     "video",
        "class_names":   list(BODY_LABELS.keys()),
        "label_full":    BODY_LABELS,
        "joint_indices": list(range(17)),
        "best_model":    "outputs/body_skeleton_full/best_model.pt",  # F1=0.623
    },
    "leg": {
        "csv_val":       "data/skeleton_dataset/leg_val.csv",
        "kp_dir":        "data/keypoints/val",
        "num_classes":   9,
        "label_col":     "leg_label",
        "video_col":     "video",
        "class_names":   list(LEG_LABELS.keys()),
        "label_full":    LEG_LABELS,
        "joint_indices": [11, 12, 13, 14, 15, 16],
        "best_model":    "outputs/leg_skeleton_leg/best_model.pt",    # F1=0.573
    },
}

# ── COCO skeleton edges ───────────────────────────────────────────────────────
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

EDGE_COLORS = {
    (0, 1): "#FF6B6B", (0, 2): "#FF6B6B", (1, 3): "#FF6B6B", (2, 4): "#FF6B6B",
    (0, 5): "#4ECDC4", (0, 6): "#4ECDC4", (5, 6):  "#4ECDC4",
    (5, 7): "#45B7D1", (7, 9): "#45B7D1", (6, 8):  "#45B7D1", (8, 10): "#45B7D1",
    (5, 11): "#96CEB4", (6, 12): "#96CEB4", (11, 12): "#96CEB4",
    (11, 13): "#FFEAA7", (13, 15): "#FFEAA7", (12, 14): "#FFEAA7", (14, 16): "#FFEAA7",
}

BG_COLOR = "#1a1a2e"


# ── Skeleton loading ──────────────────────────────────────────────────────────
def load_skeleton_raw(kp_path, joint_indices, num_frames=64):
    """Returns (T, V, 2) normalised xy for selected joints."""
    with open(kp_path) as f:
        data = json.load(f)
    all_kps = []
    for frame in data:
        kps = np.array(frame["keypoints"], dtype=np.float32)[:, :2]
        all_kps.append(kps)
    skel = np.stack(all_kps, axis=0)
    mins = skel.min(axis=(0, 1), keepdims=True)
    maxs = skel.max(axis=(0, 1), keepdims=True)
    rng  = np.where(maxs - mins > 1e-6, maxs - mins, 1.0)
    skel = (skel - mins) / rng * 2 - 1
    skel = skel[:, joint_indices, :]
    T_raw = skel.shape[0]
    if T_raw >= num_frames:
        idxs = np.linspace(0, T_raw - 1, num_frames, dtype=int)
        skel = skel[idxs]
    else:
        pad  = num_frames - T_raw
        skel = np.concatenate([skel, np.tile(skel[-1:], (pad, 1, 1))], axis=0)
    return skel


def load_skeleton_tensor(kp_path, joint_indices, num_frames=64, in_channels=2):
    """Returns (1, C, T, V, 1) float32 tensor for model inference."""
    with open(kp_path) as f:
        data = json.load(f)
    all_kps = []
    for frame in data:
        kps = np.array(frame["keypoints"], dtype=np.float32)
        all_kps.append(kps)
    skel = np.stack(all_kps, axis=0)
    xy   = skel[:, :, :2]
    conf = skel[:, :, 2:3]
    mins = xy.min(axis=(0, 1), keepdims=True)
    maxs = xy.max(axis=(0, 1), keepdims=True)
    rng  = np.where(maxs - mins > 1e-6, maxs - mins, 1.0)
    xy   = (xy - mins) / rng * 2 - 1
    skel = np.concatenate([xy, conf], axis=-1) if in_channels == 3 else xy
    skel = skel[:, joint_indices, :]
    T_raw = skel.shape[0]
    if T_raw >= num_frames:
        idxs = np.linspace(0, T_raw - 1, num_frames, dtype=int)
        skel = skel[idxs]
    else:
        pad  = num_frames - T_raw
        skel = np.concatenate([skel, np.tile(skel[-1:], (pad, 1, 1))], axis=0)
    skel = skel.transpose(2, 0, 1)[:, :, :, np.newaxis]
    return torch.from_numpy(skel.astype(np.float32)).unsqueeze(0)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_mmn(model_path, num_classes, num_joints, device):
    state      = torch.load(model_path, map_location=device)
    num_heads  = state["MFM.0.blocks.0.gconv"].shape[0]
    in_channels = 2
    model = MMN_(
        in_channels=in_channels,
        num_classes=num_classes,
        num_people=1,
        num_frames=64,
        num_points=num_joints,
        kernel_size=3,
        num_heads=num_heads,
        head_drop=0.0,
        drop=0.0,
        drop_path=0.1,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_ctrgcn(model_path, num_classes, num_joints, device):
    CTRGCN_ROOT = os.path.join(os.path.dirname(__file__), "CTR-GCN")
    GRAPH_ROOT  = os.path.join(CTRGCN_ROOT, "graph")
    sys.path.insert(0, CTRGCN_ROOT)
    sys.path.insert(0, GRAPH_ROOT)
    from model.ctrgcn import Model as CTRGCN
    model = CTRGCN(
        num_class=num_classes, num_point=num_joints, num_person=1,
        graph="coco.Graph", graph_args={"labeling_mode": "spatial"},
        in_channels=3, drop_out=0.0, adaptive=True,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, model_type, df, kp_dir, joint_indices,
                  label_col, video_col, device):
    all_true, all_pred, all_ids, all_kp_paths = [], [], [], []
    index_t = torch.arange(64, dtype=torch.long).to(device)

    for _, row in df.iterrows():
        video_id = str(row[video_col]).replace(".mp4", "")
        label    = int(row[label_col])
        kp_path  = os.path.join(kp_dir, f"{video_id}.json")
        if not os.path.exists(kp_path):
            continue

        in_ch = 2 if model_type == "mmn" else 3
        x = load_skeleton_tensor(kp_path, joint_indices,
                                 in_channels=in_ch).to(device)
        with torch.no_grad():
            out  = model(x, index_t) if model_type == "mmn" else model(x)
        pred = int(out.argmax(dim=1).item())

        all_true.append(label)
        all_pred.append(pred)
        all_ids.append(video_id)
        all_kp_paths.append(kp_path)

    return all_true, all_pred, all_ids, all_kp_paths


# ── Confusion matrix ──────────────────────────────────────────────────────────
def save_confusion_matrix(true, pred, class_names, out_path, track, f1_mean):
    cm      = confusion_matrix(true, pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, title, fmt in [
        (axes[0], cm,      "Counts",     "d"),
        (axes[1], cm_norm, "Normalised", ".2f"),
    ]:
        im = ax.imshow(data, interpolation="nearest",
                       cmap="Blues" if fmt == "d" else "RdYlGn",
                       vmin=0, vmax=(None if fmt == "d" else 1))
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(f"{track.upper()} — {title}", fontsize=12)
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = f"{data[i,j]:{fmt}}" if fmt == "d" else f"{data[i,j]:.2f}"
                ax.text(j, i, val, ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > thresh else "black")

    f1_per = f1_score(true, pred, average=None,
                      labels=list(range(len(class_names))), zero_division=0)
    fig.suptitle(
        f"{track.upper()} track  |  F1-mean = {f1_mean:.4f}  |  "
        + "  ".join(f"{n}={v:.2f}" for n, v in zip(class_names, f1_per)),
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Per-frame renderer ────────────────────────────────────────────────────────
def render_frame(skel_t, joint_indices, true_label, pred_label,
                 label_full, frame_idx, total_frames,
                 width=480, height=540):
    """Render one skeleton frame → BGR numpy array."""
    ji_set = set(joint_indices)
    ji_map = {g: l for l, g in enumerate(joint_indices)}
    local_edges = [
        (ji_map[i], ji_map[j])
        for (i, j) in COCO_EDGES
        if i in ji_set and j in ji_set
    ]
    local_colors = [
        EDGE_COLORS.get((i, j), EDGE_COLORS.get((j, i), "#888888"))
        for (i, j) in COCO_EDGES
        if i in ji_set and j in ji_set
    ]

    dpi = 96
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    xs = skel_t[:, 0]
    ys = skel_t[:, 1]
    for (i, j), color in zip(local_edges, local_colors):
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], "-", color=color, lw=2.5, zorder=2)
    ax.scatter(xs, ys, s=30, c="white", zorder=3)

    wrong = true_label != pred_label
    ax.set_title(
        f"TRUE:  {true_label} — {label_full.get(true_label, '')}\n"
        f"PRED:  {pred_label} — {label_full.get(pred_label, '')}",
        color="#FF4444" if wrong else "#00CC44",
        fontsize=9, loc="left", pad=4, fontfamily="monospace",
    )
    ax.text(0.98, 0.02, f"t {frame_idx + 1}/{total_frames}",
            transform=ax.transAxes, color="#888888", fontsize=7,
            ha="right", va="bottom")
    if wrong:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#DD0000")
            spine.set_linewidth(3)

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(cv2.resize(buf, (width, height)), cv2.COLOR_RGB2BGR)


# ── Clip writer ───────────────────────────────────────────────────────────────
def animate_clip(kp_path, dst_path, joint_indices, true_label, pred_label,
                 label_full, num_frames=64, fps=15, slow_factor=3,
                 width=480, height=540):
    try:
        skel = load_skeleton_raw(kp_path, joint_indices, num_frames)
    except Exception as e:
        print(f"  Cannot load {kp_path}: {e}")
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (width, height))
    T = skel.shape[0]
    for idx in range(T):
        frame = render_frame(skel[idx], joint_indices, true_label, pred_label,
                             label_full, idx, T, width, height)
        for _ in range(slow_factor):
            writer.write(frame)
    writer.release()
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track",       choices=["body", "leg"], required=True)
    parser.add_argument("--model_path",  default=None,
                        help="Override the best model path (default: uses hardcoded best run)")
    parser.add_argument("--model_type",  choices=["mmn", "ctrgcn"], default="mmn")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--n_per_class", type=int, default=5,
                        help="Wrong clips to animate per true class")
    parser.add_argument("--slow_factor", type=int, default=3,
                        help="Repeat each frame N times to slow playback")
    parser.add_argument("--fps",         type=int, default=15)
    parser.add_argument("--num_frames",  type=int, default=64)
    args = parser.parse_args()

    cfg        = TRACK_CFG[args.track]
    label_full = cfg["label_full"]
    joints     = cfg["joint_indices"]
    class_names = cfg["class_names"]
    model_path = args.model_path or cfg["best_model"]

    output_dir  = Path(args.output_dir)
    anim_dir    = output_dir / "animated_wrong"
    output_dir.mkdir(parents=True, exist_ok=True)
    anim_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"Loading model from {model_path} ...")
    if args.model_type == "mmn":
        model = load_mmn(model_path, cfg["num_classes"], len(joints), device)
    else:
        model = load_ctrgcn(model_path, cfg["num_classes"], len(joints), device)

    df = pd.read_csv(cfg["csv_val"])
    print(f"Running inference on {len(df)} val samples...")
    true_ids, pred_ids, video_ids, kp_paths = run_inference(
        model, args.model_type, df,
        cfg["kp_dir"], joints,
        cfg["label_col"], cfg["video_col"], device,
    )

    f1_per  = f1_score(true_ids, pred_ids, average=None,
                       labels=list(range(cfg["num_classes"])), zero_division=0)
    f1_mean = float(np.mean(f1_per))
    print(f"\nF1-mean: {f1_mean:.4f}")
    for name, val in zip(class_names, f1_per):
        print(f"  {name} ({label_full.get(name, '')}): {val:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_path = output_dir / "confusion_matrix.png"
    save_confusion_matrix(true_ids, pred_ids, class_names, cm_path,
                          args.track, f1_mean)

    # ── Save wrong_predictions.csv ────────────────────────────────────────────
    rows = []
    for t, p, vid, kp in zip(true_ids, pred_ids, video_ids, kp_paths):
        if t != p:
            rows.append({"path": kp, "true": class_names[t],
                         "predicted": class_names[p], "video_id": vid})
    df_wrong = pd.DataFrame(rows)
    csv_path = output_dir / "wrong_predictions.csv"
    df_wrong.to_csv(csv_path, index=False)
    print(f"Wrong predictions: {len(df_wrong)} / {len(true_ids)}")
    print(f"Saved: {csv_path}")

    # ── Animate ───────────────────────────────────────────────────────────────
    print(f"\nGenerating MP4s (up to {args.n_per_class} per class)...")
    total_saved = 0

    for true_class in class_names:
        class_wrong = df_wrong[df_wrong["true"] == true_class]
        if len(class_wrong) == 0:
            print(f"\n  {true_class}: no wrong predictions")
            continue

        print(f"\n  {true_class} ({label_full.get(true_class, '')}): "
              f"{len(class_wrong)} wrong total")
        for pred_cls, cnt in class_wrong["predicted"].value_counts().items():
            print(f"    -> {pred_cls} ({label_full.get(pred_cls, '')}): "
                  f"{cnt} times")

        class_dir = anim_dir / true_class
        class_dir.mkdir(exist_ok=True)

        for i, (_, row) in enumerate(class_wrong.head(args.n_per_class).iterrows()):
            kp_path = Path(row["path"])
            if not kp_path.exists():
                print(f"    Missing: {kp_path}")
                continue
            dst_name = (f"{i+1:02d}_true_{true_class}_"
                        f"pred_{row['predicted']}_{kp_path.stem}.mp4")
            dst_path = class_dir / dst_name
            success  = animate_clip(
                kp_path, dst_path, joints,
                row["true"], row["predicted"], label_full,
                args.num_frames, args.fps, args.slow_factor,
            )
            if success:
                print(f"    Saved: {dst_name}")
                total_saved += 1

    print(f"\nTotal animated clips saved: {total_saved}")
    print(f"\nOutputs in {output_dir}/")
    print(f"  confusion_matrix.png")
    print(f"  wrong_predictions.csv")
    print(f"  animated_wrong/<class>/  ({total_saved} MP4s)")
    print("\nHow to use:")
    print("  RED border + red text = wrong prediction")
    print("  Videos are slowed down for easier inspection.")


if __name__ == "__main__":
    main()
