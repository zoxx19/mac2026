"""Strip a training checkpoint down to weights only, with epoch=-1.

Used to warm-start the DSTA run from the proven adapter ep22 checkpoint while
forcing a FRESH run: epoch starts at 0, and train.py builds a fresh optimizer /
scheduler / EMA (it logs "No optimizer in checkpoint - using fresh optimizer").
The fresh DSTA adapter params (identity init) simply stay fresh under strict=False.
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument("dst", type=str)
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError(f"no 'state_dict' in {args.src}; keys={list(ckpt.keys())}")

    # train.py unconditionally reads state_dict_ema when EMA is enabled, so we
    # must keep it. Prefer the source EMA weights; fall back to raw weights.
    ema = ckpt.get("state_dict_ema", ckpt["state_dict"])
    out = {"state_dict": ckpt["state_dict"], "state_dict_ema": ema, "epoch": -1}
    torch.save(out, args.dst)
    n = sum(v.numel() for v in ckpt["state_dict"].values())
    print(f"wrote {args.dst}  (weights-only + ema, epoch=-1, {n/1e6:.1f}M params)")


if __name__ == "__main__":
    main()
