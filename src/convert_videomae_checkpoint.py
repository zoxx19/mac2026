"""
convert_videomae_checkpoint.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/convert_videomae_checkpoint.py --help` for options where applicable.
"""
import re
import torch
from safetensors.torch import load_file

INPUT = "pretrained/videomae-base/model.safetensors"
OUTPUT = "pretrained/videomae-base/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"

# Keys starting with these prefixes are silently dropped (classification head)
DROP_PREFIXES = ("fc_norm", "head", "videomae.embeddings.mask_token")

# Regex rules applied in order. First match wins.
# Each rule is (pattern, replacement) — both are passed to re.sub.
RULES = [
    # Patch embedding
    (r"^videomae\.embeddings\.patch_embeddings\.projection\.", "patch_embed.proj."),
    # CLS token and positional embedding
    (r"^videomae\.embeddings\.cls_token$", "cls_token"),
    (r"^videomae\.embeddings\.position_embeddings$", "pos_embed"),
    # Self-attention Q / K / V
    (r"^videomae\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"blocks.\1.attn.q."),
    (r"^videomae\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"blocks.\1.attn.k."),
    (r"^videomae\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"blocks.\1.attn.v."),
    # Attention output projection
    (r"^videomae\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"blocks.\1.attn.proj."),
    # Layer norms
    (r"^videomae\.encoder\.layer\.(\d+)\.layernorm_before\.", r"blocks.\1.norm1."),
    (r"^videomae\.encoder\.layer\.(\d+)\.layernorm_after\.", r"blocks.\1.norm2."),
    # MLP
    (r"^videomae\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"blocks.\1.mlp.fc1."),
    (r"^videomae\.encoder\.layer\.(\d+)\.output\.dense\.", r"blocks.\1.mlp.fc2."),
    # Final layer norm
    (r"^videomae\.layernorm\.", "norm."),
]


def convert_key(hf_key):
    for pattern, repl in RULES:
        new_key, n = re.subn(pattern, repl, hf_key)
        if n > 0:
            return new_key
    return None  # no rule matched


def main():
    print(f"Loading {INPUT}")
    src = load_file(INPUT)
    print(f"  {len(src)} keys in source checkpoint")

    converted = {}
    dropped = []
    unmapped = []

    for hf_key, tensor in src.items():
        if any(hf_key.startswith(p) for p in DROP_PREFIXES):
            dropped.append(hf_key)
            continue

        new_key = convert_key(hf_key)
        if new_key is None:
            unmapped.append(hf_key)
        else:
            converted[new_key] = tensor

    print(f"  {len(converted)} keys converted")
    print(f"  {len(dropped)} keys dropped (head / fc_norm / mask_token)")

    if unmapped:
        print(f"\n[WARNING] {len(unmapped)} keys could not be mapped:")
        for k in unmapped:
            print(f"  {k}")
    else:
        print("  No unmapped keys.")

    print(f"\nSaving to {OUTPUT}")
    torch.save({"state_dict": converted}, OUTPUT)
    print("Done.")


if __name__ == "__main__":
    main()
