"""
Environment smoke test for MAC 2026 Track 2.

Verifies the Python/CUDA/PyTorch stack and that the key dependencies import.
Run after `setup.sh` (or `pip install -r requirements.txt`):

    python tools/check_env.py
"""

import importlib
import sys


def check_import(mod, attr_version="__version__"):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, attr_version, "?")
        print(f"  [ok]   {mod:<16} {v}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [MISS] {mod:<16} ({e.__class__.__name__}: {e})")
        return False


def main():
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")

    print("\nCore deps:")
    ok = True
    for mod in ["numpy", "pandas", "scipy", "tqdm", "cv2", "yaml"]:
        ok &= check_import(mod)

    print("\nDeep-learning stack:")
    torch_ok = check_import("torch")
    check_import("torchvision")
    check_import("mmengine")
    check_import("mmcv")
    check_import("mmaction")

    print("\nOptional (auxiliary streams):")
    check_import("mediapipe")

    if torch_ok:
        import torch
        print("\nCUDA:")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram = props.total_memory / 1024**3
                print(f"    [{i}] {props.name}  ({vram:.1f} GB)")
                if vram < 18:
                    print("        WARNING: VideoMAE-Large training needs ~20 GB VRAM.")
        else:
            print("  WARNING: no GPU visible — training/inference will not run.")

    print("\nDone. Resolve any [MISS] lines before training.")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
