import torch
from transformers import VideoMAEForVideoClassification

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

print("\nLoading videomae-large-finetuned-kinetics...")
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-large-finetuned-kinetics",
    num_labels=6, ignore_mismatched_sizes=True
).cuda()

print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

for bs in [4, 8, 16]:
    try:
        torch.cuda.empty_cache()
        x = torch.randn(bs, 16, 3, 224, 224).cuda()
        with torch.no_grad():
            out = model(pixel_values=x)
        vram = torch.cuda.memory_allocated()/1e9
        print(f"batch={bs} ✅ | VRAM={vram:.2f}GB")
        del x
    except RuntimeError as e:
        print(f"batch={bs} ❌ OOM")
        break
