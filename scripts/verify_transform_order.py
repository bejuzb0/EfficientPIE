# scripts/verify_transform_order.py
"""
Visually verify that RandomOcclusion applied AFTER Normalize produces
a gray patch (not black), confirming correct transform composition.

Saves results/verify/transform_order_check.png with two columns:
  Left: CORRECT order (Normalize then RandomOcclusion) -> gray patch
  Right: WRONG order (RandomOcclusion then Normalize) -> dark patch
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from utils.occlusion_augment import RandomOcclusion

torch.manual_seed(7)

def denorm(t):
    return (t.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)

# Make a synthetic input: noise with a recognizable colored region in the middle
rng = np.random.RandomState(0)
arr = (rng.rand(300, 300, 3) * 255).astype(np.uint8)
arr[100:200, 100:200] = [200, 120, 60]
img_pil = Image.fromarray(arr)

# Correct order: Normalize first, then RandomOcclusion (writes FILL=0 in normalized space)
tfm_correct = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
    RandomOcclusion(p=1.0),   # always apply for this test
])

# Wrong order: RandomOcclusion before Normalize -> fill=0 becomes -1 after Normalize,
# which is pure black when denormalized
tfm_wrong = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    RandomOcclusion(p=1.0),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

torch.manual_seed(1); out_correct = tfm_correct(img_pil)
torch.manual_seed(1); out_wrong = tfm_wrong(img_pil)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(denorm(out_correct))
axes[0].set_title("CORRECT order:\nNormalize -> RandomOcclusion\n(patch should be GRAY)")
axes[0].axis('off')
axes[1].imshow(denorm(out_wrong))
axes[1].set_title("WRONG order:\nRandomOcclusion -> Normalize\n(patch would appear BLACK)")
axes[1].axis('off')

os.makedirs('results/verify', exist_ok=True)
plt.tight_layout()
plt.savefig('results/verify/transform_order_check.png', dpi=120, bbox_inches='tight')
print("Saved results/verify/transform_order_check.png")

# Programmatic verification
dn_correct = denorm(out_correct)
dn_wrong = denorm(out_wrong)

# Probe the center of the pedestrian region
probe_c = dn_correct[130:170, 130:170].mean()
probe_w = dn_wrong[130:170, 130:170].mean()

print(f"CORRECT composition, denormalized patch mean: {probe_c:.3f} (expect ~0.5)")
print(f"WRONG   composition, denormalized patch mean: {probe_w:.3f} (expect ~0.0)")

ok = 0.4 < probe_c < 0.6
if ok:
    print("[PASS] CORRECT composition produces gray patch")
else:
    print("[FAIL] CORRECT composition does not produce gray — check FILL value")
    sys.exit(1)

if probe_w < 0.1:
    print("[NOTE] WRONG composition produces black patch as expected "
          "(this confirms the CORRECT order matters)")
