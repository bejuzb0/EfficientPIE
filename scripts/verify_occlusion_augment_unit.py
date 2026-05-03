# scripts/verify_occlusion_augment_unit.py
"""
Pure unit test for RandomOcclusion: no JAAD data needed.
Confirms:
  - Transform imports without error.
  - With p=0.0, image is unchanged.
  - With p=1.0, image is always modified.
  - With p=0.5, ~50% of a large sample is modified.
  - Occlusion pixel values match FILL.
  - Occlusion region stays within nominal pedestrian box.
  - Multiple distinct occlusion shapes are reachable.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import torch
from utils.occlusion_augment import RandomOcclusion, TRAIN_KINDS
from utils.occlusion import PED_X1, PED_Y1, PED_X2, PED_Y2, FILL

torch.manual_seed(0)
N = 1000
img_template = torch.randn(3, 300, 300)  # arbitrary normalized-looking input

def frac_modified(tfm, n=N):
    changed = 0
    for _ in range(n):
        out = tfm(img_template)
        if not torch.equal(out, img_template):
            changed += 1
    return changed / n

# Test 1: p=0 means never modify
f0 = frac_modified(RandomOcclusion(p=0.0), n=200)
assert f0 == 0.0, f"p=0.0 but {f0:.3f} modified"
print(f"[PASS] p=0.0: {f0:.3f} modified (expected 0.0)")

# Test 2: p=1 means always modify
f1 = frac_modified(RandomOcclusion(p=1.0), n=200)
assert f1 == 1.0, f"p=1.0 but only {f1:.3f} modified"
print(f"[PASS] p=1.0: {f1:.3f} modified (expected 1.0)")

# Test 3: p=0.5 gives roughly 50%
f5 = frac_modified(RandomOcclusion(p=0.5), n=N)
assert 0.42 < f5 < 0.58, f"p=0.5 yielded {f5:.3f}, outside [0.42, 0.58]"
print(f"[PASS] p=0.5: {f5:.3f} modified (expected ~0.50)")

# Test 4: occlusion pixels are actually FILL
tfm1 = RandomOcclusion(p=1.0)
for _ in range(50):
    out = tfm1(img_template)
    diff = (out != img_template).any(dim=0)
    for c in range(3):
        ch = out[c]
        assert torch.all(ch[diff] == FILL), \
            f"Found non-FILL value in modified region: {ch[diff].unique()}"
print(f"[PASS] All modified pixels equal FILL={FILL}")

# Test 5: occlusion region stays within nominal pedestrian box
for _ in range(50):
    out = tfm1(img_template)
    diff = (out != img_template).any(dim=0)
    ys, xs = torch.nonzero(diff, as_tuple=True)
    if len(ys) == 0:
        continue
    y_min, y_max = ys.min().item(), ys.max().item()
    x_min, x_max = xs.min().item(), xs.max().item()
    assert y_min >= PED_Y1 and y_max < PED_Y2, \
        f"Occlusion y=[{y_min},{y_max}] outside [{PED_Y1},{PED_Y2})"
    assert x_min >= PED_X1 and x_max < PED_X2, \
        f"Occlusion x=[{x_min},{x_max}] outside [{PED_X1},{PED_X2})"
print(f"[PASS] All occlusions inside nominal ped box "
      f"[{PED_X1}-{PED_X2}, {PED_Y1}-{PED_Y2}]")

# Test 6: multiple distinct occlusion shapes are reachable
seen = set()
torch.manual_seed(1)
for _ in range(500):
    out = tfm1(img_template)
    diff = (out != img_template).any(dim=0)
    ys, xs = torch.nonzero(diff, as_tuple=True)
    if len(ys) == 0:
        continue
    h = ys.max().item() - ys.min().item() + 1
    w = xs.max().item() - xs.min().item() + 1
    seen.add((h // 10, w // 10))
assert len(seen) >= 4, f"Only {len(seen)} distinct shapes, expected >=4"
print(f"[PASS] Distinct occlusion shapes observed: {len(seen)} (>= 4)")

print("\nAll unit tests passed.")
