# utils/occlusion.py
import numpy as np
import torch

# Nominal pedestrian region within a 300x300 input.
# The EfficientPIE JAAD pipeline uses crop_type='context' with 2x bbox
# expansion, so the pedestrian occupies approximately the centered 150x150.
PED_X1, PED_Y1, PED_X2, PED_Y2 = 75, 75, 225, 225

# Fill value AFTER normalization. The test pipeline uses
# Normalize(mean=0.5, std=0.5), so raw gray 127.5 maps to 0.0.
FILL = 0.0

OCCLUSION_KINDS = [
    "none",
    "top_half",
    "bottom_half",
    "left_half",
    "right_half",
    "random_10pct",
    "random_25pct",
    "random_50pct",
    "random_75pct",
    "pedestrian_only",
    "context_only",
]

def _random_patch(img, pct, rng):
    ped_w, ped_h = PED_X2 - PED_X1, PED_Y2 - PED_Y1
    side = int(np.sqrt(pct * ped_w * ped_h))
    side = max(1, min(side, ped_w, ped_h))
    px = rng.randint(PED_X1, PED_X2 - side + 1)
    py = rng.randint(PED_Y1, PED_Y2 - side + 1)
    img[:, py:py+side, px:px+side] = FILL
    return img

def occlude(img, kind, rng=None):
    """
    Apply occlusion to a single normalized tensor of shape (C, 300, 300).
    Returns a new tensor; does not modify in-place.
    """
    img = img.clone()
    if kind == "none":
        return img
    if kind == "top_half":
        y_mid = (PED_Y1 + PED_Y2) // 2
        img[:, PED_Y1:y_mid, PED_X1:PED_X2] = FILL
    elif kind == "bottom_half":
        y_mid = (PED_Y1 + PED_Y2) // 2
        img[:, y_mid:PED_Y2, PED_X1:PED_X2] = FILL
    elif kind == "left_half":
        x_mid = (PED_X1 + PED_X2) // 2
        img[:, PED_Y1:PED_Y2, PED_X1:x_mid] = FILL
    elif kind == "right_half":
        x_mid = (PED_X1 + PED_X2) // 2
        img[:, PED_Y1:PED_Y2, x_mid:PED_X2] = FILL
    elif kind.startswith("random_"):
        pct = int(kind.split("_")[1].replace("pct", "")) / 100.0
        rng = rng if rng is not None else np.random.RandomState(0)
        img = _random_patch(img, pct, rng)
    elif kind == "pedestrian_only":
        out = torch.full_like(img, FILL)
        out[:, PED_Y1:PED_Y2, PED_X1:PED_X2] = img[:, PED_Y1:PED_Y2, PED_X1:PED_X2]
        img = out
    elif kind == "context_only":
        img[:, PED_Y1:PED_Y2, PED_X1:PED_X2] = FILL
    else:
        raise ValueError(f"unknown occlusion kind: {kind}")
    return img

def occlude_batch(imgs, kind, seed=0):
    """imgs: (B, C, 300, 300)"""
    rng = np.random.RandomState(seed)
    return torch.stack([occlude(imgs[i], kind, rng) for i in range(imgs.size(0))])
