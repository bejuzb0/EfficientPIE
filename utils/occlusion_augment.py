# utils/occlusion_augment.py
"""
Training-time random occlusion augmentation.
Reuses the nominal pedestrian region constants from utils.occlusion.
"""
import numpy as np
import torch
from utils.occlusion import PED_X1, PED_Y1, PED_X2, PED_Y2, FILL

# Kinds sampled during training. Intentionally excludes pedestrian_only /
# context_only which are diagnostic, not realistic augmentations.
TRAIN_KINDS = [
    "top_half", "bottom_half", "left_half", "right_half",
    "random_25pct", "random_50pct",
]

class RandomOcclusion:
    """
    Callable transform. With probability p, apply a random occlusion kind
    to a normalized (C, 300, 300) tensor. Must be placed AFTER Normalize
    in the transform pipeline so that FILL=0.0 corresponds to gray.
    """
    def __init__(self, p=0.5, kinds=None):
        self.p = p
        self.kinds = kinds if kinds is not None else TRAIN_KINDS

    def __call__(self, img):
        # Use torch.rand for proper DataLoader worker behavior
        if torch.rand(1).item() >= self.p:
            return img
        kind_idx = torch.randint(0, len(self.kinds), (1,)).item()
        kind = self.kinds[kind_idx]
        img = img.clone()
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
            ped_w, ped_h = PED_X2 - PED_X1, PED_Y2 - PED_Y1
            side = int(np.sqrt(pct * ped_w * ped_h))
            side = max(1, min(side, ped_w, ped_h))
            px = torch.randint(PED_X1, PED_X2 - side + 1, (1,)).item()
            py = torch.randint(PED_Y1, PED_Y2 - side + 1, (1,)).item()
            img[:, py:py+side, px:px+side] = FILL
        return img
