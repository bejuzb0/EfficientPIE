# scripts/verify_train_occlusion.py
"""
Load 16 training samples through the augmented pipeline and save a grid.
Expected: roughly p% of samples should show a gray patch; the rest clean.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms

from utils.jaad_data import JAAD
from utils.my_dataset import MyDataSet
from utils.occlusion_augment import RandomOcclusion
from utils.occlusion import PED_X1, PED_Y1, PED_X2, PED_Y2

def denorm(t):
    return (t.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)

def main(args):
    data_opts = {
        'fstride': 1, 'sample_type': 'all', 'height_rng': [0, float('inf')],
        'squarify_ratio': 0, 'data_split_type': 'random',
        'seq_type': 'intention', 'min_track_size': 0,
        'max_size_observe': 15, 'seq_overlap_rate': 0.5, 'balance': True,
        'crop_type': 'context', 'crop_mode': 'pad_resize',
        'encoder_input_type': [], 'decoder_input_type': ['bbox'],
        'output_type': ['intent'],
    }
    data_type = {k: data_opts[k] for k in
                 ['encoder_input_type', 'decoder_input_type', 'output_type']}
    jd = JAAD(data_path=args.data_path)
    seq = jd.generate_data_trajectory_sequence('train', **data_opts)
    seq_ds = jd.get_train_val_data(seq, data_type,
                                   data_opts['max_size_observe'],
                                   data_opts['seq_overlap_rate'])

    tfm = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        RandomOcclusion(p=args.p),
    ])
    ds = MyDataSet(images_seq=seq_ds, data_opts=data_opts, transform=tfm)

    torch.manual_seed(42)
    n = 16
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    occluded_count = 0
    
    step_size = max(1, len(ds) // n)
    indices = list(range(0, len(ds), step_size))[:n]

    for list_idx, ax in enumerate(axes.flatten()):
        if list_idx < len(indices):
            idx = indices[list_idx]
            img, lbl = ds[idx]
            ax.imshow(denorm(img))
            ax.add_patch(mpatches.Rectangle(
                (PED_X1, PED_Y1), PED_X2-PED_X1, PED_Y2-PED_Y1,
                linewidth=1.2, edgecolor='orange', facecolor='none',
                linestyle='--'))
            roi = img[:, PED_Y1:PED_Y2, PED_X1:PED_X2]
            gray_frac = ((roi.abs() < 0.02).all(dim=0)).float().mean().item()
            tag = " [OCCLUDED]" if gray_frac > 0.05 else ""
            if tag:
                occluded_count += 1
            ax.set_title(f"sample {idx}, gt={int(lbl)}{tag}", fontsize=9)
            ax.axis('off')

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir,
                            f'train_occlusion_samples_p{int(args.p*100)}.png')
    plt.suptitle(
        f"RandomOcclusion(p={args.p}) — {occluded_count}/{n} samples occluded "
        f"(expected ~{int(args.p*n)}/{n})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    print(f"Saved to {out_path}")
    print(f"Occluded: {occluded_count}/{n} (expected ~{args.p*n:.0f})")

    expected = args.p * n
    if abs(occluded_count - expected) > 4:
        print(f"[WARN] Observed rate {occluded_count}/{n} far from expected "
              f"{expected:.0f}/{n}. Small sample — re-run with another seed "
              f"to confirm before trusting.")
    else:
        print(f"[PASS] Occlusion rate roughly matches p={args.p}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--p', type=float, default=0.5)
    p.add_argument('--out-dir', default='results/verify')
    main(p.parse_args())
