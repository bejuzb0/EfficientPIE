# scripts/verify_occlusion.py
"""
Loads 3 real JAAD test samples, applies every occlusion kind to each,
and saves a grid image at results/verify/occlusion_grid.png so we can
visually confirm that occlusion regions cover the pedestrian.

Usage: python scripts/verify_occlusion.py --data-path /path/to/JAAD
"""
import argparse, os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

# Make repo root importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from utils.jaad_data import JAAD
from utils.my_dataset import MyDataSet
from utils.occlusion import occlude, OCCLUSION_KINDS, PED_X1, PED_Y1, PED_X2, PED_Y2

def denorm(img_tensor):
    """Invert Normalize(0.5, 0.5): x * 0.5 + 0.5 -> [0, 1]"""
    return (img_tensor.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)

def main(args):
    data_opts = {
        'fstride': 1, 'sample_type': 'all', 'height_rng': [0, float('inf')],
        'squarify_ratio': 0, 'data_split_type': 'random',
        'seq_type': 'intention', 'min_track_size': 0, 'max_size_observe': 15,
        'seq_overlap_rate': 0.5, 'balance': True,
        'crop_type': 'context', 'crop_mode': 'pad_resize',
        'encoder_input_type': [], 'decoder_input_type': ['bbox'],
        'output_type': ['intent'],
    }
    data_type = {k: data_opts[k] for k in
                 ['encoder_input_type', 'decoder_input_type', 'output_type']}

    JAAD_dataset = JAAD(data_path=args.data_path)
    test_seq = JAAD_dataset.generate_data_trajectory_sequence('test', **data_opts)
    test_seq_for_dataset = JAAD_dataset.get_train_val_data(
        test_seq, data_type, data_opts['max_size_observe'],
        data_opts['seq_overlap_rate'])

    tfm = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = MyDataSet(images_seq=test_seq_for_dataset, data_opts=data_opts,
                        transform=tfm)

    # Pick 3 samples with varied labels (try to skip consecutive identical frames)
    chosen = []
    seen_labels = set()
    step_size = max(1, len(dataset) // 20)
    for idx in range(0, len(dataset), step_size):
        img, label = dataset[idx]
        if label not in seen_labels or len(chosen) < 3:
            chosen.append((idx, img, label))
            seen_labels.add(label)
        if len(chosen) == 3:
            break

    # Build grid: rows = samples, cols = occlusion kinds
    n_samples = len(chosen)
    n_kinds = len(OCCLUSION_KINDS)
    fig, axes = plt.subplots(n_samples, n_kinds,
                             figsize=(2.2*n_kinds, 2.4*n_samples))
    if n_samples == 1:
        axes = axes[None, :]
    rng = np.random.RandomState(42)
    for r, (idx, img, label) in enumerate(chosen):
        for c, kind in enumerate(OCCLUSION_KINDS):
            occluded = occlude(img, kind, rng=np.random.RandomState(42))
            ax = axes[r, c]
            ax.imshow(denorm(occluded))
            ax.add_patch(patches.Rectangle(
                (PED_X1, PED_Y1), PED_X2-PED_X1, PED_Y2-PED_Y1,
                linewidth=1.2, edgecolor='orange', facecolor='none',
                linestyle='--'))
            ax.set_title(f"{kind}\nsample={idx} gt={label}", fontsize=7)
            ax.axis('off')

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'occlusion_grid.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved verification grid to {out_path}")

    # Also save one baseline image at full resolution for close inspection
    img0 = chosen[0][1]
    plt.figure(figsize=(5, 5))
    plt.imshow(denorm(img0))
    plt.gca().add_patch(patches.Rectangle(
        (PED_X1, PED_Y1), PED_X2-PED_X1, PED_Y2-PED_Y1,
        linewidth=2, edgecolor='orange', facecolor='none', linestyle='--'))
    plt.title('Baseline sample with nominal pedestrian region')
    plt.axis('off')
    plt.savefig(os.path.join(args.out_dir, 'baseline_example.png'),
                dpi=120, bbox_inches='tight')
    print(f"Saved baseline example to {args.out_dir}/baseline_example.png")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--out-dir', default='results/verify')
    main(p.parse_args())
