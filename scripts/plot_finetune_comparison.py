# scripts/plot_finetune_comparison.py
"""
Produces two figures from results/finetune_comparison.csv:
  1. Heatmap of F1 by (kind x checkpoint).
  2. Grouped bar chart of F1 per kind, grouped by checkpoint.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    df = pd.read_csv(args.input)
    pivot_f1 = df.pivot(index='kind', columns='checkpoint', values='f1')
    order = ['none', 'top_half', 'bottom_half', 'left_half', 'right_half',
             'random_10pct', 'random_25pct', 'random_50pct', 'random_75pct',
             'pedestrian_only', 'context_only']
    pivot_f1 = pivot_f1.reindex([k for k in order if k in pivot_f1.index])

    fig, ax = plt.subplots(figsize=(2 + 1.2*pivot_f1.shape[1],
                                    0.55*pivot_f1.shape[0] + 2))
    im = ax.imshow(pivot_f1.values, cmap='RdYlGn', vmin=0, vmax=0.7,
                   aspect='auto')
    ax.set_xticks(range(pivot_f1.shape[1]))
    ax.set_xticklabels(pivot_f1.columns, rotation=20, ha='right')
    ax.set_yticks(range(pivot_f1.shape[0]))
    ax.set_yticklabels(pivot_f1.index)
    for i in range(pivot_f1.shape[0]):
        for j in range(pivot_f1.shape[1]):
            v = pivot_f1.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha='center', va='center',
                        color='black' if v > 0.35 else 'white', fontsize=9)
    plt.colorbar(im, ax=ax, label='F1')
    ax.set_title('F1 by occlusion condition x checkpoint')
    os.makedirs(args.out_dir, exist_ok=True)
    p1 = os.path.join(args.out_dir, 'finetune_heatmap.png')
    plt.tight_layout()
    plt.savefig(p1, dpi=140, bbox_inches='tight')
    print(f"Saved {p1}")
    plt.close()

    kinds = pivot_f1.index.tolist()
    ckpts = pivot_f1.columns.tolist()
    x = np.arange(len(kinds))
    width = 0.8 / len(ckpts)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, c in enumerate(ckpts):
        ax.bar(x + i*width - 0.4 + width/2, pivot_f1[c].values, width,
               label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, rotation=30, ha='right')
    ax.set_ylabel('F1')
    ax.set_title('F1 by occlusion condition, grouped by checkpoint')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    p2 = os.path.join(args.out_dir, 'finetune_bars.png')
    plt.tight_layout()
    plt.savefig(p2, dpi=140, bbox_inches='tight')
    print(f"Saved {p2}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='results/finetune_comparison.csv')
    p.add_argument('--out-dir', default='results/plots')
    main(p.parse_args())
