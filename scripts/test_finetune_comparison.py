# scripts/test_finetune_comparison.py
"""
Evaluate every trained checkpoint against every occlusion condition.
Output: results/finetune_comparison.csv
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)

from utils.jaad_data import JAAD
from utils.my_dataset import MyDataSet
from utils.occlusion import occlude_batch
from models.EfficientPIE import EfficientPIE


EVAL_KINDS_DET = ['none', 'top_half', 'bottom_half', 'left_half', 'right_half',
                  'pedestrian_only', 'context_only']
EVAL_KINDS_RANDOM = ['random_10pct', 'random_25pct',
                     'random_50pct', 'random_75pct']


@torch.no_grad()
def evaluate_occluded(model, loader, device, kind, seed=42):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        imgs = occlude_batch(imgs, kind, seed=seed)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.tolist() if hasattr(labels, 'tolist')
                          else list(labels))
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
    return {'kind': kind, 'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': auc,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'n': len(all_labels)}


def build_test_loader(data_path, batch_size):
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
    jd = JAAD(data_path=data_path)
    test_seq = jd.generate_data_trajectory_sequence('test', **data_opts)
    test_seq_ds = jd.get_train_val_data(
        test_seq, data_type, data_opts['max_size_observe'],
        data_opts['seq_overlap_rate'])
    tfm = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = MyDataSet(images_seq=test_seq_ds, data_opts=data_opts, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        pin_memory=True, num_workers=min(os.cpu_count(), 4),
                        collate_fn=ds.collate_fn)
    return loader


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    test_loader = build_test_loader(args.data_path, args.batch_size)

    checkpoints = {}
    for entry in args.checkpoints.split(','):
        name, path = entry.split('=')
        checkpoints[name.strip()] = path.strip()

    rows = []
    for ckpt_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {ckpt_name}: checkpoint not found at {ckpt_path}")
            continue
        print(f"\n=== Loading {ckpt_name} from {ckpt_path} ===")
        model = EfficientPIE(num_classes=2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device),
                              strict=True)

        for kind in EVAL_KINDS_DET:
            r = evaluate_occluded(model, test_loader, device, kind, seed=42)
            r['checkpoint'] = ckpt_name
            print(f"  {kind:20s}  acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
                  f"rec={r['recall']:.4f} auc={r['auc']:.4f}")
            rows.append(r)

        for kind in EVAL_KINDS_RANDOM:
            seed_runs = [evaluate_occluded(model, test_loader, device, kind, s)
                         for s in [42, 43, 44, 45, 46]]
            agg = {'kind': kind, 'n': seed_runs[0]['n'], 'checkpoint': ckpt_name}
            for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                vals = [r[key] for r in seed_runs]
                agg[key] = float(np.mean(vals))
                agg[f'{key}_std'] = float(np.std(vals))
            for key in ['tp', 'fp', 'fn', 'tn']:
                agg[key] = int(np.mean([r[key] for r in seed_runs]))
            print(f"  {kind:20s}  acc={agg['accuracy']:.4f}±{agg['accuracy_std']:.4f} "
                  f"f1={agg['f1']:.4f}±{agg['f1_std']:.4f}")
            rows.append(agg)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")

    pivot = df.pivot(index='kind', columns='checkpoint', values='f1')
    print("\n=== F1 by kind x checkpoint ===")
    print(pivot.round(4))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--checkpoints', type=str, required=True,
                   help='Comma-separated name=path list')
    p.add_argument('--output', type=str,
                   default='results/finetune_comparison.csv')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', type=str, default='cuda:0')
    main(p.parse_args())
