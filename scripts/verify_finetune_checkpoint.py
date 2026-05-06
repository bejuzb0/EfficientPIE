# scripts/verify_finetune_checkpoint.py
"""
Fast sanity check on a finetuned checkpoint.
Loads the checkpoint, runs ~128 test samples through 3 occlusion conditions,
and reports whether clean-test accuracy is in a sensible range.

Usage:
    python scripts/verify_finetune_checkpoint.py \
        --data-path /path/to/JAAD \
        --checkpoint ./weights_occlusion_finetune/occlusion_finetuned_p50.pth
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

from utils.jaad_data import JAAD
from utils.my_dataset import MyDataSet
from utils.occlusion import occlude_batch
from models.EfficientPIE import EfficientPIE


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

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
    test_seq = jd.generate_data_trajectory_sequence('test', **data_opts)
    test_seq_ds = jd.get_train_val_data(
        test_seq, data_type, data_opts['max_size_observe'],
        data_opts['seq_overlap_rate'])
    tfm = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    full_ds = MyDataSet(images_seq=test_seq_ds, data_opts=data_opts, transform=tfm)
    
    # Step out to avoid all identical frames in subset
    n = min(128, len(full_ds))
    step = max(1, len(full_ds) // n)
    ds = Subset(full_ds, list(range(0, len(full_ds), step))[:n])
    
    # Extract the custom collate function from the underlying dataset before it was wrapped
    original_collate = getattr(full_ds, 'collate_fn', None)
    
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        pin_memory=True, num_workers=2, collate_fn=original_collate)

    model = EfficientPIE(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device),
                          strict=True)
    model.eval()
    print(f"Loaded {args.checkpoint}")

    out_lines = [f"Checkpoint: {args.checkpoint}", f"Eval subset size: {n}"]
    expected_range = {
        'none':            (0.75, 0.95),
        'bottom_half':     (0.60, 0.92),
        'pedestrian_only': (0.30, 0.90),
    }
    results = {}
    with torch.no_grad():
        for kind in ['none', 'bottom_half', 'pedestrian_only']:
            all_preds, all_labels = [], []
            for imgs, labels in loader:
                imgs = imgs.to(device)
                imgs = occlude_batch(imgs, kind, seed=42)
                logits = model(imgs)
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist() if hasattr(labels, 'tolist')
                                  else list(labels))
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            lo, hi = expected_range[kind]
            status = "[PASS]" if lo <= acc <= hi else "[WARN]"
            line = f"  {kind:20s}  acc={acc:.4f}  f1={f1:.4f}  {status} (expect [{lo},{hi}])"
            print(line)
            out_lines.append(line)
            results[kind] = (acc, f1)

    # The hard gate: clean-test (kind=none) must be in its expected range
    none_acc, _ = results['none']
    ok = expected_range['none'][0] <= none_acc <= expected_range['none'][1]

    os.makedirs('results/verify', exist_ok=True)
    report_name = ('finetune_ckpt_check_' +
                   os.path.splitext(os.path.basename(args.checkpoint))[0] +
                   '.txt')
    report_path = os.path.join('results/verify', report_name)
    if not ok:
        out_lines.append(f"[FAIL] Clean-test accuracy {none_acc:.4f} outside "
                         f"expected range. Checkpoint is likely broken.")
        with open(report_path, 'w') as f:
            f.write("\n".join(out_lines))
        print(out_lines[-1])
        print(f"Report: {report_path}")
        sys.exit(1)
    else:
        out_lines.append("[PASS] Checkpoint looks sensible; proceed to full eval.")
        with open(report_path, 'w') as f:
            f.write("\n".join(out_lines))
        print(out_lines[-1])
        print(f"Report: {report_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', type=str, default='cuda:0')
    main(p.parse_args())
