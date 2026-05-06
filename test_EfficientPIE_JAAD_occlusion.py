"""
@ Description: Adapted test script for occlusion experiment.
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.jaad_data import JAAD
from utils.my_dataset import MyDataSet
from models.EfficientPIE import EfficientPIE

from utils.occlusion import occlude_batch, OCCLUSION_KINDS
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)
import pandas as pd

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
        if hasattr(labels, 'tolist'):
            all_labels.extend(labels.tolist())
        else:
            all_labels.extend(list(labels))
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
            'f1': f1, 'auc': auc, 'tp': int(tp), 'fp': int(fp),
            'fn': int(fn), 'tn': int(tn), 'n': len(all_labels)}

def main(args):
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'random',
                 'seq_type': 'intention',
                 'min_track_size': 0,
                 'max_size_observe': 15,
                 'seq_overlap_rate': 0.5,
                 'balance': True,
                 'crop_type': 'context',
                 'crop_mode': 'pad_resize',
                 'encoder_input_type': [],
                 'decoder_input_type': ['bbox'],
                 'output_type': ['intent']
                 }

    data_type = {'encoder_input_type': data_opts['encoder_input_type'],
                 'decoder_input_type': data_opts['decoder_input_type'],
                 'output_type': data_opts['output_type']}

    JAAD_dataset = JAAD(data_path=args.data_path)
    test_seq = JAAD_dataset.generate_data_trajectory_sequence('test', **data_opts)
    seq_length = data_opts['max_size_observe']
    test_seq_for_dataset = JAAD_dataset.get_train_val_data(test_seq, data_type, seq_length, data_opts['seq_overlap_rate'])

    data_transform = {
        "test": transforms.Compose([transforms.Resize([300, 300]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }

    test_dataset = MyDataSet(images_seq=test_seq_for_dataset, data_opts=data_opts, transform=data_transform['test'])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    
    # Critical Change A - shuffle=False
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, pin_memory=True,
                             num_workers=nw, collate_fn=test_dataset.collate_fn)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = EfficientPIE(num_classes=2).to(device)
    
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(load_weights_dict, strict=False)
            print("using the weight:{}".format(args.weights))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))
    else:
        print("Warning: No weights path provided, testing with random initialization.")
    
    print("test set length:{}".format(test_dataset.__len__()))
    print("Start Testing with Occlusion sweeps!")

    # Change B - replace inference loop with occlusion sweep
    rows = []
    for kind in ['none', 'top_half', 'bottom_half', 'left_half', 'right_half',
                 'pedestrian_only', 'context_only']:
        print(f"\n=== {kind} ===")
        r = evaluate_occluded(model, test_loader, device, kind, seed=42)
        print(r)
        rows.append(r)

    for pct in [10, 25, 50, 75]:
        kind = f"random_{pct}pct"
        print(f"\n=== {kind} (5 seeds) ===")
        seed_runs = [evaluate_occluded(model, test_loader, device, kind, seed=s)
                     for s in [42, 43, 44, 45, 46]]
        agg = {'kind': kind, 'n': seed_runs[0]['n']}
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            vals = [r[key] for r in seed_runs]
            agg[key] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))
        for key in ['tp', 'fp', 'fn', 'tn']:
            agg[key] = int(np.mean([r[key] for r in seed_runs]))
        print(agg)
        rows.append(agg)

    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv('results/occlusion_results.csv', index=False)
    print("\n=== SUMMARY ===")
    print(df[['kind', 'n', 'accuracy', 'f1', 'recall', 'precision']].to_string(index=False))

    print("Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data-path', type=str,
                        default="/Users/akvma/Developer/cvproject/EfficientPIE/JAAD")
    parser.add_argument('--weights', type=str, default="./weights/transfer_noisy_model_JAAD.pth",
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
