# scripts/run_occlusion_finetune.py
"""
Fine-tune an existing EfficientPIE checkpoint with RandomOcclusion augmentation.

Produces a new checkpoint at the --output path. The ORIGINAL checkpoint at
--init-weights is read-only; it is never overwritten.

Typical usage (smoke test, 1 epoch on small subset):
    python scripts/run_occlusion_finetune.py \
        --data-path /path/to/JAAD \
        --occlusion-prob 0.5 \
        --finetune-epochs 1 \
        --smoke-test \
        --output ./weights_occlusion_finetune/smoke_test_p50.pth

Typical usage (full fine-tune):
    python scripts/run_occlusion_finetune.py \
        --data-path /path/to/JAAD \
        --occlusion-prob 0.5 \
        --finetune-epochs 10 \
        --output ./weights_occlusion_finetune/occlusion_finetuned_p50.pth
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from utils.jaad_data import JAAD
from utils.my_dataset import MyDataSet
from utils.occlusion_augment import RandomOcclusion
from models.EfficientPIE import EfficientPIE
from utils.train_val import evaluate


def build_datasets(args):
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

    train_seq = jd.generate_data_trajectory_sequence('train', **data_opts)
    val_seq = jd.generate_data_trajectory_sequence('val', **data_opts)
    train_seq_ds = jd.get_train_val_data(
        train_seq, data_type, data_opts['max_size_observe'],
        data_opts['seq_overlap_rate'])
    val_seq_ds = jd.get_train_val_data(
        val_seq, data_type, data_opts['max_size_observe'],
        data_opts['seq_overlap_rate'])

    train_tfm = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        RandomOcclusion(p=args.occlusion_prob),  # AFTER Normalize
    ])
    val_tfm = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_ds = MyDataSet(images_seq=train_seq_ds, data_opts=data_opts,
                         transform=train_tfm)
    val_ds = MyDataSet(images_seq=val_seq_ds, data_opts=data_opts,
                       transform=val_tfm)
    return train_ds, val_ds


def main(args):
    # Refuse to overwrite the init checkpoint
    init_abs = os.path.abspath(args.init_weights)
    out_abs = os.path.abspath(args.output)
    assert init_abs != out_abs, (
        f"Refusing to overwrite init checkpoint. "
        f"--output must differ from --init-weights.")
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = build_datasets(args)
    
    # Store custom collate_fn before potentially wrapping dataset in Subset
    train_collate = getattr(train_ds, 'collate_fn', None)
    val_collate = getattr(val_ds, 'collate_fn', None)

    if args.smoke_test:
        n_smoke = min(128, len(train_ds))
        train_ds = Subset(train_ds, list(range(n_smoke)))
        val_ds = Subset(val_ds, list(range(min(64, len(val_ds)))))
        print(f"[SMOKE TEST] train={len(train_ds)}, val={len(val_ds)}")

    nw = min(os.cpu_count(), 4)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True,
        num_workers=nw,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True,
        num_workers=nw,
        collate_fn=val_collate,
    )

    # Model + strict load of init weights
    model = EfficientPIE(num_classes=2).to(device)
    print(f"Loading init weights from {args.init_weights}")
    state = torch.load(args.init_weights, map_location=device)
    # strict=True: refuse to silently skip mismatched keys
    model.load_state_dict(state, strict=True)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()

    # Pre-training sanity check on val set. Catches the case where init weights
    # didn't actually populate the model (strict=True should prevent that, but
    # this catches semantic issues too).
    print("[PRE-TRAIN CHECK] Evaluating loaded checkpoint on val set...")
    pre_loss, pre_acc, pre_prec, pre_rec, pre_f1 = evaluate(model=model, dataloader=val_loader,
                                                            device=device, epoch=-1)
    print(f"[PRE-TRAIN CHECK] Val acc before fine-tuning: {pre_acc:.4f}")
    if pre_acc < 0.5 and not args.smoke_test:
        raise RuntimeError(
            f"Loaded checkpoint has val_acc={pre_acc:.4f} < 0.5. "
            f"Aborting to avoid wasting compute on a bad init.")

    best_val_acc = pre_acc
    for epoch in range(args.finetune_epochs):
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_correct += (logits.argmax(1) == labels).sum().item()
            running_n += imgs.size(0)
        train_loss = running_loss / running_n
        train_acc = running_correct / running_n

        # NaN guard
        if train_loss != train_loss:
            raise RuntimeError(
                f"Training loss became NaN at epoch {epoch+1}. "
                f"Aborting to avoid producing a broken checkpoint.")

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model=model, dataloader=val_loader,
                                                                device=device, epoch=epoch)

        scheduler.step()
        print(f"Epoch {epoch+1}/{args.finetune_epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved best checkpoint to {args.output} (val_acc={val_acc:.4f})")

    final_path = args.output.replace('.pth', '_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nDone. Best val_acc={best_val_acc:.4f}")
    print(f"Best checkpoint: {args.output}")
    print(f"Final checkpoint: {final_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--init-weights', type=str,
                   default='./weights/transfer_noisy_model_JAAD.pth',
                   help='Original checkpoint (read-only)')
    p.add_argument('--output', type=str, required=True,
                   help='Where to save the fine-tuned checkpoint. Must not '
                        'equal --init-weights.')
    p.add_argument('--occlusion-prob', type=float, default=0.5)
    p.add_argument('--finetune-epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=5e-6,
                   help='Lower than original (1e-5) for fine-tuning')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--smoke-test', action='store_true',
                   help='Tiny subset, 1 epoch — verify training loop works')
    args = p.parse_args()
    main(args)
