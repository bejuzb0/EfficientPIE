# Part 2 — Occlusion-Aware Fine-Tuning of EfficientPIE

Implementation plan for training a new EfficientPIE variant with occlusion augmentation and comparing it against the original checkpoint on both clean and occluded test sets.

Every step has a verification gate. Do not proceed past a gate that fails.

---

## Design Principles

1. **New scripts only.** No edits to `train_EfficientPIE_JAAD.py`, `test_EfficientPIE_JAAD.py`, or `utils/my_dataset.py`. Every file in this plan is new.
2. **Separate weights directory.** All new checkpoints go to `weights_occlusion_finetune/`, not the existing `weights/` directory. The original checkpoint is never at risk.
3. **Reuse, don't modify.** Import `utils/occlusion.py` (from Part 1) rather than duplicating its constants or logic.
4. **Fine-tune, don't retrain.** Start from the existing checkpoint, train for 10 epochs with a lower learning rate. Retraining from scratch would confound the comparison with random-init variation.
5. **Verify before committing compute.** Every stage has a verification gate. Do not run the full fine-tune without passing smoke tests first.

---

## File Structure

All new files; nothing existing is modified.

```
utils/
  occlusion_augment.py              NEW — random occlusion transform for training
scripts/
  verify_occlusion_augment_unit.py  NEW — Gate 1: unit tests for the transform
  verify_transform_order.py         NEW — Gate 2: Normalize-then-Occlusion check
  verify_train_occlusion.py         NEW — Gate 3: visual check on real JAAD data
  run_occlusion_finetune.py         NEW — fine-tuning script (has --smoke-test)
  verify_finetune_checkpoint.py     NEW — Gate 5: per-checkpoint sanity check
  test_finetune_comparison.py       NEW — full eval comparison
  plot_finetune_comparison.py       NEW — produce report figures
weights_occlusion_finetune/         NEW directory
  occlusion_finetuned_p25.pth
  occlusion_finetuned_p50.pth
  occlusion_finetuned_p75.pth
  smoke_test_p50.pth                (temporary)
results/
  finetune_comparison.csv           NEW
  verify/
    train_occlusion_samples_p50.png NEW
    transform_order_check.png       NEW
    finetune_ckpt_check_*.txt       NEW
  plots/
    finetune_heatmap.png            NEW
    finetune_bars.png               NEW
```

---

## Step 1 — Create `utils/occlusion_augment.py`

**Purpose:** random occlusion transform applied during training. Distinct from `utils/occlusion.py` (deterministic for inference) because training augmentation should be random.

```python
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
```

**Important:** uses `torch.rand` / `torch.randint`, not `numpy.random`. NumPy's global RNG state doesn't interact well with PyTorch DataLoader workers — you can get the same occlusion across many samples if `num_workers > 0`. Torch handles this correctly.

### Gate 1 — Unit-test the transform module

Quick unit test with no JAAD data required. Catches stupid bugs in the transform module before wiring it into training.

Create `scripts/verify_occlusion_augment_unit.py`:

```python
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
```

**Run:**

```bash
python scripts/verify_occlusion_augment_unit.py
```

**Pass criterion:** all 6 `[PASS]` lines, final "All unit tests passed."

---

## Step 2 — Verify transform composition order

**Purpose:** confirm that when `RandomOcclusion` is placed after `Normalize` in a `transforms.Compose([...])`, the FILL=0.0 actually renders as gray (127 in original image space). If someone accidentally puts `RandomOcclusion` before `Normalize`, the fill becomes black (because `Normalize` maps 0.0 to −1.0 in its output space, which denormalizes to 0.0 — pure black).

Create `scripts/verify_transform_order.py`:

```python
# scripts/verify_transform_order.py
"""
Visually verify that RandomOcclusion applied AFTER Normalize produces
a gray patch (not black), confirming correct transform composition.

Saves results/verify/transform_order_check.png with two columns:
  Left: CORRECT order (Normalize then RandomOcclusion) -> gray patch
  Right: WRONG order (RandomOcclusion then Normalize) -> dark patch
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from utils.occlusion_augment import RandomOcclusion

torch.manual_seed(7)

def denorm(t):
    return (t.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)

# Make a synthetic input: noise with a recognizable colored region in the middle
rng = np.random.RandomState(0)
arr = (rng.rand(300, 300, 3) * 255).astype(np.uint8)
arr[100:200, 100:200] = [200, 120, 60]
img_pil = Image.fromarray(arr)

# Correct order: Normalize first, then RandomOcclusion (writes FILL=0 in normalized space)
tfm_correct = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
    RandomOcclusion(p=1.0),   # always apply for this test
])

# Wrong order: RandomOcclusion before Normalize -> fill=0 becomes -1 after Normalize,
# which is pure black when denormalized
tfm_wrong = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    RandomOcclusion(p=1.0),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

torch.manual_seed(1); out_correct = tfm_correct(img_pil)
torch.manual_seed(1); out_wrong = tfm_wrong(img_pil)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(denorm(out_correct))
axes[0].set_title("CORRECT order:\nNormalize -> RandomOcclusion\n(patch should be GRAY)")
axes[0].axis('off')
axes[1].imshow(denorm(out_wrong))
axes[1].set_title("WRONG order:\nRandomOcclusion -> Normalize\n(patch would appear BLACK)")
axes[1].axis('off')

os.makedirs('results/verify', exist_ok=True)
plt.tight_layout()
plt.savefig('results/verify/transform_order_check.png', dpi=120, bbox_inches='tight')
print("Saved results/verify/transform_order_check.png")

# Programmatic verification
dn_correct = denorm(out_correct)
dn_wrong = denorm(out_wrong)

# Probe the center of the pedestrian region
probe_c = dn_correct[130:170, 130:170].mean()
probe_w = dn_wrong[130:170, 130:170].mean()

print(f"CORRECT composition, denormalized patch mean: {probe_c:.3f} (expect ~0.5)")
print(f"WRONG   composition, denormalized patch mean: {probe_w:.3f} (expect ~0.0)")

ok = 0.4 < probe_c < 0.6
if ok:
    print("[PASS] CORRECT composition produces gray patch")
else:
    print("[FAIL] CORRECT composition does not produce gray — check FILL value")
    sys.exit(1)

if probe_w < 0.1:
    print("[NOTE] WRONG composition produces black patch as expected "
          "(this confirms the CORRECT order matters)")
```

**Run:**

```bash
python scripts/verify_transform_order.py
```

**Pass criterion:** `[PASS] CORRECT composition produces gray patch`. Left panel of the saved PNG visibly shows a gray patch. If `[FAIL]`, you have a FILL value or transform-composition bug; fix before proceeding.

---

## Step 3 — Verify augmentation on real JAAD data

**Purpose:** confirm `RandomOcclusion` applies at roughly the expected rate and puts patches in the right places on actual training data.

Create `scripts/verify_train_occlusion.py`:

```python
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
    for i, ax in enumerate(axes.flatten()):
        img, lbl = ds[i]
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
        ax.set_title(f"sample {i}, gt={int(lbl)}{tag}", fontsize=9)
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
```

**Run:**

```bash
python scripts/verify_train_occlusion.py --data-path /path/to/JAAD --p 0.5
```

**Pass criterion:** `[PASS]` printed, and visual inspection of the saved image confirms 6–10 of 16 samples have a gray patch inside the orange dashed box. If 0 or all 16 are occluded, the `p` plumbing is broken.

---

## Step 4 — Create `scripts/run_occlusion_finetune.py`

**Purpose:** fine-tune an existing checkpoint with occlusion augmentation. Has a `--smoke-test` flag for end-to-end verification before committing to the full run. Includes built-in safety checks.

```python
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

    if args.smoke_test:
        n_smoke = min(128, len(train_ds))
        train_ds = Subset(train_ds, list(range(n_smoke)))
        val_ds = Subset(val_ds, list(range(min(64, len(val_ds)))))
        print(f"[SMOKE TEST] train={len(train_ds)}, val={len(val_ds)}")

    nw = min(os.cpu_count(), 4)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True,
        num_workers=nw,
        collate_fn=getattr(train_ds, 'collate_fn', None) if not args.smoke_test else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True,
        num_workers=nw,
        collate_fn=getattr(val_ds, 'collate_fn', None) if not args.smoke_test else None,
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
    pre_metrics = evaluate(model=model, dataloader=val_loader,
                           device=device, epoch=-1)
    pre_acc = (pre_metrics['accuracy']
               if isinstance(pre_metrics, dict) else pre_metrics)
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

        val_metrics = evaluate(model=model, dataloader=val_loader,
                               device=device, epoch=epoch)
        val_acc = (val_metrics['accuracy']
                   if isinstance(val_metrics, dict) else val_metrics)

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
```

### Built-in safety features

- Asserts `--output != --init-weights` (refuses to overwrite original).
- `--output` is required (no default).
- `load_state_dict(..., strict=True)`: refuses to silently skip mismatched keys.
- **Pre-training val check**: evaluates the loaded init weights on the val set before any training happens. If val_acc < 0.5, aborts. This catches bad-load scenarios.
- **NaN loss guard**: aborts if training loss becomes NaN.
- Saves both best-by-val and final checkpoints.

### Gate 4 — Smoke test the training loop

Run a smoke test before any real training:

```bash
python scripts/run_occlusion_finetune.py \
    --data-path /path/to/JAAD \
    --occlusion-prob 0.5 \
    --finetune-epochs 1 \
    --smoke-test \
    --output ./weights_occlusion_finetune/smoke_test_p50.pth
```

Should finish in 2–3 minutes on GPU. **Pass criteria:**

- `[PRE-TRAIN CHECK] Val acc before fine-tuning:` prints roughly 0.87 (close to your Part 1 baseline).
- Training loss is finite (no NaN).
- Val acc at end of epoch 1 on 128 samples is reasonable (not 0.5 or lower — don't expect improvement on this tiny subset, but it shouldn't crater).
- Checkpoint file appears at the output path.

If any of these fail, do not run full fine-tuning. Investigate.

Clean up:

```bash
rm -f ./weights_occlusion_finetune/smoke_test_p50.pth \
      ./weights_occlusion_finetune/smoke_test_p50_final.pth
```

---

## Step 5 — Create `scripts/verify_finetune_checkpoint.py`

**Purpose:** quickly sanity-check each finetuned checkpoint before moving on to the next fine-tune or the full comparison. Catches broken checkpoints early — for example if training diverged late but the "best" checkpoint is still one you wouldn't want to use.

```python
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
    n = min(128, len(full_ds))
    ds = Subset(full_ds, list(range(n)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        pin_memory=True, num_workers=2)

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
```

### Gate 5 — Verify each finetuned checkpoint

After each fine-tune run (p=0.25, p=0.50, p=0.75), run this before starting the next one:

```bash
python scripts/verify_finetune_checkpoint.py \
    --data-path /path/to/JAAD \
    --checkpoint ./weights_occlusion_finetune/occlusion_finetuned_p50.pth
```

**Pass criterion:** clean-test accuracy in [0.75, 0.95], script exits 0. If it exits 1, the checkpoint is broken — investigate before launching the next fine-tune. ~1 minute per check.

---

## Step 6 — Create `scripts/test_finetune_comparison.py`

**Purpose:** evaluate the original checkpoint AND all fine-tuned checkpoints against BOTH clean and occluded test sets. Produces one CSV with everything.

```python
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
```

### Gate 6 — Baseline reproducibility check

Before interpreting any p=X results, verify that this comparison script reproduces your Part 1 baseline on the original checkpoint. After the comparison runs, open `results/finetune_comparison.csv`:

- Row `checkpoint=original, kind=none`: accuracy should match your Part 1 CSV's `kind=none` row **within 0.001**.
- Row `checkpoint=original, kind=bottom_half`: should match Part 1's `bottom_half` row.

If these don't match, the test loader differs from Part 1's (shuffle, batch size, balance setting, data_split_type). Reconcile before trusting the comparison.

---

## Step 7 — Create `scripts/plot_finetune_comparison.py`

**Purpose:** turn the CSV into two report-ready figures.

```python
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
```

---

## Full Run Protocol with Gates

Execute in this order. **Do not skip gates.** Each gate has an explicit pass criterion; stop and debug if any fails.

```bash
mkdir -p weights_occlusion_finetune results/verify results/plots
```

### Gate 1 — Unit test the augmentation module

```bash
python scripts/verify_occlusion_augment_unit.py
```

Pass: all `[PASS]` lines, final "All unit tests passed."

### Gate 2 — Transform composition order

```bash
python scripts/verify_transform_order.py
```

Pass: `[PASS] CORRECT composition produces gray patch`. Left panel visibly gray.

### Gate 3 — Real-data augmentation check

```bash
python scripts/verify_train_occlusion.py --data-path /path/to/JAAD --p 0.5
```

Pass: `[PASS]` printed. Visual: 6–10 of 16 samples occluded inside the orange box.

### Gate 4 — Training loop smoke test

```bash
python scripts/run_occlusion_finetune.py \
    --data-path /path/to/JAAD \
    --occlusion-prob 0.5 \
    --finetune-epochs 1 \
    --smoke-test \
    --output ./weights_occlusion_finetune/smoke_test_p50.pth
```

Pass: pre-train val_acc ~0.87 printed, no NaN, checkpoint file written.

Clean up:

```bash
rm -f ./weights_occlusion_finetune/smoke_test_p50.pth \
      ./weights_occlusion_finetune/smoke_test_p50_final.pth
```

### Full fine-tuning + per-checkpoint verification

Do one fine-tune + one verify before starting the next.

```bash
# p=0.50 first (most likely to be the best config)
python scripts/run_occlusion_finetune.py \
    --data-path /path/to/JAAD --occlusion-prob 0.50 --finetune-epochs 10 \
    --output ./weights_occlusion_finetune/occlusion_finetuned_p50.pth

# Gate 5a
python scripts/verify_finetune_checkpoint.py \
    --data-path /path/to/JAAD \
    --checkpoint ./weights_occlusion_finetune/occlusion_finetuned_p50.pth
# Pass: clean-test acc in [0.75, 0.95], script exits 0

# p=0.25
python scripts/run_occlusion_finetune.py \
    --data-path /path/to/JAAD --occlusion-prob 0.25 --finetune-epochs 10 \
    --output ./weights_occlusion_finetune/occlusion_finetuned_p25.pth

python scripts/verify_finetune_checkpoint.py \
    --data-path /path/to/JAAD \
    --checkpoint ./weights_occlusion_finetune/occlusion_finetuned_p25.pth

# p=0.75
python scripts/run_occlusion_finetune.py \
    --data-path /path/to/JAAD --occlusion-prob 0.75 --finetune-epochs 10 \
    --output ./weights_occlusion_finetune/occlusion_finetuned_p75.pth

python scripts/verify_finetune_checkpoint.py \
    --data-path /path/to/JAAD \
    --checkpoint ./weights_occlusion_finetune/occlusion_finetuned_p75.pth
```

### Gate 6 — Full comparison + baseline reproducibility

```bash
python scripts/test_finetune_comparison.py \
    --data-path /path/to/JAAD \
    --checkpoints "original=./weights/transfer_noisy_model_JAAD.pth,p25=./weights_occlusion_finetune/occlusion_finetuned_p25.pth,p50=./weights_occlusion_finetune/occlusion_finetuned_p50.pth,p75=./weights_occlusion_finetune/occlusion_finetuned_p75.pth" \
    --output results/finetune_comparison.csv
```

Pass: original × none accuracy matches Part 1 CSV within 0.001.

### Plots

```bash
python scripts/plot_finetune_comparison.py \
    --input results/finetune_comparison.csv \
    --out-dir results/plots
```

---

## Gate Summary

| Gate | Script | Verifies | Pass criterion |
|---|---|---|---|
| 1 | `verify_occlusion_augment_unit.py` | `RandomOcclusion` logic, FILL value, geometry, kind variety | All unit tests pass |
| 2 | `verify_transform_order.py` | Normalize-then-Occlusion produces gray (not black) | Patch mean ~0.5 after denorm |
| 3 | `verify_train_occlusion.py` | Transform fires at rate `p` on real JAAD data | 6–10 of 16 samples occluded |
| 4 | `run_occlusion_finetune.py --smoke-test` | Training loop runs end-to-end | Pre-train val_acc ~0.87, no NaN |
| 5 | `verify_finetune_checkpoint.py` (×3) | Each finetuned ckpt produces sensible clean-test accuracy | Clean-test acc in [0.75, 0.95] |
| 6 | `test_finetune_comparison.py` | Baseline reproduces Part 1 numbers | Original × none matches Part 1 within 0.001 |

---

## What to Look For in the Results

Key cells in the final heatmap:

1. **`none` row (clean test).** Did fine-tuning hurt clean performance? If p=0.50 keeps clean F1 within 0.02 of the original, the augmentation is "safe." If it drops by 0.05+, you're trading clean accuracy for robustness.

2. **`pedestrian_only` row.** Part 1 baseline was F1=0.064, a near-total failure. If fine-tuning moves it to 0.3+, that's a real effect — the model started reading the pedestrian. If it stays near 0.06, occlusion training did not solve scene-shortcutting; it just taught the model to handle gray patches without changing where it looks.

3. **`bottom_half` and `random_50pct` rows.** Realistic occlusions. Improvement here is the headline "safety" result.

4. **Best p-value.** If p=0.75 is strictly better, the report conclusion is "aggressive occlusion augmentation is beneficial"; if p=0.25 is, "even light augmentation helps and aggressive hurts clean accuracy."

All four outcomes are publishable findings.

---

## Compute Budget Warning

Ten epochs × three configs × ~40k training samples is roughly 75% of one full training run. If Colab Pro budget is tight:

- Do **p=0.50 first** (most likely the best config).
- Only run p=0.25 and p=0.75 if you have time for a full sweep.
- A single fine-tune + comparison is enough for a report; the sweep is a robustness check.