
Part 1 — Inference-time occlusion (test-only)
Goal
Run your already-trained model against the JAAD test set with different occlusion conditions applied at inference. Produce a CSV with metrics per condition and verification images showing the occlusions are correctly placed.
Deliverables

utils/occlusion.py — occlusion transforms
scripts/verify_occlusion.py — standalone script producing ~12 annotated sample images for visual verification
test_EfficientPIE_JAAD_occlusion.py — eval loop that sweeps over occlusion conditions
results/occlusion_results.csv — metrics per condition
results/verify/*.png — verification grid images

Detailed steps for the coding agent
Step 1 — Create utils/occlusion.py
Create a new file containing the occlusion transforms. The file must define these constants and functions:
python# utils/occlusion.py
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
Step 2 — Write verification script scripts/verify_occlusion.py
This is the critical step you asked about: verify occlusions are placed correctly before running any real eval.
python# scripts/verify_occlusion.py
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

    # Pick 3 samples with varied labels (try to get at least one of each class)
    chosen = []
    seen_labels = set()
    for idx in range(min(len(dataset), 200)):
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
Verification gate: Run this script first. Open results/verify/occlusion_grid.png and confirm:

Pedestrians are roughly centered inside the orange dashed box in the "none" column.
top_half covers the head/torso of the pedestrian (not the sky above).
bottom_half covers the legs (not the road below).
pedestrian_only shows only the pedestrian visible, rest gray.
context_only shows the scene but not the pedestrian.

If any of these is visually wrong, adjust PED_X1..PED_Y2 before proceeding. If most samples look right but some are off (very tall or short pedestrians), that's the expected approximation error — move on.
Step 3 — Create test_EfficientPIE_JAAD_occlusion.py
Copy test_EfficientPIE_JAAD.py and modify as follows.
Change A — set shuffle=False on the test DataLoader so samples are in the same order across all occlusion conditions. Critical for reproducibility.
Change B — replace the single evaluate() call with a sweep over occlusion conditions. Add this import at the top:
pythonfrom utils.occlusion import occlude_batch, OCCLUSION_KINDS
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)
import pandas as pd
import numpy as np
Add this function after the existing evaluate import:
python@torch.no_grad()
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
Replace the evaluate(...) call in main with:
pythonrows = []
# Baseline: deterministic, single run
for kind in ['none', 'top_half', 'bottom_half', 'left_half', 'right_half',
             'pedestrian_only', 'context_only']:
    print(f"\n=== {kind} ===")
    r = evaluate_occluded(model, test_loader, device, kind, seed=42)
    print(r)
    rows.append(r)

# Random patches: average over 5 seeds to reduce variance
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
Change C — remove the inference-speed measurement block (the dummy_imgs / starter.record() / flops section). You don't need it for this experiment; it just adds noise.
Step 4 — Critical sanity check before running
Before trusting any result, run kind='none' first in isolation and confirm it matches your reported baseline (0.874 accuracy). If none doesn't match, something's off in the transform ordering or shuffle state, and every occluded result is meaningless until fixed.
Step 5 — Run and produce final outputs
bash# 1. Verify
python scripts/verify_occlusion.py --data-path /path/to/JAAD

# 2. Inspect results/verify/occlusion_grid.png visually. STOP and fix if wrong.

# 3. Run full sweep
python test_EfficientPIE_JAAD_occlusion.py \
    --data-path /path/to/JAAD \
    --batch_size 128 \
    --device cuda:0
Expected runtime: ~1 pass through test set per occlusion condition, total ~15-25 passes (11 kinds + random-patch seeds). On Colab Pro GPU at 0.21ms per frame, this is minutes of compute, not hours. Most time is dataloading.
Step 6 — Produce the report figure
After the CSV is produced, write scripts/plot_occlusion.py that makes two plots:

Bar chart: accuracy per occlusion kind, with a horizontal dashed line at the none baseline.
Line plot: accuracy vs. random-patch percentage, with ±1σ error bars from the 5 seeds.


Part 2 — Training with occlusion augmentation
Goal
Fine-tune your existing checkpoint with random occlusion applied to X% of training images, then evaluate the fine-tuned model on both clean and occluded test sets to see whether training-time occlusion helps.
Important choice: fine-tune vs. retrain from scratch
I strongly recommend fine-tuning from your existing checkpoint, not retraining from scratch. Reasons:

Colab Pro time is scarce; your original training was 40 epochs.
You already have the baseline you need to compare against.
Fine-tuning with occlusion for 8-12 epochs is enough to see a real effect on occlusion robustness.
If you retrain from scratch with occlusion, you confound "occlusion helps" with "different random init / different training run."

Fine-tuning answers the actually-interesting question: "If I take this pretrained model and expose it to occlusions, does it become more occlusion-robust without losing clean-image accuracy?"
Deliverables

utils/occlusion_augment.py — training-time occlusion transform (a random variant)
utils/my_dataset_augmented.py — a modified MyDataSet that applies occlusion with probability p
scripts/verify_train_occlusion.py — dumps ~10 augmented training samples for visual inspection
train_EfficientPIE_JAAD_occlusion.py — fine-tuning script
weights/occlusion_finetuned_p{X}.pth — the new checkpoint
results/occlusion_vs_finetune.csv — comparison table

Detailed steps for the coding agent
Step 1 — Create utils/occlusion_augment.py
This differs from the inference-time transform: training occlusion should be random (random kind, random position) rather than a fixed sweep, so the model sees diverse occlusions.
python# utils/occlusion_augment.py
"""
Random occlusion augmentation for training.
Applied with probability `p` to a normalized (C, 300, 300) tensor.
Reuses the nominal pedestrian region from utils.occlusion.
"""
import numpy as np
import torch
from utils.occlusion import PED_X1, PED_Y1, PED_X2, PED_Y2, FILL

# Training-time occlusion kinds. Fewer than inference-time (we don't use
# pedestrian_only / context_only since those are diagnostic, not realistic).
TRAIN_KINDS = ["top_half", "bottom_half", "left_half", "right_half",
               "random_25pct", "random_50pct"]

class RandomOcclusion:
    """
    Callable transform. With probability p, apply a random occlusion to the
    input tensor. Intended to be used AFTER Normalize in a transform pipeline.
    """
    def __init__(self, p=0.5, kinds=None, seed=None):
        self.p = p
        self.kinds = kinds if kinds is not None else TRAIN_KINDS
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def __call__(self, img):
        if self.rng.rand() >= self.p:
            return img
        kind = self.rng.choice(self.kinds)
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
            px = self.rng.randint(PED_X1, PED_X2 - side + 1)
            py = self.rng.randint(PED_Y1, PED_Y2 - side + 1)
            img[:, py:py+side, px:px+side] = FILL
        return img
Step 2 — Integrate occlusion into the training transform
You do not need to modify MyDataSet if it accepts a transform argument that gets applied after loading. Looking at train_EfficientPIE_JAAD.py, the training transform should be extended with RandomOcclusion. Locate the training transform composition — it'll look like:
pythondata_transform = {
    "train": transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(...),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ]),
    "val": transforms.Compose([...])
}
Modify it to:
pythonfrom utils.occlusion_augment import RandomOcclusion

OCCLUSION_PROB = args.occlusion_prob  # new CLI arg, default 0.5

data_transform = {
    "train": transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(...),  # whatever the existing values are
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        RandomOcclusion(p=OCCLUSION_PROB),   # <-- AFTER Normalize
    ]),
    "val": transforms.Compose([...])   # val transform UNCHANGED
}
Two important details:

Place RandomOcclusion AFTER Normalize. The transform writes FILL=0.0, which corresponds to gray (127.5) only in the normalized space. Placing it before Normalize would double-normalize the fill value and shift the gray to a random value.
Validation transform stays unchanged. You want validation accuracy to reflect clean performance, not augmented performance, during training.

Step 3 — Verification script scripts/verify_train_occlusion.py
python# scripts/verify_train_occlusion.py
"""
Saves 8 examples of training images passed through the augmented pipeline
(including RandomOcclusion). Some should be occluded, some not, roughly in
proportion to the probability p.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from utils.jaad_data import JAAD
from utils.my_dataset import MyDataSet
from utils.occlusion_augment import RandomOcclusion

OCCLUSION_PROB = 0.5

def denorm(t):
    return (t.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)

def main(data_path):
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
    jd = JAAD(data_path=data_path)
    seq = jd.generate_data_trajectory_sequence('train', **data_opts)
    seq_ds = jd.get_train_val_data(seq, data_type, 15, 0.5)

    tfm = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        RandomOcclusion(p=OCCLUSION_PROB, seed=None),
    ])
    ds = MyDataSet(images_seq=seq_ds, data_opts=data_opts, transform=tfm)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, ax in enumerate(axes.flatten()):
        img, lbl = ds[i]
        ax.imshow(denorm(img))
        ax.set_title(f"sample {i}, gt={lbl}", fontsize=9)
        ax.axis('off')
    os.makedirs('results/verify', exist_ok=True)
    plt.suptitle(f"Training samples with RandomOcclusion(p={OCCLUSION_PROB})\n"
                 f"Roughly {int(OCCLUSION_PROB*100)}% should show a gray patch",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('results/verify/train_occlusion_samples.png',
                dpi=120, bbox_inches='tight')
    print("Saved results/verify/train_occlusion_samples.png")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    main(p.parse_args().data_path)
Verification gate: Run this. Expect roughly half the 8 sample images to show a gray occluded region, half to be clean. If ALL are occluded or NONE are occluded, something's wrong with the p probability plumbing.
Step 4 — Create train_EfficientPIE_JAAD_occlusion.py
Copy train_EfficientPIE_JAAD.py and modify:

Add CLI args:

python   parser.add_argument('--occlusion_prob', type=float, default=0.5,
                       help='Fraction of training images to occlude')
   parser.add_argument('--init_weights', type=str,
                       default='./weights/transfer_noisy_model_JAAD.pth',
                       help='Checkpoint to fine-tune from')
   parser.add_argument('--finetune_epochs', type=int, default=10)
   parser.add_argument('--output_weights', type=str,
                       default='./weights/occlusion_finetuned.pth')

Load the existing checkpoint before training:

python   model = EfficientPIE(num_classes=2).to(device)
   if args.init_weights and os.path.exists(args.init_weights):
       model.load_state_dict(torch.load(args.init_weights, map_location=device))
       print(f"Loaded init weights from {args.init_weights}")

Integrate RandomOcclusion into the training transform as shown in Step 2.
Reduce learning rate for fine-tuning. The original training used lr=1e-5 with cosine annealing. For fine-tuning from a good starting point, use lr=5e-6 (half the original) for 10 epochs with cosine annealing down to 1e-7. This prevents the model from forgetting its clean-image competence.
Set total epochs to args.finetune_epochs instead of 40-50.
Save to args.output_weights instead of overwriting the original checkpoint.

Step 5 — Evaluation script test_finetuned_comparison.py
After fine-tuning, evaluate BOTH the original checkpoint and the fine-tuned checkpoint on BOTH clean and occluded test sets. This gives you a 2x2 comparison.
python# test_finetuned_comparison.py
"""
Compare original vs. occlusion-finetuned model on clean and occluded test data.
Produces a 2x2 comparison CSV.
"""
# ... same imports and data setup as test_EfficientPIE_JAAD_occlusion.py ...
from utils.occlusion import occlude_batch

CHECKPOINTS = {
    'original':   './weights/transfer_noisy_model_JAAD.pth',
    'finetuned':  './weights/occlusion_finetuned.pth',
}
TEST_CONDITIONS = ['none', 'top_half', 'bottom_half', 'random_25pct',
                   'random_50pct', 'pedestrian_only', 'context_only']

rows = []
for ckpt_name, ckpt_path in CHECKPOINTS.items():
    model = EfficientPIE(num_classes=2).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    for cond in TEST_CONDITIONS:
        r = evaluate_occluded(model, test_loader, device, cond, seed=42)
        r['checkpoint'] = ckpt_name
        print(f"{ckpt_name} x {cond}: acc={r['accuracy']:.4f} f1={r['f1']:.4f}")
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv('results/occlusion_vs_finetune.csv', index=False)

# Pivot for readable comparison
pivot = df.pivot(index='kind', columns='checkpoint', values='accuracy')
pivot['delta'] = pivot['finetuned'] - pivot['original']
print("\nAccuracy comparison (finetuned − original):")
print(pivot.round(4))
Step 6 — Experimental protocol
Don't run one fine-tuning configuration and draw conclusions. Run three:
Configocclusion_probfinetune_epochsPurposep250.2510Light augmentationp500.5010Moderate augmentationp750.7510Heavy augmentation
At p=0.75 the model sees mostly occluded images, which may hurt clean-image accuracy. At p=0.25 it may barely learn occlusion robustness. p=0.50 is the usual sweet spot — but you need the sweep to know.
Total compute: 3 fine-tuning runs × 10 epochs each ≈ 3 × 25% of your original training time. Manageable.
Step 7 — What to report
Key table:
Test conditionOriginal accFinetuned p=0.25p=0.50p=0.75none (clean)0.874???top_half????bottom_half????random_50pct????
The interesting finding(s):

If finetuned beats original on occluded conditions by 3%+ and ties on clean: occlusion training helped.
If finetuned ties original on occluded: occlusion training didn't help. Possible reason: EfficientPIE relies on context cues that weren't occluded, so occluding the pedestrian doesn't push the model to improve.
If finetuned beats original on clean too: suspect data-order effects; rerun with different seeds.
If finetuned loses on clean: classic augmentation-accuracy tradeoff. Report honestly.

Every one of those outcomes is a reportable finding. You can't lose on this experiment as long as you run it honestly.

One final meta-note
Run Part 1 end-to-end first — produce the CSV and report those results — before starting Part 2. Reason: Part 1 tells you whether occlusion is even a problem worth solving for this model. If Part 1 shows EfficientPIE barely drops accuracy under heavy occlusion (because it's relying on context, not pedestrian appearance), then Part 2's premise ("we'll train the model to be more robust to pedestrian occlusion") partly dissolves, and the interesting follow-up is investigating context reliance instead of training with occlusion.
Don't commit to 3 days of fine-tuning compute until Part 1's numbers are in hand.