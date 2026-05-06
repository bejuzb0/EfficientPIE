"""
Usage:
    python run_inference_paper_protocol.py \
        --data-path /path/to/JAAD \
        --weights weights/your_model.pth \
        --output-csv results_paper_protocol.csv \
        --batch-size 32 \
        --device cuda:0
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from utils.jaad_data import JAAD
from models.EfficientPIE import EfficientPIE


# -----------------------------------------------------------------------------
# Small unwrapping helpers
# -----------------------------------------------------------------------------
def _to_scalar(v):
    """Unwrap shapes like [[0.9]] or np.array([1]) down to a plain float/int."""
    arr = np.asarray(v).flatten()
    if arr.size == 0:
        return np.nan
    return arr[0]


def _deep_unwrap(v):
    """Repeatedly unwrap single-element lists/arrays to get the scalar value."""
    while isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0 and not isinstance(v[0], (str, bytes)):
        if len(v) == 1:
            v = v[0]
        else:
            break
    return v


# -----------------------------------------------------------------------------
# Manual overlap-clipping + balancing
# -----------------------------------------------------------------------------
def build_samples_with_clipping(seq_dict, seq_length=15, overlap_rate=0.5,
                                balance=True, rng_seed=42):
    """
    Replicates the JAAD API's internal `get_train_val_data` pipeline:
        1. slide a window of size `seq_length` over each pedestrian's track
           with stride = seq_length * (1 - overlap_rate)
        2. keep the LAST frame of each window as the sole-observation sample
        3. if balance=True, downsample the majority class to match the minority

    Returns a flat list of sample dicts with image_path, bbox, label, etc.
    """
    images = seq_dict['image']
    bboxes = seq_dict['bbox']
    intents = seq_dict['intent']
    pids = seq_dict.get('pid', [None] * len(images))
    occlusions = seq_dict.get('occlusion', None)

    stride = max(int(round(seq_length * (1 - overlap_rate))), 1)
    print(f"[clip] seq_length={seq_length}, overlap_rate={overlap_rate}, "
          f"stride={stride}")

    raw_windows = []
    skipped_short = 0
    for i in range(len(images)):
        track = images[i]
        track_len = len(track)
        if track_len < seq_length:
            skipped_short += 1
            continue

        for start in range(0, track_len - seq_length + 1, stride):
            end = start + seq_length - 1   # index of the last frame in the window

            img_path = track[end]
            if img_path is None:
                continue

            bbox = bboxes[i][end]
            x1, y1, x2, y2 = [float(v) for v in bbox]

            pid_raw = pids[i][end] if pids[i] is not None else "unknown"
            pid = _deep_unwrap(pid_raw)
            pid = str(pid) if pid is not None else "unknown"

            intent_val = intents[i][end]
            label = int(np.round(float(_to_scalar(intent_val))))

            occ = -1
            if occlusions is not None:
                try:
                    occ = int(_to_scalar(occlusions[i][end]))
                except (ValueError, TypeError):
                    occ = -1

            video = next(
                (p for p in Path(img_path).parts if p.startswith("video_")),
                "unknown"
            )

            raw_windows.append({
                "image_path": img_path,
                "video": video,
                "pid": pid,
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "occlusion": occ,
            })

    print(f"[clip] Generated {len(raw_windows)} windows "
          f"({skipped_short} tracks too short to window)")

    if balance:
        pos = [w for w in raw_windows if w["label"] == 1]
        neg = [w for w in raw_windows if w["label"] == 0]
        n = min(len(pos), len(neg))
        rng = np.random.RandomState(rng_seed)
        if len(pos) > n:
            pos = [pos[i] for i in rng.choice(len(pos), n, replace=False)]
        if len(neg) > n:
            neg = [neg[i] for i in rng.choice(len(neg), n, replace=False)]
        balanced = pos + neg
        print(f"[balance] kept {len(balanced)} samples "
              f"({len(pos)} crossing, {len(neg)} not-crossing)")
        return balanced

    return raw_windows


# -----------------------------------------------------------------------------
# Dataset + collate
# -----------------------------------------------------------------------------
class FlatJAADDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x1, y1, x2, y2 = s["bbox"]
        with Image.open(s["image_path"]).convert("RGB") as full_img:
            img_w, img_h = full_img.size
            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
            side = max(bw, bh) * 2.0
            cx1 = max(int(round(cx - side / 2.0)), 0)
            cy1 = max(int(round(cy - side / 2.0)), 0)
            cx2 = min(int(round(cx + side / 2.0)), img_w)
            cy2 = min(int(round(cy + side / 2.0)), img_h)
            cropped = full_img.crop((cx1, cy1, cx2, cy2))
        image = self.transform(cropped)

        meta = {
            **s,
            "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
            "bbox_w": bw, "bbox_h": bh,
            "bbox_area": bw * bh,
            "bbox_aspect": bh / max(bw, 1e-6),
            "bbox_cx_rel": cx / max(img_w, 1),
            "bbox_cy_rel": cy / max(img_h, 1),
            "img_w": img_w, "img_h": img_h,
        }
        return image, s["label"], meta


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    metas = [b[2] for b in batch]
    return imgs, labels, metas


# -----------------------------------------------------------------------------
# Robust CSV writer that actually tries to write before committing
# -----------------------------------------------------------------------------
def _safe_write_csv(rows, fieldnames, requested_path):
    """
    Write rows to CSV with real write-testing. Drive shortcut folders lie
    to os.access(), so we have to actually attempt the write. We try the
    requested path first, then a Drive MyDrive fallback, then /content/,
    then /tmp/.
    """
    fname = os.path.basename(requested_path) or "results.csv"
    candidates = [
        requested_path,
        f"/content/drive/MyDrive/{fname}",
        f"/content/{fname}",
        f"/tmp/{fname}",
    ]
    last_err = None
    for path in candidates:
        try:
            parent = os.path.dirname(path) or "."
            os.makedirs(parent, exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            if path != requested_path:
                print(f"[info] requested path was not writable; "
                      f"saved to {path} instead")
            return path
        except (OSError, PermissionError) as e:
            last_err = e
            print(f"[skip] {path} not writable ({e.__class__.__name__})")
            continue
    raise RuntimeError(
        f"All write candidates failed. Last error: {last_err}"
    )


# -----------------------------------------------------------------------------
# Writable-path fallback for Colab / Drive  (kept for backwards compatibility)
# -----------------------------------------------------------------------------
def _writable_path(requested):
    requested = os.path.abspath(requested)
    parent = os.path.dirname(requested) or "."
    if os.access(parent, os.W_OK):
        return requested
    for fallback_dir in ("/content", "/tmp"):
        if os.path.isdir(fallback_dir) and os.access(fallback_dir, os.W_OK):
            new = os.path.join(fallback_dir, os.path.basename(requested))
            print(f"[info] {parent} not writable; redirecting output to {new}")
            return new
    raise PermissionError(f"No writable location found for {requested}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_opts = {
        'fstride': 1,
        'sample_type': 'all',
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'data_split_type': 'default',
        'seq_type': 'crossing',          # paper protocol
        'min_track_size': 0,
        'max_size_observe': 15,
        'seq_overlap_rate': 0.5,
        'balance': True,                 # paper protocol (applied manually below)
        'crop_type': 'context',
        'crop_mode': 'pad_resize',
        'encoder_input_type': [],        # keep empty; we don't call get_train_val_data
        'decoder_input_type': ['bbox'],
        'output_type': ['intent']
    }

    # Pull the raw per-pedestrian tracks. Do NOT call get_train_val_data --
    # its internal reshaping silently drops the 'image' key we need.
    jaad = JAAD(data_path=args.data_path)
    test_seq_raw = jaad.generate_data_trajectory_sequence('test', **data_opts)
    print(f"Raw test_seq keys: {list(test_seq_raw.keys())}")
    print(f"Number of pedestrian tracks: {len(test_seq_raw['image'])}")

    # Manual overlap-clipping + balancing, preserving all metadata
    samples = build_samples_with_clipping(
        test_seq_raw,
        seq_length=data_opts['max_size_observe'],
        overlap_rate=data_opts['seq_overlap_rate'],
        balance=data_opts['balance'],
    )
    if len(samples) == 0:
        raise RuntimeError("No samples produced after clipping.")

    # Standard transform (mirrors test_EfficientPIE_JAAD.py)
    tfm = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = FlatJAADDataset(samples, transform=tfm)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True, collate_fn=_collate,
    )

    model = EfficientPIE(num_classes=2).to(device)
    state = torch.load(args.weights, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    model.eval()
    print(f"Loaded weights from {args.weights}")

    fieldnames = [
        "image_path", "video", "pid",
        "label", "prediction", "correct",
        "prob_cross", "prob_notcross", "confidence", "margin",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "bbox_w", "bbox_h", "bbox_area", "bbox_aspect",
        "bbox_cx_rel", "bbox_cy_rel",
        "img_w", "img_h", "occlusion",
    ]
    rows = []

    with torch.no_grad():
        for imgs, labels, metas in tqdm(loader, desc="Inference"):
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for i, m in enumerate(metas):
                p_cross = float(probs[i, 1])
                p_nc = float(probs[i, 0])
                pred = int(preds[i])
                label = int(m["label"])
                rows.append({
                    "image_path": m["image_path"],
                    "video": m["video"],
                    "pid": m["pid"],
                    "label": label,
                    "prediction": pred,
                    "correct": int(pred == label),
                    "prob_cross": p_cross,
                    "prob_notcross": p_nc,
                    "confidence": float(max(p_cross, p_nc)),
                    "margin": float(abs(p_cross - p_nc)),
                    "bbox_x1": m["bbox_x1"], "bbox_y1": m["bbox_y1"],
                    "bbox_x2": m["bbox_x2"], "bbox_y2": m["bbox_y2"],
                    "bbox_w": m["bbox_w"], "bbox_h": m["bbox_h"],
                    "bbox_area": m["bbox_area"],
                    "bbox_aspect": m["bbox_aspect"],
                    "bbox_cx_rel": m["bbox_cx_rel"],
                    "bbox_cy_rel": m["bbox_cy_rel"],
                    "img_w": m["img_w"], "img_h": m["img_h"],
                    "occlusion": m["occlusion"],
                })

    output_path = _safe_write_csv(rows, fieldnames, args.output_csv)

    labels = np.array([r["label"] for r in rows])
    preds = np.array([r["prediction"] for r in rows])
    acc = (labels == preds).mean() if len(labels) else 0.0
    print(f"\nWrote {len(rows)} rows to {output_path}")
    print(f"Overall accuracy on paper-protocol test set: {acc:.4f}")
    print(f"Label distribution: crossing={int((labels==1).sum())}, "
          f"not_crossing={int((labels==0).sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="results_paper_protocol.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)