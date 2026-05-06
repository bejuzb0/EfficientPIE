"""
gradcam_inference_JAAD.py

Run examples:
    # All four quadrants, 10 each:
    python gradcam_inference_JAAD.py --mode all --num-samples 10

    # Only true positive crossing samples:
    python gradcam_inference_JAAD.py --mode tp --num-samples 20

    # Only false negatives (crossing missed):
    python gradcam_inference_JAAD.py --mode fn --num-samples 20
"""

import argparse
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils.jaad_data import JAAD
from utils.my_dataset import MyDataSet
from models.EfficientPIE import EfficientPIE


MODES = {
    'tp':  'gt-crossing_pred-crossing_CORRECT',
    'tn':  'gt-not_crossing_pred-not_crossing_CORRECT',
    'fp':  'gt-not_crossing_pred-crossing_WRONG',
    'fn':  'gt-crossing_pred-not_crossing_WRONG',
    'all': 'all four quadrants, --num-samples each'
}


def unnormalize(tensor):
    img = tensor.clone().cpu()
    img = img * 0.5 + 0.5
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1).astype(np.float32)


def generate_gradcam(model, input_tensor, target_class, target_layer):
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class)]
    )[0]
    rgb_img = unnormalize(input_tensor.squeeze(0))
    return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


def save_comparison(original_tensor, heatmap_np, label, pred, sample_idx, out_dir):
    orig_uint8 = (unnormalize(original_tensor.squeeze(0)) * 255).astype(np.uint8)
    combined = Image.new("RGB", (600, 300))
    combined.paste(Image.fromarray(orig_uint8), (0, 0))
    combined.paste(Image.fromarray(heatmap_np), (300, 0))

    label_str = "crossing" if label == 1 else "not_crossing"
    pred_str  = "crossing" if pred  == 1 else "not_crossing"
    correct   = "CORRECT"  if label == pred else "WRONG"
    fname = f"sample_{sample_idx:04d}_gt-{label_str}_pred-{pred_str}_{correct}.png"
    combined.save(os.path.join(out_dir, fname))


def matches_mode(gt, pred, mode):
    """Check if a (gt, pred) pair belongs to the requested mode bucket."""
    return {
        'tp': gt == 1 and pred == 1,
        'tn': gt == 0 and pred == 0,
        'fp': gt == 0 and pred == 1,
        'fn': gt == 1 and pred == 0,
    }[mode]


def main(args):

    if args.mode not in MODES:
        raise ValueError(f"--mode must be one of: {list(MODES.keys())}")

    data_opts = {
        'fstride': 1,
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

    data_type = {
        'encoder_input_type': data_opts['encoder_input_type'],
        'decoder_input_type': data_opts['decoder_input_type'],
        'output_type':        data_opts['output_type']
    }

    JAAD_dataset = JAAD(data_path=args.data_path)
    test_seq = JAAD_dataset.generate_data_trajectory_sequence('test', **data_opts)
    seq_length = data_opts['max_size_observe']
    test_seq_for_dataset = JAAD_dataset.get_train_val_data(
        test_seq, data_type, seq_length, data_opts['seq_overlap_rate']
    )

    data_transform = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_dataset = MyDataSet(
        images_seq=test_seq_for_dataset,
        data_opts=data_opts,
        transform=data_transform
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = EfficientPIE(num_classes=2).to(device)
    model.eval()

    if args.weights:
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {
                k: v for k, v in weights_dict.items()
                if k in model.state_dict()
                and model.state_dict()[k].numel() == v.numel()
            }
            model.load_state_dict(load_weights_dict, strict=False)
            print(f"Loaded weights: {args.weights}")
        else:
            raise FileNotFoundError(f"Weights not found: {args.weights}")

    target_layer = model.commonConv1
    os.makedirs(args.output_dir, exist_ok=True)

    # ── determine which buckets to fill ──────────────────────────────────
    active_modes = ['tp', 'tn', 'fp', 'fn'] if args.mode == 'all' else [args.mode]
    buckets = {m: [] for m in active_modes}
    quota   = args.num_samples  # per bucket

    print(f"Mode: {args.mode} | Quota per bucket: {quota} | "
          f"Scanning up to {len(test_dataset)} samples...")

    # ── single pass: run inference on every sample, fill buckets ─────────
    # We must run inference to know pred — labels alone aren't enough for tp/fp/fn/tn
    for i in range(len(test_dataset)):

        # stop early if all buckets are full
        if all(len(v) >= quota for v in buckets.values()):
            break

        img_tensor, label = test_dataset[i]
        gt = int(label.item())

        input_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
        pred = int(logits.argmax(dim=1).item())

        for m in active_modes:
            if len(buckets[m]) < quota and matches_mode(gt, pred, m):
                buckets[m].append((i, img_tensor, gt, pred))
                break  # one sample goes into one bucket only

    # ── report what was found ─────────────────────────────────────────────
    for m, samples in buckets.items():
        print(f"  {m.upper()} ({MODES[m]}): {len(samples)} samples found")

    # ── generate Grad-CAM and save ────────────────────────────────────────
    total = sum(len(v) for v in buckets.values())
    done  = 0

    for m, samples in buckets.items():
        for (sample_idx, img_tensor, gt, pred) in samples:
            # Extract video ID and frame number
            frame_idx = data_opts['max_size_observe'] - 1
            img_path = test_dataset.images_seq['images'][sample_idx][frame_idx]
            path_parts = os.path.normpath(img_path).split(os.sep)
            vid_id = path_parts[-2] if len(path_parts) >= 2 else "unknown_vid"
            frame_num = path_parts[-1].split('.')[0] if len(path_parts) >= 1 else "unknown_frame"
            
            print(f"  Sample {sample_idx:04d} ({m.upper()}) | Video: {vid_id} | Frame: {frame_num}")

            input_tensor = img_tensor.unsqueeze(0).to(device)

            heatmap = generate_gradcam(
                model, input_tensor,
                target_class=pred,      # what drove THIS prediction
                target_layer=target_layer
            )

            save_comparison(input_tensor.cpu(), heatmap, gt, pred, sample_idx, args.output_dir)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{total} saved")

    print(f"\nDone. {done} images saved to: {args.output_dir}")
    print("Filename format: sample_NNNN_gt-[label]_pred-[label]_[CORRECT/WRONG].png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',   type=str,
                        default="/Users/akvma/Developer/cvproject/EfficientPIE/JAAD")
    parser.add_argument('--weights',     type=str,
                        default="./weights/transfer_noisy_model_JAAD.pth")
    parser.add_argument('--device',      type=str,  default='cuda:2')
    parser.add_argument('--num-samples', type=int,  default=10,
                        help='samples per bucket (per mode for --mode all)')
    parser.add_argument('--output-dir',  type=str,  default='./results/gradcam')
    parser.add_argument('--mode',        type=str,  default='all',
                        choices=list(MODES.keys()),
                        help=(
                            'tp  = crossing predicted correctly\n'
                            'tn  = not-crossing predicted correctly\n'
                            'fp  = not-crossing predicted as crossing\n'
                            'fn  = crossing missed (predicted not-crossing)\n'
                            'all = 10 of each'
                        ))
    opt = parser.parse_args()
    main(opt)