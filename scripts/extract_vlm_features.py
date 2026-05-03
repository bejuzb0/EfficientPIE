"""
extract_vlm_features.py

Phase 1 of Offline VLM Knowledge Distillation.
Iterates through raw dataset frames and extracts offline feature embeddings
using a pre-trained Vision-Language Model (Teacher).
Saves embeddings as .pt files mirroring the dataset directory structure.

Usage:
    python extract_vlm_features.py --data-path ../JAAD --model-id openai/clip-vit-base-patch32 --batch-size 32
"""

import os
import glob
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel

def parse_args():
    parser = argparse.ArgumentParser(description="Extract VLM features offline")
    parser.add_argument('--data-path', type=str, default="../JAAD",
                        help="Path to the base JAAD dataset directory")
    parser.add_argument('--model-id', type=str, default="openai/clip-vit-base-patch32",
                        help="HuggingFace Model ID for the Teacher VLM")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for feature extraction")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to run extraction on")
    return parser.parse_args()

def main():
    args = parse_args()

    # Define paths
    images_dir = os.path.join(args.data_path, "images")
    output_base_dir = os.path.join(args.data_path, "vlm_features")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at {images_dir}. Did you extract the dataset?")

    # Collect all image paths (assumes JAAD format: images/video_0001/00000.png)
    print(f"Scraping files from: {images_dir}")
    all_image_paths = sorted(glob.glob(os.path.join(images_dir, "*", "*.png")))
    total_images = len(all_image_paths)
    print(f"Found {total_images} frames to process.")

    if total_images == 0:
        return

    # Filter out files that have already been processed to support safe resuming
    print("Filtering already processed frames...")
    pending_paths = []
    output_mappings = []

    for img_path in all_image_paths:
        # e.g.: .../JAAD/images/video_0001/00000.png --> video_0001/00000.png
        rel_path = os.path.relpath(img_path, images_dir)
        vid_dir = os.path.dirname(rel_path)
        frame_name = os.path.basename(rel_path).split('.')[0]
        
        target_dir = os.path.join(output_base_dir, vid_dir)
        target_file = os.path.join(target_dir, f"{frame_name}_vlm.pt")

        if not os.path.exists(target_file):
            pending_paths.append(img_path)
            output_mappings.append((target_dir, target_file))

    print(f"Skipped {total_images - len(pending_paths)} previously processed frames.")
    print(f"Pending extraction: {len(pending_paths)} frames.")

    if len(pending_paths) == 0:
        print("Extraction complete.")
        return

    # Load Teacher Model
    print(f"Loading Teacher VLM: {args.model_id} on {args.device}...")
    processor = CLIPProcessor.from_pretrained(args.model_id)
    model = CLIPVisionModel.from_pretrained(args.model_id).to(args.device)
    model.eval()

    # Batch Processing
    for i in tqdm(range(0, len(pending_paths), args.batch_size), desc="Extracting"):
        batch_paths = pending_paths[i:i+args.batch_size]
        batch_mappings = output_mappings[i:i+args.batch_size]

        # Ensure output directories exist for this batch
        for target_dir, _ in batch_mappings:
            os.makedirs(target_dir, exist_ok=True)

        batch_images = []
        valid_indices = []

        # Load images gracefully
        for idx, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping damaged unreadable frame {p}: {e}")

        if not batch_images:
            continue

        # Forward pass
        with torch.no_grad():
            inputs = processor(images=batch_images, return_tensors="pt").to(args.device)
            outputs = model(**inputs)
            # Pooler output is the global representation of the image: shape [batch_size, hidden_size]
            embeddings = outputs.pooler_output 

        # Save to disk
        # We save each tensor back to the CPU to avoid locking GPU memory on disk saves
        for out_idx, valid_idx in enumerate(valid_indices):
            _, target_path = batch_mappings[valid_idx]
            emb = embeddings[out_idx].clone().cpu()
            torch.save(emb, target_path)

if __name__ == '__main__':
    main()
