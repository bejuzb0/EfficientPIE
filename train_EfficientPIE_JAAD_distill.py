"""
train_EfficientPIE_JAAD_distill.py
Phase 4 of Offline VLM Knowledge Distillation.
Entry point for distilled training.
"""
import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.jaad_data import JAAD
from utils.my_dataset_distill import JAADDatasetDistill
from models.EfficientPIE import EfficientPIE
from utils.train_val_distill import train_one_epoch_distill, load_checkpoint, save_checkpoint, evaluate_distill

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    os.makedirs("./weights", exist_ok=True)

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
        'output_type': data_opts['output_type']
    }

    JAAD_dataset = JAAD(data_path=args.data_path)
    seq_length = data_opts['max_size_observe']
    
    train_seq = JAAD_dataset.generate_data_trajectory_sequence('train', **data_opts)
    val_seq = JAAD_dataset.generate_data_trajectory_sequence('val', **data_opts)

    train_seq_for_dataset = JAAD_dataset.get_train_val_data(train_seq, data_type, seq_length, data_opts['seq_overlap_rate'])
    val_seq_for_dataset = JAAD_dataset.get_train_val_data(val_seq, data_type, seq_length, data_opts['seq_overlap_rate'])

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize([300, 300]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize([300, 300]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # Initialize distill dataset
    train_dataset = JAADDatasetDistill(images_seq=train_seq_for_dataset, data_opts=data_opts, transform=data_transform['train'])
    val_dataset = JAADDatasetDistill(images_seq=val_seq_for_dataset, data_opts=data_opts, transform=data_transform['val'])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw, collate_fn=val_dataset.collate_fn)

    model = EfficientPIE(num_classes=2).to(device)
    
    # Projector for bridging student dim (1280) to VLM dim (e.g. standard CLIP ViT is 768)
    # Using nn.Sequential allows projection and simple alignment
    projector = nn.Sequential(
        nn.Linear(1280, 1024),
        nn.ReLU(),
        nn.Linear(1024, args.vlm_dim)
    ).to(device)

    # Initialize Optimizer
    pg = [{'params': model.parameters()}, {'params': projector.parameters()}]
    optimizer = optim.RMSprop(pg, lr=args.lr, weight_decay=0.0001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Load Baseline or Pre-trained Resuming Checkpoint
    start_epoch = 0
    checkpoint_path = "weights/checkpoint_distill.pth"
    
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, projector)
        # Advance scheduler state safely
        for _ in range(start_epoch):
             scheduler.step()
    elif args.weights and os.path.exists(args.weights):
        weights_dict = torch.load(args.weights, map_location=device, weights_only=True)
        load_weights_dict = {k: v for k, v in weights_dict.items() if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
        print(f"Loaded student baseline weights: {args.weights}")

    best_val_acc = 0.0
    print("Start Distillation Training!")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch_distill(model, projector, optimizer, train_loader, device, epoch, alpha=args.alpha)
        scheduler.step()
        
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_distill(model, val_loader, device, epoch)
        
        # Fault Tolerance Checkpointing
        save_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'projector_state_dict': projector.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_acc': max(val_acc, best_val_acc)
        }
        save_checkpoint(save_state, checkpoint_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./weights/transfer_best_model_distill.pth")
            print(f"Saved optimal student distilled model at epoch {epoch} with val_acc: {val_acc:.4f}")

    print("Finished Distillation Training!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.5, help='Distillation loss weight')
    parser.add_argument('--vlm_dim', type=int, default=768, help='Dimensionality of VLM teacher embeddings')
    parser.add_argument('--data-path', type=str, default="../JAAD")
    parser.add_argument('--weights', type=str, default="./weights/transfer_noisy_model_JAAD.pth", help='Baseline weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (e.g. cuda:0 or cpu)')
    main(parser.parse_args())
