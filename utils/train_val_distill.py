"""
train_val_distill.py
Phase 3 of offline VLM Knowledge Distillation.
"""
import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm

def save_checkpoint(state, filename="weights/checkpoint_distill.pth"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None, projector=None):
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if projector and 'projector_state_dict' in checkpoint:
            projector.load_state_dict(checkpoint['projector_state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print(f"=> loaded checkpoint '{filename}' (epoch {epoch})")
        return epoch
    else:
        print(f"=> no checkpoint found at '{filename}'")
        return 0

def robust_noisy(pred, epoch):
    max_range = 0.5 * (epoch / 30)
    random_num = torch.rand(1).item() * (2 * max_range) - max_range
    noisy = torch.tensor([random_num, -random_num], dtype=pred.dtype, device=pred.device)
    robust_pred = pred + noisy
    return robust_pred

def train_one_epoch_distill(model, projector, optimizer, dataloader, device, epoch, alpha=0.5):
    model.train()
    if projector is not None:
        projector.train()
        
    loss_task_fn = nn.CrossEntropyLoss()
    loss_distill_fn = nn.MSELoss()
    
    accu_loss = torch.zeros(1).to(device)
    accu_loss_task = torch.zeros(1).to(device)
    accu_loss_distill = torch.zeros(1).to(device)

    TP = torch.zeros(1).to(device)
    TN = torch.zeros(1).to(device)
    FP = torch.zeros(1).to(device)
    FN = torch.zeros(1).to(device)

    # Dictionary to access intermediate features via forward hook
    features = {}
    def hook_fn(m, i, o):
        features['flatten'] = o
        
    # Hook into flatten output to get pooled features
    hook = model.flatten.register_forward_hook(hook_fn)
    
    optimizer.zero_grad()
    dataloader = tqdm(dataloader, file=sys.stdout)
    
    for step, data in enumerate(dataloader):
        images, labels, vlms = data
        images, labels, vlms = images.to(device), labels.to(device), vlms.to(device)
        
        # Forward pass (triggers the hook)
        pred = model(images)
        robust_pred = robust_noisy(pred, epoch)
        pred_classes = torch.max(pred, dim=1)[1]
        
        student_features = features['flatten']
        if projector is not None:
            student_features = projector(student_features)
            
        # Task Loss
        loss_task = loss_task_fn(robust_pred, labels)
        
        # Distillation Loss (MSE)
        # Ensure vlms representation is matching student feature shape
        if student_features.shape != vlms.shape:
             loss_distill = loss_distill_fn(student_features, vlms.view(student_features.shape))
        else:
             loss_distill = loss_distill_fn(student_features, vlms)
             
        # Composite Loss
        loss = loss_task + alpha * loss_distill
        
        loss.backward()
        
        accu_loss += loss.detach()
        accu_loss_task += loss_task.detach()
        accu_loss_distill += loss_distill.detach()
        
        # Calculate Performance Metrics
        TP += ((pred_classes == 1) & (labels == 1)).sum().float()
        TN += ((pred_classes == 0) & (labels == 0)).sum().float()
        FP += ((pred_classes == 1) & (labels == 0)).sum().float()
        FN += ((pred_classes == 0) & (labels == 1)).sum().float()
        
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        
        dataloader.desc = "[train epoch {}] L:{:.3f} (Task:{:.3f} Dist:{:.3f}), acc: {:.3f}".format(
            epoch, 
            accu_loss.item() / (step + 1), 
            accu_loss_task.item() / (step + 1),
            accu_loss_distill.item() / (step + 1), 
            accuracy.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        
    hook.remove()
    return accu_loss.item() / (step + 1), accuracy.item()

@torch.no_grad()
def evaluate_distill(model, dataloader, device, epoch):
    from sklearn.metrics import roc_auc_score
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)

    TP = torch.zeros(1).to(device)
    TN = torch.zeros(1).to(device)
    FP = torch.zeros(1).to(device)
    FN = torch.zeros(1).to(device)
    preds_all = []  
    labels_all = [] 

    dataloader = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(dataloader):
        images, labels, _ = data
        pred = model(images.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        labels = labels.to(device)
        
        TP += ((pred_classes == 1) & (labels == 1)).sum().float()
        TN += ((pred_classes == 0) & (labels == 0)).sum().float()
        FP += ((pred_classes == 1) & (labels == 0)).sum().float()
        FN += ((pred_classes == 0) & (labels == 1)).sum().float()

        loss = loss_function(pred, labels)
        accu_loss += loss
        
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1_score = 2 * ((precision * recall) / (precision + recall + 1e-6))

        preds_all.extend(pred[:, 1].cpu().numpy())
        labels_all.extend(labels.cpu())
        
        try:
            auc_score = roc_auc_score(labels_all, preds_all)
        except ValueError:
            auc_score = 0.0
            
        dataloader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, precision: {:.3f}, " \
                          "recall: {:.3f}, f1_score: {:.3f}, auc: {:.3f}" \
            .format(epoch, accu_loss.item() / (step + 1), accuracy.item(), precision.item(), recall.item(),
                    f1_score.item(), auc_score)

    return accu_loss.item() / (step + 1), accuracy.item(), precision.item(), recall.item(), f1_score.item()
