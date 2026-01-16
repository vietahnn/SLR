"""
Utility functions for training and evaluation
"""

import torch
import numpy as np
import random
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import json


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, args, filename=None):
    """Save model checkpoint"""
    if filename is None:
        from args_config import load_paths
        paths = load_paths()
        filename = Path(paths['checkpoint_dir']) / f"checkpoint_epoch_{epoch}.pth"
    else:
        filename = Path(filename)
    
    # Create directory if not exists
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'args': args
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, scheduler, filename, device='cuda'):
    """Load model checkpoint"""
    filename = Path(filename)
    
    if not filename.exists():
        print(f"Checkpoint not found: {filename}")
        return 0, 0.0
    
    checkpoint = torch.load(filename, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"Checkpoint loaded from {filename}")
    print(f"Resuming from epoch {epoch}, best acc: {best_acc:.4f}")
    
    return epoch, best_acc


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def compute_metrics(predictions, targets, class_names=None):
    """
    Compute classification metrics
    
    Args:
        predictions: (N,) array of predicted class indices
        targets: (N,) array of true class indices
        class_names: List of class names
        
    Returns:
        dict with accuracy, f1_score, etc.
    """
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1
    }
    
    # Classification report
    if class_names:
        report = classification_report(
            targets,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
    
    return metrics


def plot_confusion_matrix(predictions, targets, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        predictions: (N,) array of predicted class indices
        targets: (N,) array of true class indices
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def save_metrics(metrics, save_path):
    """Save metrics to JSON file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {save_path}")


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def print_model_summary(model, args):
    """Print model summary"""
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    
    print(f"\nModel configuration:")
    print(f"  - RGB feature dim: {args.rgb_feature_dim}")
    print(f"  - Pose feature dim: {args.pose_feature_dim}")
    print(f"  - Fusion dim: {args.fusion_dim}")
    print(f"  - Num transformer layers: {args.num_transformer_layers}")
    print(f"  - Num attention heads: {args.num_heads}")
    print(f"  - Dropout: {args.dropout}")
    print(f"  - Num classes: {args.num_classes}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")
    
    # Test EarlyStopping
    early_stopping = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62]
    for i, score in enumerate(scores):
        if early_stopping(score):
            print(f"Early stopping at iteration {i}")
            break
    
    print("Utilities test completed!")
