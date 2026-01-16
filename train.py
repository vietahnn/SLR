"""
Training script for Sign Language Recognition
Two-stream multimodal model: RGB + Pose
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np

from args_config import get_args, load_paths
from dataset import get_dataloader
from models import SignLanguageModel
from utils import (
    set_seed, save_checkpoint, load_checkpoint,
    AverageMeter, EarlyStopping, compute_metrics,
    plot_confusion_matrix, plot_training_history,
    save_metrics, get_lr, print_model_summary
)


class Trainer:
    """Trainer class for Sign Language Recognition"""
    
    def __init__(self, args, paths):
        self.args = args
        self.paths = paths
        
        # Set seed for reproducibility
        set_seed(self.args.seed)
        
        # Device
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create dataloaders
        print("\nLoading datasets...")
        self.train_loader, self.train_dataset = get_dataloader('train', self.args, self.paths, shuffle=True)
        self.val_loader, self.val_dataset = get_dataloader('val', self.args, self.paths, shuffle=False)
        
        # Update num_classes in args
        self.args.num_classes = len(self.train_dataset.class_to_idx)
        print(f"Number of classes: {self.args.num_classes}")
        
        # Create model
        print("\nCreating model...")
        self.model = SignLanguageModel(self.args).to(self.device)
        print_model_summary(self.model, self.args)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        
        # Learning rate scheduler
        if self.args.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.num_epochs
            )
        elif self.args.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.args.lr_step_size,
                gamma=self.args.lr_gamma
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.args.early_stopping_patience,
            mode='max'
        )
        
        # Tensorboard
        log_dir = Path(self.paths['base_dir']) / 'runs' / time.strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(log_dir)
        print(f"Tensorboard log dir: {log_dir}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_acc = 0.0
        self.start_epoch = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            rgb = batch['rgb'].to(self.device)  # (B, T, 3, H, W)
            pose = batch['pose'].to(self.device)  # (B, T, 180)
            labels = batch['label'].to(self.device)  # (B,)
            
            # Forward pass
            logits = self.model(rgb, pose)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            acc = (predicted == labels).float().mean() * 100
            
            # Update meters
            losses.update(loss.item(), rgb.size(0))
            accs.update(acc.item(), rgb.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accs.avg:.2f}%',
                'lr': f'{get_lr(self.optimizer):.6f}'
            })
            
            # Log to tensorboard
            if batch_idx % self.args.log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', losses.avg, step)
                self.writer.add_scalar('Train/Accuracy', accs.avg, step)
                self.writer.add_scalar('Train/LR', get_lr(self.optimizer), step)
        
        return losses.avg, accs.avg
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.args.num_epochs} [Val]")
        
        for batch in pbar:
            # Move data to device
            rgb = batch['rgb'].to(self.device)
            pose = batch['pose'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            logits = self.model(rgb, pose)
            loss = self.criterion(logits, labels)
            
            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            acc = (predicted == labels).float().mean() * 100
            
            # Update meters
            losses.update(loss.item(), rgb.size(0))
            accs.update(acc.item(), rgb.size(0))
            
            # Store predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accs.avg:.2f}%'
            })
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Compute metrics
        metrics = compute_metrics(
            all_predictions,
            all_targets,
            class_names=list(self.train_dataset.class_to_idx.keys())
        )
        
        return losses.avg, accs.avg, all_predictions, all_targets, metrics
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60 + "\n")
        
        for epoch in range(self.start_epoch + 1, self.args.num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.args.val_interval == 0:
                val_loss, val_acc, predictions, targets, metrics = self.validate(epoch)
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)
                
                # Log to tensorboard
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
                self.writer.add_scalar('Val/F1Score', metrics['f1_score'] * 100, epoch)
                
                # Print epoch summary
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch}/{self.args.num_epochs} Summary:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"  F1 Score: {metrics['f1_score']:.4f}")
                print(f"  Time: {epoch_time:.2f}s\n")
                
                # Save best model
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        self.best_acc,
                        self.args,
                        filename=self.paths['best_model_path']
                    )
                    print(f"New best model saved! Accuracy: {self.best_acc:.2f}%\n")
                    
                    # Save confusion matrix for best model
                    cm_path = Path(self.paths['checkpoint_dir']) / 'best_confusion_matrix.png'
                    plot_confusion_matrix(
                        predictions,
                        targets,
                        list(self.train_dataset.class_to_idx.keys()),
                        save_path=cm_path
                    )
                
                # Early stopping
                if self.early_stopping(val_acc):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
            
            # Learning rate scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Save periodic checkpoint
            if epoch % self.args.save_interval == 0:
                checkpoint_path = Path(self.paths['checkpoint_dir']) / f"checkpoint_epoch_{epoch}.pth"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.best_acc,
                    self.args,
                    filename=checkpoint_path
                )
        
        # Training completed
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Best validation accuracy: {self.best_acc:.2f}%")
        
        # Save training history plot
        history_path = Path(self.paths['checkpoint_dir']) / 'training_history.png'
        plot_training_history(self.history, save_path=history_path)
        
        # Save metrics
        metrics_path = Path(self.paths['checkpoint_dir']) / 'training_metrics.json'
        save_metrics(self.history, metrics_path)
        
        # Close tensorboard writer
        self.writer.close()
        
        print(f"\nResults saved to: {self.paths['checkpoint_dir']}")
    
    def resume_from_checkpoint(self, checkpoint_path):
        """Resume training from checkpoint"""
        self.start_epoch, self.best_acc = load_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            checkpoint_path,
            self.device
        )


def main():
    """Main function"""
    # Parse arguments and load paths
    args = get_args()
    paths = load_paths()
    
    # Create trainer
    trainer = Trainer(args, paths)
    
    # Optionally resume from checkpoint
    # trainer.resume_from_checkpoint('path/to/checkpoint.pth')
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
