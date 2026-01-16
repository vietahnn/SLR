"""
Inference script for Sign Language Recognition
Load trained model and make predictions on new videos
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from args_config import get_args, load_paths
from models import SignLanguageModel
from dataset import SignLanguageDataset, get_val_transform
from utils import compute_metrics, plot_confusion_matrix


class SignLanguageInference:
    """Inference class for Sign Language Recognition"""
    
    def __init__(self, checkpoint_path, args=None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            args: Arguments object (optional)
        """
        self.args = args if args else get_args()
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Update args from checkpoint if available
        if 'args' in checkpoint:
            checkpoint_args = checkpoint['args']
            if hasattr(checkpoint_args, 'num_classes'):
                self.args.num_classes = checkpoint_args.num_classes
        
        # Create model
        self.model = SignLanguageModel(self.args).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Num classes: {self.args.num_classes}")
    
    @torch.no_grad()
    def predict_single(self, rgb, pose):
        """
        Predict on a single video
        
        Args:
            rgb: (T, 3, H, W) RGB frames tensor
            pose: (T, 180) pose keypoints tensor
            
        Returns:
            predicted_class: int
            confidence: float
            all_probs: (num_classes,) probabilities
        """
        # Add batch dimension
        rgb = rgb.unsqueeze(0).to(self.device)  # (1, T, 3, H, W)
        pose = pose.unsqueeze(0).to(self.device)  # (1, T, 180)
        
        # Forward pass
        logits = self.model(rgb, pose)  # (1, num_classes)
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)[0]  # (num_classes,)
        
        # Get prediction
        confidence, predicted = torch.max(probs, 0)
        
        return predicted.item(), confidence.item(), probs.cpu().numpy()
    
    @torch.no_grad()
    def predict_batch(self, dataloader):
        """
        Predict on a batch of videos using dataloader
        
        Args:
            dataloader: PyTorch DataLoader
            
        Returns:
            predictions: list of predicted class indices
            confidences: list of confidence scores
            targets: list of true labels (if available)
        """
        predictions = []
        confidences = []
        targets = []
        
        for batch in tqdm(dataloader, desc="Predicting"):
            rgb = batch['rgb'].to(self.device)
            pose = batch['pose'].to(self.device)
            labels = batch['label']
            
            # Forward pass
            logits = self.model(rgb, pose)
            probs = torch.softmax(logits, dim=1)
            
            # Get predictions
            conf, pred = torch.max(probs, 1)
            
            predictions.extend(pred.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
            targets.extend(labels.numpy())
        
        return np.array(predictions), np.array(confidences), np.array(targets)
    
    @torch.no_grad()
    def extract_features(self, rgb, pose):
        """
        Extract features from intermediate layers
        
        Args:
            rgb: (T, 3, H, W) or (B, T, 3, H, W)
            pose: (T, 180) or (B, T, 180)
            
        Returns:
            dict with rgb_features, pose_features, fused_features
        """
        # Add batch dimension if needed
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(0)
            pose = pose.unsqueeze(0)
        
        rgb = rgb.to(self.device)
        pose = pose.to(self.device)
        
        return self.model.get_features(rgb, pose)


def evaluate_on_split(checkpoint_path, split='test', config=None):
    """
    Evaluate model on a specific split
    
    Args:
        checkpoint_path: Path to model checkpoint
        split: 'train', 'test', or 'val'
        config: Configuration object
    """
    config = config if config else Config()
    
    # Create inference object
    inference = SignLanguageInference(checkpoint_path, config)
    
    # Load dataset
    print(f"\nLoading {split} dataset...")
    from dataset import get_dataloader
    dataloader, dataset = get_dataloader(split, config, shuffle=False)
    
    # Get class names
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Predict
    print(f"\nEvaluating on {split} set...")
    predictions, confidences, targets = inference.predict_batch(dataloader)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets, class_names)
    
    # Print results
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ON {split.upper()} SET")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Average Confidence: {confidences.mean():.4f}")
    print("="*60 + "\n")
    
    # Print per-class accuracy
    if 'classification_report' in metrics:
        print("Per-class metrics:")
        report = metrics['classification_report']
        for class_name in class_names[:10]:  # Print first 10 classes
            if class_name in report:
                class_metrics = report[class_name]
                print(f"  {class_name}: "
                      f"Precision={class_metrics['precision']:.3f}, "
                      f"Recall={class_metrics['recall']:.3f}, "
                      f"F1={class_metrics['f1-score']:.3f}")
    
    # Plot confusion matrix
    cm_path = Path(config.CHECKPOINT_DIR) / f'{split}_confusion_matrix.png'
    plot_confusion_matrix(predictions, targets, class_names, save_path=cm_path)
    
    return metrics, predictions, targets


def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Sign Language Recognition Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                      choices=['train', 'test', 'val'],
                      help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Evaluate
    evaluate_on_split(args.checkpoint, args.split, config)


if __name__ == "__main__":
    # Example usage without command line args
    config = Config()
    checkpoint_path = config.BEST_MODEL_PATH
    
    if Path(checkpoint_path).exists():
        print("Running evaluation on test set...")
        evaluate_on_split(checkpoint_path, split='test', config=config)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using train.py")
