"""
Argument parser for Sign Language Recognition hyperparameters
"""
import argparse
import json
import os


def load_paths(config_path='config.json'):
    """Load paths from config.json file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['paths']


def get_args():
    """Parse command line arguments for hyperparameters"""
    parser = argparse.ArgumentParser(description='Sign Language Recognition Training')
    
    # Dataset settings
    parser.add_argument('--num-classes', type=int, default=100,
                        help='Number of sign language classes')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num-frames', type=int, default=15,
                        help='Number of frames per video sequence')
    
    # Pose keypoints dimensions
    parser.add_argument('--pose-dim', type=int, default=180,
                        help='Total flattened keypoints dimension')
    parser.add_argument('--left-hand-dim', type=int, default=63,
                        help='Left hand keypoints dimension (21*3)')
    parser.add_argument('--right-hand-dim', type=int, default=63,
                        help='Right hand keypoints dimension (21*3)')
    parser.add_argument('--lips-dim', type=int, default=30,
                        help='Lips keypoints dimension (10*3)')
    parser.add_argument('--body-pose-dim', type=int, default=24,
                        help='Body pose keypoints dimension (6*4)')
    
    # Model architecture
    parser.add_argument('--rgb-feature-dim', type=int, default=512,
                        help='RGB feature dimension')
    parser.add_argument('--pose-feature-dim', type=int, default=256,
                        help='Pose feature dimension')
    parser.add_argument('--fusion-dim', type=int, default=512,
                        help='Fusion layer dimension')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num-transformer-layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    
    # Learning rate scheduler
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr-step-size', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    
    # Data augmentation
    parser.add_argument('--train-augment', action='store_true', default=True,
                        help='Enable training data augmentation')
    parser.add_argument('--no-train-augment', action='store_false', dest='train_augment',
                        help='Disable training data augmentation')
    parser.add_argument('--horizontal-flip-prob', type=float, default=0.5,
                        help='Probability of horizontal flip augmentation')
    parser.add_argument('--color-jitter', action='store_true', default=True,
                        help='Enable color jitter augmentation')
    parser.add_argument('--no-color-jitter', action='store_false', dest='color_jitter',
                        help='Disable color jitter augmentation')
    parser.add_argument('--temporal-augment', action='store_true', default=True,
                        help='Enable temporal augmentation')
    parser.add_argument('--no-temporal-augment', action='store_false', dest='temporal_augment',
                        help='Disable temporal augmentation')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                        help='Patience for early stopping')
    
    # Loss weights
    parser.add_argument('--rgb-loss-weight', type=float, default=0.5,
                        help='Weight for RGB loss in multi-task learning')
    parser.add_argument('--pose-loss-weight', type=float, default=0.5,
                        help='Weight for pose loss in multi-task learning')
    parser.add_argument('--fusion-loss-weight', type=float, default=1.0,
                        help='Weight for fusion loss in multi-task learning')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Validate every N epochs')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Test the argument parser
    args = get_args()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    print("\nLoaded paths:")
    paths = load_paths()
    for key, value in paths.items():
        print(f"  {key}: {value}")
