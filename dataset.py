"""
Dataset classes for Sign Language Recognition
Handles loading RGB frames and pose keypoints from parquet files
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image



class SignLanguageDataset(Dataset):
    """
    Dataset for Sign Language Recognition
    Loads RGB frames and corresponding pose keypoints
    """
    
    def __init__(self, split='train', args=None, paths=None, transform=None):
        """
        Args:
            split: 'train', 'test', or 'val'
            args: Arguments object with hyperparameters
            paths: Dictionary with path configurations
            transform: Albumentations transform for RGB frames
        """
        self.split = split
        self.args = args
        self.paths = paths
        self.transform = transform
        
        # Paths
        self.rgb_dir = Path(self.paths['rgb_frames_dir']) / split / 'frames'
        self.landmarks_dir = Path(self.paths['landmarks_dir']) / split
        
        # Get list of all parquet files
        self.samples = self._load_dataset()
        
        # Create label mapping
        self.class_to_idx = self._create_class_mapping()
        
        print(f"Loaded {len(self.samples)} videos for {split} split")
        print(f"Number of classes: {len(self.class_to_idx)}")
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset by scanning parquet files"""
        samples = []
        
        if not self.landmarks_dir.exists():
            print(f"Warning: Landmarks directory not found: {self.landmarks_dir}")
            return samples
        
        # Get all parquet files
        parquet_files = list(self.landmarks_dir.glob("*.parquet"))
        
        for parquet_file in parquet_files:
            # Parse filename: class_video.parquet
            filename = parquet_file.stem  # Remove .parquet
            parts = filename.split('_', 1)  # Split only on first underscore
            
            if len(parts) == 2:
                class_name, video_id = parts
                
                # Check if RGB frames exist
                rgb_video_dir = self.rgb_dir / class_name / video_id
                
                if rgb_video_dir.exists():
                    samples.append({
                        'class_name': class_name,
                        'video_id': video_id,
                        'rgb_dir': rgb_video_dir,
                        'parquet_path': parquet_file
                    })
        
        return samples
    
    def _create_class_mapping(self) -> Dict[str, int]:
        """Create mapping from class name to index"""
        class_names = sorted(set([s['class_name'] for s in self.samples]))
        return {name: idx for idx, name in enumerate(class_names)}
    
    def _load_rgb_frames(self, rgb_dir: Path, num_frames: int) -> np.ndarray:
        """
        Load RGB frames from directory with uniform temporal sampling
        
        Args:
            rgb_dir: Directory containing frames
            num_frames: Number of frames to sample
            
        Returns:
            frames: (T, H, W, 3) array of RGB frames
        """
        # Get all frame files
        frame_files = sorted(rgb_dir.glob("*.jpg"), 
                           key=lambda x: int(x.stem.split('_')[-1]))
        
        total_frames = len(frame_files)
        
        if total_frames == 0:
            # Return black frames if no frames found
            return np.zeros((num_frames, self.args.img_size, self.args.img_size, 3), dtype=np.uint8)
        
        # Uniform temporal sampling
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            # Repeat frames if not enough
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            # Read frame with Unicode path support
            frame_path = frame_files[idx]
            try:
                # Use numpy for Unicode paths (Windows compatible)
                with open(frame_path, 'rb') as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img is None:
                    # Use black frame if failed
                    img = np.zeros((self.args.img_size, self.args.img_size, 3), dtype=np.uint8)
                else:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize
                    img = cv2.resize(img, (self.args.img_size, self.args.img_size))
                
                frames.append(img)
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                frames.append(np.zeros((self.args.img_size, self.args.img_size, 3), dtype=np.uint8))
        
        return np.array(frames, dtype=np.uint8)
    
    def _load_pose_keypoints(self, parquet_path: Path, num_frames: int) -> np.ndarray:
        """
        Load pose keypoints from parquet file with uniform temporal sampling
        
        Args:
            parquet_path: Path to parquet file
            num_frames: Number of frames to sample
            
        Returns:
            keypoints: (T, 180) array of flattened keypoints
        """
        try:
            df = pd.read_parquet(parquet_path)
            total_frames = len(df)
            
            if total_frames == 0:
                return np.zeros((num_frames, self.args.pose_dim), dtype=np.float32)
            
            # Uniform temporal sampling (same as RGB)
            if total_frames >= num_frames:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            else:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            # Select frames
            sampled_df = df.iloc[indices]
            keypoints = sampled_df.values.astype(np.float32)
            
            return keypoints
        
        except Exception as e:
            print(f"Error loading keypoints from {parquet_path}: {e}")
            return np.zeros((num_frames, self.args.pose_dim), dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample
        
        Returns:
            dict with keys:
                - 'rgb': (T, 3, H, W) tensor of RGB frames
                - 'pose': (T, 180) tensor of pose keypoints
                - 'label': class index
                - 'class_name': class name (for debugging)
        """
        sample = self.samples[idx]
        
        # Load RGB frames
        rgb_frames = self._load_rgb_frames(sample['rgb_dir'], self.args.num_frames)
        
        # Apply augmentation to each frame
        if self.transform:
            augmented_frames = []
            for frame in rgb_frames:
                # Convert numpy array to PIL Image
                pil_img = Image.fromarray(frame)
                # Apply transform
                augmented = self.transform(pil_img)
                augmented_frames.append(augmented)
            rgb_frames = torch.stack(augmented_frames)  # (T, 3, H, W)
        else:
            # Convert to tensor
            rgb_frames = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float() / 255.0
        
        # Load pose keypoints
        pose_keypoints = self._load_pose_keypoints(sample['parquet_path'], self.args.num_frames)
        pose_keypoints = torch.from_numpy(pose_keypoints).float()
        
        # Get label
        label = self.class_to_idx[sample['class_name']]
        
        return {
            'rgb': rgb_frames,  # (T, 3, H, W)
            'pose': pose_keypoints,  # (T, 180)
            'label': label,
            'class_name': sample['class_name']
        }


def get_train_transform(args):
    """Get training augmentation transforms using torchvision"""
    transform_list = []
    
    # Data augmentation for training
    if args.train_augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=args.horizontal_flip_prob),
        ])
        
        if args.color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            )
        
        # Additional augmentations
        transform_list.extend([
            transforms.RandomRotation(degrees=10),  # Rotate ±10 degrees
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Shift ±10%
        ])
    
    # Common transforms (always applied)
    transform_list.extend([
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)


def get_val_transform(config):
    """Get validation transforms (no augmentation) using torchvision"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_dataloader(split='train', config=None, shuffle=True):
    """
    Create dataloader for given split
    
def get_dataloader(split, args, paths, shuffle=False):
    """
    Create a DataLoader for the specified split
    
    Args:
        split: 'train', 'test', or 'val'
        args: Arguments object with hyperparameters
        paths: Dictionary with path configurations
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader, dataset
    """
    # Get transform
    if split == 'train':
        transform = get_train_transform(args)
    else:
        transform = get_val_transform()
    
    # Create dataset
    dataset = SignLanguageDataset(split=split, args=args, paths=paths, transform=transform)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        drop_last=True if split == 'train' else False
    )
    
    return dataloader, dataset


if __name__ == "__main__":
    # Test dataset
    from args_config import get_args, load_paths
    
    args = get_args()
    paths = load_paths()
    
    print("Testing train dataset...")
    train_loader, train_dataset = get_dataloader('train', args, paths, shuffle=True)
    
    print(f"\nClass to index mapping:")
    for class_name, idx in list(train_dataset.class_to_idx.items())[:10]:
        print(f"  {class_name}: {idx}")
    
    print(f"\nTesting batch loading...")
    for batch in train_loader:
        print(f"RGB shape: {batch['rgb'].shape}")  # (B, T, 3, H, W)
        print(f"Pose shape: {batch['pose'].shape}")  # (B, T, 180)
        print(f"Labels shape: {batch['label'].shape}")  # (B,)
        print(f"Sample class: {batch['class_name'][0]}")
        break
    
    print("\nDataset test completed!")
