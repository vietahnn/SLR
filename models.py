"""
Model architectures for Sign Language Recognition
Two-stream multimodal fusion: RGB + Pose
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            x with positional encoding: (B, T, D)
        """
        return x + self.pe[:, :x.size(1), :]


class RGBEncoder(nn.Module):
    """
    RGB Encoder using 2D CNN + Temporal Transformer
    Extracts spatial features per frame then models temporal dynamics
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Spatial feature extractor (ResNet50)
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the final FC layer and average pooling
        self.spatial_encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Spatial feature dimension after ResNet (2048 channels)
        self.spatial_dim = 2048
        
        # Adaptive pooling to get fixed spatial size
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Project to target dimension
        self.spatial_proj = nn.Linear(self.spatial_dim, args.rgb_feature_dim)
        
        # Temporal Transformer
        self.pos_encoding = PositionalEncoding(args.rgb_feature_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.rgb_feature_dim,
            nhead=args.num_heads,
            dim_feedforward=args.rgb_feature_dim * 4,
            dropout=args.dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_transformer_layers
        )
        
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W) RGB frames
            
        Returns:
            features: (B, T, D) temporal features
        """
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames together
        x = x.view(B * T, C, H, W)
        
        # Extract spatial features
        spatial_features = self.spatial_encoder(x)  # (B*T, 2048, h, w)
        
        # Pool spatial dimensions
        spatial_features = self.spatial_pool(spatial_features)  # (B*T, 2048, 1, 1)
        spatial_features = spatial_features.view(B * T, -1)  # (B*T, 2048)
        
        # Project to target dimension
        spatial_features = self.spatial_proj(spatial_features)  # (B*T, D)
        
        # Reshape back to temporal dimension
        spatial_features = spatial_features.view(B, T, -1)  # (B, T, D)
        
        # Add positional encoding
        spatial_features = self.pos_encoding(spatial_features)
        
        # Apply temporal transformer
        temporal_features = self.temporal_transformer(spatial_features)  # (B, T, D)
        
        return self.dropout(temporal_features)


class PoseEncoder(nn.Module):
    """
    Pose Encoder using embedding + Transformer
    Processes pose keypoints sequence
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Embedding layers for different body parts
        self.left_hand_embed = nn.Linear(args.left_hand_dim, args.pose_feature_dim // 4)
        self.right_hand_embed = nn.Linear(args.right_hand_dim, args.pose_feature_dim // 4)
        self.lips_embed = nn.Linear(args.lips_dim, args.pose_feature_dim // 4)
        self.body_pose_embed = nn.Linear(args.body_pose_dim, args.pose_feature_dim // 4)
        
        # Combined projection
        self.pose_proj = nn.Linear(args.pose_feature_dim, args.pose_feature_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(args.pose_feature_dim)
        
        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.pose_feature_dim,
            nhead=args.num_heads,
            dim_feedforward=args.pose_feature_dim * 4,
            dropout=args.dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_transformer_layers
        )
        
        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.pose_feature_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, 180) pose keypoints
                - [0:63]: left hand
                - [63:126]: right hand
                - [126:156]: lips
                - [156:180]: body pose
                
        Returns:
            features: (B, T, D) temporal features
        """
        B, T, _ = x.shape
        
        # Split into different body parts
        left_hand = x[:, :, :63]  # (B, T, 63)
        right_hand = x[:, :, 63:126]  # (B, T, 63)
        lips = x[:, :, 126:156]  # (B, T, 30)
        body_pose = x[:, :, 156:180]  # (B, T, 24)
        
        # Embed each part
        left_hand_feat = self.left_hand_embed(left_hand)  # (B, T, D/4)
        right_hand_feat = self.right_hand_embed(right_hand)  # (B, T, D/4)
        lips_feat = self.lips_embed(lips)  # (B, T, D/4)
        body_pose_feat = self.body_pose_embed(body_pose)  # (B, T, D/4)
        
        # Concatenate all features
        combined_feat = torch.cat([
            left_hand_feat,
            right_hand_feat,
            lips_feat,
            body_pose_feat
        ], dim=-1)  # (B, T, D)
        
        # Project
        combined_feat = self.pose_proj(combined_feat)
        combined_feat = self.layer_norm(combined_feat)
        
        # Add positional encoding
        combined_feat = self.pos_encoding(combined_feat)
        
        # Apply temporal transformer
        temporal_features = self.temporal_transformer(combined_feat)  # (B, T, D)
        
        return self.dropout(temporal_features)


class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion using multi-head attention
    Allows RGB and Pose features to attend to each other
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Project RGB and Pose to same dimension
        self.rgb_proj = nn.Linear(args.rgb_feature_dim, args.fusion_dim)
        self.pose_proj = nn.Linear(args.pose_feature_dim, args.fusion_dim)
        
        # Cross-attention: RGB attends to Pose
        self.rgb_to_pose_attn = nn.MultiheadAttention(
            embed_dim=args.fusion_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            batch_first=True
        )
        
        # Cross-attention: Pose attends to RGB
        self.pose_to_rgb_attn = nn.MultiheadAttention(
            embed_dim=args.fusion_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            batch_first=True
        )
        
        # Self-attention on fused features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.fusion_dim * 2,  # Concatenated RGB and Pose
            nhead=args.num_heads,
            dim_feedforward=args.fusion_dim * 4,
            dropout=args.dropout,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_transformer_layers
        )
        
        self.layer_norm = nn.LayerNorm(args.fusion_dim * 2)
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self, rgb_features, pose_features):
        """
        Args:
            rgb_features: (B, T, D1) RGB features
            pose_features: (B, T, D2) Pose features
            
        Returns:
            fused_features: (B, T, D) fused features
        """
        # Project to same dimension
        rgb_feat = self.rgb_proj(rgb_features)  # (B, T, D)
        pose_feat = self.pose_proj(pose_features)  # (B, T, D)
        
        # Cross-attention
        rgb_attended, _ = self.rgb_to_pose_attn(
            query=rgb_feat,
            key=pose_feat,
            value=pose_feat
        )  # (B, T, D)
        
        pose_attended, _ = self.pose_to_rgb_attn(
            query=pose_feat,
            key=rgb_feat,
            value=rgb_feat
        )  # (B, T, D)
        
        # Concatenate attended features
        fused = torch.cat([rgb_attended, pose_attended], dim=-1)  # (B, T, 2D)
        fused = self.layer_norm(fused)
        
        # Self-attention on fused features
        fused = self.fusion_transformer(fused)  # (B, T, 2D)
        
        return self.dropout(fused)


class SignLanguageModel(nn.Module):
    """
    Complete two-stream multimodal model for Sign Language Recognition
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Encoders
        self.rgb_encoder = RGBEncoder(args)
        self.pose_encoder = PoseEncoder(args)
        
        # Fusion module
        self.fusion = CrossModalFusion(args)
        
        # Temporal pooling (attention-based)
        self.temporal_attention = nn.Sequential(
            nn.Linear(args.fusion_dim * 2, args.fusion_dim),
            nn.Tanh(),
            nn.Linear(args.fusion_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(args.fusion_dim * 2, args.fusion_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.fusion_dim, args.num_classes)
        )
    
    def forward(self, rgb, pose):
        """
        Args:
            rgb: (B, T, 3, H, W) RGB frames
            pose: (B, T, 180) pose keypoints
            
        Returns:
            logits: (B, num_classes) classification logits
        """
        # Encode RGB and Pose
        rgb_features = self.rgb_encoder(rgb)  # (B, T, D1)
        pose_features = self.pose_encoder(pose)  # (B, T, D2)
        
        # Fuse modalities
        fused_features = self.fusion(rgb_features, pose_features)  # (B, T, D)
        
        # Temporal pooling using attention
        B, T, D = fused_features.shape
        
        # Compute attention weights
        attn_weights = self.temporal_attention(fused_features)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, T, 1)
        
        # Weighted sum
        pooled_features = torch.sum(fused_features * attn_weights, dim=1)  # (B, D)
        
        # Classification
        logits = self.classifier(pooled_features)  # (B, num_classes)
        
        return logits
    
    def get_features(self, rgb, pose):
        """
        Extract features for analysis/visualization
        
        Returns:
            dict with rgb_features, pose_features, fused_features
        """
        with torch.no_grad():
            rgb_features = self.rgb_encoder(rgb)
            pose_features = self.pose_encoder(pose)
            fused_features = self.fusion(rgb_features, pose_features)
            
            return {
                'rgb_features': rgb_features,
                'pose_features': pose_features,
                'fused_features': fused_features
            }


if __name__ == "__main__":
    # Test model
    config = Config()
    config.NUM_CLASSES = 100
    
    print("Creating model...")
    model = SignLanguageModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    B, T = 2, 16
    rgb = torch.randn(B, T, 3, 224, 224)
    pose = torch.randn(B, T, 180)
    
    logits = model(rgb, pose)
    print(f"Input RGB shape: {rgb.shape}")
    print(f"Input Pose shape: {pose.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    print("\nModel test completed!")
