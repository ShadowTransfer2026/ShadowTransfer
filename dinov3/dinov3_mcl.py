"""
DINOv3 Model with Multi-level Contrastive Learning (mCL-LC)
Modified version that exposes intermediate features for contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_backbone import DINOv3Backbone
from dinov3_decoder import DINOv3Decoder


class ContrastiveProjector(nn.Module):
    """
    Projection head for contrastive learning.
    Maps features to embedding space for contrastive loss computation.
    """
    
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        """
        Args:
            in_dim: Input feature dimension (384 for DINOv3-S)
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
        """
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Project features to embedding space.
        
        Args:
            x: Input features [B, C, H, W]
        
        Returns:
            Projected embeddings [B, out_dim, H, W]
        """
        return self.projector(x)


class DINOv3MCL(nn.Module):
    """
    DINOv3 with Multi-level Contrastive Learning support.
    
    Architecture:
    - DINOv3 backbone (extracts features from blocks 3, 6, 9, 11)
    - Lightweight decoder
    - Projection heads for feature-level and semantic-level contrastive learning
    
    Returns both segmentation outputs and intermediate features for MCL-LC.
    """
    
    def __init__(self, num_classes=2, model_name='dinov3_vits16', weights_path=None,
                 pretrained=True, frozen_stages=-1,
                 feature_proj_dim=128, semantic_proj_dim=128):
        """
        Args:
            num_classes: Number of segmentation classes
            model_name: DINOv3 variant ('dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16')
            weights_path: Path to pretrained weights
            pretrained: Use pretrained weights
            frozen_stages: Number of backbone stages to freeze
            feature_proj_dim: Dimension for feature-level projections (default: 128)
            semantic_proj_dim: Dimension for semantic-level projections (default: 128)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Initialize backbone
        print('Initializing DINOv3 backbone...')
        self.backbone = DINOv3Backbone(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            frozen_stages=frozen_stages
        )
        
        # Get embedding dimension
        self.embed_dim = self.backbone.embed_dim
        
        # Initialize decoder
        print('Initializing decoder...')
        self.decoder = DINOv3Decoder(
            num_classes=num_classes,
            embed_dim=self.embed_dim
        )
        
        # Projection heads for contrastive learning
        # Feature-level: project from block 6 (mid-level features)
        self.feature_projector = ContrastiveProjector(
            in_dim=self.embed_dim,
            hidden_dim=256,
            out_dim=feature_proj_dim
        )
        
        # Semantic-level: project from block 11 (high-level semantic features)
        self.semantic_projector = ContrastiveProjector(
            in_dim=self.embed_dim,
            hidden_dim=256,
            out_dim=semantic_proj_dim
        )
        
        print(f'\nDINOv3-MCL initialized:')
        print(f'  Model: {model_name}')
        print(f'  Embed dim: {self.embed_dim}')
        print(f'  Feature proj dim: {feature_proj_dim}')
        print(f'  Semantic proj dim: {semantic_proj_dim}')
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'  Total parameters: {total_params:,}')
        print(f'  Trainable parameters: {trainable_params:,}')
    
    def forward(self, x, return_features=False):
        """
        Forward pass with optional feature extraction.
        
        Args:
            x: Input images [B, 3, H, W]
            return_features: If True, return intermediate features for contrastive learning
        
        Returns:
            If return_features=False:
                Segmentation logits [B, num_classes, H, W]
            If return_features=True:
                Tuple of (segmentation logits, feature dict)
        """
        B, C, H, W = x.shape
        
        # Extract features from backbone
        backbone_features = self.backbone(x)
        # backbone_features contains: feat_block3, feat_block6, feat_block9, feat_block11
        # All at H/16 x W/16 resolution
        
        # Decode to segmentation mask
        output = self.decoder(backbone_features)  # [B, num_classes, H, W]
        
        # Ensure output matches input size
        if output.shape[2] != H or output.shape[3] != W:
            output = F.interpolate(
                output,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        # Return features for contrastive learning if requested
        if return_features:
            # Project features for contrastive learning
            # Feature-level: use block 6 (mid-level features, similar to ResNet conv3)
            feature_proj = self.feature_projector(backbone_features['feat_block6'])
            
            # Semantic-level: use block 11 (high-level semantic features)
            semantic_proj = self.semantic_projector(backbone_features['feat_block11'])
            
            features = {
                'feature_embeddings': feature_proj,      # [B, 128, H/16, W/16]
                'semantic_embeddings': semantic_proj,    # [B, 128, H/16, W/16]
                'backbone_feat_block6': backbone_features['feat_block6'],  # For reference
                'backbone_feat_block11': backbone_features['feat_block11']  # For reference
            }
            
            return output, features
        else:
            return output
    
    def get_predictions(self, x):
        """
        Get binary predictions for inference.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Binary predictions [B, H, W]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, return_features=False)
            preds = torch.argmax(logits, dim=1)
        return preds
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning"""
        print('Unfreezing backbone...')
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        print('Freezing backbone...')
        for param in self.backbone.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    # Test DINOv3-MCL
    print("=" * 60)
    print("Testing DINOv3-MCL")
    print("=" * 60)
    
    try:
        model = DINOv3MCL(
            num_classes=2,
            model_name='dinov3_vits16',
            weights_path=None,
            pretrained=False,  # For testing
            frozen_stages=-1,
            feature_proj_dim=128,
            semantic_proj_dim=128
        )
        
        # Test without features
        x = torch.randn(2, 3, 384, 384)
        print(f"\nInput shape: {x.shape}")
        
        output = model(x, return_features=False)
        print(f"Segmentation output: {output.shape}")
        
        # Test with features
        output, features = model(x, return_features=True)
        print("\nWith features:")
        print(f"Segmentation output: {output.shape}")
        for key, val in features.items():
            print(f"  {key}: {val.shape}")
        
        print("\n" + "=" * 60)
        print("✓ Test passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")