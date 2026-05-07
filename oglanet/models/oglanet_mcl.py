"""
OGLANet with Multi-level Contrastive Learning (mCL-LC)
Modified version that exposes intermediate features for contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .glam import GLAMEncoder
from .dffm import DFFM
from .decoder import Decoder
from .oam import OAM


class ContrastiveProjector(nn.Module):
    """
    Projection head for contrastive learning.
    Maps features to embedding space for contrastive loss computation.
    """
    
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        )
    
    def forward(self, x):
        return self.projector(x)


class OGLANetMCL(nn.Module):
    """
    OGLANet with Multi-level Contrastive Learning support.
    
    Modifications:
    - Exposes intermediate encoder and decoder features for contrastive learning
    - Adds projection heads for feature and semantic embeddings
    - Returns both segmentation outputs and intermediate features
    """
    
    def __init__(self, num_classes=2, pretrained=True, img_size=384,
                 feature_proj_dim=128, semantic_proj_dim=128, use_contrast=False):
        """
        Args:
            num_classes: Number of segmentation classes
            pretrained: Use pretrained encoder
            img_size: Input image size
            feature_proj_dim: Dimension for feature-level projections
            semantic_proj_dim: Dimension for semantic-level projections
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_contrast = use_contrast

        # 1. GLAM Encoder
        self.encoder = GLAMEncoder(pretrained=pretrained, use_contrast=self.use_contrast)
        
        # 2. Dense Feature Fusion Module
        self.dffm = DFFM()
        
        # 3. Decoder
        self.decoder = Decoder(target_size=(img_size, img_size))
        
        # 4. Omni-scale Aggregation Module
        self.oam = OAM(num_classes=num_classes, target_size=(img_size, img_size))
        
        # Projection heads for contrastive learning
        # Feature-level: project from encoder feat3 (256 channels, mid-level)
        self.feature_projector = ContrastiveProjector(
            in_dim=256,
            hidden_dim=256,
            out_dim=feature_proj_dim
        )
        
        # Semantic-level: project from decoder features (high-level semantic)
        # Using s1_d_up which has 64 channels after upsampling
        self.semantic_projector = ContrastiveProjector(
            in_dim=64,
            hidden_dim=128,
            out_dim=semantic_proj_dim
        )
    
    def forward(self, x, return_features=False):
        """
        Forward pass with optional feature extraction.
        
        Args:
            x: Input images [B, 3, H, W]
            return_features: If True, return intermediate features for contrastive learning
        
        Returns:
            If return_features=False:
                Segmentation outputs (dict with p1-p6 or just p6)
            If return_features=True:
                Tuple of (segmentation outputs, feature dict)
        """
        B, _, H, W = x.size()
        
        # 1. Encoder: Extract multi-scale features with GLAM
        encoder_features = self.encoder(x)
        # {'feat1', 'feat2', 'feat3', 'feat4', 'feat5'}
        
        # 2. DFFM: Dense feature fusion
        dffm_features = self.dffm(encoder_features)
        # {'s4_d', 's3_d', 's2_d', 's1_d'}
        
        # 3. Decoder: Upsample to original resolution
        decoder_features = self.decoder(dffm_features)
        # {'s4_d_up', 's3_d_up', 's2_d_up', 's1_d_up'}
        
        # 4. OAM: Multi-scale prediction
        predictions = self.oam(decoder_features)
        # {'p1', 'p2', 'p3', 'p4', 'p5', 'p6'}
        
        # Prepare segmentation output
        if self.training:
            seg_output = predictions  # All 6 predictions for deep supervision
        else:
            seg_output = predictions['p6']  # Only final prediction for inference
        
        # Return features for contrastive learning if requested
        if return_features:
            # Project features for contrastive learning
            # Feature-level: use encoder feat3 (mid-level features, 256 channels)
            feature_proj = self.feature_projector(encoder_features['feat3'])
            
            # Semantic-level: use decoder s1_d_up (high-level features, 64 channels)
            semantic_proj = self.semantic_projector(decoder_features['s1_d_up'])
            
            features = {
                'feature_embeddings': feature_proj,
                'semantic_embeddings': semantic_proj,
                'encoder_feat3': encoder_features['feat3'],
                'decoder_s1_d_up': decoder_features['s1_d_up']
            }
            
            return seg_output, features
        else:
            return seg_output
    
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
            outputs = self.forward(x, return_features=False)
            if isinstance(outputs, dict):
                logits = outputs['p6']
            else:
                logits = outputs
            preds = torch.argmax(logits, dim=1)
        return preds


if __name__ == "__main__":
    # Test OGLANetMCL
    model = OGLANetMCL(
        num_classes=2,
        pretrained=False,
        img_size=384,
        feature_proj_dim=128,
        semantic_proj_dim=128
    )
    
    # Test without features
    model.eval()
    x = torch.randn(2, 3, 384, 384)
    outputs = model(x, return_features=False)
    print("Segmentation outputs (inference):")
    print(f"  output: {outputs.shape}")
    
    # Test with features
    model.train()
    outputs, features = model(x, return_features=True)
    print("\nWith features (training):")
    print("Segmentation outputs:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    print("Features:")
    for key, val in features.items():
        print(f"  {key}: {val.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")