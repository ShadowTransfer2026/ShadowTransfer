"""
MAMNet with Multi-level Contrastive Learning (mCL-LC)
Modified version that exposes intermediate features for contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ResNet34Encoder
from .encoder_4ch import ResNet34Encoder4Ch
from .mscaf import MSCAF
from .decoder import Decoder
from .auxiliary import AuxiliaryModule


class ContrastiveProjector(nn.Module):
    """
    Projection head for contrastive learning.
    Maps features to embedding space for contrastive loss computation.
    """
    
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        """
        Args:
            in_dim: Input feature dimension
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


class MAMNetMCL(nn.Module):
    """
    MAMNet with Multi-level Contrastive Learning support.
    
    Modifications:
    - Exposes intermediate encoder features for contrastive learning
    - Adds projection heads for feature and semantic embeddings
    - Returns both segmentation outputs and intermediate features
    """
    
    def __init__(self, num_classes=2, pretrained=True, use_aux=True,
                 feature_proj_dim=128, semantic_proj_dim=128, use_contrast=False):
        """
        Args:
            num_classes: Number of segmentation classes
            pretrained: Use pretrained encoder
            use_aux: Use auxiliary branches
            feature_proj_dim: Dimension for feature-level projections
            semantic_proj_dim: Dimension for semantic-level projections
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_aux = use_aux
        self.use_contrast = use_contrast 
        
        # Encoder
        if use_contrast:
            self.encoder = ResNet34Encoder4Ch(pretrained=pretrained)
            print("Using 4-channel encoder (RGB + Contrast)")
        else:
            self.encoder = ResNet34Encoder(pretrained=pretrained)
        
        # MSCAF
        self.mscaf = MSCAF(in_channels=512)
        
        # Decoder
        self.decoder = Decoder(num_classes=num_classes)
        
        # Auxiliary branches
        if use_aux:
            self.aux_module = AuxiliaryModule(num_classes=num_classes, dropout_rate=0.3)
        
        # Projection heads for contrastive learning
        # Feature-level: project from encoder features (e.g., feat3: 256 channels)
        self.feature_projector = ContrastiveProjector(
            in_dim=256,
            hidden_dim=256,
            out_dim=feature_proj_dim
        )
        
        # Semantic-level: project from decoder features
        self.semantic_projector = ContrastiveProjector(
            in_dim=256,  # decoder features
            hidden_dim=256,
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
                Segmentation outputs (dict or tensor)
            If return_features=True:
                Tuple of (segmentation outputs, feature dict)
        """
        B, _, H, W = x.size()
        
        # Encoder
        enc_features = self.encoder(x)
        
        # MSCAF
        mscaf_out = self.mscaf(enc_features['feat5'])
        
        # Decoder
        decoder_outputs = self.decoder(mscaf_out, enc_features)
        main_out = decoder_outputs['main']
        
        # Prepare segmentation output
        seg_output = {'main': main_out}
        
        # Auxiliary branches (training only)
        if self.use_aux and self.training:
            aux_outputs = self.aux_module(
                decoder_outputs['dec_feat1'],
                decoder_outputs['dec_feat2'],
                decoder_outputs['dec_feat3'],
                target_size=(H, W)
            )
            seg_output.update(aux_outputs)
        
        # Return features for contrastive learning if requested
        if return_features:
            # Project features for contrastive learning
            # Feature-level: use encoder feat3 (mid-level features)
            feature_proj = self.feature_projector(enc_features['feat3'])  # [B, feature_proj_dim, H/8, W/8]
            
            # Semantic-level: use decoder feat1 (high-level features)
            semantic_proj = self.semantic_projector(decoder_outputs['dec_feat1'])  # [B, semantic_proj_dim, H/8, W/8]
            
            features = {
                'feature_embeddings': feature_proj,
                'semantic_embeddings': semantic_proj,
                'encoder_feat3': enc_features['feat3'],
                'decoder_feat1': decoder_outputs['dec_feat1']
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
                logits = outputs['main']
            else:
                logits = outputs
            preds = torch.argmax(logits, dim=1)
        return preds


if __name__ == "__main__":
    # Test MAMNetMCL
    model = MAMNetMCL(
        num_classes=2,
        pretrained=False,
        use_aux=True,
        feature_proj_dim=128,
        semantic_proj_dim=128
    )
    
    # Test without features
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    outputs = model(x, return_features=False)
    print("Segmentation outputs:")
    if isinstance(outputs, dict):
        for key, val in outputs.items():
            print(f"  {key}: {val.shape}")
    else:
        print(f"  output: {outputs.shape}")
    
    # Test with features
    model.train()
    outputs, features = model(x, return_features=True)
    print("\nWith features:")
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