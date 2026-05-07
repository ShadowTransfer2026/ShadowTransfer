"""
OGLANet with GSDPE: Ground Sample Distance Positional Encoding

Integrates Scale-MAE's GSDPE into OGLANet for cross-resolution transfer learning.

Reference:
- OGLANet: Xie et al. (2022) ISPRS
- GSDPE: Reed et al. (2023) ICCV
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gsdpe import GSDPE
from models.glam import GLAMEncoder
from models.dffm import DFFM
from models.decoder import Decoder
from models.oam import OAM


class OGLANet_GSDPE(nn.Module):
    """
    OGLANet with GSDPE for scale-aware shadow detection.
    
    Architecture:
    1. GSDPE applied at encoder input (feat1, 64 channels)
    2. GLAMEncoder (5 stages with global-local modules)
    3. DFFM (Dense Feature Fusion)
    4. Decoder (Progressive upsampling)
    5. OAM (Omni-scale Aggregation with 6 predictions)
    
    Args:
        num_classes: Number of classes (default: 2)
        pretrained: Use pretrained encoder (default: True)
        img_size: Input image size (default: 384)
        reference_gsd: Reference GSD for GSDPE (default: 1.0m)
    """
    
    def __init__(self, num_classes=2, pretrained=True, img_size=384, reference_gsd=1.0, use_contrast=False):
        super(OGLANet_GSDPE, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.reference_gsd = reference_gsd
        
        # 1. GLAM Encoder
        self.use_contrast = use_contrast
        self.encoder = GLAMEncoder(pretrained=pretrained, use_contrast=use_contrast)
        
        # 2. GSDPE: Applied to last encoder features (64 channels)
        # This is where Scale-MAE adds GSDPE (at encoder input)
        self.gsdpe = GSDPE(
            channels=64,  # feat1 has 64 channels
            reference_gsd=reference_gsd
        )
        
        # 3. Dense Feature Fusion Module
        self.dffm = DFFM()
        
        # 4. Decoder
        self.decoder = Decoder(target_size=(img_size, img_size))
        
        # 5. Omni-scale Aggregation Module
        self.oam = OAM(num_classes=num_classes, target_size=(img_size, img_size))
        
    def forward(self, x, gsd):
        """
        Forward pass with GSD conditioning.
        
        Args:
            x: Input RGB images [B, 3, H, W]
            gsd: GSD values [B] or [B, 1]
            
        Returns:
            If training: Dict with 6 predictions {'p1', ..., 'p6'}
            If inference: Final prediction P6 [B, num_classes, H, W]
        """
        B, _, H, W = x.size()
        
        # 1. Encoder: Extract multi-scale features
        enc_features = self.encoder(x)
        # Returns: {'feat1', 'feat2', 'feat3', 'feat4', 'feat5'}
        
        # 2. Apply GSDPE to first encoder features
        # This conditions the entire network on input GSD
        enc_features['feat1'] = self.gsdpe(enc_features['feat1'], gsd)
        
        # 3. DFFM: Dense feature fusion
        dffm_features = self.dffm(enc_features)
        
        # 4. Decoder: Upsample to original resolution
        decoder_features = self.decoder(dffm_features)
        
        # 5. OAM: Multi-scale prediction
        predictions = self.oam(decoder_features)
        
        if self.training:
            # Return all 6 predictions for deep supervision
            return predictions
        else:
            # Return only final prediction P6
            return predictions['p6']
    
    def get_predictions(self, x, gsd):
        """
        Get binary predictions.
        
        Args:
            x: Input images [B, 3, H, W]
            gsd: GSD values [B]
            
        Returns:
            Binary predictions [B, H, W]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, gsd)
            preds = torch.argmax(logits, dim=1)
        return preds


if __name__ == "__main__":
    print("Testing OGLANet with GSDPE")
    print("=" * 50)
    
    model = OGLANet_GSDPE(num_classes=2, pretrained=False, img_size=384, reference_gsd=1.0)
    
    # Test with different GSDs
    batch_size = 2
    x = torch.randn(batch_size, 3, 384, 384)
    gsd = torch.tensor([0.6, 0.3])  # Mix of midres and highres
    
    print(f"Input shape: {x.shape}")
    print(f"GSD values: {gsd.tolist()}")
    
    # Training mode
    model.train()
    outputs_train = model(x, gsd)
    print("\nTraining mode outputs:")
    for key, val in outputs_train.items():
        print(f"  {key}: {val.shape}")
    
    # Inference mode
    model.eval()
    outputs_eval = model(x, gsd)
    print(f"\nInference mode output: {outputs_eval.shape}")
    
    # Get predictions
    preds = model.get_predictions(x, gsd)
    print(f"Binary predictions: {preds.shape}, unique: {torch.unique(preds)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 50)
    print("OGLANet with GSDPE test completed!")