"""
DINOv3 Model with GSDPE for Shadow Detection
Integrates Ground Sample Distance Positional Encoding into DINOv3 for 
cross-resolution transfer learning.

Architecture:
1. DINOv3-S (ViT-S/16) pretrained backbone
2. GSDPE applied to patch embeddings (original Scale-MAE formula)
3. Lightweight progressive upsampling decoder
4. Single output (no auxiliary branches)

Key advantage: ViT architecture allows using original GSDPE formula directly
(unlike CNN-based MAMNet which required adaptation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsdpe import GSDPE
from dinov3_backbone import DINOv3Backbone
from dinov3_decoder import DINOv3Decoder


class DINOv3BackboneWithGSDPE(nn.Module):
    """
    DINOv3 backbone with GSDPE applied to patch embeddings.
    
    This wrapper adds GSD-aware positional encoding to the DINOv3 model
    before features are processed by the transformer blocks.
    
    Instead of reimplementing DINOv3 loading, we use the existing
    DINOv3Backbone and add GSDPE on top of it.
    """
    
    def __init__(self, model_name='dinov3_vits16', weights_path=None, 
                 pretrained=True, frozen_stages=-1, reference_gsd=1.0):
        super(DINOv3BackboneWithGSDPE, self).__init__()
        
        self.model_name = model_name
        self.reference_gsd = reference_gsd
        
        # Load existing DINOv3 backbone
        print(f'Loading DINOv3 backbone with existing implementation...')
        self.backbone = DINOv3Backbone(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            frozen_stages=frozen_stages
        )
        
        # Get model dimensions from backbone
        self.embed_dim = self.backbone.embed_dim  # 384 for ViT-S
        self.patch_size = self.backbone.patch_size  # 16
        
        # Calculate number of patches for 384x384 input
        self.num_patches_h = 384 // self.patch_size  # 24
        self.num_patches_w = 384 // self.patch_size  # 24
        
        # Initialize GSDPE
        print(f'Initializing GSDPE module...')
        self.gsdpe = GSDPE(
            embed_dim=self.embed_dim,
            reference_gsd=reference_gsd,
            num_patches_h=self.num_patches_h,
            num_patches_w=self.num_patches_w
        )
        
        print(f'DINOv3 Backbone with GSDPE initialized:')
        print(f'  Model: {model_name}')
        print(f'  Embed dim: {self.embed_dim}')
        print(f'  Patch size: {self.patch_size}')
        print(f'  Patches: {self.num_patches_h}x{self.num_patches_w}')
        print(f'  Reference GSD: {reference_gsd}m')
    
    def forward(self, x, gsd=None):
        """
        Forward pass with GSD conditioning.
        
        Strategy: We need to intercept the backbone's forward pass to inject GSDPE
        after patch embedding but before transformer blocks.
        
        Since we can't easily modify the internal forward pass, we'll use a workaround:
        1. Get patch embeddings manually
        2. Apply GSDPE
        3. Manually run through transformer blocks
        
        Args:
            x: Input RGB images [B, 3, 384, 384]
            gsd: GSD values [B] or [B, 1] or None (uses default 0.6m if None)
        
        Returns:
            Segmentation logits [B, num_classes, 384, 384]
        """
        B, C, H, W = x.shape
        
        # Use default GSD if not provided (for inference/visualization)
        if gsd is None:
            gsd = torch.tensor([0.6] * B, device=x.device)
            
        # Access the underlying DINOv3 model
        dinov3 = self.backbone.dinov3
        
        # Step 1: Get patch embeddings
        x_patches = dinov3.patch_embed(x)  # [B, N, D]
        
        # Step 2: Apply GSDPE to patch embeddings
        x_patches = self.gsdpe(x_patches, gsd)  # [B, H, W, D] or [B, N, D]

        # Flatten to [B, N, D] if needed
        if x_patches.dim() == 4:
            B_p, H_p, W_p, D_p = x_patches.shape
            x_patches = x_patches.reshape(B_p, H_p * W_p, D_p)
        
        # Step 3: Add class token
        if hasattr(dinov3, 'cls_token'):
            cls_tokens = dinov3.cls_token.expand(B, -1, -1)
            x_patches = torch.cat([cls_tokens, x_patches], dim=1)
        
        # Step 4: Add standard positional encoding (DINOv3 also uses this)
        if hasattr(dinov3, 'pos_embed'):
            x_patches = x_patches + dinov3.pos_embed
        
        # Step 5: Apply transformer blocks and extract features
        features_dict = {}
        
        for i, block in enumerate(dinov3.blocks):
            x_patches = block(x_patches)
            
            # Extract features at specified blocks
            if i in self.backbone.feature_blocks:
                # Remove class token and reshape to spatial format
                feat = x_patches[:, 1:, :]  # Remove CLS token [B, N, D]
                B_feat, N, D = feat.shape
                H_feat = W_feat = int(N ** 0.5)
                feat = feat.transpose(1, 2).reshape(B_feat, D, H_feat, W_feat)
                features_dict[f'feat_block{i}'] = feat
        
        return features_dict
    
    def get_feature_dims(self):
        """Return the dimensions of extracted features"""
        return self.backbone.get_feature_dims()


class DINOv3ShadowDetectorGSDPE(nn.Module):
    """
    DINOv3-based Shadow Detection Network with GSDPE
    
    Architecture:
    1. DINOv3-S (ViT-S/16) pretrained backbone with GSDPE
    2. Lightweight progressive upsampling decoder
    3. Single output (no auxiliary branches)
    
    Key features:
    - Input: 384×384 images (24×24 patches)
    - GSDPE: Applied to patch embeddings using original Scale-MAE formula
    - Output: 384×384 segmentation masks
    - ~22M parameters (comparable to ResNet-34)
    """
    
    def __init__(self, num_classes=2, model_name='dinov3_vits16', weights_path=None, 
                 pretrained=True, frozen_stages=-1, reference_gsd=1.0):
        """
        Args:
            num_classes: Number of output classes (default: 2 for binary shadow detection)
            model_name: DINOv3 variant (dinov3_vits16, dinov3_vitb16, dinov3_vitl16)
            weights_path: Path to pretrained weights .pth file
            pretrained: Load pretrained DINOv3 weights
            frozen_stages: Number of backbone stages to freeze (-1 = train all)
            reference_gsd: Reference GSD for GSDPE (default: 1.0m)
        """
        super(DINOv3ShadowDetectorGSDPE, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.reference_gsd = reference_gsd
        
        # Initialize backbone with GSDPE
        print('Initializing DINOv3 backbone with GSDPE...')
        self.backbone = DINOv3BackboneWithGSDPE(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            frozen_stages=frozen_stages,
            reference_gsd=reference_gsd
        )
        
        # Get embedding dimension
        embed_dim = self.backbone.embed_dim
        
        # Initialize decoder (reuse existing decoder)
        print('Initializing decoder...')
        self.decoder = DINOv3Decoder(
            num_classes=num_classes,
            embed_dim=embed_dim
        )
        
        print(f'\nDINOv3 Shadow Detector with GSDPE initialized:')
        print(f'  Model: {model_name}')
        print(f'  Input size: 384×384 (24×24 patches)')
        print(f'  Output classes: {num_classes}')
        print(f'  Reference GSD: {reference_gsd}m')
        print(f'  GSDPE: Original Scale-MAE formula for ViT')
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        gsdpe_params = sum(p.numel() for p in self.backbone.gsdpe.parameters())
        print(f'  Total parameters: {total_params:,}')
        print(f'  Trainable parameters: {trainable_params:,}')
        print(f'  GSDPE parameters: {gsdpe_params:,}')
    
    def forward(self, x, gsd=None):
        """
        Forward pass with GSD conditioning.
        
        Args:
            x: Input RGB images [B, 3, 384, 384]
            gsd: GSD values [B] or [B, 1]
        
        Returns:
            Segmentation logits [B, num_classes, 384, 384]
        """
        B, C, H, W = x.shape

        # Use default GSD if not provided
        if gsd is None:
            gsd = torch.tensor([0.6] * B, device=x.device)
        
        # Verify input size is compatible with patch size 16
        if H % 16 != 0 or W % 16 != 0:
            raise ValueError(
                f"Input size ({H}×{W}) must be divisible by patch size (16). "
                f"Expected 384×384 or other multiples of 16."
            )
        
        # Extract features from backbone with GSDPE
        features = self.backbone(x, gsd)  # Features at 1/16 resolution
        
        # Decode to segmentation mask
        output = self.decoder(features)  # [B, num_classes, H, W]
        
        # Ensure output matches input size
        if output.shape[2] != H or output.shape[3] != W:
            output = F.interpolate(
                output,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        return output
    
    def get_predictions(self, x, gsd):
        """
        Get binary predictions (for inference/evaluation).
        
        Args:
            x: Input images [B, 3, H, W]
            gsd: GSD values [B]
        
        Returns:
            Binary predictions [B, H, W] with values {0, 1}
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, gsd)  # [B, 2, H, W]
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
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
    # Test complete model
    print("=" * 60)
    print("Testing DINOv3 Shadow Detector with GSDPE")
    print("=" * 60)
    
    try:
        # Initialize model
        model = DINOv3ShadowDetectorGSDPE(
            num_classes=2,
            model_name='dinov3_vits16',
            weights_path=None,
            pretrained=False,  # Set to False for testing without weights
            frozen_stages=-1,
            reference_gsd=1.0
        )
        
        print("\n" + "=" * 60)
        print("Testing forward pass...")
        print("=" * 60)
        
        # Test with 384x384 input
        batch_size = 2
        x = torch.randn(batch_size, 3, 384, 384)
        gsd = torch.tensor([0.6, 0.3])  # Mix of midres and highres
        
        print(f"\nInput shape: {x.shape}")
        print(f"GSD values: {gsd.tolist()}")
        
        # Forward pass
        model.train()
        output = model(x, gsd)
        print(f"Output shape: {output.shape}")
        print(f"✓ Output matches input size!")
        
        # Test inference mode
        model.eval()
        preds = model.get_predictions(x, gsd)
        print(f"\nBinary predictions shape: {preds.shape}")
        print(f"Unique prediction values: {torch.unique(preds)}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()