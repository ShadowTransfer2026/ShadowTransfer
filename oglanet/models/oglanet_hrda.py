"""
OGLANet-HRDA: OGLANet with HRDA Multi-Resolution Training
Wraps OGLANet with HRDA context/detail crop processing and scale attention fusion.

Based on HRDA (ECCV 2022): https://arxiv.org/abs/2204.13132

CRITICAL ASSUMPTIONS:
1. Scale attention uses feat5 (512 channels) from GLAM encoder
2. HRDA fusion applied only to P6 (final prediction)
3. Auxiliary predictions (P1-P5) computed only on context branch
4. OGLANet core architecture (GLAM, DFFM, OAM) remains intact
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path to import OGLANet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .scale_attention import ScaleAttentionHead, HRDAFusionModule


class OGLANetHRDA(nn.Module):
    """
    OGLANet with HRDA Multi-Resolution Training.
    
    Architecture:
    1. Shared OGLANet encoder/decoder for both LR context and HR detail
    2. Scale attention head to learn fusion weights (from LR context features)
    3. HRDA fusion module to combine predictions
    4. Auxiliary predictions (P1-P5) from context branch for deep supervision
    
    ASSUMPTION: HRDA fusion applied only to P6 (final OGLANet prediction).
    P1-P5 remain as auxiliary losses on context branch only, maintaining
    OGLANet's original deep supervision strategy while adding HRDA capabilities.
    
    Args:
        base_model: OGLANet base model
        num_classes: Number of classes
        scale_attention_embed_dim: Embedding dim for scale attention (default: 256)
        hr_loss_weight: Weight for HR detail loss (default: 0.1, λ_d in paper)
    """
    
    def __init__(self, base_model, num_classes=2, 
                 scale_attention_embed_dim=256, hr_loss_weight=0.1):
        super(OGLANetHRDA, self).__init__()
        
        self.num_classes = num_classes
        self.hr_loss_weight = hr_loss_weight
        
        # Base OGLANet (shared for LR and HR)
        self.base_model = base_model
        
        # Get encoder output channels for scale attention
        # GLAM encoder (ResNet-34 based) has feat5 with 512 channels
        encoder_channels = 1024
        
        # Scale attention head
        # ASSUMPTION: Use feat5 (deepest, most semantic features) from GLAM encoder
        self.scale_attention = ScaleAttentionHead(
            in_channels=encoder_channels,
            num_classes=num_classes,
            embed_dim=scale_attention_embed_dim
        )
        
        # HRDA fusion module
        self.fusion = HRDAFusionModule(num_classes=num_classes)
        
        print("OGLANet-HRDA initialized:")
        print(f"  Base model: OGLANet (GLAM encoder + DFFM + OAM)")
        print(f"  Scale attention embed dim: {scale_attention_embed_dim}")
        print(f"  HR loss weight: {hr_loss_weight}")
        print(f"  ASSUMPTION: Scale attention from feat5 ({encoder_channels} channels)")
        print(f"  ASSUMPTION: HRDA fusion on P6 only, P1-P5 as context auxiliary")
    
    def forward(self, image_context, image_detail, detail_coords):
        """
        Forward pass with HRDA multi-resolution processing.
        
        Args:
            image_context: LR context crop [B, 3, H_c, W_c]
            image_detail: HR detail crop [B, 3, H_d, W_d]
            detail_coords: List of detail crop coordinates in context
                          [(b1, b2, b3, b4), ...] for each batch
        
        Returns:
            Dictionary with:
            - 'pred_fused': Fused P6 prediction [B, C, H_HR, W_HR]
            - 'pred_detail': Detail P6 prediction [B, C, H_d, W_d]
            - 'pred_context': Context P6 prediction [B, C, H_c, W_c]
            - 'scale_attention': Attention weights [B, C, H_c, W_c]
            - 'aux_context': Auxiliary predictions from context (P1-P5)
        """
        
        # Forward through OGLANet for context crop (LR)
        # Ensure training mode to get all 6 predictions (P1-P6)
        base_model_was_training = self.base_model.training
        if not base_model_was_training:
            self.base_model.train()
        
        # Process context (LR)
        context_output = self.base_model(image_context)
        
        # Extract predictions
        # ASSUMPTION: OGLANet returns dict with 'p1'-'p6' in training mode
        if isinstance(context_output, dict):
            pred_context = context_output['p6']  # Final prediction
            # Auxiliary predictions (P1-P5) for context branch only
            aux_context = [context_output[f'p{i}'] for i in range(1, 6)]
        else:
            # Inference mode (single tensor)
            pred_context = context_output
            aux_context = None
        
        # Forward through OGLANet for detail crop (HR)
        detail_output = self.base_model(image_detail)
        
        # Extract detail prediction (P6 only)
        if isinstance(detail_output, dict):
            pred_detail = detail_output['p6']
            # No auxiliary outputs for detail (saves computation)
            aux_detail = None
        else:
            pred_detail = detail_output
            aux_detail = None
        
        # Restore original training state
        if not base_model_was_training:
            self.base_model.eval()
        
        # Get encoder features from context for scale attention
        # ASSUMPTION: Extract feat5 from GLAM encoder for scale attention
        with torch.no_grad():
            # Get encoder features (GLAM encoder)
            enc_features = self.base_model.encoder(image_context)
            # Use deepest features (feat5) - most semantic
            context_features = enc_features['feat5']  # [B, 512, H/32, W/32]
        
        # Predict scale attention from context features
        scale_attention = self.scale_attention(context_features)  # [B, C, H', W']
        
        # Upsample attention to match context prediction size
        if scale_attention.shape[-2:] != pred_context.shape[-2:]:
            scale_attention = F.interpolate(
                scale_attention,
                size=pred_context.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Fuse predictions using scale attention
        # ASSUMPTION: Fusion applied only to P6 (final OGLANet prediction)
        pred_fused = self.fusion(
            pred_context, 
            pred_detail, 
            scale_attention, 
            detail_coords
        )
        
        return {
            'pred_fused': pred_fused,
            'pred_detail': pred_detail,
            'pred_context': pred_context,
            'scale_attention': scale_attention,
            'aux_context': aux_context  # P1-P5 from context for deep supervision
        }
    
    def forward_single(self, image):
        """
        Forward pass for single image (inference mode).
        Uses base OGLANet without HRDA multi-resolution processing.
        
        Args:
            image: Input image [B, 3, H, W]
            
        Returns:
            Prediction [B, C, H, W]
        """
        # For inference, use base OGLANet directly
        output = self.base_model(image)
        
        # Return P6 (final prediction)
        if isinstance(output, dict):
            return output['p6']
        return output
    
    def get_predictions(self, image):
        """
        Get binary predictions for inference.
        
        Args:
            image: Input images [B, 3, H, W]
            
        Returns:
            Binary predictions [B, H, W] with values {0, 1}
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward_single(image)  # [B, 2, H, W]
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
        return preds


def create_oglanet_hrda(num_classes=2, pretrained=True, 
                        img_size=384, hr_loss_weight=0.1, use_contrast=False):
    """
    Factory function to create OGLANet-HRDA.
    
    Args:
        num_classes: Number of classes
        pretrained: Use pretrained ResNet-34 encoder
        img_size: Input image size
        hr_loss_weight: Weight for HR detail loss
        use_contrast: Use 4-channel input (RGB + Contrast)
        
    Returns:
        OGLANet-HRDA model
    """
    # Import OGLANet here to avoid circular imports
    from .oglanet import OGLANet
    
    # Create base OGLANet
    base_model = OGLANet(
        num_classes=num_classes,
        pretrained=pretrained,
        img_size=img_size,
        use_contrast=use_contrast
    )
    
    # Wrap with HRDA
    model = OGLANetHRDA(
        base_model=base_model,
        num_classes=num_classes,
        hr_loss_weight=hr_loss_weight
    )
    
    return model


if __name__ == "__main__":
    # Test OGLANet-HRDA
    print("Testing OGLANet-HRDA...")
    print("="*60)
    
    # Create model
    model = create_oglanet_hrda(num_classes=2, pretrained=False, img_size=384)
    
    # Test inputs
    batch_size = 2
    image_context = torch.randn(batch_size, 3, 192, 192)  # LR context (downsampled from 384)
    image_detail = torch.randn(batch_size, 3, 192, 192)   # HR detail
    detail_coords = [(24, 120, 24, 120), (30, 126, 30, 126)]  # Example coords
    
    # Forward pass
    model.eval()
    output = model(image_context, image_detail, detail_coords)
    
    print("\nOutput shapes:")
    print(f"  pred_fused: {output['pred_fused'].shape}")
    print(f"  pred_detail: {output['pred_detail'].shape}")
    print(f"  pred_context: {output['pred_context'].shape}")
    print(f"  scale_attention: {output['scale_attention'].shape}")
    if output['aux_context'] is not None:
        print(f"  aux_context: {len(output['aux_context'])} auxiliary predictions")
        for i, aux in enumerate(output['aux_context']):
            print(f"    P{i+1}: {aux.shape}")
    
    # Test single image forward (inference)
    print("\n" + "="*60)
    print("Testing inference mode...")
    image_single = torch.randn(1, 3, 384, 384)
    pred_single = model.forward_single(image_single)
    print(f"Single image prediction: {pred_single.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("="*60)
    print("OGLANet-HRDA test completed successfully!")