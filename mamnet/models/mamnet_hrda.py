"""
MAMNet-HRDA: MAMNet with HRDA Multi-Resolution Training
Wraps MAMNet with HRDA context/detail crop processing and scale attention fusion.

Based on HRDA (ECCV 2022): https://arxiv.org/abs/2204.13132
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path to import MAMNet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .scale_attention import ScaleAttentionHead, HRDAFusionModule


class MAMNetHRDA(nn.Module):
    """
    MAMNet with HRDA Multi-Resolution Training.
    
    Architecture:
    1. Shared MAMNet encoder/decoder for both LR context and HR detail
    2. Scale attention head to learn fusion weights
    3. HRDA fusion module to combine predictions
    
    Args:
        base_model: MAMNet base model
        num_classes: Number of classes
        scale_attention_embed_dim: Embedding dim for scale attention (default: 256)
        hr_loss_weight: Weight for HR detail loss (default: 0.1, λ_d in paper)
    """
    
    def __init__(self, base_model, num_classes=2, 
                 scale_attention_embed_dim=256, hr_loss_weight=0.1):
        super(MAMNetHRDA, self).__init__()
        
        self.num_classes = num_classes
        self.hr_loss_weight = hr_loss_weight
        
        # Base MAMNet (shared for LR and HR)
        self.base_model = base_model
        
        # Get encoder output channels for scale attention
        # MAMNet uses ResNet34 encoder, deepest feature has 512 channels
        encoder_channels = 512
        
        # Scale attention head
        self.scale_attention = ScaleAttentionHead(
            in_channels=encoder_channels,
            num_classes=num_classes,
            embed_dim=scale_attention_embed_dim
        )
        
        # HRDA fusion module
        self.fusion = HRDAFusionModule(num_classes=num_classes)
        
        print("MAMNet-HRDA initialized:")
        print(f"  Base model: MAMNet")
        print(f"  Scale attention embed dim: {scale_attention_embed_dim}")
        print(f"  HR loss weight: {hr_loss_weight}")
    
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
            - 'pred_fused': Fused prediction [B, C, H_HR, W_HR]
            - 'pred_detail': Detail prediction [B, C, H_d, W_d]
            - 'pred_context': Context prediction [B, C, H_c, W_c]
            - 'scale_attention': Attention weights [B, C, H_c, W_c]
        """
        
        # Forward through MAMNet for context crop (LR)
        # Only force training mode if we're actually training (not during eval/test)
        base_model_was_training = self.base_model.training
        needs_aux = self.training  # Only need aux outputs during actual training

        if needs_aux and not base_model_was_training:
            self.base_model.train()

        # Forward through MAMNet for context crop (LR)
        context_output = self.base_model(image_context)

        if isinstance(context_output, dict):
            pred_context = context_output['main']
            # Extract aux1, aux2, aux3 into a list
            aux_context = [context_output.get('aux1'), context_output.get('aux2'), context_output.get('aux3')]
            # Filter out None values
            aux_context = [aux for aux in aux_context if aux is not None]
            if len(aux_context) == 0:
                aux_context = None
        else:
            pred_context = context_output
            aux_context = None

        # Forward through MAMNet for detail crop (HR)
        detail_output = self.base_model(image_detail)

        if isinstance(detail_output, dict):
            pred_detail = detail_output['main']
            # Extract aux1, aux2, aux3 into a list
            aux_detail = [detail_output.get('aux1'), detail_output.get('aux2'), detail_output.get('aux3')]
            # Filter out None values
            aux_detail = [aux for aux in aux_detail if aux is not None]
            if len(aux_detail) == 0:
                aux_detail = None
        else:
            pred_detail = detail_output
            aux_detail = None

        # Restore original training state
        if needs_aux and not base_model_was_training:
            self.base_model.eval()
        
        # Get encoder features from context for scale attention
        # We need to extract features from the encoder
        # For MAMNet, we can get features by hooking into encoder
        with torch.no_grad():
            enc_features = self.base_model.encoder(image_context)
            # Get deepest features (feat4 from ResNet34)
            context_features = enc_features['feat4']  # [B, 512, H/16, W/16]
        
        # Predict scale attention from context features
        scale_attention = self.scale_attention(context_features)  # [B, C, H/8, W/8]
        
        # Upsample attention to match context prediction size
        if scale_attention.shape[-2:] != pred_context.shape[-2:]:
            scale_attention = F.interpolate(
                scale_attention,
                size=pred_context.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Fuse predictions using scale attention
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
            'aux_context': aux_context,  # NEW
            'aux_detail': aux_detail
        }
    
    def forward_single(self, image):
        """
        Forward pass for single image (inference mode).
        Uses overlapping sliding window for HR prediction.
        
        Args:
            image: Input image [B, 3, H, W]
            
        Returns:
            Prediction [B, C, H, W]
        """
        # For inference, we can either:
        # 1. Use only HR prediction (simple)
        # 2. Use HRDA sliding window (complex, better quality)
        
        # For now, use simple HR prediction
        output = self.base_model(image)
        
        if isinstance(output, dict):
            return output['main']
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


def create_mamnet_hrda(num_classes=2, pretrained=True, 
                       use_aux=True, hr_loss_weight=0.1, use_contrast=False):
    """
    Factory function to create MAMNet-HRDA.
    
    Args:
        num_classes: Number of classes
        pretrained: Use pretrained ResNet encoder
        use_aux: Use auxiliary branches (for MAMNet)
        hr_loss_weight: Weight for HR detail loss
        
    Returns:
        MAMNet-HRDA model
    """
    # Import MAMNet here to avoid circular imports
    from .mamnet import MAMNet
    
    # Create base MAMNet
    base_model = MAMNet(
        num_classes=num_classes,
        pretrained=pretrained,
        use_aux=use_aux,
        use_contrast=use_contrast
    )
    
    # Wrap with HRDA
    model = MAMNetHRDA(
        base_model=base_model,
        num_classes=num_classes,
        hr_loss_weight=hr_loss_weight
    )
    
    return model


if __name__ == "__main__":
    # Test MAMNet-HRDA
    print("Testing MAMNet-HRDA...")
    
    # Create model
    model = create_mamnet_hrda(num_classes=2, pretrained=False)
    
    # Test inputs
    batch_size = 2
    image_context = torch.randn(batch_size, 3, 192, 192)  # LR context (downsampled from 384)
    image_detail = torch.randn(batch_size, 3, 192, 192)   # HR detail
    detail_coords = [(24, 120, 24, 120), (30, 126, 30, 126)]  # Example coords
    
    # Forward pass
    model.eval()
    output = model(image_context, image_detail, detail_coords)
    
    print("\nOutput shapes:")
    for key, val in output.items():
        print(f"  {key}: {val.shape}")
    
    # Test single image forward (inference)
    image_single = torch.randn(1, 3, 384, 384)
    pred_single = model.forward_single(image_single)
    print(f"\nSingle image prediction: {pred_single.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")