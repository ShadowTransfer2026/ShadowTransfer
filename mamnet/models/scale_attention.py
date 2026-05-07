"""
Scale Attention Module for HRDA
Learns to fuse predictions from LR context and HR detail crops.

Based on HRDA (ECCV 2022): https://arxiv.org/abs/2204.13132
The scale attention predicts per-pixel weights to combine LR and HR predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAttentionHead(nn.Module):
    """
    Scale Attention Decoder for HRDA.
    Predicts attention weights to fuse LR context and HR detail predictions.
    
    Architecture: Lightweight MLP decoder similar to SegFormer
    Input: Features from encoder
    Output: Per-class attention weights in [0, 1] (1 = focus on HR detail)
    
    Args:
        in_channels: Number of input channels from encoder
        num_classes: Number of segmentation classes
        embed_dim: Embedding dimension for MLP (default: 256)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_channels, num_classes, embed_dim=256, dropout=0.1):
        super(ScaleAttentionHead, self).__init__()
        
        self.num_classes = num_classes
        
        # MLP layers for scale attention
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        self.attention_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: Encoder features [B, C, H, W]
            
        Returns:
            attention: Scale attention weights [B, num_classes, H, W] in [0, 1]
                      1 = focus on HR detail, 0 = focus on LR context
        """
        # Fuse features
        x = self.linear_fuse(features)  # [B, embed_dim, H, W]
        
        # Predict attention weights
        attention = self.attention_head(x)  # [B, num_classes, H, W]
        
        # Apply sigmoid to get weights in [0, 1]
        attention = torch.sigmoid(attention)
        
        return attention


class HRDAFusionModule(nn.Module):
    """
    HRDA Fusion Module - combines LR context and HR detail predictions
    using learned scale attention.
    
    Args:
        num_classes: Number of segmentation classes
    """
    
    def __init__(self, num_classes):
        super(HRDAFusionModule, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, pred_context, pred_detail, attention, detail_crop_coords):
        """
        Fuses LR context and HR detail predictions using scale attention.
        
        Args:
            pred_context: LR context prediction [B, C, H_c, W_c]
            pred_detail: HR detail prediction [B, C, H_d, W_d]
            attention: Scale attention from context [B, C, H_c, W_c]
            detail_crop_coords: Coordinates of detail crop in context
                               [(b1, b2, b3, b4), ...] for each batch item
        
        Returns:
            fused_pred: Fused prediction at HR resolution [B, C, H_HR, W_HR]
        """
        B, C, H_c, W_c = pred_context.shape
        _, _, H_d, W_d = pred_detail.shape
        
        # Determine scale factor
        # Context is at 0.5x resolution, need to upsample to HR (2x)
        scale_factor = 2
        H_HR = H_c * scale_factor
        W_HR = W_c * scale_factor
        
        # Upsample context prediction to HR resolution
        pred_context_hr = F.interpolate(
            pred_context, 
            size=(H_HR, W_HR), 
            mode='bilinear', 
            align_corners=False
        )  # [B, C, H_HR, W_HR]
        
        # Upsample scale attention to HR resolution
        attention_hr = F.interpolate(
            attention,
            size=(H_HR, W_HR),
            mode='bilinear',
            align_corners=False
        )  # [B, C, H_HR, W_HR]
        
        # Create masked attention (only where detail crop exists)
        # Initialize attention mask with zeros (all context)
        attention_masked = torch.zeros_like(attention_hr)  # [B, C, H_HR, W_HR]
        
        # For each batch item, fill in the detail crop region
        for i in range(B):
            if detail_crop_coords is not None and i < len(detail_crop_coords):
                b1, b2, b3, b4 = detail_crop_coords[i]
                
                # Map coordinates from context resolution to HR resolution
                b1_hr = b1 * scale_factor
                b2_hr = b2 * scale_factor
                b3_hr = b3 * scale_factor
                b4_hr = b4 * scale_factor
                
                # Ensure coordinates are within bounds
                b1_hr = max(0, min(b1_hr, H_HR))
                b2_hr = max(0, min(b2_hr, H_HR))
                b3_hr = max(0, min(b3_hr, W_HR))
                b4_hr = max(0, min(b4_hr, W_HR))
                
                # Set attention in detail crop region
                attention_masked[i, :, b1_hr:b2_hr, b3_hr:b4_hr] = \
                    attention_hr[i, :, b1_hr:b2_hr, b3_hr:b4_hr]
        
        # Pad and align detail prediction
        pred_detail_aligned = torch.zeros_like(pred_context_hr)  # [B, C, H_HR, W_HR]
        
        for i in range(B):
            if detail_crop_coords is not None and i < len(detail_crop_coords):
                b1, b2, b3, b4 = detail_crop_coords[i]
                
                # Map coordinates to HR
                b1_hr = b1 * scale_factor
                b2_hr = b2 * scale_factor
                b3_hr = b3 * scale_factor
                b4_hr = b4 * scale_factor
                
                # Ensure detail prediction fits
                b1_hr = max(0, min(b1_hr, H_HR))
                b2_hr = max(0, min(b2_hr, H_HR))
                b3_hr = max(0, min(b3_hr, W_HR))
                b4_hr = max(0, min(b4_hr, W_HR))
                
                # Resize detail prediction to fit the crop region
                detail_h = b2_hr - b1_hr
                detail_w = b4_hr - b3_hr
                
                if detail_h > 0 and detail_w > 0:
                    pred_detail_resized = F.interpolate(
                        pred_detail[i:i+1],
                        size=(detail_h, detail_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    pred_detail_aligned[i, :, b1_hr:b2_hr, b3_hr:b4_hr] = \
                        pred_detail_resized[0]
        
        # Fuse predictions: (1 - attention) * context + attention * detail
        # Eq. 12 from HRDA paper
        fused_pred = (1 - attention_masked) * pred_context_hr + \
                     attention_masked * pred_detail_aligned
        
        return fused_pred


if __name__ == "__main__":
    # Test scale attention module
    batch_size = 2
    num_classes = 2
    
    # Simulate encoder features
    features = torch.randn(batch_size, 512, 24, 24)
    
    # Create scale attention head
    scale_attn = ScaleAttentionHead(
        in_channels=512,
        num_classes=num_classes,
        embed_dim=256
    )
    
    # Forward pass
    attention = scale_attn(features)
    
    print(f"Input features: {features.shape}")
    print(f"Scale attention: {attention.shape}")
    print(f"Attention range: [{attention.min():.3f}, {attention.max():.3f}]")
    
    # Test fusion module
    fusion = HRDAFusionModule(num_classes=num_classes)
    
    # Simulate predictions
    pred_context = torch.randn(batch_size, num_classes, 24, 24)
    pred_detail = torch.randn(batch_size, num_classes, 48, 48)
    detail_coords = [(4, 20, 4, 20), (6, 22, 6, 22)]
    
    fused = fusion(pred_context, pred_detail, attention, detail_coords)
    
    print(f"\nContext prediction: {pred_context.shape}")
    print(f"Detail prediction: {pred_detail.shape}")
    print(f"Fused prediction: {fused.shape}")