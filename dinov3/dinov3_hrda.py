"""
DINOv3-HRDA Model for Cross-Resolution Shadow Detection
Integrates DINOv3 backbone with HRDA multi-resolution training.

Based on HRDA (ECCV 2022): https://arxiv.org/abs/2204.13132
Architecture combines:
1. Shared DINOv3 encoder for both LR context and HR detail crops
2. Separate prediction heads for context and detail
3. Scale attention module for learned fusion
4. Auxiliary branches for additional supervision

Key Design Decisions (see ASSUMPTIONS.md for details):
- DINOv3 ViT-S/16 with embed_dim=384
- Feature extraction at 1/16 resolution
- Auxiliary outputs for both context and detail branches
- Scale attention operates on shared encoder features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_backbone import DINOv3Backbone
from dinov3_decoder import DINOv3Decoder
from scale_attention_dinov3 import ScaleAttentionHeadViT, HRDAFusionModule


class DINOv3DecoderHRDA(nn.Module):
    """
    Modified DINOv3 Decoder with auxiliary outputs for HRDA.
    
    Adds intermediate predictions at different scales for better supervision,
    similar to auxiliary losses in semantic segmentation (e.g., PSPNet, DeepLabV3).
    
    Args:
        num_classes: Number of output classes
        embed_dim: Embedding dimension from ViT backbone
    """
    
    def __init__(self, num_classes=2, embed_dim=384):
        super(DINOv3DecoderHRDA, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Main decoder path (progressive upsampling)
        # Stage 1: 1/16 -> 1/8 (initial upsampling)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: 1/8 -> 1/4
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: 1/4 -> 1/2
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 4: 1/2 -> 1/1 (full resolution)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Main prediction head
        self.main_head = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Auxiliary prediction heads (for intermediate supervision)
        # Auxiliary at 1/8 resolution
        self.aux_head1 = nn.Conv2d(256, num_classes, kernel_size=1)
        # Auxiliary at 1/4 resolution
        self.aux_head2 = nn.Conv2d(128, num_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features, return_aux=False):
        """
        Forward pass with optional auxiliary outputs.
        
        Args:
            features: Encoder features [B, embed_dim, H/16, W/16]
            return_aux: Whether to return auxiliary predictions
            
        Returns:
            If return_aux=False: main prediction [B, num_classes, H, W]
            If return_aux=True: dict with 'main' and 'aux' predictions
        """
        # Progressive upsampling
        x1 = self.up1(features)  # [B, 256, H/8, W/8]
        x2 = self.up2(x1)        # [B, 128, H/4, W/4]
        x3 = self.up3(x2)        # [B, 64, H/2, W/2]
        x4 = self.up4(x3)        # [B, 32, H, W]
        
        # Main prediction
        main_pred = self.main_head(x4)  # [B, num_classes, H, W]
        
        if return_aux:
            # Auxiliary predictions at intermediate scales
            aux1 = self.aux_head1(x1)  # [B, num_classes, H/8, W/8]
            aux2 = self.aux_head2(x2)  # [B, num_classes, H/4, W/4]
            
            return {
                'main': main_pred,
                'aux': [aux1, aux2]
            }
        else:
            return main_pred


class DINOv3HRDA(nn.Module):
    """
    DINOv3-HRDA: Multi-Resolution Shadow Detection with HRDA.
    
    Architecture:
    1. Shared DINOv3 backbone for context and detail crops
    2. Separate decoders for context and detail predictions
    3. Scale attention module for learning fusion weights
    4. Fusion module to combine predictions
    
    Training Strategy:
    - Context crop: Low-resolution (192×192 after 0.5× downsampling)
    - Detail crop: High-resolution (192×192 at full resolution)
    - Scale attention learns per-pixel weights for fusion
    - EMA teacher generates pseudo-labels for target domain
    
    Args:
        num_classes: Number of classes (default: 2 for shadow detection)
        model_name: DINOv3 variant ('dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16')
        weights_path: Path to pretrained DINOv3 weights
        pretrained: Whether to use pretrained weights
        frozen_stages: Number of backbone stages to freeze (-1 = train all)
        use_aux: Whether to use auxiliary supervision
        hr_loss_weight: Weight for HR detail loss (λ_d in HRDA paper)
    """
    
    def __init__(self, num_classes=2, model_name='dinov3_vits16',
                 weights_path=None, pretrained=True, frozen_stages=-1,
                 use_aux=True, hr_loss_weight=0.1):
        super(DINOv3HRDA, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.use_aux = use_aux
        self.hr_loss_weight = hr_loss_weight
        
        # Initialize shared backbone
        print('Initializing DINOv3 backbone (shared)...')
        self.backbone = DINOv3Backbone(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            frozen_stages=frozen_stages
        )
        
        # Get embedding dimension
        self.embed_dim = self.backbone.embed_dim
        
        # Context decoder (for LR context predictions)
        print('Initializing context decoder...')
        self.context_decoder = DINOv3DecoderHRDA(
            num_classes=num_classes,
            embed_dim=self.embed_dim
        )
        
        # Detail decoder (for HR detail predictions)
        print('Initializing detail decoder...')
        self.detail_decoder = DINOv3DecoderHRDA(
            num_classes=num_classes,
            embed_dim=self.embed_dim
        )
        
        # Scale attention head
        print('Initializing scale attention...')
        self.scale_attention = ScaleAttentionHeadViT(
            embed_dim=self.embed_dim,
            num_classes=num_classes,
            hidden_dim=256,
            dropout=0.1
        )
        
        # Fusion module
        self.fusion = HRDAFusionModule(num_classes=num_classes)
        
        print(f'\nDINOv3-HRDA initialized:')
        print(f'  Model: {model_name}')
        print(f'  Embed dim: {self.embed_dim}')
        print(f'  Num classes: {num_classes}')
        print(f'  Auxiliary supervision: {use_aux}')
        print(f'  HR loss weight: {hr_loss_weight}')
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'  Total parameters: {total_params:,}')
        print(f'  Trainable parameters: {trainable_params:,}')
    
    def forward(self, image_context, image_detail, detail_coords):
        """
        Forward pass through HRDA network.
        
        Args:
            image_context: LR context crop [B, 3, H_c, W_c]
            image_detail: HR detail crop [B, 3, H_d, W_d]
            detail_coords: List of tuples with detail crop coordinates
                          [(b1, b2, b3, b4), ...] for each batch item
        
        Returns:
            Dictionary with:
            - 'pred_fused': Fused prediction at HR resolution [B, C, H_HR, W_HR]
            - 'pred_context': Context prediction [B, C, H_c, W_c]
            - 'pred_detail': Detail prediction [B, C, H_d, W_d]
            - 'attention': Scale attention weights [B, C, H_c, W_c]
            - 'aux_context': Auxiliary context predictions (if use_aux)
            - 'aux_detail': Auxiliary detail predictions (if use_aux)
        """
        # Extract features from context crop
        features_context_dict = self.backbone(image_context)
        features_context = features_context_dict['feat_block11']  # Use deepest features [B, embed_dim, H_c/16, W_c/16]

        # Extract features from detail crop
        features_detail_dict = self.backbone(image_detail)
        features_detail = features_detail_dict['feat_block11']    # Use deepest features [B, embed_dim, H_d/16, W_d/16]
        
        # Decode context features to predictions
        if self.use_aux:
            context_output = self.context_decoder(features_context, return_aux=True)
            pred_context = context_output['main']        # [B, C, H_c, W_c]
            aux_context = context_output['aux']          # List of auxiliary predictions
        else:
            pred_context = self.context_decoder(features_context, return_aux=False)
            aux_context = None
        
        # Decode detail features to predictions
        if self.use_aux:
            detail_output = self.detail_decoder(features_detail, return_aux=True)
            pred_detail = detail_output['main']          # [B, C, H_d, W_d]
            aux_detail = detail_output['aux']            # List of auxiliary predictions
        else:
            pred_detail = self.detail_decoder(features_detail, return_aux=False)
            aux_detail = None
        
        # Predict scale attention from context features
        attention = self.scale_attention(features_context)  # This is already the tensor now, so this line is fine
        
        # Fuse predictions using scale attention
        pred_fused = self.fusion(
            pred_context,
            pred_detail,
            attention,
            detail_coords
        )  # [B, C, H_HR, W_HR]
        
        result = {
            'pred_fused': pred_fused,
            'pred_context': pred_context,
            'pred_detail': pred_detail,
            'attention': attention
        }
        
        if self.use_aux:
            result['aux_context'] = aux_context
            result['aux_detail'] = aux_detail
        
        return result
    
    def get_predictions(self, image_context, image_detail, detail_coords):
        """
        Get binary predictions for inference.
        
        Args:
            image_context: LR context crop [B, 3, H_c, W_c]
            image_detail: HR detail crop [B, 3, H_d, W_d]
            detail_coords: List of detail crop coordinates
        
        Returns:
            Binary predictions [B, H_HR, W_HR] with values {0, 1}
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(image_context, image_detail, detail_coords)
            logits = output['pred_fused']  # [B, C, H_HR, W_HR]
            preds = torch.argmax(logits, dim=1)  # [B, H_HR, W_HR]
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


def create_dinov3_hrda(num_classes=2, model_name='dinov3_vits16',
                       weights_path=None, pretrained=True,
                       use_aux=True, hr_loss_weight=0.1):
    """
    Factory function to create DINOv3-HRDA model.
    
    Args:
        num_classes: Number of classes
        model_name: DINOv3 variant
        weights_path: Path to pretrained weights
        pretrained: Whether to use pretrained weights
        use_aux: Whether to use auxiliary supervision
        hr_loss_weight: Weight for HR detail loss
    
    Returns:
        DINOv3HRDA model
    """
    model = DINOv3HRDA(
        num_classes=num_classes,
        model_name=model_name,
        weights_path=weights_path,
        pretrained=pretrained,
        frozen_stages=-1,  # Train all stages
        use_aux=use_aux,
        hr_loss_weight=hr_loss_weight
    )
    return model


if __name__ == "__main__":
    # Test DINOv3-HRDA model
    print("=" * 60)
    print("Testing DINOv3-HRDA Model")
    print("=" * 60)
    
    try:
        # Initialize model
        model = create_dinov3_hrda(
            num_classes=2,
            model_name='dinov3_vits16',
            weights_path=None,
            pretrained=True,
            use_aux=True,
            hr_loss_weight=0.1
        )
        
        print("\n" + "=" * 60)
        print("Testing forward pass...")
        print("=" * 60)
        
        batch_size = 2
        
        # Simulate HRDA inputs
        # Context: 192×192 (after 0.5× downsampling from 384×384)
        image_context = torch.randn(batch_size, 3, 192, 192)
        # Detail: 192×192 (full resolution crop)
        image_detail = torch.randn(batch_size, 3, 192, 192)
        # Detail coordinates in context space
        detail_coords = [(4, 20, 4, 20), (6, 18, 6, 18)]
        
        print(f"\nInput shapes:")
        print(f"  Context: {image_context.shape}")
        print(f"  Detail: {image_detail.shape}")
        print(f"  Detail coords: {detail_coords}")
        
        # Forward pass
        model.train()
        output = model(image_context, image_detail, detail_coords)
        
        print(f"\nOutput shapes:")
        print(f"  Fused prediction: {output['pred_fused'].shape}")
        print(f"  Context prediction: {output['pred_context'].shape}")
        print(f"  Detail prediction: {output['pred_detail'].shape}")
        print(f"  Attention: {output['attention'].shape}")
        
        if 'aux_context' in output and output['aux_context'] is not None:
            print(f"  Auxiliary context: {[aux.shape for aux in output['aux_context']]}")
        if 'aux_detail' in output and output['aux_detail'] is not None:
            print(f"  Auxiliary detail: {[aux.shape for aux in output['aux_detail']]}")
        
        # Test inference mode
        model.eval()
        preds = model.get_predictions(image_context, image_detail, detail_coords)
        print(f"\nBinary predictions shape: {preds.shape}")
        print(f"Unique prediction values: {torch.unique(preds)}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()