"""
MAMNet with GSDPE: Multi-Scale Spatial Channel Attention Network with 
Ground Sample Distance Positional Encoding

Integrates Scale-MAE's GSDPE into MAMNet for cross-resolution transfer learning.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gsdpe import GSDPE
from models.encoder import ResNet34Encoder
from models.encoder_4ch import ResNet34Encoder4Ch
from models.decoder import Decoder


class SimpleMSCAF(nn.Module):
    """Simplified MSCAF - replace with your actual mscaf.py implementation"""
    def __init__(self, in_channels=512):
        super(SimpleMSCAF, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# class SimpleDecoder(nn.Module):
#     """Simplified Decoder - replace with your actual decoder.py implementation"""
#     def __init__(self, num_classes=2):
#         super(SimpleDecoder, self).__init__()
#         self.num_classes = num_classes
        
#         # Upsampling layers
#         self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.conv4 = nn.Conv2d(512, 256, 3, padding=1)
        
#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(256, 128, 3, padding=1)
        
#         self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        
#         self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
#         self.conv1 = nn.Conv2d(128, 64, 3, padding=1)
        
#         self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
#     def forward(self, x, enc_features):
#         # Decoder stage 4
#         x = self.up4(x)
#         x = torch.cat([x, enc_features['feat4']], dim=1)
#         x = self.conv4(x)
#         dec_feat1 = x
        
#         # Decoder stage 3
#         x = self.up3(x)
#         x = torch.cat([x, enc_features['feat3']], dim=1)
#         x = self.conv3(x)
#         dec_feat2 = x
        
#         # Decoder stage 2
#         x = self.up2(x)
#         x = torch.cat([x, enc_features['feat2']], dim=1)
#         x = self.conv2(x)
#         dec_feat3 = x
        
#         # Decoder stage 1
#         x = self.up1(x)
#         x = torch.cat([x, enc_features['feat1']], dim=1)
#         x = self.conv1(x)
        
#         # Final prediction
#         main_out = self.final(x)
        
#         return {
#             'main': main_out,
#             'dec_feat1': dec_feat1,
#             'dec_feat2': dec_feat2,
#             'dec_feat3': dec_feat3
#         }


class SimpleAuxiliary(nn.Module):
    """Simplified Auxiliary - replace with your actual auxiliary.py implementation"""
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(SimpleAuxiliary, self).__init__()
        self.aux1 = nn.Sequential(
            nn.Conv2d(256, num_classes, 1),
        )
        self.aux2 = nn.Sequential(
            nn.Conv2d(128, num_classes, 1),
        )
        self.aux3 = nn.Sequential(
            nn.Conv2d(64, num_classes, 1),
        )
        
    def forward(self, dec_feat1, dec_feat2, dec_feat3, target_size):
        aux1 = nn.functional.interpolate(self.aux1(dec_feat1), size=target_size, mode='bilinear', align_corners=False)
        aux2 = nn.functional.interpolate(self.aux2(dec_feat2), size=target_size, mode='bilinear', align_corners=False)
        aux3 = nn.functional.interpolate(self.aux3(dec_feat3), size=target_size, mode='bilinear', align_corners=False)
        
        return {
            'aux1': aux1,
            'aux2': aux2,
            'aux3': aux3
        }


class MAMNet_GSDPE(nn.Module):
    """
    MAMNet with Ground Sample Distance Positional Encoding (GSDPE).
    
    Integrates Scale-MAE's GSDPE at the encoder input to enable
    cross-resolution transfer learning.
    
    Architecture:
    1. GSDPE applied at encoder input (first conv layer output)
    2. ResNet-34 Encoder (pretrained ImageNet)
    3. MSCAF module for multi-scale feature fusion
    4. Decoder with CCA modules at each stage
    5. Auxiliary branches for deep supervision
    
    Args:
        num_classes: Number of output classes (default: 2)
        pretrained: Use pretrained ResNet-34 encoder (default: True)
        use_aux: Use auxiliary branches during training (default: True)
        reference_gsd: Reference GSD for GSDPE (default: 1.0m)
    """
    
    def __init__(self, num_classes=2, pretrained=True, use_aux=True, reference_gsd=1.0, use_contrast=False):
        super(MAMNet_GSDPE, self).__init__()
        
        self.num_classes = num_classes
        self.use_aux = use_aux
        self.reference_gsd = reference_gsd
        self.use_contrast = use_contrast  # ADD THIS

        # GSDPE: Applied to INPUT (3 or 4 channels) before encoder
        input_channels = 4 if use_contrast else 3
        self.gsdpe_input = GSDPE(
            channels=input_channels,  # Match input image channels
            reference_gsd=reference_gsd
        )

        # Encoder: ResNet-34 (3ch or 4ch)
        if use_contrast:
            self.encoder = ResNet34Encoder4Ch(pretrained=pretrained)
            print("Using 4-channel encoder (RGB + Contrast)")
        else:
            self.encoder = ResNet34Encoder(pretrained=pretrained)
        
        # GSDPE: Applied to first encoder features (64 channels)
        # This is where GSDPE is added in Scale-MAE (at encoder input)
        self.gsdpe = GSDPE(
            channels=64,  # First conv layer output has 64 channels
            reference_gsd=reference_gsd
        )
        
        # MSCAF: Multi-Scale Spatial Channel Attention Fusion
        self.mscaf = SimpleMSCAF(in_channels=512)
        
        # Decoder with CCA modules
        self.decoder = Decoder(num_classes=num_classes)
        
        # Auxiliary branches
        if use_aux:
            self.aux_module = SimpleAuxiliary(num_classes=num_classes, dropout_rate=0.3)
        
    def forward(self, x, gsd):
        """
        Forward pass with GSD conditioning.
        
        Args:
            x: Input RGB images [B, 3, H, W]
            gsd: GSD values for each sample [B] or [B, 1]
            
        Returns:
            If training (use_aux=True):
                Dictionary with 'main' and auxiliary outputs ('aux1', 'aux2', 'aux3')
            If inference (use_aux=False):
                Main prediction only [B, num_classes, H, W]
        """
        B, _, H, W = x.size()
        # print(f"Input size: {H}x{W}")

        # Encoder
        enc_features = self.encoder(x)
        
        # DEBUG: Print all feature sizes
        # for k, v in enc_features.items():
        #     print(f"{k}: {v.shape}")
        
        # Apply GSDPE
        enc_features['feat1'] = self.gsdpe(enc_features['feat1'], gsd)
        # print(f"After GSDPE, feat1: {enc_features['feat1'].shape}")
        
        # MSCAF
        mscaf_out = self.mscaf(enc_features['feat5'])
        # print(f"MSCAF out: {mscaf_out.shape}")
        
        # Decoder
        decoder_outputs = self.decoder(mscaf_out, enc_features)
        # print(f"Decoder main output: {decoder_outputs['main'].shape}")

        main_out = decoder_outputs['main']
        
        # Auxiliary branches (only during training)
        if self.use_aux and self.training:
            aux_outputs = self.aux_module(
                decoder_outputs['dec_feat1'],
                decoder_outputs['dec_feat2'],
                decoder_outputs['dec_feat3'],
                target_size=(H, W)
            )
            
            return {
                'main': main_out,
                'aux1': aux_outputs['aux1'],
                'aux2': aux_outputs['aux2'],
                'aux3': aux_outputs['aux3']
            }
        else:
            # Inference: return only main output
            return main_out
    
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
            logits = self.forward(x, gsd)
            preds = torch.argmax(logits, dim=1)
        return preds


if __name__ == "__main__":
    print("Testing MAMNet with GSDPE")
    print("=" * 50)
    
    # Create model
    model = MAMNet_GSDPE(num_classes=2, pretrained=False, use_aux=True, reference_gsd=1.0)
    
    # Test with different GSDs
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
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
    print(f"Binary predictions: {preds.shape}, unique values: {torch.unique(preds)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 50)
    print("MAMNet with GSDPE test completed successfully!")