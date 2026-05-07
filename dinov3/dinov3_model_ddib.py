"""
DINOv3 + DDIB Model for Shadow Detection

Integrates the Disentangled Domain-Invariant Bottleneck between the
DINOv3 backbone (encoder) and a lightweight progressive upsampling
decoder.

Architecture flow:
    Image [B,3,384,384]
      → DINOv3 Backbone → 4 feature maps [B,384,24,24] each
      → Concatenate → [B,1536,24,24]
      → DDIB module → [B,384,24,24]
      → Decoder upsampling → [B, num_classes, 384, 384]

The decoder is identical to DINOv3Decoder's upsampling stages.
The DDIB replaces the decoder's feature_fusion layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_backbone import DINOv3Backbone
from dinov3_decoder import ConvBlock          # reuse existing conv block
from ddib import DDIB


class DINOv3ShadowDetectorDDIB(nn.Module):
    """
    DINOv3-based shadow detector with DDIB at the encoder-decoder interface.

    The three DDIB components (C1, C2, C3) are independently toggleable
    via constructor flags, enabling clean ablation studies.

    Args:
        num_classes:      output segmentation classes (default 2).
        model_name:       DINOv3 variant ('dinov3_vits16', 'dinov3_vitb16',
                          'dinov3_vitl16').
        weights_path:     path to DINOv3 pretrained backbone weights.
        pretrained:       load DINOv3 pretrained weights.
        frozen_stages:    number of backbone stages to freeze (-1 = train all).
        use_disentangle:  enable DDIB Component 1.
        use_vib:          enable DDIB Component 2.
        use_feat_aug:     enable DDIB Component 3.
        num_domains:      number of source cities during training.
        hsic_samples:     spatial samples for HSIC computation.
        vib_beta_base:    VIB minimum compression weight.
        vib_beta_scale:   VIB intensity-adaptive compression range.
        aug_sigma_style:  C3 random-style perturbation strength.
        aug_sigma_shift:  C3 random-shift perturbation strength.
        aug_p_aug:        probability of C3 style perturbation.
        aug_p_mix:        probability of C3 cross-domain mixing.
    """

    def __init__(
        self,
        num_classes=2,
        model_name='dinov3_vits16',
        weights_path=None,
        pretrained=True,
        frozen_stages=-1,
        # DDIB toggles
        use_disentangle=True,
        use_vib=True,
        use_feat_aug=True,
        # DDIB hyper-parameters
        num_domains=2,
        hsic_samples=1024,
        vib_beta_base=0.001,
        vib_beta_scale=0.01,
        aug_sigma_style=0.5,
        aug_sigma_shift=0.3,
        aug_p_aug=0.5,
        aug_p_mix=0.3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # ---- Backbone (frozen DINOv2/v3 ViT) ----
        print('Initialising DINOv3 backbone …')
        self.backbone = DINOv3Backbone(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            frozen_stages=frozen_stages,
        )
        embed_dim = self.backbone.embed_dim          # 384 for ViT-S
        num_feature_blocks = len(self.backbone.feature_blocks)  # 4
        in_channels = embed_dim * num_feature_blocks  # 1536

        # ---- DDIB ----
        print('Initialising DDIB …')
        self.ddib = DDIB(
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_disentangle=use_disentangle,
            use_vib=use_vib,
            use_feat_aug=use_feat_aug,
            num_domains=num_domains,
            hsic_samples=hsic_samples,
            vib_beta_base=vib_beta_base,
            vib_beta_scale=vib_beta_scale,
            aug_sigma_style=aug_sigma_style,
            aug_sigma_shift=aug_sigma_shift,
            aug_p_aug=aug_p_aug,
            aug_p_mix=aug_p_mix,
        )

        # ---- Decoder (upsampling stages only — fusion handled by DDIB) ----
        print('Initialising decoder …')
        dec = [embed_dim, 256, 128, 64, 32]

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(dec[0], dec[1], 2, stride=2),
            nn.BatchNorm2d(dec[1]),
            nn.ReLU(inplace=True),
            ConvBlock(dec[1], dec[1]),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(dec[1], dec[2], 2, stride=2),
            nn.BatchNorm2d(dec[2]),
            nn.ReLU(inplace=True),
            ConvBlock(dec[2], dec[2]),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(dec[2], dec[3], 2, stride=2),
            nn.BatchNorm2d(dec[3]),
            nn.ReLU(inplace=True),
            ConvBlock(dec[3], dec[3]),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(dec[3], dec[4], 2, stride=2),
            nn.BatchNorm2d(dec[4]),
            nn.ReLU(inplace=True),
            ConvBlock(dec[4], dec[4]),
        )
        self.final_conv = nn.Conv2d(dec[4], num_classes, kernel_size=1)

        # ---- Summary ----
        total  = sum(p.numel() for p in self.parameters())
        train_ = sum(p.numel() for p in self.parameters() if p.requires_grad)
        bb     = sum(p.numel() for p in self.backbone.parameters())
        ddib_p = sum(p.numel() for p in self.ddib.parameters())
        dec_p  = (sum(p.numel() for p in self.up1.parameters())
                + sum(p.numel() for p in self.up2.parameters())
                + sum(p.numel() for p in self.up3.parameters())
                + sum(p.numel() for p in self.up4.parameters())
                + sum(p.numel() for p in self.final_conv.parameters()))
        print(f'\nDINOv3-DDIB Shadow Detector:')
        print(f'  Total params:     {total:,}')
        print(f'  Trainable params: {train_:,}')
        print(f'  Backbone:         {bb:,}')
        print(f'  DDIB:             {ddib_p:,}')
        print(f'  Decoder:          {dec_p:,}')

    # ------------------------------------------------------------------
    def forward(self, x, intensity_map=None, city_ids=None):
        """
        Args:
            x:             [B, 3, H, W]  input RGB (H, W multiples of 16).
            intensity_map: [B, 1, H, W]  pre-normalisation grayscale
                           intensity in [0, 1].  Optional (can be None at
                           inference).
            city_ids:      [B] int64 domain labels.  Optional (can be None
                           at inference).
        Returns:
            logits:      [B, num_classes, H, W]
            ddib_losses: dict  (empty at inference / when all components off)
        """
        B, C, H, W = x.shape

        # 1. Encoder ----------------------------------------------------------
        features = self.backbone(x)   # dict of [B, embed_dim, H/16, W/16]

        feat_concat = torch.cat([
            features['feat_block3'],
            features['feat_block6'],
            features['feat_block9'],
            features['feat_block11'],
        ], dim=1)                     # [B, 4*embed_dim, H/16, W/16]

        # 2. DDIB -------------------------------------------------------------
        task_feat, ddib_losses = self.ddib(
            feat_concat, intensity_map, city_ids)   # [B, embed_dim, H/16, W/16]

        # 3. Decoder upsampling -----------------------------------------------
        out = self.up1(task_feat)      # → H/8
        out = self.up2(out)            # → H/4
        out = self.up3(out)            # → H/2
        out = self.up4(out)            # → H
        out = self.final_conv(out)     # → [B, num_classes, H, W]

        if out.shape[2] != H or out.shape[3] != W:
            out = F.interpolate(out, size=(H, W),
                                mode='bilinear', align_corners=False)

        return out, ddib_losses

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def get_predictions(self, x, intensity_map=None):
        """Inference helper — returns [B, H, W] integer predictions."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, intensity_map)
            return torch.argmax(logits, dim=1)

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        print('Backbone frozen.')

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print('Backbone unfrozen.')


# ======================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('Testing DINOv3-DDIB Shadow Detector')
    print('=' * 60)

    try:
        model = DINOv3ShadowDetectorDDIB(
            num_classes=2,
            model_name='dinov3_vits16',
            pretrained=True,
            use_disentangle=True,
            use_vib=True,
            use_feat_aug=True,
            num_domains=2,
        )

        x   = torch.randn(4, 3, 384, 384)
        im  = torch.rand(4, 1, 384, 384)     # intensity map
        cid = torch.tensor([0, 1, 0, 1])      # city ids

        model.train()
        out, losses = model(x, im, cid)
        print(f'\nTrain mode  →  output {out.shape}')
        for k, v in losses.items():
            print(f'  {k}: {v.item():.6f}')

        model.eval()
        out_e, losses_e = model(x)
        print(f'\nEval mode   →  output {out_e.shape}')
        print(f'  losses: {losses_e}')

        print('\n✓ All tests passed!')

    except Exception as e:
        print(f'\nTest failed: {e}')