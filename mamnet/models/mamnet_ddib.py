"""
MAMNet + DDIB (Option B): Multi-Scale Attention Network with
Disentangled Domain-Invariant Bottleneck for Shadow Detection.

Option B architecture:
  - Full DDIB at the bottleneck (after MSCAF, before decoder stage 1)
  - Lightweight SkipDomainFilter on each skip connection (feat1-feat4)

The skip filters are small 1x1-conv bottleneck modules with:
  - Channel compression (reduction ratio r=4)
  - Optional mini-VIB reparameterisation for information compression
  - Gated residual connection (gate initialised to 0 for stable start)

This ensures domain-specific information is suppressed not only in
the bottleneck but also in the skip connections that flow directly
into the decoder's CCA blocks.

Architecture flow:
    Image [B, 3/4, 384, 384]
      -> ResNet-34 Encoder -> feat1..feat5
      -> MSCAF(feat5)      -> [B, 512, H/16, W/16]
      -> DDIB              -> [B, 512, H/16, W/16]   (bottleneck)
      -> SkipFilter(feat1) -> [B, 64,  H,    W   ]   (skip 1)
      -> SkipFilter(feat2) -> [B, 128, H/2,  W/2 ]   (skip 2)
      -> SkipFilter(feat3) -> [B, 256, H/4,  W/4 ]   (skip 3)
      -> SkipFilter(feat4) -> [B, 512, H/8,  W/8 ]   (skip 4)
      -> Decoder (CCA with filtered skips)
      -> [B, num_classes, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResNet34Encoder
from .encoder_4ch import ResNet34Encoder4Ch
from .mscaf import MSCAF
from .decoder import Decoder
from .auxiliary import AuxiliaryModule

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddib import DDIB


# ======================================================================
# Lightweight skip-connection domain filter
# ======================================================================

class SkipDomainFilter(nn.Module):
    """
    Lightweight domain-invariant filter for encoder skip connections.

    Architecture:
        x -> 1x1 conv (C -> C//r) -> BN -> ReLU
          -> [optional mini-VIB reparameterisation]
          -> 1x1 conv (C//r -> C)
          -> gated residual:  output = x + gate * filtered

    The learnable ``gate`` scalar is initialised to 0 so the module
    starts as an identity, ensuring stable early training.

    When ``use_vib=True``, the bottleneck representation is treated
    as a variational distribution (mu, logvar) with KL regularisation,
    providing an information-compression effect analogous to DDIB's C2
    but much lighter (no intensity-adaptive beta, just a fixed prior).

    Args:
        channels:   number of input/output channels.
        reduction:  channel reduction ratio for the bottleneck (default 4).
        use_vib:    add mini-VIB reparameterisation (default False).
    """

    def __init__(self, channels, reduction=4, use_vib=False):
        super().__init__()
        mid = max(channels // reduction, 16)

        self.compress = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.expand = nn.Conv2d(mid, channels, 1, bias=False)

        # Gated residual — starts at 0 (identity)
        self.gate = nn.Parameter(torch.zeros(1))

        # Optional mini-VIB
        self.use_vib = use_vib
        if use_vib:
            self.mu_proj = nn.Conv2d(mid, mid, 1)
            self.logvar_proj = nn.Conv2d(mid, mid, 1)
            # Initialise logvar projection to produce small negative values
            # so initial variance ~ 1 (exp(-small) approx 1)
            nn.init.constant_(self.logvar_proj.bias, -2.0)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] encoder skip feature.

        Returns:
            filtered: [B, C, H, W] filtered features.
            kl_loss:  scalar tensor (0 if use_vib=False or eval mode).
        """
        h = self.compress(x)
        h = self.bn(h)
        h = F.relu(h, inplace=True)

        kl_loss = torch.tensor(0.0, device=x.device)

        if self.use_vib and self.training:
            mu = self.mu_proj(h)
            logvar = self.logvar_proj(h)
            # Clamp for numerical stability
            logvar = torch.clamp(logvar, min=-10.0, max=2.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            h = mu + eps * std
            # KL(q(z|x) || N(0,I))
            kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        elif self.use_vib:
            # Eval: use the mean (no sampling)
            h = self.mu_proj(h)

        h = self.expand(h)

        return x + self.gate * h, kl_loss


# ======================================================================
# MAMNet + DDIB (Option B)
# ======================================================================

class MAMNetDDIB(nn.Module):
    """
    MAMNet with full DDIB at the bottleneck and lightweight
    SkipDomainFilters on encoder skip connections.

    Args:
        num_classes:      output classes (default 2).
        pretrained:       use ImageNet-pretrained ResNet-34 encoder.
        use_aux:          enable auxiliary decoder branches.
        use_contrast:     4-channel input (RGB + contrast).

        -- DDIB component toggles --
        use_disentangle:  enable C1 (feature disentanglement / HSIC).
        use_vib:          enable C2 (variational information bottleneck).
        use_feat_aug:     enable C3 (stochastic feature augmentation).
        use_skip_filter:  enable lightweight skip-connection filtering.
                          When True AND use_vib=True, skip filters include
                          mini-VIB.  When True AND use_vib=False, skip
                          filters are plain bottleneck projections.

        -- DDIB hyper-parameters --
        num_domains, hsic_samples, vib_beta_base, vib_beta_scale,
        aug_sigma_style, aug_sigma_shift, aug_p_aug, aug_p_mix:
            Forwarded to the main DDIB module.

        -- Skip filter hyper-parameters --
        skip_reduction:   channel reduction ratio (default 4).
        skip_kl_weight:   weight for aggregated skip KL losses relative
                          to the main VIB KL (default 0.1, lighter than
                          the bottleneck VIB since skips are auxiliary).
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        use_aux=True,
        use_contrast=False,
        # DDIB toggles
        use_disentangle=True,
        use_vib=True,
        use_feat_aug=True,
        use_skip_filter=True,
        # DDIB hyper-parameters
        num_domains=2,
        hsic_samples=1024,
        vib_beta_base=0.001,
        vib_beta_scale=0.01,
        aug_sigma_style=0.5,
        aug_sigma_shift=0.3,
        aug_p_aug=0.5,
        aug_p_mix=0.3,
        # Skip filter hyper-parameters
        skip_reduction=4,
        skip_kl_weight=0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_aux = use_aux
        self.use_contrast = use_contrast
        self.use_skip_filter = use_skip_filter
        self.skip_kl_weight = skip_kl_weight

        # ---- Encoder ----
        if use_contrast:
            self.encoder = ResNet34Encoder4Ch(pretrained=pretrained)
            print("Using 4-channel encoder (RGB + Contrast)")
        else:
            self.encoder = ResNet34Encoder(pretrained=pretrained)

        # ---- MSCAF ----
        self.mscaf = MSCAF(in_channels=512)

        # ---- DDIB at bottleneck (512 -> 512) ----
        print('Initialising DDIB at bottleneck ...')
        self.ddib = DDIB(
            in_channels=512,
            embed_dim=512,
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

        # ---- Skip domain filters ----
        # feat1: 64ch, feat2: 128ch, feat3: 256ch, feat4: 512ch
        skip_channels = [64, 128, 256, 512]
        skip_vib = use_vib and use_skip_filter  # mini-VIB only if both flags on

        if use_skip_filter:
            print(f'Initialising skip domain filters '
                  f'(reduction={skip_reduction}, vib={skip_vib}) ...')
            self.skip_filters = nn.ModuleList([
                SkipDomainFilter(
                    channels=ch,
                    reduction=skip_reduction,
                    use_vib=skip_vib,
                ) for ch in skip_channels
            ])
        else:
            self.skip_filters = None

        # ---- Decoder with CCA ----
        self.decoder = Decoder(num_classes=num_classes)

        # ---- Auxiliary branches ----
        if use_aux:
            self.aux_module = AuxiliaryModule(
                num_classes=num_classes, dropout_rate=0.3)

        # ---- Summary ----
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        mscaf_p = sum(p.numel() for p in self.mscaf.parameters())
        ddib_p = sum(p.numel() for p in self.ddib.parameters())
        skip_p = (sum(p.numel() for p in self.skip_filters.parameters())
                  if self.skip_filters is not None else 0)
        dec_p = sum(p.numel() for p in self.decoder.parameters())
        aux_p = (sum(p.numel() for p in self.aux_module.parameters())
                 if use_aux else 0)

        print(f'\nMAMNet-DDIB (Option B) Shadow Detector:')
        print(f'  Total params:     {total:,}')
        print(f'  Trainable params: {trainable:,}')
        print(f'  Encoder:          {enc_p:,}')
        print(f'  MSCAF:            {mscaf_p:,}')
        print(f'  DDIB (bottleneck):{ddib_p:,}')
        print(f'  Skip filters:     {skip_p:,}')
        print(f'  Decoder:          {dec_p:,}')
        print(f'  Auxiliary:        {aux_p:,}')
        if self.skip_filters is not None:
            for i, (ch, sf) in enumerate(zip(skip_channels, self.skip_filters)):
                sf_p = sum(p.numel() for p in sf.parameters())
                print(f'    skip{i+1} ({ch:>3}ch): {sf_p:,} params  '
                      f'(vib={sf.use_vib})')

    def forward(self, x, intensity_map=None, city_ids=None):
        """
        Args:
            x:             [B, 3, H, W] or [B, 4, H, W] if use_contrast.
            intensity_map: [B, 1, H, W] grayscale in [0, 1] (for C2).
            city_ids:      [B] int64 domain labels (for C1/C3).

        Returns:
            During training (use_aux=True):
                outputs:     dict {'main', 'aux1', 'aux2', 'aux3'}
                all_losses:  dict {'hsic_loss', 'domain_loss', 'kl_loss',
                                   'skip_kl_loss', ...}
            During inference:
                outputs:     tensor [B, num_classes, H, W]
                all_losses:  dict (mostly zeros)
        """
        B, _, H, W = x.size()

        # 1. Encoder
        enc_features = self.encoder(x)
        feat1 = enc_features['feat1']  # [B, 64,  H,    W   ]
        feat2 = enc_features['feat2']  # [B, 128, H/2,  W/2 ]
        feat3 = enc_features['feat3']  # [B, 256, H/4,  W/4 ]
        feat4 = enc_features['feat4']  # [B, 512, H/8,  W/8 ]

        # 2. MSCAF on deepest features
        mscaf_out = self.mscaf(enc_features['feat5'])  # [B, 512, H/16, W/16]

        # 3. Full DDIB at the bottleneck
        task_feat, ddib_losses = self.ddib(
            mscaf_out, intensity_map, city_ids)

        # 4. Skip domain filters
        skip_kl_total = torch.tensor(0.0, device=x.device)

        if self.skip_filters is not None:
            skips = [feat1, feat2, feat3, feat4]
            filtered_skips = []
            for i, (skip, filt) in enumerate(zip(skips, self.skip_filters)):
                filtered, kl_i = filt(skip)
                filtered_skips.append(filtered)
                skip_kl_total = skip_kl_total + kl_i
            # Average skip KL across the 4 levels
            skip_kl_total = skip_kl_total / len(self.skip_filters)
            # Rebuild enc_features with filtered skips
            enc_features_filtered = {
                'feat1': filtered_skips[0],
                'feat2': filtered_skips[1],
                'feat3': filtered_skips[2],
                'feat4': filtered_skips[3],
            }
        else:
            enc_features_filtered = {
                'feat1': feat1,
                'feat2': feat2,
                'feat3': feat3,
                'feat4': feat4,
            }

        # 5. Decoder with filtered skip connections
        decoder_outputs = self.decoder(task_feat, enc_features_filtered)
        main_out = decoder_outputs['main']

        # 6. Aggregate all losses
        all_losses = dict(ddib_losses)  # copy
        all_losses['skip_kl_loss'] = skip_kl_total * self.skip_kl_weight

        # 7. Auxiliary branches (training only)
        if self.use_aux and self.training:
            aux_outputs = self.aux_module(
                decoder_outputs['dec_feat1'],
                decoder_outputs['dec_feat2'],
                decoder_outputs['dec_feat3'],
                target_size=(H, W),
            )
            outputs = {
                'main': main_out,
                'aux1': aux_outputs['aux1'],
                'aux2': aux_outputs['aux2'],
                'aux3': aux_outputs['aux3'],
            }
            return outputs, all_losses
        else:
            return main_out, all_losses

    def get_predictions(self, x, intensity_map=None):
        """Inference helper -- returns [B, H, W] integer predictions."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, intensity_map)
            return torch.argmax(logits, dim=1)

    def get_skip_gate_values(self):
        """Return current skip filter gate values (for monitoring)."""
        if self.skip_filters is None:
            return {}
        return {
            f'skip{i+1}_gate': sf.gate.item()
            for i, sf in enumerate(self.skip_filters)
        }


# ======================================================================
# Quick test
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing MAMNet-DDIB (Option B)")
    print("=" * 60)

    # ---- Full DDIB + skip filters ----
    model = MAMNetDDIB(
        num_classes=2, pretrained=True, use_aux=True,
        use_contrast=False,
        use_disentangle=True, use_vib=True, use_feat_aug=True,
        use_skip_filter=True,
        num_domains=2, skip_reduction=4, skip_kl_weight=0.1,
    )

    x = torch.randn(4, 3, 384, 384)
    im = torch.rand(4, 1, 384, 384)
    cid = torch.tensor([0, 1, 0, 1])

    model.train()
    outputs, losses = model(x, im, cid)
    print("\nTraining outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")
    print("Skip gates:", model.get_skip_gate_values())

    model.eval()
    out_e, losses_e = model(x, im)
    print(f"\nEval output: {out_e.shape}")
    print(f"Eval skip_kl_loss: {losses_e['skip_kl_loss'].item():.6f}")

    # ---- Skip filters OFF (Option A equivalent) ----
    print("\n--- Option A mode (skip_filter=False) ---")
    model_a = MAMNetDDIB(
        num_classes=2, pretrained=True, use_aux=True,
        use_disentangle=True, use_vib=True, use_feat_aug=True,
        use_skip_filter=False,
        num_domains=2,
    )
    model_a.train()
    out_a, l_a = model_a(x, im, cid)
    print(f"Main: {out_a['main'].shape}")
    print(f"skip_kl_loss: {l_a['skip_kl_loss'].item():.6f}  (should be 0)")

    # ---- 4-channel test ----
    print("\n--- 4-channel + skip filters ---")
    model_4ch = MAMNetDDIB(
        num_classes=2, pretrained=True, use_contrast=True,
        use_disentangle=True, use_vib=True, use_feat_aug=True,
        use_skip_filter=True,
        num_domains=2,
    )
    model_4ch.train()
    x4 = torch.randn(4, 4, 384, 384)
    out4, l4 = model_4ch(x4, im, cid)
    print(f"4ch main: {out4['main'].shape}")

    # ---- No DDIB, no skip (baseline) ----
    print("\n--- Baseline (no DDIB, no skip filters) ---")
    model_base = MAMNetDDIB(
        num_classes=2, pretrained=True,
        use_disentangle=False, use_vib=False, use_feat_aug=False,
        use_skip_filter=False,
        num_domains=2,
    )
    model_base.train()
    out_b, l_b = model_base(x, im, cid)
    print(f"Baseline main: {out_b['main'].shape}")
    total_ddib_loss = sum(v.item() for v in l_b.values())
    print(f"Total DDIB+skip loss: {total_ddib_loss:.6f}  (should be ~0)")

    print("\nAll tests passed!")