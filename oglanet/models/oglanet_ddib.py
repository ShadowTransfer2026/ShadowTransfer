"""
OGLANet + DDIB (Option B): Disentangled Domain-Invariant Bottleneck
with lightweight skip-connection domain filters for OGLANet.

Option B architecture:
  - Full DDIB at the bottleneck (feat5: 1024ch, 6×6)
  - Lightweight SkipDomainFilter on each skip connection (feat1-feat4)

The skip filters are small 1×1-conv bottleneck modules with:
  - Channel compression (reduction ratio r=4)
  - Optional mini-VIB reparameterisation for information compression
  - Gated residual connection (gate initialised to 0 for stable start)

This ensures domain-specific information is suppressed not only in
the bottleneck but also in the skip connections that flow directly
into the DFFM dense feature fusion bridge.

Architecture flow:
    Image [B, 3/4, 384, 384]
      → GLAMEncoder       → {feat1..feat5}
      → DDIB(feat5)       → feat5_processed  [B,1024, 6, 6]  (bottleneck)
      → SkipFilter(feat1) → [B, 64,  96,96]  (skip 1)
      → SkipFilter(feat2) → [B, 128, 48,48]  (skip 2)
      → SkipFilter(feat3) → [B, 256, 24,24]  (skip 3)
      → SkipFilter(feat4) → [B, 512, 12,12]  (skip 4)
      → DFFM({filtered feat1..feat4, feat5_processed})
      → Decoder → OAM → P1..P6

Requires:
    Copy  dinov3/ddib.py  →  oglanet/models/ddib.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .glam import GLAMEncoder
from .dffm import DFFM
from .decoder import Decoder
from .oam import OAM
from .ddib import DDIB


# ======================================================================
# Lightweight skip-connection domain filter (same as MAMNet Option B)
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
            logvar = torch.clamp(logvar, min=-10.0, max=2.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            h = mu + eps * std
            kl_loss = -0.5 * torch.mean(
                1.0 + logvar - mu.pow(2) - logvar.exp())
        elif self.use_vib:
            h = self.mu_proj(h)

        h = self.expand(h)

        return x + self.gate * h, kl_loss


# ======================================================================
# OGLANet + DDIB (Option B)
# ======================================================================

class OGLANetDDIB(nn.Module):
    """
    OGLANet with full DDIB at the bottleneck and lightweight
    SkipDomainFilters on encoder skip connections.

    Args:
        num_classes:      output classes (default 2).
        pretrained:       load pretrained ResNet-34 weights.
        img_size:         input spatial size (default 384).
        use_contrast:     4-channel RGBC input (default False).

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
                          to the main VIB KL (default 0.1).
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        img_size=384,
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
        super(OGLANetDDIB, self).__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.use_contrast = use_contrast
        self.use_skip_filter = use_skip_filter
        self.skip_kl_weight = skip_kl_weight

        # 1. GLAM Encoder (unchanged) -----------------------------------------
        print('Initialising GLAM Encoder …')
        self.encoder = GLAMEncoder(
            pretrained=pretrained, use_contrast=use_contrast)

        # 2. DDIB at bottleneck ------------------------------------------------
        #    feat5 is [B, 1024, 6, 6] for 384×384 input
        print('Initialising DDIB at bottleneck …')
        self.ddib = DDIB(
            in_channels=1024,
            embed_dim=1024,
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

        # 3. Skip domain filters -----------------------------------------------
        # feat1: 64ch, feat2: 128ch, feat3: 256ch, feat4: 512ch
        skip_channels = [64, 128, 256, 512]
        skip_vib = use_vib and use_skip_filter

        if use_skip_filter:
            print(f'Initialising skip domain filters '
                  f'(reduction={skip_reduction}, vib={skip_vib}) …')
            self.skip_filters = nn.ModuleList([
                SkipDomainFilter(
                    channels=ch,
                    reduction=skip_reduction,
                    use_vib=skip_vib,
                ) for ch in skip_channels
            ])
        else:
            self.skip_filters = None

        # 4. Dense Feature Fusion Module (unchanged) ---------------------------
        print('Initialising DFFM …')
        self.dffm = DFFM()

        # 5. Decoder (unchanged) -----------------------------------------------
        print('Initialising Decoder …')
        self.decoder = Decoder(target_size=(img_size, img_size))

        # 6. Omni-scale Aggregation Module (unchanged) -------------------------
        print('Initialising OAM …')
        self.oam = OAM(num_classes=num_classes, target_size=(img_size, img_size))

        # ---- Summary ---------------------------------------------------------
        total  = sum(p.numel() for p in self.parameters())
        train_ = sum(p.numel() for p in self.parameters() if p.requires_grad)
        enc_p  = sum(p.numel() for p in self.encoder.parameters())
        ddib_p = sum(p.numel() for p in self.ddib.parameters())
        skip_p = (sum(p.numel() for p in self.skip_filters.parameters())
                  if self.skip_filters is not None else 0)
        rest_p = total - enc_p - ddib_p - skip_p
        print(f'\nOGLANet-DDIB (Option B) Shadow Detector:')
        print(f'  Total params:     {total:,}')
        print(f'  Trainable params: {train_:,}')
        print(f'  Encoder (GLAM):   {enc_p:,}')
        print(f'  DDIB (bottleneck):{ddib_p:,}')
        print(f'  Skip filters:     {skip_p:,}')
        print(f'  DFFM+Decoder+OAM: {rest_p:,}')
        if self.skip_filters is not None:
            for i, (ch, sf) in enumerate(zip(skip_channels, self.skip_filters)):
                sf_p = sum(p.numel() for p in sf.parameters())
                print(f'    skip{i+1} ({ch:>3}ch): {sf_p:,} params  '
                      f'(vib={sf.use_vib})')

    # ------------------------------------------------------------------
    def forward(self, x, intensity_map=None, city_ids=None):
        """
        Args:
            x:             [B, 3/4, H, W] input image (RGB or RGBC).
            intensity_map: [B, 1, H, W] pre-normalisation grayscale [0,1].
            city_ids:      [B] int64 domain labels.

        Returns:
            If training:
                predictions: dict {'p1'…'p6'}, each [B, num_classes, H, W]
                all_losses:  dict with DDIB + skip loss terms
            If eval:
                predictions: P6 tensor [B, num_classes, H, W]
                all_losses:  dict (mostly zeros at eval)
        """
        # 1. Encoder -----------------------------------------------------------
        encoder_features = self.encoder(x)
        # feat1 [B,64,96,96], feat2 [B,128,48,48],
        # feat3 [B,256,24,24], feat4 [B,512,12,12],
        # feat5 [B,1024,6,6]

        # 2. DDIB on bottleneck ------------------------------------------------
        feat5_processed, ddib_losses = self.ddib(
            encoder_features['feat5'], intensity_map, city_ids)
        encoder_features['feat5'] = feat5_processed

        # 3. Skip domain filters -----------------------------------------------
        skip_kl_total = torch.tensor(0.0, device=x.device)

        if self.skip_filters is not None:
            skip_keys = ['feat1', 'feat2', 'feat3', 'feat4']
            for i, key in enumerate(skip_keys):
                filtered, kl_i = self.skip_filters[i](encoder_features[key])
                encoder_features[key] = filtered
                skip_kl_total = skip_kl_total + kl_i
            skip_kl_total = skip_kl_total / len(self.skip_filters)

        # 4. Aggregate all losses ----------------------------------------------
        all_losses = dict(ddib_losses)
        all_losses['skip_kl_loss'] = skip_kl_total * self.skip_kl_weight

        # 5. DFFM (receives filtered features) ---------------------------------
        dffm_features = self.dffm(encoder_features)

        # 6. Decoder -----------------------------------------------------------
        decoder_features = self.decoder(dffm_features)

        # 7. OAM --------------------------------------------------------------
        predictions = self.oam(decoder_features)

        if self.training:
            return predictions, all_losses
        else:
            return predictions['p6'], all_losses

    # ------------------------------------------------------------------
    def get_predictions(self, x, intensity_map=None):
        """Inference helper — returns [B, H, W] integer predictions."""
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
    print("Testing OGLANet-DDIB (Option B)")
    print("=" * 60)

    # ---- Full DDIB + skip filters ----
    model = OGLANetDDIB(
        num_classes=2, pretrained=False, img_size=384,
        use_contrast=False,
        use_disentangle=True, use_vib=True, use_feat_aug=True,
        use_skip_filter=True,
        num_domains=2, skip_reduction=4, skip_kl_weight=0.1,
    )

    x   = torch.randn(2, 3, 384, 384)
    im  = torch.rand(2, 1, 384, 384)
    cid = torch.tensor([0, 1])

    model.train()
    preds, losses = model(x, im, cid)
    print("\nTraining outputs:")
    for k, v in preds.items():
        print(f"  {k}: {v.shape}")
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")
    print("Skip gates:", model.get_skip_gate_values())

    model.eval()
    p6, losses_e = model(x)
    print(f"\nInference output: {p6.shape}")
    print(f"  skip_kl_loss: {losses_e['skip_kl_loss'].item():.6f}")

    # ---- Skip filters OFF (Option A equivalent) ----
    print("\n--- Option A mode (skip_filter=False) ---")
    model_a = OGLANetDDIB(
        num_classes=2, pretrained=False, img_size=384,
        use_disentangle=True, use_vib=True, use_feat_aug=True,
        use_skip_filter=False,
        num_domains=2,
    )
    model_a.train()
    preds_a, l_a = model_a(x, im, cid)
    print(f"P6: {preds_a['p6'].shape}")
    print(f"skip_kl_loss: {l_a['skip_kl_loss'].item():.6f}  (should be 0)")

    # ---- No DDIB baseline ----
    print("\n--- Baseline (no DDIB, no skip filters) ---")
    model_b = OGLANetDDIB(
        num_classes=2, pretrained=False, img_size=384,
        use_disentangle=False, use_vib=False, use_feat_aug=False,
        use_skip_filter=False,
        num_domains=2,
    )
    model_b.train()
    preds_b, l_b = model_b(x, im, cid)
    total_loss = sum(v.item() for v in l_b.values())
    print(f"Baseline P6: {preds_b['p6'].shape}")
    print(f"Total DDIB+skip loss: {total_loss:.6f}  (should be ~0)")

    print("\nAll tests passed!")