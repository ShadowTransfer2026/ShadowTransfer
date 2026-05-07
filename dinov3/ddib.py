"""
Disentangled Domain-Invariant Bottleneck (DDIB)

A plug-and-play module inserted at the encoder-decoder interface of any
dense prediction architecture. Three independently toggleable components:

  C1 — Feature Disentanglement (FeatureDisentangler)
       Dual projection heads split features into task-relevant and
       domain-specific subspaces. HSIC independence penalty ensures
       statistical independence. Domain classifier trains the domain
       subspace to capture city identity.

  C2 — Variational Information Bottleneck (VariationalInformationBottleneck)
       Per-pixel VIB compresses task features, discarding residual
       domain information. Intensity-adaptive beta applies stronger
       compression at bright surfaces (where Thread 1b showed the
       largest transfer gap).

  C3 — Stochastic Feature Augmentation (StochasticFeatureAugmentation)
       Random AdaIN perturbation + cross-domain statistic mixing during
       training. Forces the decoder to handle diverse feature distributions,
       preventing decoder miscalibration (Experiment A finding).

Usage:
    ddib = DDIB(
        in_channels=1536,       # 4 x 384 concatenated encoder features
        embed_dim=384,          # output dim matching decoder expectation
        use_disentangle=True,   # C1
        use_vib=True,           # C2
        use_feat_aug=True,      # C3
        num_domains=2           # number of training cities
    )

    task_features, ddib_losses = ddib(features_concat, intensity_map, city_ids)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# HSIC utilities
# ---------------------------------------------------------------------------

def compute_rbf_kernel(X, sigma_sq=None):
    """
    Compute Gaussian RBF kernel matrix.

    Args:
        X: [N, D] feature matrix.
        sigma_sq: Bandwidth squared. If None, uses the median heuristic
                  (computed without gradient to avoid instability).
    Returns:
        K: [N, N] kernel matrix.
    """
    # Pairwise squared Euclidean distances via expansion trick
    XXT = X @ X.t()
    diag = XXT.diag().unsqueeze(1)
    dist_sq = (diag + diag.t() - 2 * XXT).clamp(min=0)

    if sigma_sq is None:
        # Median heuristic — detach so bandwidth is treated as a constant
        with torch.no_grad():
            positive = dist_sq[dist_sq > 1e-10]
            if positive.numel() > 0:
                sigma_sq = positive.median() / (2 * np.log(X.shape[0] + 1))
            else:
                sigma_sq = torch.tensor(1.0, device=X.device)
            sigma_sq = sigma_sq.clamp(min=1e-10)

    K = torch.exp(-dist_sq / (2 * sigma_sq))
    return K


def compute_hsic(X, Y, num_samples=1024):
    """
    Biased HSIC estimator with RBF kernel.

    Deterministic subsampling: when the number of spatial locations exceeds
    *num_samples*, evenly-spaced indices are selected (no randomness).

    Args:
        X: [N, D1] features from subspace 1.
        Y: [N, D2] features from subspace 2.
        num_samples: maximum number of samples for kernel computation.
    Returns:
        Scalar HSIC estimate (differentiable w.r.t. X and Y).
    """
    N = X.shape[0]

    # Deterministic thinning
    if N > num_samples:
        indices = torch.linspace(0, N - 1, num_samples).long().to(X.device)
        X = X[indices]
        Y = Y[indices]
        N = num_samples

    if N < 5:
        return torch.tensor(0.0, device=X.device, requires_grad=True)

    K = compute_rbf_kernel(X)   # [N, N]
    L = compute_rbf_kernel(Y)   # [N, N]

    # Centering matrix  H = I - (1/n) 11^T
    H = torch.eye(N, device=X.device) - 1.0 / N

    # tr(KHLH) = sum( (KH) ⊙ (LH) )   (both H and L symmetric)
    KH = K @ H
    LH = L @ H
    hsic = (KH * LH).sum() / ((N - 1) ** 2)

    return hsic


# ---------------------------------------------------------------------------
# Component 1 — Feature Disentanglement
# ---------------------------------------------------------------------------

class FeatureDisentangler(nn.Module):
    """
    Dual-head projection that separates encoder features into a
    task-relevant subspace and a domain-specific subspace, with an
    HSIC independence penalty and a domain classification objective.

    During inference only the task pathway is used; the domain pathway
    (and its losses) can be ignored.
    """

    def __init__(self, in_channels, embed_dim, num_domains=2,
                 hsic_samples=1024):
        """
        Args:
            in_channels: dimensionality of the concatenated encoder features.
            embed_dim:   output dimensionality for the task subspace
                         (must match decoder expectation).
            num_domains: number of source domains (cities) during training.
            hsic_samples: number of spatial samples for HSIC estimation.
        """
        super().__init__()
        self.hsic_samples = hsic_samples

        half = in_channels // 2

        # Task projection  in_channels → half
        self.proj_task = nn.Sequential(
            nn.Conv2d(in_channels, half, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )

        # Domain projection  in_channels → half
        self.proj_domain = nn.Sequential(
            nn.Conv2d(in_channels, half, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )

        # Map task features to the decoder's expected channel count
        self.task_to_embed = nn.Sequential(
            nn.Conv2d(half, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Lightweight domain classifier (operates on domain features)
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(half, half // 4),
            nn.ReLU(inplace=True),
            nn.Linear(half // 4, num_domains),
        )
        self.domain_ce = nn.CrossEntropyLoss()

    def forward(self, x, city_ids=None):
        """
        Args:
            x:        [B, in_channels, H, W]
            city_ids: [B] integer city labels (None during inference).
        Returns:
            task_out: [B, embed_dim, H, W]
            losses:   dict  {'hsic_loss': …, 'domain_loss': …}
        """
        F_task   = self.proj_task(x)       # [B, half, H, W]
        F_domain = self.proj_domain(x)     # [B, half, H, W]

        task_out = self.task_to_embed(F_task)  # [B, embed_dim, H, W]

        losses = {}

        # ---- HSIC independence penalty ----
        B, C_t, H, W = F_task.shape
        C_d = F_domain.shape[1]
        ft = F_task.permute(0, 2, 3, 1).reshape(-1, C_t)   # [BHW, C_t]
        fd = F_domain.permute(0, 2, 3, 1).reshape(-1, C_d)  # [BHW, C_d]
        losses['hsic_loss'] = compute_hsic(ft, fd,
                                           num_samples=self.hsic_samples)

        # ---- Domain classification loss ----
        if city_ids is not None:
            logits = self.domain_classifier(F_domain)   # [B, num_domains]
            losses['domain_loss'] = self.domain_ce(logits, city_ids)
        else:
            losses['domain_loss'] = torch.tensor(0.0, device=x.device)

        return task_out, losses


# ---------------------------------------------------------------------------
# Component 2 — Variational Information Bottleneck
# ---------------------------------------------------------------------------

class VariationalInformationBottleneck(nn.Module):
    """
    Per-pixel VIB with intensity-adaptive compression.

    At bright surfaces (high median intensity inside the shadow mask),
    the bottleneck is tighter, matching the Thread 1b observation that
    the transfer gap widens monotonically with surface brightness.

    During inference the stochastic reparameterisation is skipped;
    only the deterministic mean μ is used.
    """

    def __init__(self, embed_dim, beta_base=0.001, beta_scale=0.01):
        super().__init__()
        self.beta_base = beta_base
        self.beta_scale = beta_scale

        self.fc_mu     = nn.Conv2d(embed_dim, embed_dim, 1)
        self.fc_logvar = nn.Conv2d(embed_dim, embed_dim, 1)

        # Intensity → per-pixel beta scaling
        self.intensity_to_beta = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, intensity_map=None, training=True):
        """
        Args:
            x:             [B, C, H_f, W_f]
            intensity_map: [B, 1, H_img, W_img]  range [0, 1]  (can be None)
            training:      bool
        Returns:
            z:      [B, C, H_f, W_f]
            losses: {'kl_loss': …}
        """
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        if training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        # KL(q(z|x) || N(0,I)) per pixel, summed over channels
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)   # [B,C,H,W]
        kl = kl.sum(dim=1, keepdim=True)                       # [B,1,H,W]

        # Adaptive beta from surface intensity
        if intensity_map is not None:
            _, _, H_f, W_f = x.shape
            inten = F.adaptive_avg_pool2d(intensity_map, (H_f, W_f))  # [B,1,H_f,W_f]
            B, _, H, W = inten.shape
            flat = inten.permute(0, 2, 3, 1).reshape(-1, 1)           # [BHW, 1]
            scale = self.intensity_to_beta(flat)                       # [BHW, 1]
            beta_map = (self.beta_base
                        + self.beta_scale * scale.reshape(B, H, W, 1).permute(0, 3, 1, 2))
        else:
            beta_map = self.beta_base

        kl_loss = (beta_map * kl).mean()
        return z, {'kl_loss': kl_loss}


# ---------------------------------------------------------------------------
# Component 3 — Stochastic Feature Augmentation
# ---------------------------------------------------------------------------

class StochasticFeatureAugmentation(nn.Module):
    """
    During training, randomly perturbs feature statistics to make the
    decoder robust to distribution shift.  Adds zero learnable parameters.

    Two augmentation modes (applied stochastically):
      1. Random style: re-normalise channel statistics with random γ, β.
      2. Cross-domain mixing: swap channel stats between samples from
         different source cities in the same batch.
    """

    def __init__(self, sigma_style=0.5, sigma_shift=0.3,
                 p_aug=0.5, p_mix=0.3):
        super().__init__()
        self.sigma_style = sigma_style
        self.sigma_shift = sigma_shift
        self.p_aug = p_aug
        self.p_mix = p_mix

    def forward(self, z, city_ids=None, training=True):
        """
        Args:
            z:        [B, C, H, W]
            city_ids: [B]  (optional; needed for cross-domain mixing)
            training: bool — augmentation is a no-op at inference.
        Returns:
            z (potentially augmented): [B, C, H, W]
        """
        if not training:
            return z

        B, C, H, W = z.shape
        eps = 1e-5

        # --- Random style perturbation ---
        if torch.rand(1).item() < self.p_aug:
            mu    = z.mean(dim=(2, 3), keepdim=True)              # [B,C,1,1]
            sigma = z.std(dim=(2, 3), keepdim=True).clamp(min=eps)

            gamma = torch.empty(B, C, 1, 1, device=z.device).log_normal_(
                0, self.sigma_style)
            beta  = torch.empty(B, C, 1, 1, device=z.device).normal_(
                0, self.sigma_shift)

            z = gamma * (z - mu) / sigma + mu + beta

        # --- Cross-domain statistic mixing ---
        if city_ids is not None and B > 1 and torch.rand(1).item() < self.p_mix:
            unique = city_ids.unique()
            if len(unique) > 1:
                mu    = z.mean(dim=(2, 3), keepdim=True)
                sigma = z.std(dim=(2, 3), keepdim=True).clamp(min=eps)
                z_out = z.clone()
                for i in range(B):
                    others = torch.where(city_ids != city_ids[i])[0]
                    if others.numel() > 0:
                        j = others[torch.randint(others.numel(), (1,)).item()]
                        z_out[i] = sigma[j] * (z[i] - mu[i]) / sigma[i] + mu[j]
                z = z_out

        return z


# ---------------------------------------------------------------------------
# Full DDIB wrapper
# ---------------------------------------------------------------------------

class DDIB(nn.Module):
    """
    Disentangled Domain-Invariant Bottleneck.

    Sits between the encoder output (concatenated multi-block features)
    and the decoder upsampling stages.

    Args:
        in_channels:      concatenated encoder feature channels (e.g. 1536).
        embed_dim:        output channels the decoder expects (e.g. 384).
        use_disentangle:  enable Component 1.
        use_vib:          enable Component 2.
        use_feat_aug:     enable Component 3.
        num_domains:      number of source domains during training.
        hsic_samples:     spatial samples for HSIC (default 1024).
        vib_beta_base:    VIB minimum compression.
        vib_beta_scale:   VIB intensity-adaptive compression range.
        aug_sigma_style:  C3 random-style log-normal σ.
        aug_sigma_shift:  C3 random-shift normal σ.
        aug_p_aug:        C3 probability of random-style perturbation.
        aug_p_mix:        C3 probability of cross-domain mixing.
    """

    def __init__(self, in_channels=1536, embed_dim=384,
                 use_disentangle=True, use_vib=True, use_feat_aug=True,
                 num_domains=2,
                 hsic_samples=1024,
                 vib_beta_base=0.001, vib_beta_scale=0.01,
                 aug_sigma_style=0.5, aug_sigma_shift=0.3,
                 aug_p_aug=0.5, aug_p_mix=0.3):
        super().__init__()

        self.use_disentangle = use_disentangle
        self.use_vib         = use_vib
        self.use_feat_aug    = use_feat_aug

        # C1
        if use_disentangle:
            self.disentangler = FeatureDisentangler(
                in_channels=in_channels,
                embed_dim=embed_dim,
                num_domains=num_domains,
                hsic_samples=hsic_samples,
            )
        else:
            # Simple linear fusion when C1 is off
            self.simple_fusion = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )

        # C2
        if use_vib:
            self.vib = VariationalInformationBottleneck(
                embed_dim=embed_dim,
                beta_base=vib_beta_base,
                beta_scale=vib_beta_scale,
            )

        # C3
        if use_feat_aug:
            self.feat_aug = StochasticFeatureAugmentation(
                sigma_style=aug_sigma_style,
                sigma_shift=aug_sigma_shift,
                p_aug=aug_p_aug,
                p_mix=aug_p_mix,
            )

        # Log configuration
        active = []
        if use_disentangle: active.append('C1-Disentangle')
        if use_vib:         active.append('C2-VIB')
        if use_feat_aug:    active.append('C3-FeatAug')
        print(f'DDIB initialised  |  in={in_channels} → out={embed_dim}  |'
              f'  active: {", ".join(active) if active else "none (pass-through)"}')

    # ------------------------------------------------------------------
    def forward(self, features_concat, intensity_map=None, city_ids=None):
        """
        Args:
            features_concat: [B, in_channels, H, W]
            intensity_map:   [B, 1, H_img, W_img]  (for C2; None at eval is OK)
            city_ids:        [B] int64  (for C1/C3; None at eval is OK)
        Returns:
            x:           [B, embed_dim, H, W]  — features ready for decoder.
            ddib_losses: dict with keys that are present only for active
                         components  (e.g. 'hsic_loss', 'domain_loss',
                         'kl_loss').  Empty dict when everything is off.
        """
        training = self.training
        ddib_losses = {}

        # --- C1 or simple fusion ---
        if self.use_disentangle:
            x, c1_losses = self.disentangler(features_concat, city_ids)
            ddib_losses.update(c1_losses)
        else:
            x = self.simple_fusion(features_concat)

        # --- C2 ---
        if self.use_vib:
            x, c2_losses = self.vib(x, intensity_map, training=training)
            ddib_losses.update(c2_losses)

        # --- C3 ---
        if self.use_feat_aug:
            x = self.feat_aug(x, city_ids, training=training)

        return x, ddib_losses