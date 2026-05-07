"""
Multi-level Contrastive Learning Losses for mCL-LC
WACV 2023 - Tang et al.

Implements:
- Feature-level contrastive loss (style features)
- Semantic-level contrastive loss
- Boundary-aware negative sampling (BANE)
- Local consistency via mutual information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    Used in SimCLR and other contrastive learning methods.
    """
    
    def __init__(self, temperature=0.5):
        """
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss between two sets of embeddings.
        
        Args:
            z_i: Embeddings from view 1 [B, D]
            z_j: Embeddings from view 2 [B, D]
        
        Returns:
            Scalar loss value
        """
        B = z_i.size(0)
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t())  # [2B, 2B]
        sim_matrix = sim_matrix / self.temperature
        
        # Create labels: positive pairs are (i, B+i) and (B+i, i)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Positive similarities
        pos_sim = torch.cat([
            torch.diag(sim_matrix, B),
            torch.diag(sim_matrix, -B)
        ], dim=0)  # [2B]
        
        # Compute loss
        loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
        loss = loss.mean()
        
        return loss


class FeatureLevelContrastiveLoss(nn.Module):
    """
    Feature-level contrastive loss using style features (mean & std).
    Learns domain-invariant low-level representations.
    """
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.ntxent = NTXentLoss(temperature=temperature)
    
    def extract_style_features(self, features):
        """
        Extract style features (channel-wise mean and std).
        
        Args:
            features: Feature maps [B, C, H, W]
        
        Returns:
            Style features [B, 2*C]
        """
        # Channel-wise mean and std
        mean = features.mean(dim=(2, 3))  # [B, C]
        std = features.std(dim=(2, 3))    # [B, C]
        
        # Concatenate
        style = torch.cat([mean, std], dim=1)  # [B, 2*C]
        
        return style
    
    def forward(self, features_i, features_j):
        """
        Compute feature-level contrastive loss.
        
        Args:
            features_i: Features from augmented view 1 [B, C, H, W]
            features_j: Features from augmented view 2 [B, C, H, W]
        
        Returns:
            Scalar loss value
        """
        # Extract style features
        style_i = self.extract_style_features(features_i)
        style_j = self.extract_style_features(features_j)
        
        # Compute contrastive loss
        loss = self.ntxent(style_i, style_j)
        
        return loss


class BoundaryAwareNegativeSampling:
    """
    BANE: Boundary-Aware Negative Sampling
    
    Identifies hard negatives near class boundaries for better contrastive learning.
    """
    
    def __init__(self, boundary_threshold=5):
        """
        Args:
            boundary_threshold: Distance threshold for boundary pixels (in pixels)
        """
        self.boundary_threshold = boundary_threshold
    
    def extract_boundaries(self, masks):
        """
        Extract boundary pixels from segmentation masks.
        
        Args:
            masks: Binary masks [B, H, W]
        
        Returns:
            Boundary masks [B, H, W] (1 = boundary, 0 = non-boundary)
        """
        B, H, W = masks.shape
        
        # Simple boundary detection using dilation
        import torch.nn.functional as F
        
        # Create kernel for dilation (3x3)
        kernel = torch.ones(1, 1, 3, 3, device=masks.device)
        
        # Dilate and erode
        masks_float = masks.float().unsqueeze(1)  # [B, 1, H, W]
        dilated = F.conv2d(masks_float, kernel, padding=1)
        dilated = (dilated > 0).float()
        
        eroded = F.conv2d(masks_float, kernel, padding=1)
        eroded = (eroded == 9).float()  # All 9 neighbors are 1
        
        # Boundary = dilated - eroded
        boundaries = (dilated - eroded).squeeze(1)  # [B, H, W]
        boundaries = (boundaries > 0).float()
        
        return boundaries
    
    def sample_negatives(self, embeddings, masks, num_negatives=256):
        """
        Sample hard negatives near class boundaries.
        
        Args:
            embeddings: Semantic embeddings [B, D, H, W]
            masks: Ground truth masks [B, H, W]
            num_negatives: Number of negative samples to extract
        
        Returns:
            Negative samples [B, num_negatives, D]
        """
        B, D, H, W = embeddings.shape

        # Resize masks to match embedding spatial dimensions
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(
                masks.unsqueeze(1).float(),  # [B, 1, H_mask, W_mask]
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()  # [B, H, W]
        
        # Extract boundaries
        boundaries = self.extract_boundaries(masks)  # [B, H, W]
        
        # Diagnostic print
        total_boundary_pixels = torch.sum(boundaries).item()
        # print(f"[BANE] Total boundary pixels: {total_boundary_pixels} / {B*H*W} ({100*total_boundary_pixels/(B*H*W):.2f}%)")
        
        # Sample from boundary regions
        negatives_list = []
        
        for b in range(B):
            boundary_mask = boundaries[b]
            boundary_indices = torch.nonzero(boundary_mask, as_tuple=False)  # [N, 2]
            
            if len(boundary_indices) == 0:
                # If no boundaries, sample randomly
                # print(f"[BANE] Warning: No boundaries found for image {b}, sampling randomly")
                random_indices = torch.randint(0, H * W, (num_negatives,), device=embeddings.device)
                h_idx = random_indices // W
                w_idx = random_indices % W
            else:
                # Sample from boundary pixels
                if len(boundary_indices) < num_negatives:
                    # Repeat if not enough boundary pixels
                    sample_indices = torch.randint(0, len(boundary_indices), (num_negatives,), device=embeddings.device)
                else:
                    sample_indices = torch.randperm(len(boundary_indices), device=embeddings.device)[:num_negatives]
                
                sampled_coords = boundary_indices[sample_indices]  # [num_negatives, 2]
                h_idx = sampled_coords[:, 0]
                w_idx = sampled_coords[:, 1]
            
            # Extract embeddings at sampled locations
            neg_samples = embeddings[b, :, h_idx, w_idx].t()  # [num_negatives, D]
            negatives_list.append(neg_samples)
        
        negatives = torch.stack(negatives_list, dim=0)  # [B, num_negatives, D]
        
        return negatives


class SemanticLevelContrastiveLoss(nn.Module):
    """
    Semantic-level contrastive loss with BANE sampling.
    Learns high-level semantic representations.
    """
    
    def __init__(self, temperature=0.5, use_bane=True, num_samples=256):
        super().__init__()
        self.temperature = temperature
        self.use_bane = use_bane
        self.num_samples = num_samples  # Number of spatial samples per image
        
        if use_bane:
            self.bane = BoundaryAwareNegativeSampling()
    
    def sample_positive_features(self, features, masks, num_samples=256):
        """
        Sample positive features from same-class regions (non-boundary).
        
        Args:
            features: Feature maps [B, C, H, W]
            masks: Ground truth masks [B, H, W]
            num_samples: Number of samples per image
        
        Returns:
            Sampled features [B, num_samples, C]
        """
        B, C, H, W = features.shape
        
        # Resize masks to match feature spatial dimensions
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(
                masks.unsqueeze(1).float(),  # [B, 1, H_mask, W_mask]
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()  # [B, H, W]
        
        sampled_features = []
        
        for b in range(B):
            # For each class, sample from interior regions
            mask = masks[b]
            
            # Sample from shadow regions (class 1)
            shadow_mask = (mask == 1)
            shadow_indices = torch.nonzero(shadow_mask, as_tuple=False)
            
            # Sample from background regions (class 0)
            bg_mask = (mask == 0)
            bg_indices = torch.nonzero(bg_mask, as_tuple=False)
            
            # Sample half from shadow, half from background
            samples_per_class = num_samples // 2
            
            if len(shadow_indices) > 0:
                if len(shadow_indices) < samples_per_class:
                    sample_idx = torch.randint(0, len(shadow_indices), (samples_per_class,), device=features.device)
                else:
                    sample_idx = torch.randperm(len(shadow_indices), device=features.device)[:samples_per_class]
                shadow_coords = shadow_indices[sample_idx]
                shadow_feats = features[b, :, shadow_coords[:, 0], shadow_coords[:, 1]].t()
            else:
                shadow_feats = torch.zeros(samples_per_class, C, device=features.device)
            
            if len(bg_indices) > 0:
                if len(bg_indices) < samples_per_class:
                    sample_idx = torch.randint(0, len(bg_indices), (samples_per_class,), device=features.device)
                else:
                    sample_idx = torch.randperm(len(bg_indices), device=features.device)[:samples_per_class]
                bg_coords = bg_indices[sample_idx]
                bg_feats = features[b, :, bg_coords[:, 0], bg_coords[:, 1]].t()
            else:
                bg_feats = torch.zeros(samples_per_class, C, device=features.device)
            
            # Combine shadow and background samples
            batch_samples = torch.cat([shadow_feats, bg_feats], dim=0)
            sampled_features.append(batch_samples)
        
        sampled_features = torch.stack(sampled_features, dim=0)  # [B, num_samples, C]
        return sampled_features
    
    def forward(self, features_i, features_j, masks_i=None, masks_j=None):
        """
        Compute semantic-level contrastive loss with optional BANE.
        
        Args:
            features_i: Features from view 1 [B, C, H, W]
            features_j: Features from view 2 [B, C, H, W]
            masks_i: Masks for view 1 [B, H, W] (optional, for BANE)
            masks_j: Masks for view 2 [B, H, W] (optional, for BANE)
        
        Returns:
            Scalar loss value
        """
        B, C, H, W = features_i.shape
        
        if self.use_bane and masks_i is not None:
            # BANE: Sample positives from interior, negatives from boundaries
            
            # Sample positive features from both views
            pos_features_i = self.sample_positive_features(features_i, masks_i, self.num_samples)  # [B, K, C]
            pos_features_j = self.sample_positive_features(features_j, masks_j, self.num_samples)  # [B, K, C]
            
            # Average over spatial samples for each image
            semantic_i = pos_features_i.mean(dim=1)  # [B, C]
            semantic_j = pos_features_j.mean(dim=1)  # [B, C]
            
            # Sample negative features from boundaries (hard negatives)
            neg_features = self.bane.sample_negatives(features_i, masks_i, num_negatives=self.num_samples)  # [B, K, C]
            
            # Compute contrastive loss with hard negatives
            loss = self.contrastive_loss_with_negatives(semantic_i, semantic_j, neg_features)
            
            # if torch.distributed.is_initialized() or True:  # Always print for debugging
            #     print(f"[BANE] Sampled {self.num_samples} positives and {neg_features.shape[1]} negatives from boundaries")
            
        else:
            # Standard global pooling (no BANE)
            semantic_i = F.adaptive_avg_pool2d(features_i, 1).view(B, C)
            semantic_j = F.adaptive_avg_pool2d(features_j, 1).view(B, C)
            
            # Standard NT-Xent loss
            ntxent = NTXentLoss(temperature=self.temperature)
            loss = ntxent(semantic_i, semantic_j)
        
        return loss
    
    def contrastive_loss_with_negatives(self, z_i, z_j, neg_features):
        """
        Compute contrastive loss with explicit negative samples.
        
        Args:
            z_i: Positive features from view 1 [B, C]
            z_j: Positive features from view 2 [B, C]
            neg_features: Negative features from boundaries [B, K, C]
        
        Returns:
            Scalar loss
        """
        B, C = z_i.shape
        K = neg_features.shape[1]
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)  # [B, C]
        z_j = F.normalize(z_j, dim=1)  # [B, C]
        neg_features = F.normalize(neg_features, dim=2)  # [B, K, C]
        
        # Positive similarity
        pos_sim = torch.sum(z_i * z_j, dim=1) / self.temperature  # [B]
        
        # Negative similarities
        # For each positive pair (i,j), compute similarity with all negatives
        neg_sim_i = torch.bmm(neg_features, z_i.unsqueeze(2)).squeeze(2) / self.temperature  # [B, K]
        neg_sim_j = torch.bmm(neg_features, z_j.unsqueeze(2)).squeeze(2) / self.temperature  # [B, K]
        
        # Also include the other view as negative
        neg_sim_cross = torch.sum(z_i * z_j, dim=1, keepdim=True) / self.temperature  # [B, 1] (but this is positive, skip)
        
        # Combine all negative similarities
        all_neg_sim = torch.cat([neg_sim_i, neg_sim_j], dim=1)  # [B, 2K]
        
        # InfoNCE loss
        loss = -pos_sim + torch.logsumexp(torch.cat([pos_sim.unsqueeze(1), all_neg_sim], dim=1), dim=1)
        
        return loss.mean()


class LocalConsistencyLoss(nn.Module):
    """
    Local Consistency Loss via Mutual Information Maximization.
    
    Encourages local spatial consistency in representations.
    """
    
    def __init__(self, num_patches=49):
        """
        Args:
            num_patches: Number of local patches to sample (7x7 grid)
        """
        super().__init__()
        self.num_patches = num_patches
    
    def compute_local_mi(self, features, patch_size=7):
        """
        Compute local mutual information between neighboring patches.
        
        Args:
            features: Feature maps [B, C, H, W]
            patch_size: Size of local patches
        
        Returns:
            Local MI loss (lower is better consistency)
        """
        B, C, H, W = features.shape
        
        # Divide into patches using unfold
        patches = F.unfold(features, kernel_size=patch_size, stride=patch_size)  # [B, C*patch_size^2, num_patches]
        patches = patches.permute(0, 2, 1)  # [B, num_patches, C*patch_size^2]
        
        # Compute pairwise cosine similarity between patches
        patches_norm = F.normalize(patches, dim=2)  # [B, num_patches, C*patch_size^2]
        
        # Similarity matrix for each batch
        similarity = torch.bmm(patches_norm, patches_norm.transpose(1, 2))  # [B, num_patches, num_patches]
        
        # Local consistency: higher similarity = lower loss
        # We want neighboring patches to be similar
        # Create adjacency matrix for neighboring patches (simplified: all patches)
        consistency_loss = 1 - similarity.mean()
        
        return consistency_loss
    
    def forward(self, features_i, features_j):
        """
        Compute local consistency loss between two views.
        
        Args:
            features_i: Features from view 1 [B, C, H, W]
            features_j: Features from view 2 [B, C, H, W]
        
        Returns:
            Scalar loss value
        """
        # Compute MI for each view
        mi_i = self.compute_local_mi(features_i)
        mi_j = self.compute_local_mi(features_j)
        
        # Average
        loss = (mi_i + mi_j) / 2.0
        
        return loss


class mCLLCLoss(nn.Module):
    """
    Complete mCL-LC Loss combining all components.
    
    Total Loss = seg_loss + aux_loss + lambda_fl * feature_loss + lambda_sl * semantic_loss + lambda_lc * local_loss
    """
    
    def __init__(self, seg_criterion, lambda_fl=0.1, lambda_sl=0.1, lambda_lc=0.05, 
                 aux_weight=0.4, temperature=0.5, use_bane=True):  # ADD aux_weight parameter
        """
        Args:
            seg_criterion: Segmentation loss criterion
            lambda_fl: Weight for feature-level contrastive loss
            lambda_sl: Weight for semantic-level contrastive loss
            lambda_lc: Weight for local consistency loss
            aux_weight: Weight for auxiliary losses
            temperature: Temperature for contrastive losses
            use_bane: Whether to use boundary-aware negative sampling
        """
        super().__init__()
        
        self.seg_criterion = seg_criterion
        self.lambda_fl = lambda_fl
        self.lambda_sl = lambda_sl
        self.lambda_lc = lambda_lc
        self.aux_weight = aux_weight  # ADD THIS
        
        # Contrastive losses
        self.feature_loss = FeatureLevelContrastiveLoss(temperature=temperature)
        self.semantic_loss = SemanticLevelContrastiveLoss(temperature=temperature, use_bane=use_bane)
        self.local_loss = LocalConsistencyLoss()
    
    def forward(self, outputs, masks, features_aug1=None, features_aug2=None):
        """
        Compute complete mCL-LC loss.
        
        Args:
            outputs: Segmentation outputs - can be:
                    - Dict with 'main' (MAMNet style)
                    - Dict with 'p1'-'p6' (OGLANet style)
                    - Tensor [B, num_classes, H, W]
            masks: Ground truth masks [B, H, W]
            features_aug1: Features from augmented view 1 [B, C, H, W] (optional)
            features_aug2: Features from augmented view 2 [B, C, H, W] (optional)
        
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Determine main output based on architecture
        if isinstance(outputs, dict):
            # Check for MAMNet format (has 'main' key)
            if 'main' in outputs:
                main_output = outputs['main']
            # Check for OGLANet format (has 'p6' key - final prediction)
            elif 'p6' in outputs:
                main_output = outputs['p6']
            # Fallback: try to find any reasonable output
            else:
                # Get first available output
                available_keys = list(outputs.keys())
                main_output = outputs[available_keys[0]]
                print(f"Warning: Expected 'main' or 'p6' key, using '{available_keys[0]}' instead")
        else:
            main_output = outputs
        
        # Segmentation loss - MAIN OUTPUT
        seg_loss = self.seg_criterion(main_output, masks)
        losses['seg_loss'] = seg_loss
        
        total_loss = seg_loss
        
        # ADD AUXILIARY LOSSES
        if isinstance(outputs, dict):
            aux_loss = 0.0
            num_aux = 0
            
            # Check for MAMNet-style auxiliary outputs (aux1, aux2, aux3)
            for aux_key in ['aux1', 'aux2', 'aux3']:
                if aux_key in outputs:
                    aux_output = outputs[aux_key]
                    aux_loss += self.seg_criterion(aux_output, masks)
                    num_aux += 1
            
            # Check for OGLANet-style predictions (p1-p5, excluding p6 which is main)
            for pred_key in ['p1', 'p2', 'p3', 'p4', 'p5']:
                if pred_key in outputs:
                    pred_output = outputs[pred_key]
                    aux_loss += self.seg_criterion(pred_output, masks)
                    num_aux += 1
            
            if num_aux > 0:
                aux_loss = aux_loss / num_aux  # Average over auxiliary branches
                losses['aux_loss'] = aux_loss
                total_loss = total_loss + self.aux_weight * aux_loss
        
        # Contrastive losses (if augmented features provided)
        if features_aug1 is not None and features_aug2 is not None:
            # Feature-level contrastive loss
            fl_loss = self.feature_loss(features_aug1, features_aug2)
            losses['feature_loss'] = fl_loss
            total_loss = total_loss + self.lambda_fl * fl_loss
            
            # Semantic-level contrastive loss
            sl_loss = self.semantic_loss(features_aug1, features_aug2, masks, masks)
            losses['semantic_loss'] = sl_loss
            total_loss = total_loss + self.lambda_sl * sl_loss
            
            # Local consistency loss
            lc_loss = self.local_loss(features_aug1, features_aug2)
            losses['local_loss'] = lc_loss
            total_loss = total_loss + self.lambda_lc * lc_loss
        
        losses['total'] = total_loss
        
        return losses


if __name__ == "__main__":
    # Test losses
    print("Testing mCL-LC losses...")
    
    # Test NT-Xent
    ntxent = NTXentLoss(temperature=0.5)
    z_i = F.normalize(torch.randn(4, 128), dim=1)
    z_j = F.normalize(torch.randn(4, 128), dim=1)
    loss = ntxent(z_i, z_j)
    print(f"NT-Xent loss: {loss.item():.4f}")
    
    # Test Feature-level loss
    fl_loss = FeatureLevelContrastiveLoss(temperature=0.5)
    features_i = torch.randn(2, 256, 32, 32)
    features_j = torch.randn(2, 256, 32, 32)
    loss = fl_loss(features_i, features_j)
    print(f"Feature-level loss: {loss.item():.4f}")
    
    # Test Semantic-level loss
    sl_loss = SemanticLevelContrastiveLoss(temperature=0.5, use_bane=True)
    masks = torch.randint(0, 2, (2, 32, 32))
    loss = sl_loss(features_i, features_j, masks, masks)
    print(f"Semantic-level loss: {loss.item():.4f}")
    
    # Test Local consistency loss
    lc_loss = LocalConsistencyLoss()
    loss = lc_loss(features_i, features_j)
    print(f"Local consistency loss: {loss.item():.4f}")
    
    # Test complete mCL-LC loss
    seg_criterion = nn.CrossEntropyLoss()
    mcl_loss = mCLLCLoss(seg_criterion, lambda_fl=0.1, lambda_sl=0.1, lambda_lc=0.05)
    
    outputs = torch.randn(2, 2, 256, 256)
    masks = torch.randint(0, 2, (2, 256, 256))
    features_1 = torch.randn(2, 256, 64, 64)
    features_2 = torch.randn(2, 256, 64, 64)
    
    losses = mcl_loss(outputs, masks, features_1, features_2)
    print("\nComplete mCL-LC losses:")
    for key, val in losses.items():
        print(f"  {key}: {val.item():.4f}")