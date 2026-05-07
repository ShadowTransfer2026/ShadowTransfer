"""
HRDA Loss Functions
Implements losses for HRDA multi-resolution training and pseudo-label adaptation.

Based on HRDA (ECCV 2022): https://arxiv.org/abs/2204.13132
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HRDALoss(nn.Module):
    """
    HRDA Loss for Multi-Resolution Training.
    
    Combines:
    1. Fused prediction loss (context + detail fusion)
    2. Detail prediction loss (additional supervision)
    
    Loss = (1 - λ_d) * L_fused + λ_d * L_detail
    
    Based on HRDA Eq. 13 (source) and Eq. 14 (target with pseudo-labels)
    
    Args:
        hr_loss_weight: Weight for detail loss (λ_d in paper, default: 0.1)
        ignore_index: Index to ignore in loss computation (default: 255)
    """
    
    def __init__(self, hr_loss_weight=0.1, ignore_index=255, use_class_weights=False):
        super(HRDALoss, self).__init__()
        
        self.hr_loss_weight = hr_loss_weight
        self.ignore_index = ignore_index
        
        if use_class_weights:
            class_weights = torch.tensor([1.0, 5.0])
            self.register_buffer('class_weights', class_weights)
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index, reduction='mean')
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    
    def forward(self, pred_fused, pred_detail, target_fused, target_detail,
                confidence_fused=None, confidence_detail=None,
                aux_context=None, aux_detail=None):
        """
        Compute HRDA loss.
        
        Args:
            pred_fused: Fused prediction [B, C, H, W]
            pred_detail: Detail prediction [B, C, H_d, W_d]
            target_fused: Target for fused (labels or pseudo-labels) [B, H, W]
            target_detail: Target for detail [B, H_d, W_d]
            confidence_fused: Confidence weights for fused [B, H, W] (optional)
            confidence_detail: Confidence weights for detail [B, H_d, W_d] (optional)
        
        Returns:
            Dictionary with:
            - 'total': Total loss
            - 'fused': Fused prediction loss
            - 'detail': Detail prediction loss
        """
        
        # Resize targets to match predictions if needed
        if target_fused.shape[-2:] != pred_fused.shape[-2:]:
            target_fused = F.interpolate(
                target_fused.unsqueeze(1).float(),
                size=pred_fused.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()
        
        if target_detail.shape[-2:] != pred_detail.shape[-2:]:
            target_detail = F.interpolate(
                target_detail.unsqueeze(1).float(),
                size=pred_detail.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()
        
        # Compute losses
        if confidence_fused is not None:
            # Weighted loss for pseudo-labels
            loss_fused = self._weighted_ce_loss(
                pred_fused, target_fused, confidence_fused
            )
        else:
            # Standard cross-entropy for labeled data
            loss_fused = self.ce_loss(pred_fused, target_fused)
        
        if confidence_detail is not None:
            # Weighted loss for pseudo-labels
            loss_detail = self._weighted_ce_loss(
                pred_detail, target_detail, confidence_detail
            )
        else:
            # Standard cross-entropy for labeled data
            loss_detail = self.ce_loss(pred_detail, target_detail)

        # NEW: Auxiliary losses
        # Compute auxiliary losses (with proper resizing)
        loss_aux = 0.0
        aux_weight = 0.4  # Standard auxiliary loss weight
        num_aux = 0

        if aux_context is not None and len(aux_context) > 0:
            for aux_pred in aux_context:
                # Resize target to match auxiliary prediction
                if aux_pred.shape[-2:] != target_fused.shape[-2:]:
                    target_aux = F.interpolate(
                        target_fused.unsqueeze(1).float(),
                        size=aux_pred.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    target_aux = target_fused
                
                # Apply confidence weighting if available
                if confidence_fused is not None:
                    conf_aux = F.interpolate(
                        confidence_fused.unsqueeze(1),
                        size=aux_pred.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    loss_aux += self._weighted_ce_loss(aux_pred, target_aux, conf_aux) * aux_weight
                else:
                    loss_aux += self.ce_loss(aux_pred, target_aux) * aux_weight
                num_aux += 1

        if aux_detail is not None and len(aux_detail) > 0:
            for aux_pred in aux_detail:
                # Resize target to match auxiliary prediction
                if aux_pred.shape[-2:] != target_detail.shape[-2:]:
                    target_aux = F.interpolate(
                        target_detail.unsqueeze(1).float(),
                        size=aux_pred.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    target_aux = target_detail
                
                # Apply confidence weighting if available
                if confidence_detail is not None:
                    conf_aux = F.interpolate(
                        confidence_detail.unsqueeze(1),
                        size=aux_pred.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    loss_aux += self._weighted_ce_loss(aux_pred, target_aux, conf_aux) * aux_weight
                else:
                    loss_aux += self.ce_loss(aux_pred, target_aux) * aux_weight
                num_aux += 1

        # Combined loss
        total_loss = (1 - self.hr_loss_weight) * loss_fused + \
                    self.hr_loss_weight * loss_detail + \
                    loss_aux

        return {
            'total': total_loss,
            'fused': loss_fused,
            'detail': loss_detail,
            'aux': loss_aux  # Add for logging
        }
    
    def _weighted_ce_loss(self, pred, target, confidence):
        """
        Compute confidence-weighted cross-entropy loss.
        
        Args:
            pred: Predictions [B, C, H, W]
            target: Targets [B, H, W]
            confidence: Confidence weights [B, H, W]
        
        Returns:
            Weighted loss
        """
        # Resize confidence to match prediction
        if confidence.shape[-2:] != pred.shape[-2:]:
            confidence = F.interpolate(
                confidence.unsqueeze(1),
                size=pred.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # Compute per-pixel cross-entropy
        log_probs = F.log_softmax(pred, dim=1)
        
        # Gather log probabilities for target classes
        target_long = target.long()
        target_one_hot = F.one_hot(target_long, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Weighted loss
        loss = -(target_one_hot * log_probs).sum(dim=1)
        loss = (loss * confidence).sum() / (confidence.sum() + 1e-8)
        
        return loss


class PseudoLabelGenerator(nn.Module):
    """
    Generates pseudo-labels with confidence estimation for target domain.
    
    Args:
        num_classes: Number of classes
        confidence_threshold: Threshold for pseudo-label confidence (default: 0.968)
    """
    
    def __init__(self, num_classes, confidence_threshold=0.968):
        super(PseudoLabelGenerator, self).__init__()
        
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
    
    def forward(self, predictions):
        """
        Generate pseudo-labels and confidence weights.
        
        Args:
            predictions: Model predictions [B, C, H, W]
        
        Returns:
            Dictionary with:
            - 'pseudo_labels': Pseudo-labels [B, H, W]
            - 'confidence': Confidence weights [B, H, W]
        """
        # Get softmax probabilities
        probs = F.softmax(predictions, dim=1)  # [B, C, H, W]
        
        # Get max probability and corresponding class
        max_probs, pseudo_labels = torch.max(probs, dim=1)  # [B, H, W]
        
        # Confidence: use max probability as weight
        confidence = max_probs
        
        # Apply confidence threshold
        confidence = (confidence > self.confidence_threshold).float() * confidence
        
        return {
            'pseudo_labels': pseudo_labels,
            'confidence': confidence,
            'max_probs': max_probs
        }


if __name__ == "__main__":
    # Test HRDA loss
    batch_size = 2
    num_classes = 2
    H, W = 384, 384
    H_d, W_d = 192, 192
    
    # Simulate predictions
    pred_fused = torch.randn(batch_size, num_classes, H, W)
    pred_detail = torch.randn(batch_size, num_classes, H_d, W_d)
    
    # Simulate targets
    target_fused = torch.randint(0, num_classes, (batch_size, H, W))
    target_detail = torch.randint(0, num_classes, (batch_size, H_d, W_d))
    
    # Test with labels (source domain)
    print("Testing HRDA Loss with labels (source domain)...")
    hrda_loss = HRDALoss(hr_loss_weight=0.1)
    losses = hrda_loss(pred_fused, pred_detail, target_fused, target_detail)
    
    print("Losses:")
    for key, val in losses.items():
        print(f"  {key}: {val.item():.4f}")
    
    # Test with pseudo-labels (target domain)
    print("\nTesting HRDA Loss with pseudo-labels (target domain)...")
    
    # Generate confidence weights
    confidence_fused = torch.rand(batch_size, H, W)
    confidence_detail = torch.rand(batch_size, H_d, W_d)
    
    losses_pseudo = hrda_loss(
        pred_fused, pred_detail, target_fused, target_detail,
        confidence_fused, confidence_detail
    )
    
    print("Losses with pseudo-labels:")
    for key, val in losses_pseudo.items():
        print(f"  {key}: {val.item():.4f}")
    
    # Test pseudo-label generator
    print("\nTesting Pseudo-Label Generator...")
    pseudo_gen = PseudoLabelGenerator(num_classes=num_classes)
    
    pred = torch.randn(batch_size, num_classes, H, W)
    pseudo_output = pseudo_gen(pred)
    
    print(f"Pseudo-labels shape: {pseudo_output['pseudo_labels'].shape}")
    print(f"Confidence shape: {pseudo_output['confidence'].shape}")
    print(f"Confidence range: [{pseudo_output['confidence'].min():.3f}, {pseudo_output['confidence'].max():.3f}]")
    print(f"Mean confidence: {pseudo_output['confidence'].mean():.3f}")