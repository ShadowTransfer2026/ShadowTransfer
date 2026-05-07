"""
Training script for DINOv3 with Multi-level Contrastive Learning (mCL-LC)
Implements contrastive learning at feature and semantic levels with local consistency.

Decision metrics (best checkpoint, early stopping) are based on Tolerant mIOU
when --eval_boundary_tolerant is enabled, so noisy boundary pixels don't cause
premature stopping or bad checkpoint picks.
"""

import os
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import DINOv3-MCL model
from dinov3_mcl import DINOv3MCL

# Import utilities (reuse from MAMNet)
from data.dataset import get_dataloaders
from utils.losses import CrossEntropyLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization import (
    plot_loss_curves,
    plot_metrics_curves,
    save_best_worst_visualizations
)
from utils.evaluation_detailed import DetailedEvaluator

# Import contrastive losses (assuming you'll copy from MAMNet)
sys.path.append('../mamnet')  # Adjust path as needed
from utils.contrastive_losses import mCLLCLoss


print("="*50)
print("GPU DIAGNOSTICS")
print("="*50)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
print("="*50)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DINOv3 with mCL-LC for Shadow Detection')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=False, default=None,
                      help='Root directory of dataset (required for single mode)')
    parser.add_argument('--img_size', type=int, default=384,
                      help='Input image size (default: 384)')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size (default: 4)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of data loading workers')
    
    # LOCO and multi-city parameters
    parser.add_argument('--mode', type=str, default='loco',
                      choices=['single', 'all', 'loco'],
                      help='Training mode')
    parser.add_argument('--base_data_root', type=str, default=None,
                      help='Base directory for all/loco modes')
    parser.add_argument('--resolution', type=str, default=None,
                      choices=['highres', 'midres'],
                      help='Resolution for all/loco modes')
    parser.add_argument('--fold_id', type=int, default=None,
                      choices=[0, 1, 2],
                      help='Fold ID for LOCO mode')
    parser.add_argument('--cities', type=str, nargs='+', default=None,
                      help='List of cities for all mode')
    
    # mCL-LC parameters
    parser.add_argument('--lambda_fl', type=float, default=0.001,
                      help='Weight for feature-level contrastive loss')
    parser.add_argument('--lambda_sl', type=float, default=0.001,
                      help='Weight for semantic-level contrastive loss')
    parser.add_argument('--lambda_lc', type=float, default=0.0005,
                      help='Weight for local consistency loss')
    parser.add_argument('--temperature', type=float, default=0.5,
                      help='Temperature for contrastive losses')
    parser.add_argument('--use_bane', action='store_true',
                      help='Use boundary-aware negative sampling')
    parser.add_argument('--feature_proj_dim', type=int, default=128,
                      help='Feature projection dimension')
    parser.add_argument('--semantic_proj_dim', type=int, default=128,
                      help='Semantic projection dimension')
    parser.add_argument('--use_pseudo_cloud', action='store_true',
                      help='Use pseudo-cloud augmentation')
    parser.add_argument('--cloud_p', type=float, default=0.3,
                      help='Probability of cloud augmentation')
    parser.add_argument('--shadow_p', type=float, default=0.2,
                      help='Probability of cloud shadow augmentation')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes')
    parser.add_argument('--model_name', type=str, default='dinov3_vits16',
                      choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'],
                      help='DINOv3 model variant')
    parser.add_argument('--weights_path', type=str, default=None,
                      help='Path to DINOv3 pretrained weights')
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Use pretrained encoder')
    parser.add_argument('--frozen_stages', type=int, default=-1,
                      help='Number of backbone stages to freeze')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=120,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                      help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                      help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                      help='Minimum learning rate')
    
    # Checkpoint and logging
    parser.add_argument('--output_dir', type=str, default='./outputs',
                      help='Directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=5,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                      help='Only evaluate the model')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use')
    parser.add_argument('--use_mcl', action='store_true',
                      help='Use MCL dataset with augmented views')
    
    # Boundary tolerant evaluation
    parser.add_argument('--eval_boundary_tolerant', action='store_true',
                        help='Compute boundary-tolerant metrics during training/validation. '
                             'When enabled, best-checkpoint and early-stopping decisions '
                             'are driven by Tolerant mIOU (±5 px boundary excluded).')
    
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (epochs without mIOU improvement). 0 to disable.')

    # Comparison / inference dirs (parity with DDIB pipeline)
    parser.add_argument('--comparison_inference_dir', type=str, default=None,
                        help='Directory for comparison inference results')
    parser.add_argument('--comparison_data_root', type=str, default=None,
                        help='Data root used during comparison inference')
    
    return parser.parse_args()


class CosineWarmupScheduler:
    """Cosine learning rate schedule with warmup"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class MCLTrainer:
    """Trainer class for DINOv3 with mCL-LC"""
    
    def __init__(self, args):
        self.args = args
        
        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = f'dinov3_mcl_loco_holdout_{test_city}_{args.resolution}_{1}'
        else:
            exp_name = f'dinov3_mcl_{args.mode}_{1}'
        
        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Initialize model
        print('Initializing DINOv3 with mCL-LC...')
        self.model = DINOv3MCL(
            num_classes=args.num_classes,
            model_name=args.model_name,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            feature_proj_dim=args.feature_proj_dim,
            semantic_proj_dim=args.semantic_proj_dim
        ).to(self.device)
        
        # Setup loss functions
        base_criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Segmentation loss only (no auxiliary for DINOv3)
        self.seg_criterion = CrossEntropyLoss()
        
        # Contrastive losses (only if needed)
        self.contrastive_criterion = None
        if args.lambda_fl > 0 or args.lambda_sl > 0 or args.lambda_lc > 0:
            self.contrastive_criterion = mCLLCLoss(
                seg_criterion=base_criterion,
                lambda_fl=args.lambda_fl,
                lambda_sl=args.lambda_sl,
                lambda_lc=args.lambda_lc,
                aux_weight=0.0,  # No auxiliary for DINOv3
                temperature=args.temperature,
                use_bane=args.use_bane
            )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Setup scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.min_lr
        )
        
        # ── Tracking ────────────────────────────────────────────
        self.start_epoch = 0
        self.best_miou = 0.0              # strict best (always tracked)
        self.best_tolerant_miou = 0.0     # tolerant best (used for decisions when enabled)
        self.best_shadow_iou = 0.0
        self.best_f1 = 0.0
        self.epochs_without_improvement = 0
        
        # Resume from checkpoint if specified
        if args.resume:
            self.load_checkpoint(args.resume)
        
        # Load datasets with MCL support
        self.dataloaders = get_dataloaders(
            data_root=args.data_root,
            base_data_root=args.base_data_root,
            mode=args.mode,
            cities=args.cities,
            resolution=args.resolution,
            fold_id=args.fold_id,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            use_mcl=args.use_mcl,
            use_pseudo_cloud=args.use_pseudo_cloud
        )
        
        print(f'Training samples: {len(self.dataloaders["train"].dataset)}')
        print(f'Validation samples: {len(self.dataloaders["val"].dataset)}')
        print(f'Test samples: {len(self.dataloaders["test"].dataset)}')
        
        # Tracking for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }

        # Boundary-tolerant evaluation
        if args.eval_boundary_tolerant:
            self.detailed_evaluator_train = DetailedEvaluator()
            self.detailed_evaluator_val = DetailedEvaluator()
            print("Boundary-tolerant evaluation enabled")
            print("  → Best-checkpoint and early-stopping decisions use Tolerant mIOU")
    
    def train_epoch(self, epoch):
        """Train for one epoch with contrastive learning"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_fl_loss = 0.0
        epoch_sl_loss = 0.0
        epoch_lc_loss = 0.0
        
        train_metrics = ShadowMetrics()
        
        num_batches = len(self.dataloaders['train'])
        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.dataloaders['train']):
            if batch_idx == 0:  # First batch only
                print("\n=== BATCH KEYS DIAGNOSTIC ===")
                print(f"Batch keys: {batch.keys()}")
                print(f"Has image_aug1: {'image_aug1' in batch}")
                print(f"Has image_aug2: {'image_aug2' in batch}")
                print(f"use_mcl flag: {self.args.use_mcl}")
                print("============================\n")
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Check if augmented views are available (MCL mode)
            use_contrastive = 'image_aug1' in batch and 'image_aug2' in batch
            
            # Forward pass on main image (for segmentation)
            outputs = self.model(images, return_features=False)
            
            # Compute segmentation loss
            loss = self.seg_criterion(outputs, masks)
            losses = {'main': loss}
            
            # Forward pass on augmented views (for contrastive learning) - only if available
            if use_contrastive and self.contrastive_criterion is not None:
                images_aug1 = batch['image_aug1'].to(self.device)
                images_aug2 = batch['image_aug2'].to(self.device)
                
                _, features_aug1 = self.model(images_aug1, return_features=True)
                _, features_aug2 = self.model(images_aug2, return_features=True)
                
                # Extract feature embeddings
                feat_emb1 = features_aug1['feature_embeddings']
                feat_emb2 = features_aug2['feature_embeddings']
                
                # Compute contrastive losses
                contrastive_losses = self.contrastive_criterion(
                    outputs,
                    masks,
                    features_aug1=feat_emb1,
                    features_aug2=feat_emb2
                )
                
                # Add contrastive components
                if 'feature_loss' in contrastive_losses:
                    losses['feature_loss'] = contrastive_losses['feature_loss']
                    loss = loss + self.args.lambda_fl * contrastive_losses['feature_loss']
                
                if 'semantic_loss' in contrastive_losses:
                    losses['semantic_loss'] = contrastive_losses['semantic_loss']
                    loss = loss + self.args.lambda_sl * contrastive_losses['semantic_loss']
                
                if 'local_loss' in contrastive_losses:
                    losses['local_loss'] = contrastive_losses['local_loss']
                    loss = loss + self.args.lambda_lc * contrastive_losses['local_loss']
            
            losses['total'] = loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            filtered_outputs = filter_small_predictions(outputs, min_pixels=10)
            train_metrics.update(filtered_outputs, masks)

            if self.args.eval_boundary_tolerant:
                preds = torch.argmax(outputs, dim=1)
                self.detailed_evaluator_train.update(preds, masks, images)
            
            # Track losses
            epoch_loss += loss.item()
            epoch_seg_loss += losses['main'].item()
            if 'feature_loss' in losses:
                epoch_fl_loss += losses['feature_loss'].item()
            if 'semantic_loss' in losses:
                epoch_sl_loss += losses['semantic_loss'].item()
            if 'local_loss' in losses:
                epoch_lc_loss += losses['local_loss'].item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'Batch [{batch_idx + 1}/{num_batches}] | '
                      f'Loss: {loss.item():.4f} | '
                      f'Seg: {losses["main"].item():.4f} | '
                      f'FL: {losses.get("feature_loss", torch.tensor(0.0)).item():.4f} | '
                      f'SL: {losses.get("semantic_loss", torch.tensor(0.0)).item():.4f} | '
                      f'LC: {losses.get("local_loss", torch.tensor(0.0)).item():.4f}')
        
        # Compute averages
        epoch_loss /= num_batches
        epoch_seg_loss /= num_batches
        epoch_fl_loss /= num_batches if epoch_fl_loss > 0 else 1
        epoch_sl_loss /= num_batches if epoch_sl_loss > 0 else 1
        epoch_lc_loss /= num_batches if epoch_lc_loss > 0 else 1
        
        # Compute metrics
        metrics = train_metrics.compute()
        
        # Time taken
        epoch_time = time.time() - start_time
        
        print(f'\nTraining Results:')
        print(f'Time: {epoch_time:.2f}s | Loss: {epoch_loss:.4f}')
        print(f'Seg: {epoch_seg_loss:.4f} | FL: {epoch_fl_loss:.4f} | '
              f'SL: {epoch_sl_loss:.4f} | LC: {epoch_lc_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}% | F1: {metrics["F1"]:.2f}% | '
              f'mIOU: {metrics["mIOU"]:.2f}% | Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')
        
        # Log to tensorboard
        self.writer.add_scalar('Train/TotalLoss', epoch_loss, epoch)
        self.writer.add_scalar('Train/SegLoss', epoch_seg_loss, epoch)
        self.writer.add_scalar('Train/FeatureLoss', epoch_fl_loss, epoch)
        self.writer.add_scalar('Train/SemanticLoss', epoch_sl_loss, epoch)
        self.writer.add_scalar('Train/LocalLoss', epoch_lc_loss, epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)
        
        # Store for plotting
        self.train_losses.append(epoch_loss)
        for key in self.train_metrics_history.keys():
            self.train_metrics_history[key].append(metrics[key])

        if self.args.eval_boundary_tolerant:
            detailed_results = self.detailed_evaluator_train.compute_metrics()
            
            self.writer.add_scalar('Train/F1_Tolerant',
                                detailed_results['boundary_tolerant']['tolerant_5px']['f1'], epoch)
            self.writer.add_scalar('Train/mIOU_Tolerant',
                                detailed_results['boundary_tolerant']['tolerant_5px']['iou'], epoch)
            
            self.detailed_evaluator_train.reset()
            
            print(f'Boundary-Tolerant: F1: {detailed_results["boundary_tolerant"]["tolerant_5px"]["f1"]:.2f}% | '
                f'mIOU: {detailed_results["boundary_tolerant"]["tolerant_5px"]["iou"]:.2f}%')
        
        return epoch_loss, metrics
    
    def validate(self, epoch):
        """Validate the model.

        Returns
        -------
        val_loss : float
        metrics  : dict          (strict metrics from ShadowMetrics)
        tolerant_miou : float    Tolerant mIOU when --eval_boundary_tolerant is
                                 enabled, else None.  This is the value that
                                 drives best-checkpoint / early-stopping decisions.
        """
        print('\nValidating...')
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = ShadowMetrics()
        
        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, return_features=False)
                
                # Compute loss
                loss = self.seg_criterion(outputs, masks)
                val_loss += loss.item()
                
                # Update metrics
                filtered_outputs = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered_outputs, masks)

                if self.args.eval_boundary_tolerant:
                    preds = torch.argmax(outputs, dim=1)
                    self.detailed_evaluator_val.update(preds, masks, images)
        
        # Compute averages
        val_loss /= len(self.dataloaders['val'])
        metrics = val_metrics.compute()
        
        print(f'Validation Results:')
        print(f'Loss: {val_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}% | F1: {metrics["F1"]:.2f}% | '
              f'mIOU: {metrics["mIOU"]:.2f}% | Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Val/{key}', val, epoch)
        
        # Store for plotting
        self.val_losses.append(val_loss)
        for key in self.val_metrics_history.keys():
            self.val_metrics_history[key].append(metrics[key])

        # ── Tolerant metrics ────────────────────────────────────
        tolerant_miou = None
        if self.args.eval_boundary_tolerant:
            detailed_results = self.detailed_evaluator_val.compute_metrics()
            
            tolerant_miou = detailed_results['boundary_tolerant']['tolerant_5px']['iou']
            tolerant_f1   = detailed_results['boundary_tolerant']['tolerant_5px']['f1']

            self.writer.add_scalar('Val/F1_Tolerant', tolerant_f1, epoch)
            self.writer.add_scalar('Val/mIOU_Tolerant', tolerant_miou, epoch)
            
            self.detailed_evaluator_val.reset()
            
            print(f'Boundary-Tolerant: F1: {tolerant_f1:.2f}% | '
                  f'mIOU: {tolerant_miou:.2f}%')
        
        return val_loss, metrics, tolerant_miou
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'best_tolerant_miou': self.best_tolerant_miou,
            'best_shadow_iou': self.best_shadow_iou,
            'best_f1': self.best_f1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'args': vars(self.args)
        }
        
        # Save latest
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved to {best_path}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        self.best_tolerant_miou = checkpoint.get('best_tolerant_miou', 0.0)
        self.best_shadow_iou = checkpoint.get('best_shadow_iou', 0.0)
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        
        print(f'Resumed from epoch {checkpoint["epoch"]}')
    
    def train(self):
        """Main training loop.

        When --eval_boundary_tolerant is enabled the *decision metric* for
        best-checkpoint selection and early-stopping is the **Tolerant mIOU**
        (±5 px boundary band excluded from computation).  Strict metrics are
        still logged for reference.
        """
        print('\n' + '='*50)
        print('Starting training with mCL-LC...')
        if self.args.eval_boundary_tolerant:
            print('Decision metric: Tolerant mIOU (±5px boundary excluded)')
        else:
            print('Decision metric: Strict mIOU')
        print('='*50)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Update learning rate
            current_lr = self.scheduler.step(epoch)
            print(f'\nLearning rate: {current_lr:.2e}')
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch + 1)
            
            # Validate — now also returns tolerant_miou (or None)
            val_loss, val_metrics, tolerant_miou = self.validate(epoch + 1)
            
            # ── Decision metric selection ───────────────────────
            # When boundary-tolerant eval is on, all decisions key off
            # tolerant mIOU so noisy boundary pixels don't mislead us.
            if self.args.eval_boundary_tolerant and tolerant_miou is not None:
                decision_miou = tolerant_miou
                decision_label = "Tolerant mIOU"
            else:
                decision_miou = val_metrics['mIOU']
                decision_label = "Strict mIOU"

            # ── Best checkpoint ─────────────────────────────────
            is_best = False
            if decision_miou > (self.best_tolerant_miou if self.args.eval_boundary_tolerant
                                else self.best_miou):
                if self.args.eval_boundary_tolerant:
                    self.best_tolerant_miou = decision_miou
                else:
                    self.best_miou = decision_miou
                is_best = True
                self.epochs_without_improvement = 0
                print(f'★ New best {decision_label}: {decision_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1

            # Always track strict best for logging even when decisions
            # are driven by tolerant metric
            if val_metrics['mIOU'] > self.best_miou:
                self.best_miou = val_metrics['mIOU']
                if self.args.eval_boundary_tolerant:
                    print(f'  (Strict mIOU also improved: {self.best_miou:.2f}%)')
            
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
                print(f'New best Shadow IoU: {self.best_shadow_iou:.2f}%')
            
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']
                print(f'New best F1: {self.best_f1:.2f}%')
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best=is_best)
            
            # Log learning rate
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch + 1)
            
            print('='*50)
            
            # ── Early stopping ──────────────────────────────────
            if self.args.early_stopping_patience > 0 and \
               self.epochs_without_improvement >= self.args.early_stopping_patience:
                print(f'\nEarly stopping triggered! No {decision_label} improvement for '
                      f'{self.args.early_stopping_patience} epochs.')
                break
        
        print('\nTraining completed!')
        print(f'Best Strict mIOU: {self.best_miou:.2f}%')
        if self.args.eval_boundary_tolerant:
            print(f'Best Tolerant mIOU: {self.best_tolerant_miou:.2f}%')
        print(f'Best Shadow IoU: {self.best_shadow_iou:.2f}%')
        print(f'Best F1: {self.best_f1:.2f}%')
        
        # Generate plots
        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png')
        )
        
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png')
        )
        
        self.writer.close()
    
    def test(self):
        """Test the model"""
        print('\n' + '='*50)
        print('Testing model...')
        print('='*50)
        
        # Load best checkpoint
        best_checkpoint = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_checkpoint):
            self.load_checkpoint(best_checkpoint)
        
        self.model.eval()
        test_metrics = ShadowMetrics()
        
        # Detailed evaluator
        detailed_eval = DetailedEvaluator() if self.args.eval_boundary_tolerant else None
        
        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, return_features=False)
                
                # Update standard metrics
                filtered_outputs = filter_small_predictions(outputs, min_pixels=10)
                test_metrics.update(filtered_outputs, masks)
                
                # Update detailed metrics if enabled
                if self.args.eval_boundary_tolerant:
                    preds = torch.argmax(outputs, dim=1)
                    detailed_eval.update(preds, masks, images)
        
        # Compute standard metrics
        metrics = test_metrics.compute()
        
        print('\n' + '='*50)
        print('Standard Test Results:')
        print('='*50)
        for key, val in metrics.items():
            print(f'{key}: {val:.2f}%')
        
        # Save results
        results_path = os.path.join(self.output_dir, 'test_results.json')
        results_to_save = {'standard': metrics}
        
        # Compute and print detailed metrics if enabled
        if self.args.eval_boundary_tolerant:
            detailed_results = detailed_eval.compute_metrics()
            
            print('\n' + '='*50)
            print('Boundary-Tolerant Evaluation:')
            print('='*50)
            
            print('\nOverall Metrics:')
            print(f"  Strict F1:        {detailed_results['boundary_tolerant']['strict']['f1']:.2f}%")
            print(f"  Strict mIOU:      {detailed_results['boundary_tolerant']['strict']['iou']:.2f}%")
            print(f"  Tolerant F1:      {detailed_results['boundary_tolerant']['tolerant_5px']['f1']:.2f}%")
            print(f"  Tolerant mIOU:    {detailed_results['boundary_tolerant']['tolerant_5px']['iou']:.2f}%")
            
            if 'size_stratified' in detailed_results:
                print('\nSize-Stratified (Strict):')
                for category in ['tiny', 'small', 'medium', 'large']:
                    if category in detailed_results['size_stratified']:
                        m = detailed_results['size_stratified'][category]
                        print(f"  {category:8s}: Miss Rate = {m['miss_rate']:5.1f}%, "
                              f"IoU = {m['avg_iou']:5.1f}% ({m['total']} shadows)")
            
            if 'size_stratified_tolerant' in detailed_results:
                print('\nSize-Stratified (Tolerant 5px):')
                for category in ['tiny', 'small', 'medium', 'large']:
                    if category in detailed_results['size_stratified_tolerant']:
                        m = detailed_results['size_stratified_tolerant'][category]
                        print(f"  {category:8s}: Miss Rate = {m['miss_rate']:5.1f}%, "
                              f"IoU = {m['avg_iou']:5.1f}% ({m['total']} shadows)")
            
            if 'fp_fn_analysis' in detailed_results and 'fp' in detailed_results['fp_fn_analysis']:
                fp_info = detailed_results['fp_fn_analysis']['fp']
                print('\nFP Spatial Distribution:')
                print(f"  Within 1px:  {fp_info['pct_within_1px']:.1f}%")
                print(f"  Within 5px:  {fp_info['pct_within_5px']:.1f}%")
                print(f"  Within 10px: {fp_info['pct_within_10px']:.1f}%")
            
            results_to_save['detailed'] = detailed_results
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f'\nResults saved to {results_path}')
        
        # Generate visualizations
        save_best_worst_visualizations(
            self.model,
            self.dataloaders['test'],
            self.device,
            self.output_dir,
            num_images=10
        )
        
        return metrics


def main():
    args = get_args()
    
    # Create trainer
    trainer = MCLTrainer(args)
    
    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()