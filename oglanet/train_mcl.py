"""
Training script for OGLANet with Multi-level Contrastive Learning (mCL-LC)
WACV 2023 - Tang et al. (adapted for OGLANet)

Implements contrastive learning at feature and semantic levels with local consistency.

Decision metrics (LR scheduler, best checkpoint, early stopping) are based on
**Tolerant mIOU** (±5 px boundary exclusion) when --eval_boundary_tolerant is set,
falling back to strict mIOU otherwise.
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

from models.oglanet_mcl import OGLANetMCL
from data.dataset import get_dataloaders
from utils.contrastive_losses import mCLLCLoss
from utils.metrics import ShadowMetrics
from utils.losses import OGLANetLoss
from utils.postprocessing import filter_small_predictions
from utils.visualization import plot_loss_curves, plot_metrics_curves, save_best_worst_visualizations
from utils.evaluation_detailed import DetailedEvaluator


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train OGLANet with mCL-LC for Shadow Detection')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=False, default=None,
                      help='Root directory of dataset')
    parser.add_argument('--img_size', type=int, default=384,
                      help='Input image size (default: 384)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
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
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Use pretrained encoder')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=120,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                      help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamax',
                      choices=['adam', 'adamax'],
                      help='Optimizer')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                      help='Early stopping patience (epochs without improvement)')
    
    # Checkpoint and logging
    parser.add_argument('--output_dir', type=str, default='./outputs',
                      help='Directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=10,
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
    
    # Contrast channel
    parser.add_argument('--use_contrast', action='store_true',
                        help='Use contrast as 4th input channel')

    # Boundary tolerant evaluation
    parser.add_argument('--eval_boundary_tolerant', action='store_true',
                        help='Compute boundary-tolerant metrics (and use tolerant mIOU for decisions)')

    # Comparison / external inference dirs (for post-hoc analysis)
    parser.add_argument('--comparison_inference_dir', type=str, default=None,
                        help='Directory with inference results from other methods for comparison')
    parser.add_argument('--comparison_data_root', type=str, default=None,
                        help='Data root used by comparison methods')
    
    return parser.parse_args()


class MCLTrainer:
    """Trainer class for OGLANet with mCL-LC"""
    
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
            exp_name = f'oglanet_mcl_loco_holdout_{test_city}_{args.resolution}_{1}'
        else:
            exp_name = f'oglanet_mcl_{args.mode}_{1}'
        
        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Initialize model
        print('Initializing OGLANet with mCL-LC...')
        self.model = OGLANetMCL(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size,
            feature_proj_dim=args.feature_proj_dim,
            semantic_proj_dim=args.semantic_proj_dim,
            use_contrast=args.use_contrast
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        
        # Setup loss functions
        self.criterion = OGLANetLoss()  # Base segmentation loss
        self.contrastive_criterion = None
        
        # Only create contrastive losses if needed
        if args.lambda_fl > 0 or args.lambda_sl > 0 or args.lambda_lc > 0:
            base_criterion = nn.CrossEntropyLoss(ignore_index=255)
            self.contrastive_criterion = mCLLCLoss(
                seg_criterion=base_criterion,
                lambda_fl=args.lambda_fl,
                lambda_sl=args.lambda_sl,
                lambda_lc=args.lambda_lc,
                aux_weight=0.0,  # OGLANetLoss handles aux
                temperature=args.temperature,
                use_bane=args.use_bane
            )
        
        # Setup optimizer
        if args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(
                self.model.parameters(),
                lr=args.lr
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=args.lr
            )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize tracking
        self.start_epoch = 0
        self.best_miou = 0.0
        self.best_shadow_iou = 0.0
        self.best_f1 = 0.0
        self.best_tolerant_miou = 0.0        # <-- tolerant tracking
        self.patience = args.early_stopping_patience
        self.patience_counter = 0
        
        # Resume from checkpoint if specified
        if args.resume:
            self.load_checkpoint(args.resume)
        
        # Load datasets with MCL support
        if args.use_contrast:
            # Use enhanced dataset with contrast AND MCL
            from data.dataset_enhanced import ShadowDatasetEnhanced
            from torch.utils.data import DataLoader
            
            # Determine paths
            if args.mode == 'single':
                if args.data_root is None:
                    raise ValueError("data_root must be provided for single city mode")
                train_paths = [args.data_root]
                val_paths = [args.data_root]
                test_paths = [args.data_root]
            elif args.mode == 'all':
                if args.base_data_root is None or args.resolution is None:
                    raise ValueError("base_data_root and resolution required for 'all' mode")
                cities = args.cities if args.cities else ['chicago', 'miami', 'phoenix']
                train_paths = [os.path.join(args.base_data_root, city, args.resolution) for city in cities]
                val_paths = train_paths
                test_paths = train_paths
            elif args.mode == 'loco':
                if args.base_data_root is None or args.resolution is None or args.fold_id is None:
                    raise ValueError("base_data_root, resolution, and fold_id required for LOCO mode")
                from data.dataset import LOCO_FOLDS
                fold_config = LOCO_FOLDS[args.fold_id]
                train_cities = fold_config['train']
                test_city = fold_config['test']
                train_paths = [os.path.join(args.base_data_root, city, args.resolution) for city in train_cities]
                val_paths = train_paths
                test_paths = [os.path.join(args.base_data_root, test_city, args.resolution)]
            
            train_dataset = ShadowDatasetEnhanced(
                root_dirs=train_paths,
                split='train',
                img_size=args.img_size,
                augment=True,
                use_contrast=True,
                use_mcl=args.use_mcl
            )
            val_dataset = ShadowDatasetEnhanced(
                root_dirs=val_paths,
                split='val',
                img_size=args.img_size,
                augment=False,
                use_contrast=True,
                use_mcl=False
            )
            test_dataset = ShadowDatasetEnhanced(
                root_dirs=test_paths,
                split='test',
                img_size=args.img_size,
                augment=False,
                use_contrast=True,
                use_mcl=False
            )
            
            self.dataloaders = {
                'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True),
                'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True),
                'test': DataLoader(test_dataset, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            }
        else:
            # Use original dataset
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
                use_mcl=args.use_mcl
            )
        
        print(f'Training samples: {len(self.dataloaders["train"].dataset)}')
        print(f'Validation samples: {len(self.dataloaders["val"].dataset)}')
        print(f'Test samples: {len(self.dataloaders["test"].dataset)}')
        
        # Detailed evaluators for boundary-tolerant metrics
        if args.eval_boundary_tolerant:
            self.detailed_evaluator_train = DetailedEvaluator()
            self.detailed_evaluator_val = DetailedEvaluator()
            print("Boundary-tolerant evaluation enabled")
            print("  -> Decision metric: Tolerant mIOU (±5 px boundary exclusion)")

        # Tracking for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
    
    # ------------------------------------------------------------------
    # Helper: extract the scalar used for all decisions
    # ------------------------------------------------------------------
    def _decision_miou(self, strict_metrics, detailed_results):
        """Return the mIOU that drives scheduler / checkpoint / early-stop.

        If boundary-tolerant evaluation is enabled *and* detailed_results is
        available, use the tolerant mIOU.  Otherwise fall back to strict.
        """
        if self.args.eval_boundary_tolerant and detailed_results is not None:
            return detailed_results['boundary_tolerant']['tolerant_5px']['iou']
        return strict_metrics['mIOU']

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
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Check if augmented views are available (MCL mode)
            use_contrastive = 'image_aug1' in batch and 'image_aug2' in batch
            
            # Forward pass on main image
            outputs = self.model(images, return_features=False)
            
            # Forward pass on augmented views (for contrastive learning) - only if available
            feat_emb1, feat_emb2 = None, None
            if use_contrastive:
                images_aug1 = batch['image_aug1'].to(self.device)
                images_aug2 = batch['image_aug2'].to(self.device)
                
                _, features_aug1 = self.model(images_aug1, return_features=True)
                _, features_aug2 = self.model(images_aug2, return_features=True)
                
                feat_emb1 = features_aug1['feature_embeddings']
                feat_emb2 = features_aug2['feature_embeddings']
            
            # Compute segmentation loss
            losses = self.criterion(outputs, masks)
            loss = losses['total']
            
            # Add contrastive losses if enabled
            if self.contrastive_criterion is not None and use_contrastive:
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
            
            # Update metrics (using P6 - final prediction)
            if isinstance(outputs, dict):
                main_output = outputs['p6']
            else:
                main_output = outputs
            filtered_outputs = filter_small_predictions(main_output, min_pixels=10)
            train_metrics.update(filtered_outputs, masks)

            # Update detailed evaluator if enabled
            if self.args.eval_boundary_tolerant:
                preds = torch.argmax(main_output, dim=1)
                self.detailed_evaluator_train.update(preds, masks, images)
            
            # Track losses
            epoch_loss += loss.item()
            if 'loss1' in losses:
                epoch_seg_loss += losses['loss1'].item()
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
        print(f'FL: {epoch_fl_loss:.4f} | SL: {epoch_sl_loss:.4f} | LC: {epoch_lc_loss:.4f}')
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

        # Compute and log boundary-tolerant metrics
        if self.args.eval_boundary_tolerant:
            detailed_results = self.detailed_evaluator_train.compute_metrics()
            
            self.writer.add_scalar('Train/F1_Tolerant', 
                                detailed_results['boundary_tolerant']['tolerant_5px']['f1'], epoch)
            self.writer.add_scalar('Train/mIOU_Tolerant',
                                detailed_results['boundary_tolerant']['tolerant_5px']['iou'], epoch)
            
            print(f'Boundary-Tolerant: F1: {detailed_results["boundary_tolerant"]["tolerant_5px"]["f1"]:.2f}% | '
                f'mIOU: {detailed_results["boundary_tolerant"]["tolerant_5px"]["iou"]:.2f}%')
            
            self.detailed_evaluator_train.reset()
        
        # Store for plotting
        self.train_losses.append(epoch_loss)
        for key in self.train_metrics_history.keys():
            self.train_metrics_history[key].append(metrics[key])
        
        return epoch_loss, metrics
    
    def validate(self, epoch):
        """Validate the model.

        Returns
        -------
        val_loss : float
        metrics  : dict          (strict metrics from ShadowMetrics)
        detailed_results : dict | None
            Boundary-tolerant results when --eval_boundary_tolerant is set.
        """
        print('\nValidating...')
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = ShadowMetrics()
        
        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass (inference mode returns P6 only)
                outputs = self.model(images, return_features=False)
                
                # Compute loss on P6
                if isinstance(outputs, dict):
                    main_output = outputs['p6']
                else:
                    main_output = outputs
                
                loss = nn.CrossEntropyLoss()(main_output, masks)
                val_loss += loss.item()
                
                # Update metrics
                filtered_outputs = filter_small_predictions(main_output, min_pixels=10)
                val_metrics.update(filtered_outputs, masks)

                # Update detailed evaluator if enabled
                if self.args.eval_boundary_tolerant:
                    preds = torch.argmax(main_output, dim=1)
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

        # Compute and log boundary-tolerant metrics
        detailed_results = None
        if self.args.eval_boundary_tolerant:
            detailed_results = self.detailed_evaluator_val.compute_metrics()
            
            self.writer.add_scalar('Val/F1_Tolerant',
                                detailed_results['boundary_tolerant']['tolerant_5px']['f1'], epoch)
            self.writer.add_scalar('Val/mIOU_Tolerant',
                                detailed_results['boundary_tolerant']['tolerant_5px']['iou'], epoch)
            
            print(f'Boundary-Tolerant: F1: {detailed_results["boundary_tolerant"]["tolerant_5px"]["f1"]:.2f}% | '
                f'mIOU: {detailed_results["boundary_tolerant"]["tolerant_5px"]["iou"]:.2f}%')
            
            self.detailed_evaluator_val.reset()
        
        # Store for plotting
        self.val_losses.append(val_loss)
        for key in self.val_metrics_history.keys():
            self.val_metrics_history[key].append(metrics[key])
        
        return val_loss, metrics, detailed_results
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        self.best_tolerant_miou = checkpoint.get('best_tolerant_miou', 0.0)
        self.best_shadow_iou = checkpoint.get('best_shadow_iou', 0.0)
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        
        print(f'Resumed from epoch {checkpoint["epoch"]}')
    
    def train(self):
        """Main training loop.

        Decision metric (scheduler plateau, best checkpoint, early stopping)
        is the **Tolerant mIOU** when --eval_boundary_tolerant is set,
        otherwise strict mIOU.
        """
        print('\n' + '='*50)
        print('Starting training with mCL-LC...')
        if self.args.eval_boundary_tolerant:
            print('Decision metric: Tolerant mIOU (±5 px)')
        else:
            print('Decision metric: Strict mIOU')
        print('='*50)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch + 1)
            
            # Validate — now also returns detailed_results (or None)
            val_loss, val_metrics, val_detailed = self.validate(epoch + 1)
            
            # ---- Decision metric ----
            decision_miou = self._decision_miou(val_metrics, val_detailed)
            metric_label = "Tolerant mIOU" if (self.args.eval_boundary_tolerant and val_detailed) else "Strict mIOU"
            
            # Update scheduler on decision metric
            self.scheduler.step(decision_miou)
            
            # Check if best (using decision metric)
            is_best = False
            if decision_miou > self.best_tolerant_miou if self.args.eval_boundary_tolerant else decision_miou > self.best_miou:
                if self.args.eval_boundary_tolerant:
                    self.best_tolerant_miou = decision_miou
                else:
                    self.best_miou = decision_miou
                is_best = True
                print(f'New best {metric_label}: {decision_miou:.2f}%')
            
            # Always track strict bests for logging (informational only)
            if val_metrics['mIOU'] > self.best_miou:
                self.best_miou = val_metrics['mIOU']
                if not self.args.eval_boundary_tolerant:
                    pass  # already handled above
                else:
                    print(f'  (Strict mIOU also improved: {self.best_miou:.2f}%)')
            
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
                print(f'New best Shadow IoU: {self.best_shadow_iou:.2f}%')
            
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']
                print(f'New best F1: {self.best_f1:.2f}%')
            
            # Early stopping on decision metric
            if is_best:
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                    print(f'  No improvement in {metric_label} for {self.patience} epochs')
                    break
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best=is_best)
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch + 1)
            
            print('='*50)
        
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

        # Detailed evaluator (always instantiate, will only use if flag set)
        detailed_eval = DetailedEvaluator() if self.args.eval_boundary_tolerant else None

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, return_features=False)
                
                # Get main output
                if isinstance(outputs, dict):
                    main_output = outputs['p6']
                else:
                    main_output = outputs
                
                # Update standard metrics
                filtered_outputs = filter_small_predictions(main_output, min_pixels=10)
                test_metrics.update(filtered_outputs, masks)
                
                # Update detailed metrics if enabled
                if self.args.eval_boundary_tolerant:
                    preds = torch.argmax(main_output, dim=1)
                    detailed_eval.update(preds, masks, images)

        # Compute standard metrics
        metrics = test_metrics.compute()

        print('\n' + '='*50)
        print('Standard Test Results:')
        print('='*50)
        for key, val in metrics.items():
            print(f'{key}: {val:.2f}%')

        # Save standard results
        results_path = os.path.join(self.output_dir, 'test_results.json')
        results_to_save = {'standard': metrics}

        # Compute and print detailed metrics if enabled
        if self.args.eval_boundary_tolerant:
            detailed_results = detailed_eval.compute_metrics()
            
            print('\n' + '='*50)
            print('Boundary-Tolerant Evaluation:')
            print('='*50)
            
            # Overall metrics
            print('\nOverall Metrics:')
            print(f"  Strict F1:        {detailed_results['boundary_tolerant']['strict']['f1']:.2f}%")
            print(f"  Strict mIOU:      {detailed_results['boundary_tolerant']['strict']['iou']:.2f}%")
            print(f"  Tolerant F1:      {detailed_results['boundary_tolerant']['tolerant_5px']['f1']:.2f}%")
            print(f"  Tolerant mIOU:    {detailed_results['boundary_tolerant']['tolerant_5px']['iou']:.2f}%")
            
            # Size-stratified (strict)
            if 'size_stratified' in detailed_results:
                print('\nSize-Stratified (Strict):')
                for category in ['tiny', 'small', 'medium', 'large']:
                    if category in detailed_results['size_stratified']:
                        m = detailed_results['size_stratified'][category]
                        print(f"  {category:8s}: Miss Rate = {m['miss_rate']:5.1f}%, "
                            f"IoU = {m['avg_iou']:5.1f}% ({m['total']} shadows)")
            
            # Size-stratified (tolerant)
            if 'size_stratified_tolerant' in detailed_results:
                print('\nSize-Stratified (Tolerant 5px):')
                for category in ['tiny', 'small', 'medium', 'large']:
                    if category in detailed_results['size_stratified_tolerant']:
                        m = detailed_results['size_stratified_tolerant'][category]
                        print(f"  {category:8s}: Miss Rate = {m['miss_rate']:5.1f}%, "
                            f"IoU = {m['avg_iou']:5.1f}% ({m['total']} shadows)")
            
            # FP analysis
            if 'fp_fn_analysis' in detailed_results and 'fp' in detailed_results['fp_fn_analysis']:
                fp_info = detailed_results['fp_fn_analysis']['fp']
                print('\nFP Spatial Distribution:')
                print(f"  Within 1px:  {fp_info['pct_within_1px']:.1f}%")
                print(f"  Within 5px:  {fp_info['pct_within_5px']:.1f}%")
                print(f"  Within 10px: {fp_info['pct_within_10px']:.1f}%")
            
            # Add to save dict
            results_to_save['detailed'] = detailed_results

        # Save all results
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