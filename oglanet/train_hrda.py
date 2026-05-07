"""
Training Script for OGLANet-HRDA
Implements multi-resolution training with domain adaptation for shadow detection.

Based on HRDA (ECCV 2022): https://arxiv.org/abs/2204.13132

TRAINING STRATEGY:
- Source domain (labeled): One resolution (e.g., midres)
- Target domain (unlabeled): Other resolution (e.g., highres)  
- Cross-resolution adaptation within single city
- EMA teacher generates pseudo-labels for target domain
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

import torch.nn.functional as F

# Add parent to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.oglanet_hrda import create_oglanet_hrda
from data.dataset_hrda import HRDAShadowDataset
from utils.hrda_losses import HRDALoss, PseudoLabelGenerator
from utils.ema import EMATeacher
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization import plot_loss_curves, plot_metrics_curves, save_best_worst_visualizations
from utils.evaluation_detailed import DetailedEvaluator

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train OGLANet-HRDA for Shadow Detection')

    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                    help='Path to city-resolution data (e.g., ./dataset/chicago/midres)')
    parser.add_argument('--target_res', type=str, required=True,
                    choices=['midres', 'highres'],
                    help='Target resolution for adaptation (other res will be auto-detected as source)')
    
    # HRDA crop parameters
    parser.add_argument('--img_size', type=int, default=384,
                      help='Base image size')
    parser.add_argument('--context_size', type=int, default=384,
                      help='Context crop size (before downsampling)')
    parser.add_argument('--detail_size', type=int, default=192,
                      help='Detail crop size')
    parser.add_argument('--scale_factor', type=float, default=0.5,
                      help='LR downsampling factor')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Use pretrained ResNet encoder')
    parser.add_argument('--hr_loss_weight', type=float, default=0.1,
                      help='Weight for HR detail loss (λ_d)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size (HRDA is memory-intensive, use smaller batch)')
    parser.add_argument('--lr', type=float, default=0.0003,
                      help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamax',
                      choices=['adam', 'adamax'],
                      help='Optimizer (default: adamax, as per OGLANet paper)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                      help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of data loading workers')
    
    # EMA and pseudo-label parameters
    parser.add_argument('--ema_alpha', type=float, default=0.99,
                      help='EMA momentum for teacher model')
    parser.add_argument('--confidence_threshold', type=float, default=0.968,
                      help='Confidence threshold for pseudo-labels')
    parser.add_argument('--lambda_target', type=float, default=1.0,
                      help='Weight for target domain loss')
    
    # OGLANet auxiliary loss weight
    parser.add_argument('--aux_weight', type=float, default=0.4,
                      help='Weight for auxiliary losses (P1-P5)')
    
    # Checkpoint and logging
    parser.add_argument('--output_dir', type=str, default='./outputs_hrda',
                      help='Directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=10,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    # Contrast channel
    parser.add_argument('--use_contrast', action='store_true',
                    help='Use contrast as 4th input channel')

    # Boundary tolerant evaluation
    parser.add_argument('--eval_boundary_tolerant', action='store_true',
                    help='Compute boundary-tolerant metrics')

    # Pretrained checkpoint for initialization
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                    help='Path to pretrained OGLANet checkpoint to initialize from')

    parser.add_argument('--auto_load_pretrained', action='store_true',
                    help='Automatically load checkpoint from source domain (city_resolution_1/checkpoint_best.pth)')
    
    return parser.parse_args()


def get_confidence_threshold(epoch, total_epochs=100):
    """Gradually increase confidence threshold during training"""
    # Start low to bootstrap
    start_threshold = 0.52
    # End at paper value
    end_threshold = 0.968
    
    # Linear warmup over first 60% of training
    warmup_epochs = int(total_epochs * 0.6)
    
    if epoch <= warmup_epochs:
        return start_threshold + (end_threshold - start_threshold) * (epoch / warmup_epochs)
    else:
        return end_threshold


class HRDATrainer:
    """Trainer for OGLANet-HRDA"""
    
    def __init__(self, args):
        self.args = args
        
        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Parse city name from data_root
        city_name = os.path.basename(os.path.dirname(args.data_root.rstrip('/')))
        exp_name = f'oglanet_hrda_{city_name}_{1}'
        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Initialize model (student)
        print('Initializing OGLANet-HRDA...')
        self.model = create_oglanet_hrda(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size,
            hr_loss_weight=args.hr_loss_weight,
            use_contrast=args.use_contrast
        ).to(self.device)


        # Determine checkpoint path
        checkpoint_path = args.pretrained_checkpoint
        if checkpoint_path is None and args.auto_load_pretrained:
            # Auto-construct path from source domain
            data_root = args.data_root.rstrip('/')
            current_res = os.path.basename(data_root)
            city_root = os.path.dirname(data_root)
            city_name = os.path.basename(city_root)
            
            # Determine source resolution
            if current_res == args.target_res:
                source_res = 'highres' if args.target_res == 'midres' else 'midres'
            else:
                source_res = current_res
            
            # Construct checkpoint path
            checkpoint_path = os.path.join(
                args.output_dir,  # Base outputs directory
                f'oglanet_{city_name}_{source_res}_1',
                'checkpoint_best.pth'
            )
            print(f'\n{"="*60}')
            print(f'Auto-loading pretrained checkpoint from source domain:')
            print(f'  City: {city_name}')
            print(f'  Resolution: {source_res}')
            print(f'  Path: {checkpoint_path}')
            print(f'{"="*60}')

        # Load pretrained OGLANet weights if path exists
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                print(f'\n✗ ERROR: Checkpoint not found at {checkpoint_path}')
                print(f'  Please either:')
                print(f'    1. Train source domain first: python train.py --data_root {os.path.dirname(args.data_root)}/{source_res}')
                print(f'    2. Provide explicit path: --pretrained_checkpoint /path/to/checkpoint.pth')
                raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
            
            print(f'\n{"="*60}')
            print(f'Loading pretrained OGLANet from: {checkpoint_path}')
            print(f'{"="*60}')
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Extract info from checkpoint
            ckpt_args = checkpoint.get('args', {})
            ckpt_epoch = checkpoint.get('epoch', 'unknown')
            ckpt_miou = checkpoint.get('best_miou', 'unknown')
            
            print(f'Checkpoint info:')
            print(f'  Epoch: {ckpt_epoch}')
            print(f'  Best mIoU: {ckpt_miou}')
            print(f'  Use contrast: {ckpt_args.get("use_contrast", "unknown")}')
            
            # Load weights into base_model
            try:
                self.model.base_model.load_state_dict(checkpoint['model_state_dict'])
                print(f'✓ Successfully loaded pretrained weights into base_model')
            except RuntimeError as e:
                print(f'✗ Error loading checkpoint: {e}')
                print(f'Attempting to load compatible layers only...')
                
                # Load only matching keys
                pretrained_dict = checkpoint['model_state_dict']
                model_dict = self.model.base_model.state_dict()
                
                matched_dict = {k: v for k, v in pretrained_dict.items() 
                            if k in model_dict and v.size() == model_dict[k].size()}
                
                model_dict.update(matched_dict)
                self.model.base_model.load_state_dict(model_dict)
                
                print(f'✓ Loaded {len(matched_dict)}/{len(pretrained_dict)} layers')
                
                # Show mismatched keys
                mismatched = set(pretrained_dict.keys()) - set(matched_dict.keys())
                if mismatched:
                    print(f'⚠ Skipped layers: {mismatched}')
            
            print(f'{"="*60}\n')
        else:
            print('\n⚠ WARNING: No pretrained checkpoint provided!')
            print('  Teacher will start from random initialization.')
            print('  This may result in poor pseudo-label quality.\n')
        
        # Initialize EMA teacher
        self.teacher = EMATeacher(self.model, alpha=args.ema_alpha)
        self.teacher.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        
        # Setup loss functions
        self.hrda_loss = HRDALoss(hr_loss_weight=args.hr_loss_weight)
        self.pseudo_gen = PseudoLabelGenerator(
            num_classes=args.num_classes,
            confidence_threshold=args.confidence_threshold
        )
        
        # Setup optimizer (Adamax as per OGLANet paper)
        if args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(
                self.model.parameters(),
                lr=args.lr
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize tracking variables
        self.start_epoch = 0
        self.best_miou = 0.0
        self.best_val_loss = float('inf')
        # Tracking for plotting (add after line with self.best_miou)
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        
        # Load checkpoint if specified
        if args.resume:
            self.load_checkpoint(args.resume)
        
        # Load datasets
        self._setup_dataloaders()
    
    def get_confidence_threshold(self, epoch):
        """
        Gradually increase confidence threshold during training.
        
        Args:
            epoch: Current epoch number (1-indexed)
            
        Returns:
            float: Confidence threshold for this epoch
        """
        # Start low to bootstrap learning
        start_threshold = 0.52
        # End at a high value for quality pseudo-labels
        end_threshold = 0.90
        
        # Linear warmup over first 60% of training
        warmup_epochs = int(self.args.epochs * 0.6)
        
        if epoch <= warmup_epochs:
            # Linear increase from start to end
            progress = epoch / warmup_epochs
            threshold = start_threshold + (end_threshold - start_threshold) * progress
        else:
            # Stay at end threshold
            threshold = end_threshold
        
        return threshold
    
    def _setup_dataloaders(self):
        """Setup source and target dataloaders for single city"""
        args = self.args
        
        # Parse data_root to get city and current resolution
        # Expected: .../dataset/chicago/midres
        data_root = args.data_root.rstrip('/')
        current_res = os.path.basename(data_root)
        city_root = os.path.dirname(data_root)
        city_name = os.path.basename(city_root)
        
        # Determine source and target resolutions
        if current_res == args.target_res:
            # User provided target_res path, need to find source_res
            source_res = 'highres' if args.target_res == 'midres' else 'midres'
            source_path = os.path.join(city_root, source_res)
            target_path = data_root
        else:
            # User provided source_res path
            source_path = data_root
            target_path = os.path.join(city_root, args.target_res)
            source_res = current_res
        
        # Verify both paths exist
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source resolution path not found: {source_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target resolution path not found: {target_path}")
        
        print("\n" + "="*50)
        print(f"HRDA Single City Training: {city_name}")
        print("="*50)
        print(f"Source domain (labeled): {source_path}")
        print(f"Target domain (unlabeled): {target_path}")
        print(f"  Using train + val splits only (no test)")
        print("="*50 + "\n")
        
        # Source domain datasets (labeled)
        self.source_train_dataset = HRDAShadowDataset(
            source_path, split='train', is_source=True,
            img_size=args.img_size, context_size=args.context_size,
            detail_size=args.detail_size, scale_factor=args.scale_factor,
            augment=True, use_contrast=args.use_contrast
        )
        
        self.source_val_dataset = HRDAShadowDataset(
            source_path, split='val', is_source=True,
            img_size=args.img_size, context_size=args.context_size,
            detail_size=args.detail_size, scale_factor=args.scale_factor,
            augment=False, use_contrast=args.use_contrast
        )
        
        # Target domain datasets (unlabeled for adaptation)
        self.target_train_dataset = HRDAShadowDataset(
            target_path, split='train', is_source=False,
            img_size=args.img_size, context_size=args.context_size,
            detail_size=args.detail_size, scale_factor=args.scale_factor,
            augment=True, use_contrast=args.use_contrast
        )
        
        self.target_val_dataset = HRDAShadowDataset(
            target_path, split='val', is_source=False,
            img_size=args.img_size, context_size=args.context_size,
            detail_size=args.detail_size, scale_factor=args.scale_factor,
            augment=False, use_contrast=args.use_contrast
        )
        
        # Create dataloaders
        self.source_train_loader = torch.utils.data.DataLoader(
            self.source_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.source_val_loader = torch.utils.data.DataLoader(
            self.source_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        self.target_train_loader = torch.utils.data.DataLoader(
            self.target_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.target_val_loader = torch.utils.data.DataLoader(
            self.target_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Initialize detailed evaluator if enabled
        if args.eval_boundary_tolerant:
            self.detailed_evaluator_val = DetailedEvaluator()
            print("Boundary-tolerant evaluation enabled (source validation)")
        
        print(f'Source train samples: {len(self.source_train_dataset)}')
        print(f'Source val samples: {len(self.source_val_dataset)}')
        print(f'Target train samples (unlabeled): {len(self.target_train_dataset)}')
        print(f'Target val samples (unlabeled): {len(self.target_val_dataset)}')
    
    def train_epoch(self, epoch):
        """Train for one epoch with HRDA"""
        self.model.train()
        self.teacher.eval()

        # Update confidence threshold for this epoch
        current_threshold = self.get_confidence_threshold(epoch)
        self.pseudo_gen.confidence_threshold = current_threshold
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}: Confidence threshold = {current_threshold:.3f}")
        print(f"{'='*60}\n")
        
        epoch_loss_source = 0.0
        epoch_loss_target = 0.0
        
        # ADD THIS: Initialize metrics tracker
        train_metrics = ShadowMetrics()
        
        # Iterate over both source and target
        source_iter = iter(self.source_train_loader)
        target_iter = iter(self.target_train_loader)
        
        num_batches = min(len(self.source_train_loader), len(self.target_train_loader))
        
        for batch_idx in range(num_batches):
            try:
                source_batch = next(source_iter)
                target_batch = next(target_iter)
            except StopIteration:
                break
            
            # ============ Source Domain (Labeled) ============
            image_context_src = source_batch['image_context'].to(self.device)
            image_detail_src = source_batch['image_detail'].to(self.device)
            mask_context_src = source_batch['mask_context'].to(self.device)
            mask_detail_src = source_batch['mask_detail'].to(self.device)
            detail_coords_src = source_batch['detail_coords']

            # Convert batched coords
            if isinstance(detail_coords_src, (list, tuple)) and len(detail_coords_src) == 4:
                batch_size = len(detail_coords_src[0])
                detail_coords_src = [
                    (detail_coords_src[0][i].item(), detail_coords_src[1][i].item(),
                    detail_coords_src[2][i].item(), detail_coords_src[3][i].item())
                    for i in range(batch_size)
                ]

            # Forward pass
            output_src = self.model(image_context_src, image_detail_src, detail_coords_src)
            
            # Compute source loss
            loss_src_dict = self.hrda_loss(
                output_src['pred_fused'],
                output_src['pred_detail'],
                mask_context_src,
                mask_detail_src,
                aux_context=output_src.get('aux_context'),
                aux_detail=None
            )

            loss_source = loss_src_dict['total']

            # ADD THIS: Update training metrics (using fused prediction)
            with torch.no_grad():
                filtered_outputs = filter_small_predictions(output_src['pred_fused'], min_pixels=10)
                # Resize mask if needed to match prediction size
                if mask_context_src.shape[-2:] != filtered_outputs.shape[-2:]:
                    mask_context_resized = F.interpolate(
                        mask_context_src.unsqueeze(1).float(),
                        size=filtered_outputs.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    mask_context_resized = mask_context_src
                train_metrics.update(filtered_outputs, mask_context_resized)

            # ============ Target Domain (Pseudo-labels) ============
            image_context_tgt = target_batch['image_context'].to(self.device)
            image_detail_tgt = target_batch['image_detail'].to(self.device)
            detail_coords_tgt = target_batch['detail_coords']

            # Convert batched coords
            if isinstance(detail_coords_tgt, (list, tuple)) and len(detail_coords_tgt) == 4:
                batch_size = len(detail_coords_tgt[0])
                detail_coords_tgt = [
                    (detail_coords_tgt[0][i].item(), detail_coords_tgt[1][i].item(),
                    detail_coords_tgt[2][i].item(), detail_coords_tgt[3][i].item())
                    for i in range(batch_size)
                ]
            
            # Generate pseudo-labels with teacher model
            with torch.no_grad():
                teacher_output = self.teacher(
                    image_context_tgt, image_detail_tgt, detail_coords_tgt
                )
                
                # Generate pseudo-labels for fused prediction
                pseudo_fused = self.pseudo_gen(teacher_output['pred_fused'])
                pseudo_detail = self.pseudo_gen(teacher_output['pred_detail'])
            
            # Forward through student
            output_tgt = self.model(image_context_tgt, image_detail_tgt, detail_coords_tgt)
            
            # Compute target loss (with pseudo-labels and confidence)
            loss_tgt_dict = self.hrda_loss(
                output_tgt['pred_fused'],
                output_tgt['pred_detail'],
                pseudo_fused['pseudo_labels'],
                pseudo_detail['pseudo_labels'],
                pseudo_fused['confidence'],
                pseudo_detail['confidence'],
                aux_context=output_tgt.get('aux_context'),
                aux_detail=None
            )
            loss_target = loss_tgt_dict['total']
            
            # ============ Combined Loss ============
            total_loss = loss_source + self.args.lambda_target * loss_target
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update teacher model (EMA)
            self.teacher.update(self.model)
            
            # Track losses
            epoch_loss_source += loss_source.item()
            epoch_loss_target += loss_target.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch}/{self.args.epochs}] '
                    f'Batch [{batch_idx + 1}/{num_batches}] '
                    f'Loss_src: {loss_source.item():.4f} '
                    f'Loss_tgt: {loss_target.item():.4f} '
                    f'Aux_src: {loss_src_dict.get("aux", 0.0):.4f}')
                print(f'  Conf_fused: {pseudo_fused["confidence"].mean().item():.3f} '
                    f'Conf_detail: {pseudo_detail["confidence"].mean().item():.3f}')
        
        # Average losses
        epoch_loss_source /= num_batches
        epoch_loss_target /= num_batches
        
        # ADD THIS: Compute training metrics
        metrics = train_metrics.compute()
        
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Source Loss: {epoch_loss_source:.4f}')
        print(f'  Target Loss: {epoch_loss_target:.4f}')
        # ADD THIS: Print training metrics
        print(f'  Train mIoU: {metrics["mIOU"]:.2f}% | Shadow IoU: {metrics["Shadow_IOU"]:.2f}%')
        
        # Log to tensorboard
        self.writer.add_scalar('Train/Loss_Source', epoch_loss_source, epoch)
        self.writer.add_scalar('Train/Loss_Target', epoch_loss_target, epoch)
        self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
        # ADD THIS: Log training metrics
        for key, val in metrics.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)
        
        # ADD THIS: Store for plotting
        self.train_losses.append(epoch_loss_source)
        for key in self.train_metrics_history.keys():
            self.train_metrics_history[key].append(metrics[key])
        
        # MODIFY THIS: Return metrics as well
        return epoch_loss_source, epoch_loss_target, metrics
    
    def validate(self, epoch):
        """Validate on source domain validation set"""
        self.model.eval()
        
        val_loss = 0.0
        num_batches = 0
        
        # ADD THIS: Initialize metrics tracker
        val_metrics = ShadowMetrics()
        
        with torch.no_grad():
            for batch in self.source_val_loader:
                image_context = batch['image_context'].to(self.device)
                image_detail = batch['image_detail'].to(self.device)
                mask_context = batch['mask_context'].to(self.device)
                mask_detail = batch['mask_detail'].to(self.device)
                detail_coords = batch['detail_coords']

                # Convert batched coords
                if isinstance(detail_coords, (list, tuple)) and len(detail_coords) == 4:
                    batch_size = len(detail_coords[0])
                    detail_coords = [
                        (detail_coords[0][i].item(), detail_coords[1][i].item(),
                        detail_coords[2][i].item(), detail_coords[3][i].item())
                        for i in range(batch_size)
                    ]
                
                # Forward pass
                output = self.model(image_context, image_detail, detail_coords)
                
                # Compute loss
                loss_dict = self.hrda_loss(
                    output['pred_fused'],
                    output['pred_detail'],
                    mask_context,
                    mask_detail
                )
                
                val_loss += loss_dict['total'].item()
                
                # ADD THIS: Update validation metrics
                pred_fused = output['pred_fused']
                filtered_outputs = filter_small_predictions(pred_fused, min_pixels=10)
                
                # Resize mask_context to match pred_fused if needed
                if mask_context.shape[-2:] != filtered_outputs.shape[-2:]:
                    mask_context_resized = F.interpolate(
                        mask_context.unsqueeze(1).float(),
                        size=filtered_outputs.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    mask_context_resized = mask_context
                
                val_metrics.update(filtered_outputs, mask_context_resized)
                
                num_batches += 1
        
        val_loss /= num_batches
        
        # ADD THIS: Compute metrics
        metrics = val_metrics.compute()
        
        print(f'\nValidation - Epoch {epoch}:')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val mIoU: {metrics["mIOU"]:.2f}% | Shadow IoU: {metrics["Shadow_IOU"]:.2f}%')
        
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        # ADD THIS: Log all validation metrics
        for key, val in metrics.items():
            self.writer.add_scalar(f'Val/{key}', val, epoch)

        # Boundary-tolerant evaluation if enabled
        if self.args.eval_boundary_tolerant:
            # Reset evaluator for this epoch
            self.detailed_evaluator_val = DetailedEvaluator()
            
            # Second pass through validation for detailed metrics
            with torch.no_grad():
                for batch in self.source_val_loader:
                    image_context = batch['image_context'].to(self.device)
                    image_detail = batch['image_detail'].to(self.device)
                    mask_context = batch['mask_context'].to(self.device)
                    detail_coords = batch['detail_coords']
                    
                    # Convert batched coords
                    if isinstance(detail_coords, (list, tuple)) and len(detail_coords) == 4:
                        batch_size = len(detail_coords[0])
                        detail_coords = [
                            (detail_coords[0][i].item(), detail_coords[1][i].item(),
                            detail_coords[2][i].item(), detail_coords[3][i].item())
                            for i in range(batch_size)
                        ]
                    
                    # Forward pass
                    output = self.model(image_context, image_detail, detail_coords)
                    preds = torch.argmax(output['pred_fused'], dim=1)
                    
                    # Resize mask if needed to match predictions
                    if mask_context.shape[-2:] != preds.shape[-2:]:
                        mask_context_resized = F.interpolate(
                            mask_context.unsqueeze(1).float(),
                            size=preds.shape[-2:],
                            mode='nearest'
                        ).squeeze(1).long()
                    else:
                        mask_context_resized = mask_context
                    
                    # Resize image_context to match predictions for evaluator
                    if image_context.shape[-2:] != preds.shape[-2:]:
                        image_context_resized = F.interpolate(
                            image_context,
                            size=preds.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    else:
                        image_context_resized = image_context
                    
                    self.detailed_evaluator_val.update(preds, mask_context_resized, image_context_resized)
            
            # Compute detailed metrics
            detailed_results = self.detailed_evaluator_val.compute_metrics()
            
            print(f'  Tolerant F1 (5px): {detailed_results["boundary_tolerant"]["tolerant_5px"]["f1"]:.2f}%')
            print(f'  Tolerant mIOU (5px): {detailed_results["boundary_tolerant"]["tolerant_5px"]["iou"]:.2f}%')
            
            # Log to tensorboard
            self.writer.add_scalar('Val/Tolerant_F1', detailed_results["boundary_tolerant"]["tolerant_5px"]["f1"], epoch)
            self.writer.add_scalar('Val/Tolerant_mIOU', detailed_results["boundary_tolerant"]["tolerant_5px"]["iou"], epoch)
        
        # ADD THIS: Store for plotting - THIS IS THE KEY PART THAT'S MISSING
        self.val_losses.append(val_loss)
        for key in self.val_metrics_history.keys():
            self.val_metrics_history[key].append(metrics[key])
        
        # MODIFY THIS: Return both loss and metrics
        return val_loss, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'best_val_loss': getattr(self, 'best_val_loss', float('inf')),
            'args': vars(self.args)
        }
        
        # Save latest
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved: {best_path}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_miou = checkpoint.get('best_miou', 0.0)

        print(f'Resumed from epoch {checkpoint["epoch"]}')
        print(f'  Best val loss: {self.best_val_loss:.4f}')
        print(f'  Best mIoU: {self.best_miou:.4f}')
    
    def train(self):
        """Main training loop"""
        print('\n' + '='*50)
        print('Starting HRDA Training for OGLANet...')
        print('='*50)

        print(f"\nTraining Configuration:")
        print(f"  Learning rate: {self.args.lr}")
        print(f"  Optimizer: Adamax")
        print(f"  EMA alpha: {self.args.ema_alpha}")
        print(f"  Use contrast: {self.args.use_contrast}")
        print(f"  Pretrained: {self.args.pretrained}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Epochs: {self.args.epochs}")
        print(f"  Confidence threshold: {self.args.confidence_threshold}")
        print(f"  HR loss weight: {self.args.hr_loss_weight}")
        print(f"  Auxiliary weight: {self.args.aux_weight}")
        print('='*50 + '\n')
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train one epoch
            loss_source, loss_target, train_metrics = self.train_epoch(epoch + 1)

            # Validate - unpack as val_loss and val_metrics (dictionary)
            val_loss, val_metrics = self.validate(epoch + 1)

            # Update learning rate based on validation mIoU
            self.scheduler.step(val_metrics['mIOU'])

            # Check if best model
            is_best = val_metrics['mIOU'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['mIOU']
                print(f'New best mIoU: {self.best_miou:.2f}%')
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best=is_best)
        
        print('\nTraining completed!')
        print(f'Best mIoU: {self.best_miou:.4f}')

        # Generate plots
        print('\nGenerating visualization plots...')
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
        
        with torch.no_grad():
            for batch in self.source_val_loader:  # or create a test loader
                image_context = batch['image_context'].to(self.device)
                image_detail = batch['image_detail'].to(self.device)
                mask_context = batch['mask_context'].to(self.device)
                detail_coords = batch['detail_coords']
                
                # Convert batched coords
                if isinstance(detail_coords, (list, tuple)) and len(detail_coords) == 4:
                    batch_size = len(detail_coords[0])
                    detail_coords = [
                        (detail_coords[0][i].item(), detail_coords[1][i].item(),
                        detail_coords[2][i].item(), detail_coords[3][i].item())
                        for i in range(batch_size)
                    ]
                
                # Forward pass
                output = self.model(image_context, image_detail, detail_coords)
                
                # Update metrics
                from utils.postprocessing import filter_small_predictions
                filtered_outputs = filter_small_predictions(output['pred_fused'], min_pixels=10)
                test_metrics.update(filtered_outputs, mask_context)
        
        # Compute metrics
        metrics = test_metrics.compute()

        print('\nStandard Test Results:')
        for key, val in metrics.items():
            print(f'{key}: {val:.2f}%')

        # Prepare results dictionary
        results_to_save = {'standard': metrics}

        # Boundary-tolerant evaluation if enabled
        if self.args.eval_boundary_tolerant:
            detailed_evaluator_test = DetailedEvaluator()
            
            with torch.no_grad():
                for batch in self.target_test_loader:
                    image_context = batch['image_context'].to(self.device)
                    image_detail = batch['image_detail'].to(self.device)
                    mask_context = batch['mask_context'].to(self.device)
                    detail_coords = batch['detail_coords']
                    
                    # Convert coords
                    if isinstance(detail_coords, (list, tuple)) and len(detail_coords) == 4:
                        batch_size = len(detail_coords[0])
                        detail_coords = [
                            (detail_coords[0][i].item(), detail_coords[1][i].item(),
                            detail_coords[2][i].item(), detail_coords[3][i].item())
                            for i in range(batch_size)
                        ]
                    
                    output = self.model(image_context, image_detail, detail_coords)
                    preds = torch.argmax(output['pred_fused'], dim=1)
                    
                    # Resize mask if needed
                    if mask_context.shape[-2:] != preds.shape[-2:]:
                        mask_context_resized = F.interpolate(
                            mask_context.unsqueeze(1).float(),
                            size=preds.shape[-2:],
                            mode='nearest'
                        ).squeeze(1).long()
                    else:
                        mask_context_resized = mask_context
                    
                    # Resize image_context to match predictions for evaluator
                    if image_context.shape[-2:] != preds.shape[-2:]:
                        image_context_resized = F.interpolate(
                            image_context,
                            size=preds.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    else:
                        image_context_resized = image_context
                    
                    detailed_evaluator_test.update(preds, mask_context_resized, image_context_resized)
            
            detailed_results = detailed_evaluator_test.compute_metrics()
            
            print('\n' + '='*50)
            print('Boundary-Tolerant Evaluation:')
            print('='*50)
            print(f"Strict F1:     {detailed_results['boundary_tolerant']['strict']['f1']:.2f}%")
            print(f"Strict mIOU:   {detailed_results['boundary_tolerant']['strict']['iou']:.2f}%")
            print(f"Tolerant F1:   {detailed_results['boundary_tolerant']['tolerant_5px']['f1']:.2f}%")
            print(f"Tolerant mIOU: {detailed_results['boundary_tolerant']['tolerant_5px']['iou']:.2f}%")
            
            results_to_save['detailed'] = detailed_results

        # Save all results
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f'\nResults saved to {results_path}')
        
        # Generate visualizations
        save_best_worst_visualizations(
            self.model,
            self.source_val_loader,  # or test loader
            self.device,
            self.output_dir,
            num_images=10,
            is_hrda=True  # Flag to handle HRDA-specific forward pass
        )
        
        return metrics


def main():
    args = get_args()
    
    # Create trainer
    trainer = HRDATrainer(args)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()