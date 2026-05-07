"""
Training Script for DINOv3-HRDA
Implements multi-resolution training with domain adaptation for shadow detection.

Based on HRDA (ECCV 2022): https://arxiv.org/abs/2204.13132

Key Features:
- Single city, cross-resolution adaptation (e.g., midres → highres)
- Multi-resolution training with LR context and HR detail crops
- EMA teacher for pseudo-label generation
- Scale attention for learned fusion
- Auxiliary supervision for better training

Hyperparameters adapted for DINOv3:
- Learning rate: 3e-5 (lower than MAMNet due to pretrained ViT)
- Weight decay: 0.05 (standard for ViT fine-tuning)
- EMA alpha: 0.999 (as per HRDA paper)
- Confidence threshold: Dynamic warmup from 0.52 to 0.90
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

from dinov3_hrda import create_dinov3_hrda
from data.dataset_hrda import HRDAShadowDataset
from utils.hrda_losses import HRDALoss, PseudoLabelGenerator
from utils.ema import EMATeacher
from utils.metrics import ShadowMetrics
from utils.visualization import plot_loss_curves, plot_metrics_curves, save_best_worst_visualizations


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DINOv3-HRDA for Shadow Detection')

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
    parser.add_argument('--model_name', type=str, default='dinov3_vits16',
                      choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'],
                      help='DINOv3 model variant')
    parser.add_argument('--weights_path', type=str, default=None,
                      help='Path to DINOv3 pretrained weights')
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Use pretrained DINOv3 weights')
    parser.add_argument('--hr_loss_weight', type=float, default=0.1,
                      help='Weight for HR detail loss (λ_d)')
    
    # Training parameters (adapted for DINOv3)
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size (HRDA is memory-intensive, DINOv3 even more so)')
    parser.add_argument('--lr', type=float, default=3e-5,
                      help='Learning rate (lower for pretrained ViT)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                      help='Weight decay (standard for ViT fine-tuning)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of data loading workers')
    
    # EMA and pseudo-label parameters
    parser.add_argument('--ema_alpha', type=float, default=0.999,
                      help='EMA momentum for teacher model')
    parser.add_argument('--confidence_threshold', type=float, default=0.52,
                      help='Initial confidence threshold for pseudo-labels')
    parser.add_argument('--lambda_target', type=float, default=1.0,
                      help='Weight for target domain loss')
    
    # Checkpoint and logging
    parser.add_argument('--output_dir', type=str, default='./outputs_hrda_dinov3',
                      help='Directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=5,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


class DINOv3HRDATrainer:
    """Trainer for DINOv3-HRDA"""
    
    def __init__(self, args):
        self.args = args
        
        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Parse city name from data_root
        city_name = os.path.basename(os.path.dirname(args.data_root.rstrip('/')))
        exp_name = f'dinov3_hrda_{city_name}_{timestamp}'
        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Initialize model (student)
        print('Initializing DINOv3-HRDA...')
        self.model = create_dinov3_hrda(
            num_classes=args.num_classes,
            model_name=args.model_name,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            use_aux=True,
            hr_loss_weight=args.hr_loss_weight
        ).to(self.device)
        
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
        
        # Setup optimizer (AdamW for ViT)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        
        # Initialize tracking variables
        self.start_epoch = 0
        self.best_miou = 0.0
        self.best_val_loss = float('inf')
        # Tracking for plotting
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
        data_root = args.data_root.rstrip('/')
        current_res = os.path.basename(data_root)
        city_root = os.path.dirname(data_root)
        city_name = os.path.basename(city_root)
        
        # Determine source and target resolutions
        if current_res == args.target_res:
            source_res = 'highres' if args.target_res == 'midres' else 'midres'
            source_path = os.path.join(city_root, source_res)
            target_path = data_root
        else:
            source_path = data_root
            target_path = os.path.join(city_root, args.target_res)
            source_res = current_res
        
        # Verify both paths exist
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source resolution path not found: {source_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target resolution path not found: {target_path}")
        
        print("\n" + "="*50)
        print(f"DINOv3-HRDA Single City Training: {city_name}")
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
            augment=True
        )
        
        self.source_val_dataset = HRDAShadowDataset(
            source_path, split='val', is_source=True,
            img_size=args.img_size, context_size=args.context_size,
            detail_size=args.detail_size, scale_factor=args.scale_factor,
            augment=False
        )
        
        # Target domain datasets (unlabeled for adaptation)
        self.target_train_dataset = HRDAShadowDataset(
            target_path, split='train', is_source=False,
            img_size=args.img_size, context_size=args.context_size,
            detail_size=args.detail_size, scale_factor=args.scale_factor,
            augment=True
        )
        
        self.target_val_dataset = HRDAShadowDataset(
            target_path, split='val', is_source=False,
            img_size=args.img_size, context_size=args.context_size,
            detail_size=args.detail_size, scale_factor=args.scale_factor,
            augment=False
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
                aux_detail=output_src.get('aux_detail')
            )
            loss_source = loss_src_dict['total']
            
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
                
                # Generate pseudo-labels for fused and detail predictions
                pseudo_fused = self.pseudo_gen(teacher_output['pred_fused'])
                pseudo_detail = self.pseudo_gen(teacher_output['pred_detail'])
            
            # Forward through student
            output_tgt = self.model(image_context_tgt, image_detail_tgt, detail_coords_tgt)
            
            # Compute target loss with pseudo-labels
            loss_tgt_dict = self.hrda_loss(
                output_tgt['pred_fused'],
                output_tgt['pred_detail'],
                pseudo_fused['pseudo_labels'],
                pseudo_detail['pseudo_labels'],
                pseudo_fused['confidence'],
                pseudo_detail['confidence'],
                aux_context=output_tgt.get('aux_context'),
                aux_detail=output_tgt.get('aux_detail')
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
                    f'Aux_src: {loss_src_dict.get("aux", 0.0):.4f} '
                    f'Aux_tgt: {loss_tgt_dict.get("aux", 0.0):.4f}')
                print(f'  Conf_fused: {pseudo_fused["confidence"].mean().item():.3f} '
                    f'Conf_detail: {pseudo_detail["confidence"].mean().item():.3f}')
        
        # Average losses
        epoch_loss_source /= num_batches
        epoch_loss_target /= num_batches

        # Compute training metrics on source domain
        train_metrics = ShadowMetrics()
        with torch.no_grad():
            for batch in self.source_train_loader:
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
                
                output = self.model(image_context, image_detail, detail_coords)
                
                # Resize mask for metrics
                pred_fused = output['pred_fused']
                if mask_context.shape[-2:] != pred_fused.shape[-2:]:
                    mask_resized = F.interpolate(
                        mask_context.unsqueeze(1).float(),
                        size=pred_fused.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    mask_resized = mask_context
                
                from utils.postprocessing import filter_small_predictions
                filtered_outputs = filter_small_predictions(pred_fused, min_pixels=10)
                train_metrics.update(filtered_outputs, mask_resized)

        metrics = train_metrics.compute()

        print(f'\nEpoch {epoch} Summary:')
        print(f'  Source Loss: {epoch_loss_source:.4f}')
        print(f'  Target Loss: {epoch_loss_target:.4f}')
        print(f'  OA: {metrics["OA"]:.2f}% | F1: {metrics["F1"]:.2f}% | '
            f'mIOU: {metrics["mIOU"]:.2f}% | Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # Log to tensorboard
        self.writer.add_scalar('Train/Loss_Source', epoch_loss_source, epoch)
        self.writer.add_scalar('Train/Loss_Target', epoch_loss_target, epoch)
        self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)

        # Store for plotting
        self.train_losses.append(epoch_loss_source)
        for key in self.train_metrics_history.keys():
            self.train_metrics_history[key].append(metrics[key])

        return epoch_loss_source, epoch_loss_target, metrics
    
    def validate(self, epoch):
        """Validate on source domain validation set"""
        self.model.eval()
        
        val_loss = 0.0
        num_batches = 0
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
                
                # Resize mask for metrics
                pred_fused = output['pred_fused']
                if mask_context.shape[-2:] != pred_fused.shape[-2:]:
                    mask_resized = F.interpolate(
                        mask_context.unsqueeze(1).float(),
                        size=pred_fused.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    mask_resized = mask_context
                
                # Update metrics
                from utils.postprocessing import filter_small_predictions
                filtered_outputs = filter_small_predictions(pred_fused, min_pixels=10)
                val_metrics.update(filtered_outputs, mask_resized)
                
                num_batches += 1
        
        val_loss /= num_batches
        metrics = val_metrics.compute()
        
        print(f'\nValidation - Epoch {epoch}:')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  OA: {metrics["OA"]:.2f}% | F1: {metrics["F1"]:.2f}% | '
            f'mIOU: {metrics["mIOU"]:.2f}% | Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')
        
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Val/{key}', val, epoch)
        
        # Store for plotting
        self.val_losses.append(val_loss)
        for key in self.val_metrics_history.keys():
            self.val_metrics_history[key].append(metrics[key])
        
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
            'best_val_loss': self.best_val_loss,
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
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
        print('Starting DINOv3-HRDA Training...')
        print('='*50)
        
        print(f"\nTraining Configuration:")
        print(f"  Model: {self.args.model_name}")
        print(f"  Learning rate: {self.args.lr}")
        print(f"  Weight decay: {self.args.weight_decay}")
        print(f"  EMA alpha: {self.args.ema_alpha}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Epochs: {self.args.epochs}")
        print(f"  Initial confidence threshold: {self.args.confidence_threshold}")
        print(f"  HR loss weight: {self.args.hr_loss_weight}")
        print('='*50 + '\n')
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train one epoch
            loss_source, loss_target, train_metrics = self.train_epoch(epoch + 1)

            # Validate
            val_loss, val_metrics = self.validate(epoch + 1)

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint
            # Check if this is the best model
            is_best = val_metrics['mIOU'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['mIOU']
                # Save best checkpoint immediately
                self.save_checkpoint(epoch + 1, is_best=True)
                print(f'  ✓ New best model saved! mIOU: {self.best_miou:.4f}')

            # Save periodic checkpoint
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
                print(f'  ✓ Periodic checkpoint saved at epoch {epoch + 1}')
        
        print('\nTraining completed!')
        print(f'Best mIoU: {self.best_miou:.4f}')

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
        """Test the model on source validation set"""
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
                pred_fused = output['pred_fused']
                
                # Resize mask for metrics
                if mask_context.shape[-2:] != pred_fused.shape[-2:]:
                    mask_resized = F.interpolate(
                        mask_context.unsqueeze(1).float(),
                        size=pred_fused.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    mask_resized = mask_context
                
                # Update metrics
                from utils.postprocessing import filter_small_predictions
                filtered_outputs = filter_small_predictions(pred_fused, min_pixels=10)
                test_metrics.update(filtered_outputs, mask_resized)
        
        # Compute metrics
        metrics = test_metrics.compute()
        
        print('\nTest Results:')
        for key, val in metrics.items():
            print(f'{key}: {val:.2f}%')
        
        # Save results
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f'\nResults saved to {results_path}')
        
        # Generate visualizations
        # Note: You'll need to adapt the dataloader for save_best_worst_visualizations
        # or create a wrapper that handles HRDA's dual-crop format
        print('\nNote: Visual outputs require adaptation for HRDA dual-crop format')
        
        return metrics


def main():
    args = get_args()
    
    # Create trainer
    trainer = DINOv3HRDATrainer(args)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()