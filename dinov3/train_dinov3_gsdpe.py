"""
Training script for DINOv3 with GSDPE (Ground Sample Distance Positional Encoding)

Implements cross-resolution transfer learning using original Scale-MAE GSDPE formula:
- Train on one resolution (e.g., midres 0.6m)
- Test on another resolution (e.g., highres 0.3m)

Key differences from full Scale-MAE:
- No MAE pretraining (direct supervised training)
- No multi-scale reconstruction
- Only GSDPE for scale-awareness
- Single-stage training (vs two-stage)
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

# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_gsdpe import DINOv3ShadowDetectorGSDPE
from dataset_gsdpe import get_dataloaders_gsdpe, LOCO_FOLDS

# Import utilities (reuse from existing DINOv3)
from utils.losses import CrossEntropyLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization import (
    plot_loss_curves,
    plot_metrics_curves,
    save_best_worst_visualizations
)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DINOv3 with GSDPE for Cross-Resolution Transfer')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default=None,
                      help='Root directory of dataset (for single mode)')
    parser.add_argument('--base_data_root', type=str, default=None,
                      help='Base directory for all/loco modes')
    parser.add_argument('--mode', type=str, default='single',
                      choices=['single', 'all', 'loco'],
                      help='Training mode')
    parser.add_argument('--cities', type=str, nargs='+', default=['chicago', 'miami', 'phoenix'],
                      help='List of cities for all mode')
    parser.add_argument('--resolution', type=str, default=None,
                      choices=['highres', 'midres'],
                      help='Resolution for all/loco modes')
    parser.add_argument('--fold_id', type=int, default=None,
                      choices=[0, 1, 2],
                      help='Fold ID for LOCO mode')
    parser.add_argument('--img_size', type=int, default=384,
                      help='Input image size (default: 384 for DINOv3)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes')
    parser.add_argument('--model_name', type=str, default='dinov3_vits16',
                      choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'],
                      help='DINOv3 model variant')
    parser.add_argument('--weights_path', type=str, required=True,
                      help='Path to DINOv3 pretrained weights .pth file')
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Use pretrained DINOv3 weights')
    parser.add_argument('--frozen_stages', type=int, default=-1,
                      help='Number of backbone stages to freeze (-1 = train all)')
    parser.add_argument('--reference_gsd', type=float, default=1.0,
                      help='Reference GSD for GSDPE (default: 1.0m)')
    
    # Training parameters (optimized for ViT finetuning with GSDPE)
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='Learning rate (0.0001 recommended for ViT+GSDPE finetuning)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                      help='Weight decay (0.05 recommended for ViT)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                      help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                      help='Minimum learning rate for cosine schedule')
    
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
                      help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


class CosineWarmupScheduler:
    """Cosine learning rate schedule with warmup (standard for ViT)"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self, epoch):
        """Update learning rate"""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class Trainer:
    """Trainer class for DINOv3 with GSDPE"""
    
    def __init__(self, args):
        self.args = args
        
        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.mode == 'single':
            city = args.data_root.rstrip('/').split("/")[-2]
            res = args.data_root.rstrip('/').split("/")[-1]
            exp_name = f'dinov3_gsdpe_{city}_{res}_{timestamp}'
        elif args.mode == 'all':
            exp_name = f'dinov3_gsdpe_all_{args.resolution}_{timestamp}'
        elif args.mode == 'loco':
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = f'dinov3_gsdpe_loco_holdout_{test_city}_{args.resolution}_{timestamp}'
        
        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Initialize model with GSDPE
        print('\nInitializing DINOv3 with GSDPE...')
        self.model = DINOv3ShadowDetectorGSDPE(
            num_classes=args.num_classes,
            model_name=args.model_name,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            reference_gsd=args.reference_gsd
        ).to(self.device)
        
        # Setup loss function
        self.criterion = CrossEntropyLoss()
        
        # Setup optimizer (AdamW as per ViT best practices)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler (Cosine with warmup)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.min_lr
        )
        
        # Initialize tracking variables
        self.start_epoch = 0
        self.best_miou = 0.0
        self.best_shadow_iou = 0.0
        self.best_f1 = 0.0
        
        # Resume from checkpoint if specified
        if args.resume:
            self.load_checkpoint(args.resume)
        
        # Load datasets
        print('\nLoading datasets...')
        self.dataloaders = get_dataloaders_gsdpe(
            data_root=args.data_root,
            base_data_root=args.base_data_root,
            mode=args.mode,
            cities=args.cities,
            resolution=args.resolution,
            fold_id=args.fold_id,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size
        )
        
        print(f'\nDataset sizes:')
        print(f'  Training samples: {len(self.dataloaders["train"].dataset)}')
        print(f'  Validation samples: {len(self.dataloaders["val"].dataset)}')
        print(f'  Test samples: {len(self.dataloaders["test"].dataset)}')
        
        # Tracking for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        train_metrics = ShadowMetrics()
        
        num_batches = len(self.dataloaders['train'])
        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            gsd = batch['gsd'].to(self.device)  # GSD values
            
            # Forward pass with GSD conditioning
            outputs = self.model(images, gsd)
            
            # Compute loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics (with postprocessing)
            filtered_outputs = filter_small_predictions(outputs, min_pixels=10)
            train_metrics.update(filtered_outputs, masks)
            
            # Track losses
            epoch_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'Batch [{batch_idx + 1}/{num_batches}] | Loss: {loss.item():.4f}')
        
        # Compute average loss
        epoch_loss /= num_batches
        
        # Compute metrics
        metrics = train_metrics.compute()
        
        # Time taken
        epoch_time = time.time() - start_time
        
        print(f'\nTraining Results:')
        print(f'Time: {epoch_time:.2f}s | Loss: {epoch_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}% | Precision: {metrics["Precision"]:.2f}% | '
              f'F1: {metrics["F1"]:.2f}% | BER: {metrics["BER"]:.2f}% | '
              f'mIOU: {metrics["mIOU"]:.2f}% | Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')
        
        # Log to tensorboard
        self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
        for key in metrics:
            self.writer.add_scalar(f'Train/{key}', metrics[key], epoch)
        
        # Store for plotting
        self.train_losses.append(epoch_loss)
        for key in self.train_metrics_history.keys():
            self.train_metrics_history[key].append(metrics[key])
        
        return epoch_loss, metrics
    
    def validate(self, epoch):
        """Validate the model"""
        print('\nValidating...')
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = ShadowMetrics()
        
        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                gsd = batch['gsd'].to(self.device)
                
                # Forward pass with GSD
                outputs = self.model(images, gsd)
                
                # Compute loss
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                
                # Update metrics (with postprocessing)
                filtered_outputs = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered_outputs, masks)
        
        # Compute average loss
        val_loss /= len(self.dataloaders['val'])
        
        # Compute metrics
        metrics = val_metrics.compute()
        
        print(f'Validation Results:')
        print(f'Loss: {val_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}% | Precision: {metrics["Precision"]:.2f}% | '
              f'F1: {metrics["F1"]:.2f}% | BER: {metrics["BER"]:.2f}% | '
              f'mIOU: {metrics["mIOU"]:.2f}% | Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        for key in metrics:
            self.writer.add_scalar(f'Val/{key}', metrics[key], epoch)
        
        # Store for plotting
        self.val_losses.append(val_loss)
        for key in self.val_metrics_history.keys():
            self.val_metrics_history[key].append(metrics[key])
        
        return val_loss, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'best_shadow_iou': self.best_shadow_iou,
            'best_f1': self.best_f1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'args': vars(self.args)
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved to {best_path}')
        
        # Save epoch checkpoint
        if epoch % self.args.save_freq == 0:
            epoch_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        self.best_shadow_iou = checkpoint.get('best_shadow_iou', 0.0)
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        
        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_metrics_history = checkpoint.get('train_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })
        self.val_metrics_history = checkpoint.get('val_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })
        
        print(f'Resumed from epoch {checkpoint["epoch"]}')
        print(f'Best mIOU: {self.best_miou:.2f}%, Best Shadow IoU: {self.best_shadow_iou:.2f}%, Best F1: {self.best_f1:.2f}%')
    
    def train(self):
        """Main training loop"""
        print('\n' + '='*50)
        print('Starting training...')
        print('='*50)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Update learning rate
            current_lr = self.scheduler.step(epoch)
            print(f'\nLearning rate: {current_lr:.2e}')
            
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(epoch + 1)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch + 1)
            
            # Check if this is the best model
            is_best = False
            if val_metrics['mIOU'] > self.best_miou:
                self.best_miou = val_metrics['mIOU']
                is_best = True
                print(f'New best mIOU: {self.best_miou:.2f}%')
            
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
        
        print('\nTraining completed!')
        print(f'Best mIOU: {self.best_miou:.2f}%')
        print(f'Best Shadow IoU: {self.best_shadow_iou:.2f}%')
        print(f'Best F1: {self.best_f1:.2f}%')
        
        # Generate plots
        print('\nGenerating plots...')
        
        # Plot loss curves
        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png')
        )
        
        # Plot metrics curves
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png')
        )
        
        # Close tensorboard writer
        self.writer.close()
    
    def test(self):
        """Test the model with postprocessing and visualization"""
        print('\n' + '='*50)
        print('Testing model...')
        print('='*50)
        
        # Load best checkpoint
        best_checkpoint = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_checkpoint):
            self.load_checkpoint(best_checkpoint)
        else:
            print('Warning: Best checkpoint not found, using current model weights')
        
        self.model.eval()
        test_metrics = ShadowMetrics()
        
        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                gsd = batch['gsd'].to(self.device)
                
                # Forward pass with GSD
                outputs = self.model(images, gsd)
                
                # Update metrics (with postprocessing)
                filtered_outputs = filter_small_predictions(outputs, min_pixels=10)
                test_metrics.update(filtered_outputs, masks)
        
        # Compute metrics
        metrics = test_metrics.compute()
        
        print('\nTest Results:')
        print(f'OA: {metrics["OA"]:.2f}%')
        print(f'Precision: {metrics["Precision"]:.2f}%')
        print(f'F1: {metrics["F1"]:.2f}%')
        print(f'BER: {metrics["BER"]:.2f}%')
        print(f'mIOU: {metrics["mIOU"]:.2f}%')
        print(f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')
        
        # Save results
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f'\nResults saved to {results_path}')
        
        # Generate best/worst visualizations
        print('\nGenerating best/worst predictions visualizations...')
        
        # Need to modify save_best_worst_visualizations to accept GSD
        # For now, we'll create a wrapper
        # def model_forward_with_gsd(images):
        #     # Extract GSD from batch (assuming test loader provides it)
        #     # This is a simplified version - you may need to adapt based on your needs
        #     with torch.no_grad():
        #         # For visualization, we'll use the mean GSD from the dataset
        #         # Or you can pass it explicitly
        #         gsd = torch.tensor([0.6] * images.size(0)).to(images.device)  # Default to midres
        #         return self.model(images, gsd)
        
        # # Save visualizations
        # save_best_worst_visualizations(
        #     model_forward_with_gsd,
        #     self.dataloaders['test'],
        #     self.device,
        #     self.output_dir,
        #     num_images=10
        # )

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
    
    # Print key settings
    print("\n" + "="*70)
    print("DINOv3 + GSDPE Training")
    print("="*70)
    print(f"Mode: {args.mode}")
    if args.mode == 'loco':
        print(f"Fold ID: {args.fold_id}")
        print(f"Train cities: {LOCO_FOLDS[args.fold_id]['train']}")
        print(f"Test city: {LOCO_FOLDS[args.fold_id]['test']}")
    print(f"Resolution: {args.resolution}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Reference GSD: {args.reference_gsd}m")
    print("="*70 + "\n")
    
    # Create trainer
    trainer = Trainer(args)
    
    if args.eval_only:
        # Only evaluate
        trainer.test()
    else:
        # Train and then test
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()