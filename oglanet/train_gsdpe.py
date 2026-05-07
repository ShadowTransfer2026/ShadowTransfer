"""
Training script for OGLANet with GSDPE

Implements single city/resolution training with GSD-aware positional encoding.
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
from utils.evaluation_detailed import DetailedEvaluator
from models.gsdpe import get_gsd_from_filename

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.oglanet_gsdpe import OGLANet_GSDPE
from data.dataset_gsdpe import get_dataloaders_gsdpe
from utils.losses import OGLANetLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization import (
    plot_loss_curves, 
    plot_metrics_curves, 
    save_best_worst_visualizations
)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train OGLANet with GSDPE')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                      help='Root directory of dataset (e.g., /path/to/chicago/midres)')
    parser.add_argument('--img_size', type=int, default=384,
                      help='Input image size')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Use pretrained encoder')
    parser.add_argument('--reference_gsd', type=float, default=1.0,
                      help='Reference GSD for GSDPE (default: 1.0m)')
    
    # Training parameters (adjusted for GSDPE based on Scale-MAE)
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate (Scale-MAE-inspired: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                      help='Weight decay (Scale-MAE: 0.005)')
    parser.add_argument('--optimizer', type=str, default='adamax',
                      choices=['adam', 'adamax'],
                      help='Optimizer')
    
    # Checkpoint and logging
    parser.add_argument('--output_dir', type=str, default='./outputs',
                      help='Output directory')
    parser.add_argument('--save_freq', type=int, default=10,
                      help='Save every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                      help='Resume from checkpoint')
    parser.add_argument('--eval_only', action='store_true',
                      help='Evaluation only')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cuda/cpu)')
    
    parser.add_argument('--use_contrast', action='store_true',
                    help='Use contrast as 4th input channel')
    parser.add_argument('--eval_boundary_tolerant', action='store_true',
                        help='Compute boundary-tolerant metrics during evaluation')
    
    return parser.parse_args()


class Trainer:
    """Trainer for OGLANet with GSDPE"""
    
    def __init__(self, args):
        self.args = args
        
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        city = args.data_root.rstrip('/').split("/")[-2]
        res = args.data_root.rstrip('/').split("/")[-1]
        exp_name = f'oglanet_gsdpe_{city}_{res}_{1}'
        
        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Initialize model with GSDPE
        print('Initializing OGLANet with GSDPE...')
        self.model = OGLANet_GSDPE(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size,
            reference_gsd=args.reference_gsd,
            use_contrast=args.use_contrast
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        
        # Setup loss
        self.criterion = OGLANetLoss()
        
        # Setup optimizer (Adamax as per OGLANet paper)
        if args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Tracking
        self.start_epoch = 0
        self.best_miou = 0.0
        self.best_shadow_iou = 0.0
        self.best_f1 = 0.0
        
        if args.resume:
            self.load_checkpoint(args.resume)
        
        # Load datasets
        self.dataloaders = get_dataloaders_gsdpe(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            use_contrast=args.use_contrast
        )

        if args.eval_boundary_tolerant:
            self.detailed_evaluator_test = DetailedEvaluator()
            self.detailed_evaluator_val = DetailedEvaluator()
            print("Boundary-tolerant evaluation enabled")
        
        print(f'Training samples: {len(self.dataloaders["train"].dataset)}')
        print(f'Validation samples: {len(self.dataloaders["val"].dataset)}')
        print(f'Test samples: {len(self.dataloaders["test"].dataset)}')
        
        # History for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
    
    def train_epoch(self, epoch):
        """Train one epoch"""
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
            
            # Forward pass with GSD
            predictions = self.model(images, gsd)
            
            # Compute loss
            losses = self.criterion(predictions, masks)
            loss = losses['total']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics (use P6)
            filtered_predictions = filter_small_predictions(predictions['p6'], min_pixels=10)
            train_metrics.update(filtered_predictions, masks)
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'Batch [{batch_idx + 1}/{num_batches}] | Loss: {loss.item():.4f}')
        
        epoch_loss /= num_batches
        metrics = train_metrics.compute()
        epoch_time = time.time() - start_time
        
        print(f'\nTraining Results:')
        print(f'Time: {epoch_time:.2f}s | Loss: {epoch_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}% | F1: {metrics["F1"]:.2f}% | '
              f'mIOU: {metrics["mIOU"]:.2f}% | Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')
        
        # Log to tensorboard
        self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)
        
        # Store for plotting
        self.train_losses.append(epoch_loss)
        for key in self.train_metrics_history.keys():
            self.train_metrics_history[key].append(metrics[key])
        
        return epoch_loss, metrics
    
    def validate(self, epoch):
        """Validate"""
        print('\nValidating...')
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = ShadowMetrics()
        
        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                gsd = batch['gsd'].to(self.device)
                
                # Forward with GSD
                predictions = self.model(images, gsd)
                
                loss = self.criterion.criterion(predictions, masks)
                val_loss += loss.item()
                
                filtered_predictions = filter_small_predictions(predictions, min_pixels=10)
                val_metrics.update(filtered_predictions, masks)
        
        val_loss /= len(self.dataloaders['val'])
        metrics = val_metrics.compute()
        
        print(f'Validation Results:')
        print(f'Loss: {val_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}% | F1: {metrics["F1"]:.2f}% | '
              f'mIOU: {metrics["mIOU"]:.2f}% | Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')
        
        # Log
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Val/{key}', val, epoch)

        if self.args.eval_boundary_tolerant:
            self.detailed_evaluator_val = DetailedEvaluator()
            with torch.no_grad():
                for batch in self.dataloaders['val']:
                    images = batch['image'].to(self.device)
                    masks  = batch['mask'].to(self.device)
                    gsd    = batch['gsd'].to(self.device)
                    predictions = self.model(images, gsd)
                    preds = torch.argmax(predictions, dim=1)
                    self.detailed_evaluator_val.update(preds, masks, images)
            detailed_results = self.detailed_evaluator_val.compute_metrics()
            print(f'  Tolerant F1 (5px):   {detailed_results["boundary_tolerant"]["tolerant_5px"]["f1"]:.2f}%')
            print(f'  Tolerant mIOU (5px): {detailed_results["boundary_tolerant"]["tolerant_5px"]["iou"]:.2f}%')
            self.writer.add_scalar('Val/Tolerant_F1',   detailed_results["boundary_tolerant"]["tolerant_5px"]["f1"],  epoch)
            self.writer.add_scalar('Val/Tolerant_mIOU', detailed_results["boundary_tolerant"]["tolerant_5px"]["iou"], epoch)
        
        # Store
        self.val_losses.append(val_loss)
        for key in self.val_metrics_history.keys():
            self.val_metrics_history[key].append(metrics[key])
        
        return val_loss, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'best_shadow_iou': self.best_shadow_iou,
            'best_f1': self.best_f1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'args': vars(self.args)
        }
        
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved')
        
        if epoch % self.args.save_freq == 0:
            epoch_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        self.best_shadow_iou = checkpoint.get('best_shadow_iou', 0.0)
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_metrics_history = checkpoint.get('train_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })
        self.val_metrics_history = checkpoint.get('val_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })
        
        print(f'Resumed from epoch {checkpoint["epoch"]}')
    
    def train(self):
        """Main training loop"""
        print('\n' + '='*50)
        print('Starting training...')
        print('='*50)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            train_loss, train_metrics = self.train_epoch(epoch + 1)
            val_loss, val_metrics = self.validate(epoch + 1)
            
            self.scheduler.step(val_metrics['mIOU'])
            
            is_best = False
            if val_metrics['mIOU'] > self.best_miou:
                self.best_miou = val_metrics['mIOU']
                is_best = True
                print(f'New best mIOU: {self.best_miou:.2f}%')
            
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
            
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']
            
            self.save_checkpoint(epoch + 1, is_best=is_best)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch + 1)
            
            print('='*50)
        
        print('\nGenerating plots...')
        
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
        
        print(f'\nTraining completed!')
        print(f'Best mIOU: {self.best_miou:.2f}%')
        print(f'Best Shadow IoU: {self.best_shadow_iou:.2f}%')
        print(f'Best F1: {self.best_f1:.2f}%')
        
        self.writer.close()
    
    def test(self):
        """Test"""
        print('\n' + '='*50)
        print('Testing model...')
        print('='*50)
        
        best_checkpoint = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_checkpoint):
            self.load_checkpoint(best_checkpoint)
        
        self.model.eval()
        test_metrics = ShadowMetrics()
        
        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                gsd = batch['gsd'].to(self.device)
                
                predictions = self.model(images, gsd)
                
                filtered_predictions = filter_small_predictions(predictions, min_pixels=10)
                test_metrics.update(filtered_predictions, masks)

                if self.args.eval_boundary_tolerant:
                    preds = torch.argmax(filtered_predictions, dim=1)
                    self.detailed_evaluator_test.update(preds, masks, images)
        
        metrics = test_metrics.compute()
        
        print('\nTest Results:')
        for key, value in metrics.items():
            print(f'{key}: {value:.2f}%')

        results_to_save = {'standard': metrics}

        if self.args.eval_boundary_tolerant:
            detailed_results = self.detailed_evaluator_test.compute_metrics()
            print('\n' + '='*50)
            print('Boundary-Tolerant Evaluation:')
            print('='*50)
            print(f"Strict F1:     {detailed_results['boundary_tolerant']['strict']['f1']:.2f}%")
            print(f"Strict mIOU:   {detailed_results['boundary_tolerant']['strict']['iou']:.2f}%")
            print(f"Tolerant F1:   {detailed_results['boundary_tolerant']['tolerant_5px']['f1']:.2f}%")
            print(f"Tolerant mIOU: {detailed_results['boundary_tolerant']['tolerant_5px']['iou']:.2f}%")
            results_to_save['detailed'] = detailed_results
        
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        
        # Generate visualizations
        print('\nGenerating visualizations...')

        # Get the actual GSD from dataset
        dataset_gsd = get_gsd_from_filename(self.dataloaders['test'].dataset.img_files[0])  # All same in single city/res

        class ModelWrapper:
            def __init__(self, model, gsd_value):
                self.model = model
                self.gsd_value = gsd_value
                self.eval = model.eval
                self.train = model.train
            
            def __call__(self, images):
                batch_size = images.size(0)
                gsd = torch.tensor([self.gsd_value] * batch_size).to(images.device)
                return self.model(images, gsd)

        wrapped_model = ModelWrapper(self.model, dataset_gsd)

        save_best_worst_visualizations(
            wrapped_model,
            self.dataloaders['test'],
            self.device,
            self.output_dir,
            num_images=10
        )


def main():
    args = get_args()
    trainer = Trainer(args)
    
    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()