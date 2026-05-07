"""
Training script for MAMNet with GSDPE (Ground Sample Distance Positional Encoding)

Implements cross-resolution transfer learning:
- Train on one resolution (e.g., midres 0.6m)
- Test on another resolution (e.g., highres 0.3m)
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
import torch.nn.functional as F

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet_gsdpe import MAMNet_GSDPE
from data.dataset_gsdpe import get_dataloaders_gsdpe

# Import utility functions from original MAMNet
# NOTE: You'll need to copy these from your original implementation
# For now, I'll create simplified versions

class MAMNetLoss(nn.Module):
    """Simplified loss - use your actual losses.py implementation"""
    def __init__(self, aux_weight=0.4):
        super(MAMNetLoss, self).__init__()
        self.aux_weight = aux_weight
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            main_loss = self.criterion(outputs['main'], targets)
            losses = {'main': main_loss}
            
            if 'aux1' in outputs:
                aux1_loss = self.criterion(outputs['aux1'], targets)
                aux2_loss = self.criterion(outputs['aux2'], targets)
                aux3_loss = self.criterion(outputs['aux3'], targets)
                aux_loss = (aux1_loss + aux2_loss + aux3_loss) / 3
                losses['aux'] = aux_loss
                losses['total'] = main_loss + self.aux_weight * aux_loss
            else:
                losses['total'] = main_loss
            
            return losses
        else:
            return {'total': self.criterion(outputs, targets)}


class ShadowMetrics:
    """Simplified metrics - use your actual metrics.py implementation"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
    
    def update(self, preds, targets):
        if isinstance(preds, dict):
            preds = preds['main']
        
        preds = torch.argmax(preds, dim=1)
        
        self.tp += ((preds == 1) & (targets == 1)).sum().item()
        self.fp += ((preds == 1) & (targets == 0)).sum().item()
        self.tn += ((preds == 0) & (targets == 0)).sum().item()
        self.fn += ((preds == 0) & (targets == 1)).sum().item()
    
    def compute(self):
        epsilon = 1e-7
        
        # Overall Accuracy
        oa = 100 * (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + epsilon)
        
        # Precision
        precision = 100 * self.tp / (self.tp + self.fp + epsilon)
        
        # Recall
        recall = 100 * self.tp / (self.tp + self.fn + epsilon)
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        
        # Balanced Error Rate
        ber = 100 * (1 - 0.5 * (self.tp / (self.tp + self.fn + epsilon) + 
                                  self.tn / (self.tn + self.fp + epsilon)))
        
        # IoU for shadow class
        shadow_iou = 100 * self.tp / (self.tp + self.fp + self.fn + epsilon)
        
        # Mean IoU
        bg_iou = 100 * self.tn / (self.tn + self.fp + self.fn + epsilon)
        miou = (shadow_iou + bg_iou) / 2
        
        return {
            'OA': oa,
            'Precision': precision,
            'F1': f1,
            'BER': ber,
            'mIOU': miou,
            'Shadow_IOU': shadow_iou
        }


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MAMNet with GSDPE for Cross-Resolution Transfer')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default=None,
                      help='Root directory of dataset (for single mode)')
    parser.add_argument('--base_data_root', type=str, default=None,
                      help='Base directory for multi-city mode')
    parser.add_argument('--mode', type=str, default='single',
                      choices=['single', 'all'],
                      help='Training mode')
    parser.add_argument('--cities', type=str, nargs='+', default=['chicago', 'miami', 'phoenix'],
                      help='List of cities')
    parser.add_argument('--train_resolution', type=str, default='midres',
                      choices=['highres', 'midres'],
                      help='Resolution for training')
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
                      help='Use pretrained ResNet-34 encoder')
    parser.add_argument('--aux_weight', type=float, default=0.4,
                      help='Weight for auxiliary loss')
    parser.add_argument('--reference_gsd', type=float, default=1.0,
                      help='Reference GSD for GSDPE (default: 1.0m)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=15,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    
    # Checkpoint and logging
    parser.add_argument('--output_dir', type=str, default='./outputs',
                      help='Directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=1,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                      help='Only evaluate the model')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    # Contrast channel
    parser.add_argument('--use_contrast', action='store_true',
                    help='Use contrast as 4th input channel')

    # Boundary tolerant evaluation
    parser.add_argument('--eval_boundary_tolerant', action='store_true',
                    help='Compute boundary-tolerant metrics')
    
    return parser.parse_args()


class Trainer:
    """Trainer class for MAMNet with GSDPE"""
    
    def __init__(self, args):
        self.args = args
        
        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        city_name = os.path.basename(os.path.dirname(args.data_root.rstrip('/')))
        exp_name = f'mamnet_gsdpe_{city_name}_{args.train_resolution}_{1}'
        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Initialize model with GSDPE
        print('Initializing MAMNet with GSDPE...')
        self.model = MAMNet_GSDPE(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            use_aux=True,
            reference_gsd=args.reference_gsd,
            use_contrast=args.use_contrast
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        
        # Setup loss function
        self.criterion = MAMNetLoss(aux_weight=args.aux_weight)
        
        # Setup optimizer
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
            patience=3,
            verbose=True
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
        self.dataloaders = get_dataloaders_gsdpe(
            data_root=args.data_root,
            base_data_root=args.base_data_root,
            mode=args.mode,
            cities=args.cities,
            resolution=args.train_resolution,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            use_contrast=args.use_contrast
        )
        
        print(f'Training samples: {len(self.dataloaders["train"].dataset)}')
        print(f'Validation samples: {len(self.dataloaders["val"].dataset)}')
        print(f'Test samples: {len(self.dataloaders["test"].dataset)}')

        # Initialize detailed evaluator if enabled
        if args.eval_boundary_tolerant:
            self.detailed_evaluator_test = DetailedEvaluator()
            self.detailed_evaluator_val = DetailedEvaluator()
            print("Boundary-tolerant evaluation enabled")
        
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
        epoch_main_loss = 0.0
        epoch_aux_loss = 0.0
        
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
            losses = self.criterion(outputs, masks)
            loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            train_metrics.update(outputs, masks)
            
            # Track losses
            epoch_loss += loss.item()
            epoch_main_loss += losses['main'].item()
            if 'aux' in losses:
                epoch_aux_loss += losses['aux'].item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'Batch [{batch_idx + 1}/{num_batches}] | '
                      f'Loss: {loss.item():.4f} | '
                      f'Main: {losses["main"].item():.4f} | '
                      f'Aux: {losses.get("aux", torch.tensor(0.0)).item():.4f}')
        
        # Compute average losses
        epoch_loss /= num_batches
        epoch_main_loss /= num_batches
        epoch_aux_loss /= num_batches
        
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
        for key, value in metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
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
                loss = self.criterion.criterion(outputs, masks)
                val_loss += loss.item()
                
                # Update metrics
                val_metrics.update(outputs, masks)
        
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
        for key, value in metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)

        # Boundary-tolerant evaluation if enabled
        if self.args.eval_boundary_tolerant:
            # Reset evaluator for this epoch
            self.detailed_evaluator_val = DetailedEvaluator()
            
            # Second pass through validation for detailed metrics
            with torch.no_grad():
                for batch in self.dataloaders['val']:
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device)
                    gsd = batch['gsd'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images, gsd)
                    
                    # Get predictions
                    if isinstance(outputs, dict):
                        preds = torch.argmax(outputs['main'], dim=1)
                    else:
                        preds = torch.argmax(outputs, dim=1)
                    
                    # Update detailed evaluator
                    self.detailed_evaluator_val.update(preds, masks, images)
            
            # Compute detailed metrics
            detailed_results = self.detailed_evaluator_val.compute_metrics()
            
            print(f'  Tolerant F1 (5px): {detailed_results["boundary_tolerant"]["tolerant_5px"]["f1"]:.2f}%')
            print(f'  Tolerant mIOU (5px): {detailed_results["boundary_tolerant"]["tolerant_5px"]["iou"]:.2f}%')
            
            # Log to tensorboard
            self.writer.add_scalar('Val/Tolerant_F1', detailed_results["boundary_tolerant"]["tolerant_5px"]["f1"], epoch)
            self.writer.add_scalar('Val/Tolerant_mIOU', detailed_results["boundary_tolerant"]["tolerant_5px"]["iou"], epoch)
        
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'best_shadow_iou': self.best_shadow_iou,
            'best_f1': self.best_f1,
            'args': vars(self.args)
        }
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved to {best_path}')
        
        # Save every 10 epochs
        if epoch % 10 == 0:
            epoch_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
            print(f'Epoch checkpoint saved to {epoch_path}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        self.best_shadow_iou = checkpoint.get('best_shadow_iou', 0.0)
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        
        print(f'Resumed from epoch {checkpoint["epoch"]}')
    
    def train(self):
        """Main training loop"""
        print('\n' + '='*50)
        print('Starting training...')
        print('='*50)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch + 1)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch + 1)
            
            # Update scheduler
            self.scheduler.step(val_metrics['mIOU'])
            
            # Check if best model
            is_best = False
            if val_metrics['mIOU'] > self.best_miou:
                self.best_miou = val_metrics['mIOU']
                is_best = True
                print(f'New best mIOU: {self.best_miou:.2f}%')
            
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
            
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best=is_best)
            
            print('='*50)
        
        print('\nTraining completed!')
        print(f'Best mIOU: {self.best_miou:.2f}%')
        print(f'Best Shadow IoU: {self.best_shadow_iou:.2f}%')
        print(f'Best F1: {self.best_f1:.2f}%')

        print(f'Best F1: {self.best_f1:.2f}%')

        from utils.visualization import plot_loss_curves, plot_metrics_curves

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

        print(f'Loss and metrics curves saved to {self.output_dir}')

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
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                gsd = batch['gsd'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, gsd)
                
                # Update standard metrics
                test_metrics.update(outputs, masks)
                
                # Update detailed metrics if enabled
                if self.args.eval_boundary_tolerant:
                    if isinstance(outputs, dict):
                        preds = torch.argmax(outputs['main'], dim=1)
                    else:
                        preds = torch.argmax(outputs, dim=1)
                    self.detailed_evaluator_test.update(preds, masks, images)
        
        # Compute standard metrics
        metrics = test_metrics.compute()
        
        print('\nStandard Test Results:')
        for key, value in metrics.items():
            print(f'{key}: {value:.2f}%')
        
        # Save results
        results_path = os.path.join(self.output_dir, 'test_results.json')
        results_to_save = {'standard': metrics}
        
        # Compute detailed metrics if enabled
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
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
    
        # Generate visualizations manually (GSDPE requires GSD input)
        print('\nGenerating best/worst visualizations...')
        from utils.visualization import visualize_predictions, compute_per_image_iou
        from utils.postprocessing import filter_small_predictions
        
        all_images = []
        all_masks_gt = []
        all_masks_pred = []
        all_probs = []
        all_ious = []
        all_filenames = []
        
        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                gsd = batch['gsd'].to(self.device)
                filenames = batch['filename']
                
                # Forward pass with GSD
                outputs = self.model(images, gsd)
                
                # Handle dict or tensor output
                if isinstance(outputs, dict):
                    logits = outputs['main']
                else:
                    logits = outputs
                
                # Get probability map
                probs_batch = F.softmax(logits, dim=1)[:, 1, :, :]
                
                # Apply filtering
                outputs_filtered = filter_small_predictions(logits, min_pixels=10)
                preds_batch = torch.argmax(outputs_filtered, dim=1)
                
                # Process each image
                for i in range(images.size(0)):
                    img = images[i]
                    mask_gt = masks[i]
                    mask_pred = preds_batch[i]
                    prob = probs_batch[i]
                    
                    # Compute IoU
                    iou = compute_per_image_iou(mask_pred, mask_gt)
                    
                    all_images.append(img.cpu())
                    all_masks_gt.append(mask_gt.cpu())
                    all_masks_pred.append(mask_pred.cpu())
                    all_probs.append(prob.cpu())
                    all_ious.append(iou)
                    all_filenames.append(filenames[i])
        
        # Filter images with shadows
        indices_with_shadows = [i for i in range(len(all_masks_gt)) 
                                if torch.sum(all_masks_gt[i]) > 0]
        
        if len(indices_with_shadows) > 0:
            ious_with_shadows = [all_ious[i] for i in indices_with_shadows]
            sorted_positions = np.argsort(ious_with_shadows)
            sorted_indices = [indices_with_shadows[i] for i in sorted_positions]
            
            # Best and worst
            num_vis = min(10, len(sorted_indices))
            worst_indices = sorted_indices[:num_vis]
            best_indices = sorted_indices[-num_vis:][::-1]
            
            # Visualize best
            visualize_predictions(
                [all_images[i] for i in best_indices],
                [all_masks_gt[i] for i in best_indices],
                [all_masks_pred[i] for i in best_indices],
                [all_probs[i] for i in best_indices],
                [all_ious[i] for i in best_indices],
                os.path.join(self.output_dir, 'best_predictions.png'),
                f'Top {num_vis} Best Predictions (Highest Shadow IoU)',
                num_images=num_vis
            )
            
            # Visualize worst
            visualize_predictions(
                [all_images[i] for i in worst_indices],
                [all_masks_gt[i] for i in worst_indices],
                [all_masks_pred[i] for i in worst_indices],
                [all_probs[i] for i in worst_indices],
                [all_ious[i] for i in worst_indices],
                os.path.join(self.output_dir, 'worst_predictions.png'),
                f'Top {num_vis} Worst Predictions (Lowest Shadow IoU)',
                num_images=num_vis
            )
            
            # Save statistics
            stats = {
                'mean_iou': np.mean(all_ious),
                'std_iou': np.std(all_ious),
                'min_iou': np.min(all_ious),
                'max_iou': np.max(all_ious),
                'median_iou': np.median(all_ious),
                'best_files': [all_filenames[i] for i in best_indices],
                'best_ious': [all_ious[i] for i in best_indices],
                'worst_files': [all_filenames[i] for i in worst_indices],
                'worst_ious': [all_ious[i] for i in worst_indices]
            }
            
            with open(os.path.join(self.output_dir, 'iou_statistics.json'), 'w') as f:
                json.dump(stats, f, indent=4)
            
            print(f'Visualizations saved!')
        
        return metrics


def main():
    args = get_args()
    
    # Create trainer
    trainer = Trainer(args)
    
    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()