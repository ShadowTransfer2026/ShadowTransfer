"""
Fine-tuning script for cross-city transfer with spatial sampling strategies
Fine-tunes LOCO checkpoints on target city data using different sampling strategies
"""

import os
import argparse
import time
import json
import glob
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.mamnet import MAMNet
from utils.losses import MAMNetLoss
from utils.metrics import ShadowMetrics
# Import fine-tuning specific modules
sys.path.append('/mnt/user-data/outputs')  # For spatial_sampling
from utils.spatial_sampling import select_patches_by_strategy
from utils.postprocessing import filter_small_predictions
from data.dataset_finetuning import get_finetune_dataloaders
from typing import List, Dict, Tuple, Optional


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune MAMNet with Spatial Sampling')
    
    # Fine-tuning specific parameters
    parser.add_argument('--target_city', type=str, required=True,
                       choices=['chicago', 'miami', 'phoenix'],
                       help='Target city for fine-tuning')
    parser.add_argument('--resolution', type=str, required=True,
                       choices=['highres', 'midres'],
                       help='Resolution')
    parser.add_argument('--n_samples', type=int, required=True,
                       help='Number of samples for fine-tuning (0 for no fine-tuning)')
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['random', 'clustered', 'dispersed'],
                       help='Spatial sampling strategy')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Checkpoint parameters
    parser.add_argument('--loco_checkpoint_dir', type=str, 
                       default='/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/outputs',
                       help='Directory containing LOCO checkpoints')
    parser.add_argument('--base_data_root', type=str, required=True,
                       help='Base directory for data (e.g., /path/to/Final_data_test/)')
    parser.add_argument('--metadata_dir', type=str,
                       default='/scratch/gilbreth/mittal53/ShadeMaps/data/Final_data_test/metadata/',
                       help='Directory containing metadata files')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001, 10x smaller than training)')
    parser.add_argument('--max_epochs', type=int, default=10,
                       help='Maximum number of epochs (default: 10)')
    parser.add_argument('--early_stop_patience', type=int, default=3,
                       help='Early stopping patience (default: 3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--aux_weight', type=float, default=0.4,
                       help='Weight for auxiliary loss (default: 0.4)')
    
    # Other parameters
    parser.add_argument('--img_size', type=int, default=384,
                       help='Input image size (default: 384)')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, 
                       default='/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/outputs',
                       help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--split_file', type=str, default=None,
                       help='Path to pre-generated split JSON file (optional)')
    
    return parser.parse_args()


def find_latest_loco_checkpoint(checkpoint_dir: str, target_city: str, 
                                resolution: str) -> str:
    """
    Find the latest LOCO checkpoint for the target city
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        target_city: Target city (holdout city in LOCO)
        resolution: 'highres' or 'midres'
    
    Returns:
        Path to checkpoint file
    """
    # Pattern: mamnet_loco_holdout_{city}_{resolution}_*
    pattern = os.path.join(checkpoint_dir, 
                          f'mamnet_loco_holdout_{target_city}_{resolution}_*')
    
    folders = glob.glob(pattern)
    
    if not folders:
        raise FileNotFoundError(
            f"No LOCO checkpoint found matching pattern: {pattern}\n"
            f"Make sure you have trained a LOCO model with {target_city} as holdout."
        )
    
    # Sort by timestamp in folder name
    folders.sort(key=lambda x: x.split('_')[-2] + '_' + x.split('_')[-1])
    latest_folder = folders[-1]
    
    checkpoint_path = os.path.join(latest_folder, 'checkpoint_best.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )
    
    print(f"Found LOCO checkpoint: {checkpoint_path}")
    return checkpoint_path


class FinetuneTrainer:
    """Trainer class for fine-tuning MAMNet"""
    
    def __init__(self, args):
        self.args = args
        
        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = (f'finetune_{args.target_city}_{args.resolution}_'
                   f'{args.strategy}_N{args.n_samples:03d}_seed{args.random_seed}_{timestamp}')
        
        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f'\nOutput directory: {self.output_dir}')
        
        # Save arguments
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Initialize model
        print('\nInitializing model...')
        self.model = MAMNet(
            num_classes=2,
            pretrained=False,
            use_aux=True
        ).to(self.device)
        
        # Load LOCO checkpoint
        if args.n_samples > 0:
            loco_checkpoint_path = find_latest_loco_checkpoint(
                args.loco_checkpoint_dir, args.target_city, args.resolution
            )
            self.load_checkpoint(loco_checkpoint_path)
        else:
            # N=0: Just evaluate LOCO model, no fine-tuning
            loco_checkpoint_path = find_latest_loco_checkpoint(
                args.loco_checkpoint_dir, args.target_city, args.resolution
            )
            self.load_checkpoint(loco_checkpoint_path)
            print("\nN=0: No fine-tuning, will only evaluate LOCO checkpoint")
        
        # Setup loss and optimizer (only if fine-tuning)
        if args.n_samples > 0:
            self.criterion = MAMNetLoss(aux_weight=args.aux_weight)
            
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=2,
                verbose=True
            )
        
        # Initialize tracking variables
        self.best_f1 = 0.0
        self.epochs_no_improve = 0
        
        # Setup data
        self.setup_data()

    def check_existing_results(self) -> Optional[str]:
        """
        Check if equivalent results already exist to avoid redundant training
        - For N=600: Check if random/seed1 results exist
        - For Miami dispersed N>=350: Check if random (same seed) results exist
        
        Returns:
            Path to existing results directory if found, None otherwise
        """
        args = self.args
        
        # N=600: Use random/seed1 as canonical
        if args.n_samples == 600:
            # Skip if this IS the canonical job
            if args.strategy == 'random' and args.random_seed == 1:
                return None
            
            canonical_pattern = f'finetune_{args.target_city}_{args.resolution}_random_N{args.n_samples:03d}_seed1_*'
        
        else:
            return None
        
        # Search for canonical results
        import glob
        matching_dirs = glob.glob(os.path.join(args.output_dir, canonical_pattern))
        
        if matching_dirs:
            matching_dirs.sort()
            canonical_dir = matching_dirs[-1]
            
            results_file = os.path.join(canonical_dir, 'results.json')
            if os.path.exists(results_file):
                print(f"\n{'='*60}")
                print(f"Found existing equivalent results:")
                print(f"  {canonical_dir}")
                print(f"Will copy instead of training")
                print(f"{'='*60}\n")
                return canonical_dir
        
        return None

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        print(f'Loading checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('Checkpoint loaded successfully')
        
    def setup_data(self):
        """Setup dataloaders with spatial sampling"""
        args = self.args
        
        # Test set (always use all test images)
        test_data_root = os.path.join(args.base_data_root, args.target_city, args.resolution)
        
        from data.dataset import get_dataloaders
        test_loaders = get_dataloaders(
            data_root=test_data_root,
            mode='single',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size
        )
        self.test_loader = test_loaders['test']
        
        if args.n_samples == 0:
            # N=0: No training/validation data needed
            self.train_loader = None
            self.val_loader = None
            self.spatial_metrics = {
                'strategy': args.strategy,
                'n_samples': 0,
                'note': 'No fine-tuning (N=0)'
            }
            print(f"\nTest samples: {len(self.test_loader.dataset)}")
            return
        
        # Check if using pre-generated split
        if args.split_file is not None:
            # Load from pre-generated split file
            print(f"\nLoading pre-generated split from: {args.split_file}")
            
            with open(args.split_file, 'r') as f:
                selection_result = json.load(f)
            
            # Verify split matches arguments
            if (selection_result['city'] != args.target_city or
                selection_result['resolution'] != args.resolution or
                selection_result['n_samples'] != args.n_samples or
                selection_result['random_seed'] != args.random_seed):
                raise ValueError(f"Split file does not match arguments!\n"
                               f"  File: {selection_result['city']} {selection_result['resolution']} "
                               f"N={selection_result['n_samples']} seed={selection_result['random_seed']}\n"
                               f"  Args: {args.target_city} {args.resolution} "
                               f"N={args.n_samples} seed={args.random_seed}")
            
            # Verify split matches arguments
            if args.n_samples == 600:
                # N=600 always uses original split
                if selection_result['strategy'] != 'original':
                    raise ValueError(f"N=600 must use original split, but got '{selection_result['strategy']}'")
                print(f"  Note: Using original split for all strategies (N=600)")
            elif selection_result['strategy'] != args.strategy:
                raise ValueError(f"Split strategy mismatch: file has '{selection_result['strategy']}', "
                               f"args specify '{args.strategy}'")
            
            self.spatial_metrics = selection_result['spatial_metrics']
            
            print(f"  Loaded {len(selection_result['train_filenames'])} train files")
            print(f"  Loaded {len(selection_result['val_filenames'])} val files")
            
        else:
            # Generate split on-the-fly (original behavior)
            print(f"\nSelecting {args.n_samples} patches using '{args.strategy}' strategy...")
            
            # Create visualization path
            viz_path = os.path.join(self.output_dir, 'sampling_visualization.png')
            
            selection_result = select_patches_by_strategy(
                city=args.target_city,
                resolution=args.resolution,
                n_samples=args.n_samples,
                strategy=args.strategy,
                random_seed=args.random_seed,
                metadata_dir=args.metadata_dir,
                base_data_root=args.base_data_root,
                split_ratio=(0.75, 0.25),
                save_visualization=viz_path
            )
            
            self.spatial_metrics = selection_result['spatial_metrics']
        
        # Save spatial metrics
        with open(os.path.join(self.output_dir, 'spatial_metrics.json'), 'w') as f:
            json.dump(self.spatial_metrics, f, indent=4)
        
        print(f"\nSpatial Metrics:")
        for key, value in self.spatial_metrics.items():
            print(f"  {key}: {value}")
        
        # Create dataloaders with selected filenames
        loaders = get_finetune_dataloaders(
            data_root=test_data_root,
            train_filenames=selection_result['train_filenames'],
            val_filenames=selection_result['val_filenames'],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size
        )
        
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        train_metrics = ShadowMetrics()
        
        print(f'\nEpoch {epoch}/{self.args.max_epochs}')
        print('-' * 50)
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            losses = self.criterion(outputs, masks)
            loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            filtered_outputs = filter_small_predictions(outputs['main'], min_pixels=10)
            train_metrics.update(filtered_outputs, masks)
            
            epoch_loss += loss.item()
        
        # Compute metrics
        epoch_loss /= len(self.train_loader)
        metrics = train_metrics.compute()
        
        print(f'Train Loss: {epoch_loss:.4f} | F1: {metrics["F1"]:.2f}%')
        
        # Log to tensorboard
        self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
        self.writer.add_scalar('Train/F1', metrics['F1'], epoch)
        
        return epoch_loss, metrics
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = ShadowMetrics()
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion.criterion(outputs, masks)
                val_loss += loss.item()
                
                filtered_outputs = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered_outputs, masks)
        
        val_loss /= len(self.val_loader)
        metrics = val_metrics.compute()
        
        print(f'Val Loss: {val_loss:.4f} | F1: {metrics["F1"]:.2f}%')
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        self.writer.add_scalar('Val/F1', metrics['F1'], epoch)
        
        return val_loss, metrics
    
    def test(self):
        """Test the model on target city test set"""
        print('\n' + '='*60)
        print('Testing on target city test set...')
        print('='*60)
        
        self.model.eval()
        test_metrics = ShadowMetrics()
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                filtered_outputs = filter_small_predictions(outputs, min_pixels=10)
                test_metrics.update(filtered_outputs, masks)
        
        metrics = test_metrics.compute()

        print('\nTest Results:')
        print(f'Available metrics: {list(metrics.keys())}')  # DEBUG
        
        for metric_name in ['F1', 'Shadow_IOU', 'mIOU', 'OA', 'Precision', 'Recall', 'BER']:
            if metric_name in metrics:
                print(f'{metric_name}: {metrics[metric_name]:.2f}%')
            else:
                print(f'{metric_name}: Not available')
        
        print('\nTest Results:')
        # Print only metrics that exist
        for metric_name in ['F1', 'Shadow_IOU', 'mIOU', 'OA', 'Precision', 'Recall', 'BER']:
            if metric_name in metrics:
                print(f'{metric_name}: {metrics[metric_name]:.2f}%')
        
        # Save results
        results = {
            'test_metrics': metrics,
            'spatial_metrics': self.spatial_metrics,
            'args': vars(self.args)
        }
        
        results_path = os.path.join(self.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f'\nResults saved to {results_path}')
        
        return metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_f1': self.best_f1,
            'args': vars(self.args)
        }
        
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved: {best_path}')
    
    def train(self):

        # Check if equivalent results exist
        existing_results_dir = self.check_existing_results()
        if existing_results_dir:
            # Copy results instead of training
            import shutil
            
            # Copy results.json
            src_results = os.path.join(existing_results_dir, 'results.json')
            dst_results = os.path.join(self.output_dir, 'results.json')
            shutil.copy2(src_results, dst_results)
            
            # Copy checkpoint if exists
            src_checkpoint = os.path.join(existing_results_dir, 'checkpoint_best.pth')
            if os.path.exists(src_checkpoint):
                dst_checkpoint = os.path.join(self.output_dir, 'checkpoint_best.pth')
                shutil.copy2(src_checkpoint, dst_checkpoint)
            
            # Copy spatial_metrics.json if exists
            src_spatial = os.path.join(existing_results_dir, 'spatial_metrics.json')
            if os.path.exists(src_spatial):
                dst_spatial = os.path.join(self.output_dir, 'spatial_metrics.json')
                shutil.copy2(src_spatial, dst_spatial)
            
            # Create updated args.json with correct strategy/seed
            updated_args = vars(self.args).copy()
            updated_args['note'] = f'Results copied from {os.path.basename(existing_results_dir)}'
            with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
                json.dump(updated_args, f, indent=4)
            
            print(f"\nResults copied successfully to {self.output_dir}")
            
            # Load and return the test metrics
            with open(dst_results, 'r') as f:
                results = json.load(f)
            return results['test_metrics']
        
        """Main fine-tuning loop"""
        if self.args.n_samples == 0:
            # No fine-tuning, just test
            print("\nSkipping fine-tuning (N=0)")
            return self.test()
        
        print('\n' + '='*60)
        print('Starting fine-tuning...')
        print('='*60)
        
        for epoch in range(1, self.args.max_epochs + 1):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step(val_metrics['F1'])
            
            # Check for improvement
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f'New best F1: {self.best_f1:.2f}%')
            else:
                self.epochs_no_improve += 1
                print(f'No improvement for {self.epochs_no_improve} epochs')
            
            # Early stopping
            if self.epochs_no_improve >= self.args.early_stop_patience:
                print(f'\nEarly stopping triggered after {epoch} epochs')
                break
        
        print(f'\nFine-tuning completed!')
        print(f'Best validation F1: {self.best_f1:.2f}%')
        
        # Load best checkpoint and test
        best_checkpoint = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_checkpoint):
            self.load_checkpoint(best_checkpoint)
        
        # Test on target city
        return self.test()


def main():
    args = get_args()
    
    print('\n' + '='*60)
    print('Fine-tuning Configuration:')
    print('='*60)
    print(f'Target City: {args.target_city}')
    print(f'Resolution: {args.resolution}')
    print(f'N Samples: {args.n_samples}')
    print(f'Strategy: {args.strategy}')
    print(f'Random Seed: {args.random_seed}')
    print(f'Learning Rate: {args.lr}')
    print(f'Max Epochs: {args.max_epochs}')
    print('='*60)
    
    # Create trainer and run
    trainer = FinetuneTrainer(args)
    trainer.train()
    
    # Close tensorboard
    trainer.writer.close()
    
    print('\n' + '='*60)
    print('Fine-tuning complete!')
    print('='*60)


if __name__ == '__main__':
    main()