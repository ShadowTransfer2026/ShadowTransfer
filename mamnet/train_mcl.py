"""
Training script for MAMNet with Multi-level Contrastive Learning (mCL-LC)
WACV 2023 - Tang et al.

Implements contrastive learning at feature and semantic levels with local consistency.

Decision metrics (LR scheduler, best checkpoint, early stopping) use
**per-image** mIOU from DetailedEvaluator — never pooled ShadowMetrics.

When --eval_boundary_tolerant is set, decisions use per-image TOLERANT mIOU
(boundary band width controlled by --boundary_tolerance, default 2px).
Otherwise decisions use per-image STRICT mIOU from DetailedEvaluator.

ShadowMetrics (pooled) is still computed and logged to TensorBoard for
reference, but is NOT used for any decisions.
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
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet_mcl import MAMNetMCL
from data.dataset import get_dataloaders
from data.dataset_enhanced import ShadowDatasetEnhanced
from utils.evaluation_detailed import DetailedEvaluator
from utils.contrastive_losses import mCLLCLoss
from utils.metrics import ShadowMetrics
from utils.losses import MAMNetLoss
from utils.postprocessing import filter_small_predictions
from utils.visualization import plot_metrics_curves
from utils.visualization_mcl import (
    plot_loss_curves_mcl,
    save_best_worst_visualizations_mcl,
)

print("="*50)
print("GPU DIAGNOSTICS")
print("="*50)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print("="*50)


def compute_augmentation_diversity(images, images_aug1, images_aug2):
    """Compute metrics to quantify augmentation diversity"""
    mse_1 = F.mse_loss(images, images_aug1).item()
    mse_2 = F.mse_loss(images, images_aug2).item()
    mse_aug = F.mse_loss(images_aug1, images_aug2).item()

    images_flat = images.view(images.size(0), -1)
    aug1_flat = images_aug1.view(images_aug1.size(0), -1)
    aug2_flat = images_aug2.view(images_aug2.size(0), -1)

    cos_sim_1 = F.cosine_similarity(images_flat, aug1_flat, dim=1).mean().item()
    cos_sim_2 = F.cosine_similarity(images_flat, aug2_flat, dim=1).mean().item()
    cos_sim_aug = F.cosine_similarity(aug1_flat, aug2_flat, dim=1).mean().item()

    return {
        'mse_orig_aug1': mse_1,
        'mse_orig_aug2': mse_2,
        'mse_aug1_aug2': mse_aug,
        'cosine_orig_aug1': cos_sim_1,
        'cosine_orig_aug2': cos_sim_2,
        'cosine_aug1_aug2': cos_sim_aug
    }


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train MAMNet with mCL-LC for Shadow Detection')

    # Data parameters
    parser.add_argument('--data_root', type=str, required=False, default=None,
                        help='Root directory of dataset (required for single mode)')
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
    parser.add_argument('--lambda_fl', type=float, default=0,
                        help='Weight for feature-level contrastive loss')
    parser.add_argument('--lambda_sl', type=float, default=0,
                        help='Weight for semantic-level contrastive loss')
    parser.add_argument('--lambda_lc', type=float, default=0,
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
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained encoder')
    parser.add_argument('--aux_weight', type=float, default=0.4,
                        help='Weight for auxiliary loss')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience (0 = disabled)')

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
                        help='Device to use')
    parser.add_argument('--use_mcl', action='store_true',
                        help='Use MCL dataset with augmented views')

    # Contrast channel
    parser.add_argument('--use_contrast', action='store_true',
                        help='Use contrast as 4th input channel')

    # Boundary-tolerant evaluation
    parser.add_argument('--eval_boundary_tolerant', action='store_true',
                        help='Use tolerant mIOU (instead of strict) for all decisions')
    # CHANGE: --boundary_tolerance added so K can be set from the bash scripts.
    # DetailedEvaluator always runs; this sets the band half-width in pixels.
    parser.add_argument('--boundary_tolerance', type=int, default=2,
                        help='Don\'t-care band half-width in pixels (default: 2). '
                             'Controls DetailedEvaluator for both strict and tolerant '
                             'per-image metrics.')

    # Comparison / inference directories
    parser.add_argument('--comparison_inference_dir', type=str, default=None,
                        help='Directory with comparison inference results')
    parser.add_argument('--comparison_data_root', type=str, default=None,
                        help='Data root for comparison evaluation')

    return parser.parse_args()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class MCLTrainer:
    """Trainer class for MAMNet with mCL-LC"""

    def __init__(self, args):
        self.args = args

        # Setup device
        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # CHANGE: dynamic tolerant key — replaces every hardcoded 'tolerant_5px'
        # (the old hardcoded value caused a KeyError since DetailedEvaluator
        # defaults to boundary_tolerance=2, producing key 'tolerant_2px').
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # Create output directory
        if args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = (f'mamnet_mcl_loco_holdout_{test_city}'
                        f'_{args.resolution}_{1}')
        else:
            exp_name = f'mamnet_mcl_{args.mode}_{1}'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(
            os.path.join(self.output_dir, 'tensorboard'))

        # Initialize model
        print('Initializing MAMNet with mCL-LC...')
        self.model = MAMNetMCL(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            use_aux=True,
            feature_proj_dim=args.feature_proj_dim,
            semantic_proj_dim=args.semantic_proj_dim,
            use_contrast=args.use_contrast
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters:     {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')

        # Setup losses
        self.criterion = MAMNetLoss(aux_weight=args.aux_weight)
        self.contrastive_criterion = None
        if args.lambda_fl > 0 or args.lambda_sl > 0 or args.lambda_lc > 0:
            base_criterion = nn.CrossEntropyLoss(ignore_index=255)
            self.contrastive_criterion = mCLLCLoss(
                seg_criterion=base_criterion,
                lambda_fl=args.lambda_fl,
                lambda_sl=args.lambda_sl,
                lambda_lc=args.lambda_lc,
                aux_weight=0.0,
                temperature=args.temperature,
                use_bane=args.use_bane
            )

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Setup scheduler (monitors decision metric)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True)

        # ---- Decision-metric tracking ----
        # DetailedEvaluator ALWAYS runs (not guarded by eval_boundary_tolerant).
        # eval_boundary_tolerant controls WHICH per-image metric drives decisions:
        #   True  → tolerant mIOU (±boundary_tolerance px band excluded)
        #   False → strict  mIOU  (all pixels, per-image mean)
        # ShadowMetrics (pooled) is logged for reference only.
        self.use_tolerant_decision = args.eval_boundary_tolerant
        if self.use_tolerant_decision:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px boundary excluded)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU '
                  f'(DetailedEvaluator, not pooled ShadowMetrics)')

        # CHANGE: DetailedEvaluators always instantiated, with correct K.
        # The original only created them when eval_boundary_tolerant was set,
        # causing AttributeError when the flag was off.
        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # Initialize tracking variables
        # CHANGE: renamed best_miou → best_decision_miou (clearer intent);
        # added best_miou_pooled for reference logging only.
        self.start_epoch = 0
        self.best_decision_miou = 0.0       # drives checkpoint/early-stop/scheduler
        self.best_miou_pooled = 0.0         # reference only (pooled ShadowMetrics)
        self.best_shadow_iou = 0.0          # reference
        self.best_f1 = 0.0                  # reference
        self.epochs_without_improvement = 0  # for early stopping

        # CHANGE: per-component loss histories added so plot_loss_curves_mcl
        # can render individual subplots at their own y-axis scale.
        # Previously only train_losses (total) was stored, so component panels
        # had no data.
        self.train_losses = []       # total
        self.train_seg_losses = []   # main CE
        self.train_aux_losses = []   # weighted auxiliary
        self.train_fl_losses = []    # feature-level contrastive
        self.train_sl_losses = []    # semantic-level contrastive
        self.train_lc_losses = []    # local consistency
        self.val_losses = []         # val main CE (eval mode, no aux)

        # Metric histories (pooled ShadowMetrics, reference plots only)
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [],
            'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [],
            'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }

        if args.resume:
            self.load_checkpoint(args.resume)

        # Load datasets
        if args.use_contrast:
            from torch.utils.data import DataLoader

            if args.mode == 'loco':
                from data.dataset import LOCO_FOLDS
                fold_config = LOCO_FOLDS[args.fold_id]
                train_cities = fold_config['train']
                test_city = fold_config['test']
                train_paths = [
                    os.path.join(args.base_data_root, c, args.resolution)
                    for c in train_cities]
                val_paths = train_paths
                test_paths = [
                    os.path.join(args.base_data_root, test_city, args.resolution)]
            elif args.mode == 'single':
                train_paths = val_paths = test_paths = [args.data_root]
            else:
                cities = args.cities or ['chicago', 'miami', 'phoenix']
                train_paths = [
                    os.path.join(args.base_data_root, c, args.resolution)
                    for c in cities]
                val_paths = test_paths = train_paths

            train_dataset = ShadowDatasetEnhanced(
                root_dir=train_paths, split='train', img_size=args.img_size,
                task_id=2, augment=True, use_mcl=args.use_mcl)
            val_dataset = ShadowDatasetEnhanced(
                root_dir=val_paths, split='val', img_size=args.img_size,
                task_id=2, augment=False)
            test_dataset = ShadowDatasetEnhanced(
                root_dir=test_paths, split='test', img_size=args.img_size,
                task_id=2, augment=False)

            self.dataloaders = {
                'train': DataLoader(
                    train_dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True),
                'val': DataLoader(
                    val_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                    pin_memory=True),
                'test': DataLoader(
                    test_dataset, batch_size=1, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True),
            }
        else:
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
                use_pseudo_cloud=args.use_pseudo_cloud,
            )

        print(f'Training samples:   {len(self.dataloaders["train"].dataset)}')
        print(f'Validation samples: {len(self.dataloaders["val"].dataset)}')
        print(f'Test samples:       {len(self.dataloaders["test"].dataset)}')

    # ------------------------------------------------------------------
    # Decision metric
    # ------------------------------------------------------------------

    def _get_decision_miou(self, detailed_results):
        """
        Return the mIOU driving all decisions (LR scheduler, best checkpoint,
        early stopping).

        Both options are per-image means from DetailedEvaluator — never the
        pooled ShadowMetrics value.

        Args:
            detailed_results: dict from DetailedEvaluator.compute_metrics()

        Returns:
            float mIOU (%)
        """
        bt = detailed_results['boundary_tolerant']
        if self.use_tolerant_decision:
            return bt[self.tol_key]['iou']
        else:
            return bt['strict']['iou']

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        """
        Train for one epoch with contrastive learning.

        Returns
        -------
        epoch_loss     : float  total loss
        epoch_seg_loss : float  main CE loss
        epoch_aux_loss : float  weighted auxiliary loss
        epoch_fl_loss  : float  feature-level contrastive loss
        epoch_sl_loss  : float  semantic-level contrastive loss
        epoch_lc_loss  : float  local-consistency loss
        metrics        : dict   pooled ShadowMetrics (reference only)
        """
        self.model.train()

        # Mask diagnostic on first epoch
        if epoch == 1:
            print("\n=== MASK DIAGNOSTIC ===")
            sample_batch = next(iter(self.dataloaders['train']))
            sample_masks = sample_batch['mask']
            print(f"Mask dtype: {sample_masks.dtype}")
            print(f"Mask shape: {sample_masks.shape}")
            print(f"Mask unique values: {torch.unique(sample_masks)}")
            print(f"Mask value counts: "
                  f"0={torch.sum(sample_masks==0).item()}, "
                  f"1={torch.sum(sample_masks==1).item()}")
            print(f"Shadow percentage: "
                  f"{100.0 * torch.sum(sample_masks==1).float() / sample_masks.numel():.2f}%")
            print("======================\n")

        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_aux_loss = 0.0
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

            # Forward pass on augmented views for contrastive learning
            feat_emb1, feat_emb2 = None, None
            if use_contrastive:
                images_aug1 = batch['image_aug1'].to(self.device)
                images_aug2 = batch['image_aug2'].to(self.device)

                _, features_aug1 = self.model(images_aug1, return_features=True)
                _, features_aug2 = self.model(images_aug2, return_features=True)

                feat_emb1 = features_aug1['feature_embeddings']
                feat_emb2 = features_aug2['feature_embeddings']

            # Segmentation loss (main + auxiliary)
            losses = self.criterion(outputs, masks)
            loss = losses['total']

            # Contrastive losses
            if self.contrastive_criterion is not None:
                contrastive_losses = self.contrastive_criterion(
                    outputs,
                    masks,
                    features_aug1=feat_emb1,
                    features_aug2=feat_emb2
                )

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

            # CHANGE: use filtered_outputs consistently for BOTH ShadowMetrics
            # and DetailedEvaluator.  The original used unfiltered preds for the
            # DetailedEvaluator, causing a metric mismatch between the two.
            filtered_outputs = filter_small_predictions(
                outputs['main'], min_pixels=10)
            train_metrics.update(filtered_outputs, masks)

            # CHANGE: DetailedEvaluator updated ALWAYS (not guarded by
            # eval_boundary_tolerant).  The original guard meant per-image
            # metrics were never collected when the flag was off, so
            # _get_decision_miou would have crashed.
            preds = torch.argmax(filtered_outputs, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            # Track losses
            epoch_loss += loss.item()
            epoch_seg_loss += losses['main'].item()
            epoch_aux_loss += losses.get('aux', torch.tensor(0.0)).item()
            epoch_fl_loss += losses.get(
                'feature_loss', torch.tensor(0.0)).item()
            epoch_sl_loss += losses.get(
                'semantic_loss', torch.tensor(0.0)).item()
            epoch_lc_loss += losses.get(
                'local_loss', torch.tensor(0.0)).item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'Batch [{batch_idx + 1}/{num_batches}] | '
                      f'Loss: {loss.item():.4f} | '
                      f'Main: {losses["main"].item():.4f} | '
                      f'Aux: {losses.get("aux", torch.tensor(0.0)).item():.4f} | '
                      f'FL: {losses.get("feature_loss", torch.tensor(0.0)).item():.4f} | '
                      f'SL: {losses.get("semantic_loss", torch.tensor(0.0)).item():.4f} | '
                      f'LC: {losses.get("local_loss", torch.tensor(0.0)).item():.4f}')

        # Averages
        epoch_loss /= num_batches
        epoch_seg_loss /= num_batches
        epoch_aux_loss /= num_batches
        # Avoid divide-by-zero when a contrastive component was never used
        epoch_fl_loss = epoch_fl_loss / num_batches if epoch_fl_loss > 0 else 0.0
        epoch_sl_loss = epoch_sl_loss / num_batches if epoch_sl_loss > 0 else 0.0
        epoch_lc_loss = epoch_lc_loss / num_batches if epoch_lc_loss > 0 else 0.0

        metrics = train_metrics.compute()
        epoch_time = time.time() - start_time

        print(f'\nTraining Results (pooled ShadowMetrics — reference):')
        print(f'Time: {epoch_time:.2f}s | Total Loss: {epoch_loss:.4f}')
        print(f'Seg: {epoch_seg_loss:.4f} | Aux: {epoch_aux_loss:.4f} | '
              f'FL: {epoch_fl_loss:.4f} | SL: {epoch_sl_loss:.4f} | '
              f'LC: {epoch_lc_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  F1: {metrics["F1"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — losses
        self.writer.add_scalar('Train/TotalLoss',    epoch_loss,     epoch)
        self.writer.add_scalar('Train/SegLoss',      epoch_seg_loss, epoch)
        self.writer.add_scalar('Train/AuxLoss',      epoch_aux_loss, epoch)
        self.writer.add_scalar('Train/FeatureLoss',  epoch_fl_loss,  epoch)
        self.writer.add_scalar('Train/SemanticLoss', epoch_sl_loss,  epoch)
        self.writer.add_scalar('Train/LocalLoss',    epoch_lc_loss,  epoch)
        # TensorBoard — pooled metrics (reference)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)

        # CHANGE: append ALL component losses to their history lists.
        # Previously only epoch_loss (total) was appended, so the individual
        # component panels in the plot had no data.
        self.train_losses.append(epoch_loss)
        self.train_seg_losses.append(epoch_seg_loss)
        self.train_aux_losses.append(epoch_aux_loss)
        self.train_fl_losses.append(epoch_fl_loss)
        self.train_sl_losses.append(epoch_sl_loss)
        self.train_lc_losses.append(epoch_lc_loss)
        for key in self.train_metrics_history:
            self.train_metrics_history[key].append(metrics[key])

        # DetailedEvaluator — per-image metrics (logged, not used for decisions here)
        detailed_results_train = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()

        strict_tr = detailed_results_train['boundary_tolerant']['strict']
        tol_tr = detailed_results_train['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Train/mIOU_strict_perimage',   strict_tr['iou'],  epoch)
        self.writer.add_scalar('Train/F1_strict_perimage',     strict_tr['f1'],   epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_perimage', tol_tr['iou'],     epoch)
        self.writer.add_scalar('Train/F1_tolerant_perimage',   tol_tr['f1'],      epoch)

        print(f'Per-image Strict:   F1={strict_tr["f1"]:.2f}%  '
              f'mIOU={strict_tr["iou"]:.2f}%')
        print(f'Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tol_tr["f1"]:.2f}%  mIOU={tol_tr["iou"]:.2f}%')

        return (epoch_loss, epoch_seg_loss, epoch_aux_loss,
                epoch_fl_loss, epoch_sl_loss, epoch_lc_loss, metrics)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, epoch):
        """
        Validate the model.

        Returns
        -------
        val_loss         : float
        metrics          : dict  — pooled ShadowMetrics (reference only)
        detailed_results : dict  — DetailedEvaluator per-image metrics
                                   (ALWAYS populated; used for decisions via
                                   _get_decision_miou)
        """
        print('\nValidating...')
        self.model.eval()

        val_loss = 0.0
        val_metrics = ShadowMetrics()

        # Val CE criterion — explicit to avoid dependency on MAMNetLoss internals
        _val_criterion = nn.CrossEntropyLoss(ignore_index=255)

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                # MAMNetMCL always returns a dict (unlike MAMNet which returns
                # a raw tensor in eval mode).
                outputs = self.model(images, return_features=False)

                loss = _val_criterion(outputs['main'], masks)
                val_loss += loss.item()

                # CHANGE: use filtered_outputs consistently for BOTH evaluators.
                filtered_outputs = filter_small_predictions(
                    outputs['main'], min_pixels=10)
                val_metrics.update(filtered_outputs, masks)

                # CHANGE: DetailedEvaluator updated ALWAYS (not guarded).
                preds = torch.argmax(filtered_outputs, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders['val'])
        metrics = val_metrics.compute()

        print(f'Validation Results (pooled ShadowMetrics — reference):')
        print(f'Loss: {val_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  F1: {metrics["F1"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — loss + pooled metrics (reference)
        self.writer.add_scalar('Val/Loss',          val_loss,             epoch)
        self.writer.add_scalar('Val/mIOU_pooled',   metrics['mIOU'],      epoch)
        self.writer.add_scalar('Val/Shadow_IOU',    metrics['Shadow_IOU'], epoch)
        self.writer.add_scalar('Val/F1',            metrics['F1'],         epoch)
        self.writer.add_scalar('Val/BER',           metrics['BER'],        epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Val/{key}', val, epoch)

        self.val_losses.append(val_loss)
        for key in self.val_metrics_history:
            self.val_metrics_history[key].append(metrics[key])

        # DetailedEvaluator — per-image metrics (ALWAYS; drive all decisions)
        # CHANGE: validate() now returns detailed_results instead of a
        # pre-computed decision scalar, so the caller can call
        # _get_decision_miou() in one place.
        detailed_results = self.detailed_evaluator_val.compute_metrics()
        self.detailed_evaluator_val.reset()

        strict_v = detailed_results['boundary_tolerant']['strict']
        tol_v = detailed_results['boundary_tolerant'][self.tol_key]

        self.writer.add_scalar('Val/mIOU_strict_perimage',   strict_v['iou'],  epoch)
        self.writer.add_scalar('Val/F1_strict_perimage',     strict_v['f1'],   epoch)
        self.writer.add_scalar('Val/mIOU_tolerant_perimage', tol_v['iou'],     epoch)
        self.writer.add_scalar('Val/F1_tolerant_perimage',   tol_v['f1'],      epoch)

        print(f'Per-image Strict:   F1={strict_v["f1"]:.2f}%  '
              f'mIOU={strict_v["iou"]:.2f}%')
        print(f'Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tol_v["f1"]:.2f}%  mIOU={tol_v["iou"]:.2f}%')

        return val_loss, metrics, detailed_results

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch':                        epoch,
            'model_state_dict':             self.model.state_dict(),
            'optimizer_state_dict':         self.optimizer.state_dict(),
            'scheduler_state_dict':         self.scheduler.state_dict(),
            # Decision-metric tracking
            'best_decision_miou':           self.best_decision_miou,
            'best_miou_pooled':             self.best_miou_pooled,
            'best_shadow_iou':              self.best_shadow_iou,
            'best_f1':                      self.best_f1,
            'epochs_without_improvement':   self.epochs_without_improvement,
            # CHANGE: persist all component loss histories
            'train_losses':                 self.train_losses,
            'train_seg_losses':             self.train_seg_losses,
            'train_aux_losses':             self.train_aux_losses,
            'train_fl_losses':              self.train_fl_losses,
            'train_sl_losses':              self.train_sl_losses,
            'train_lc_losses':              self.train_lc_losses,
            'val_losses':                   self.val_losses,
            # Metric histories
            'train_metrics_history':        self.train_metrics_history,
            'val_metrics_history':          self.val_metrics_history,
            'args':                         vars(self.args),
        }

        # Always save latest
        latest_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved → {best_path}')

        if epoch % self.args.save_freq == 0:
            epoch_path = os.path.join(
                self.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1

        # CHANGE: load best_decision_miou (renamed from best_miou); fall back to
        # old key name for backward compatibility with earlier checkpoints.
        self.best_decision_miou = checkpoint.get(
            'best_decision_miou',
            checkpoint.get('best_miou', 0.0))
        self.best_miou_pooled = checkpoint.get('best_miou_pooled', 0.0)
        self.best_shadow_iou = checkpoint.get('best_shadow_iou', 0.0)
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        self.epochs_without_improvement = checkpoint.get(
            'epochs_without_improvement',
            checkpoint.get('patience_counter', 0))

        # CHANGE: restore component loss histories (backward-compat defaults)
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_seg_losses = checkpoint.get('train_seg_losses', [])
        self.train_aux_losses = checkpoint.get('train_aux_losses', [])
        self.train_fl_losses = checkpoint.get('train_fl_losses', [])
        self.train_sl_losses = checkpoint.get('train_sl_losses', [])
        self.train_lc_losses = checkpoint.get('train_lc_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        self.train_metrics_history = checkpoint.get(
            'train_metrics_history',
            {'OA': [], 'Precision': [], 'F1': [],
             'BER': [], 'mIOU': [], 'Shadow_IOU': []})
        self.val_metrics_history = checkpoint.get(
            'val_metrics_history',
            {'OA': [], 'Precision': [], 'F1': [],
             'BER': [], 'mIOU': [], 'Shadow_IOU': []})

        print(f'Resumed from epoch {checkpoint["epoch"]}')
        print(f'Best decision mIOU: {self.best_decision_miou:.2f}%  '
              f'Epochs w/o improvement: {self.epochs_without_improvement}')

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """
        Main training loop.

        Decision metrics (LR scheduler, best-checkpoint, early stopping) are
        driven by per-image mIOU from DetailedEvaluator:
          - TOLERANT mIOU when --eval_boundary_tolerant is set
          - STRICT   mIOU otherwise
        Pooled ShadowMetrics values are never used for any decision.
        """
        print('\n' + '='*50)
        print('Starting training with mCL-LC...')
        metric_label = (f'Tolerant ({self.tol_key}) per-image mIOU'
                        if self.use_tolerant_decision
                        else 'Strict per-image mIOU')
        print(f'*** Decisions based on: {metric_label} ***')
        print('='*50)

        patience = self.args.early_stopping_patience
        if patience > 0:
            print(f'Early stopping: patience={patience}  metric={metric_label}')

        for epoch in range(self.start_epoch, self.args.epochs):
            # --- Train ---
            (train_loss, train_seg_loss, train_aux_loss,
             train_fl_loss, train_sl_loss, train_lc_loss,
             train_metrics) = self.train_epoch(epoch + 1)

            # --- Validate ---
            # CHANGE: validate() now returns detailed_results; decision scalar
            # computed via _get_decision_miou() below (not inside validate()).
            val_loss, val_metrics, detailed_results = self.validate(epoch + 1)

            # --- Decision metric ---
            decision_miou = self._get_decision_miou(detailed_results)
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, epoch + 1)

            # LR scheduler uses decision metric
            self.scheduler.step(decision_miou)

            # Best checkpoint
            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou = decision_miou
                is_best = True
                self.epochs_without_improvement = 0
                print(f'>> New best {metric_label}: {self.best_decision_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1

            # Reference trackers (not used for decisions)
            if val_metrics['mIOU'] > self.best_miou_pooled:
                self.best_miou_pooled = val_metrics['mIOU']
                print(f'   New best pooled mIOU (ref): {self.best_miou_pooled:.2f}%')
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
                print(f'   New best Shadow IoU (ref):  {self.best_shadow_iou:.2f}%')
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']
                print(f'   New best F1 (ref):          {self.best_f1:.2f}%')

            self.save_checkpoint(epoch + 1, is_best=is_best)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch + 1)

            # Early stopping
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping after {epoch + 1} epochs '
                      f'(no improvement in {metric_label} for {patience} epochs)')
                break

            print('='*50)

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best pooled mIOU (ref):  {self.best_miou_pooled:.2f}%')
        print(f'Best Shadow IoU (ref):   {self.best_shadow_iou:.2f}%')
        print(f'Best F1 (ref):           {self.best_f1:.2f}%')

        print('\nGenerating plots...')
        # CHANGE: call plot_loss_curves_mcl with all component histories.
        # The original called visualization.py's plot_loss_curves which only
        # accepted train_main_losses and train_aux_losses — FL/SL/LC had no
        # panel.  Also no data was ever stored for the components.
        plot_loss_curves_mcl(
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            save_path=os.path.join(self.output_dir, 'loss_curves.png'),
            train_seg_losses=self.train_seg_losses,
            train_aux_losses=self.train_aux_losses,
            train_fl_losses=self.train_fl_losses,
            train_sl_losses=self.train_sl_losses,
            train_lc_losses=self.train_lc_losses,
        )
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png')
        )

        self.writer.close()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self):
        """Test the model using best checkpoint"""
        print('\n' + '='*50)
        print('Testing model...')
        print('='*50)

        best_checkpoint = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_checkpoint):
            self.load_checkpoint(best_checkpoint)
        else:
            print('Warning: Best checkpoint not found, using current model weights')

        self.model.eval()
        test_metrics = ShadowMetrics()
        # DetailedEvaluator always instantiated
        detailed_eval = DetailedEvaluator(
            boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                # MAMNetMCL always returns a dict
                outputs = self.model(images, return_features=False)

                # CHANGE: use filtered_outputs consistently for both evaluators
                filtered_outputs = filter_small_predictions(
                    outputs['main'], min_pixels=10)
                test_metrics.update(filtered_outputs, masks)

                preds = torch.argmax(filtered_outputs, dim=1)
                detailed_eval.update(preds, masks, images)

        metrics = test_metrics.compute()
        detailed_results = detailed_eval.compute_metrics()

        print('\n' + '='*50)
        print('Pooled Test Results (reference):')
        print('='*50)
        for key, val in metrics.items():
            print(f'  {key}: {val:.2f}%')

        print('\n' + '='*50)
        print('Per-Image Test Results (DetailedEvaluator):')
        print('='*50)
        strict = detailed_results['boundary_tolerant']['strict']
        tolerant = detailed_results['boundary_tolerant'][self.tol_key]
        print(f"  Strict   — F1: {strict['f1']:.2f}%   mIOU: {strict['iou']:.2f}%")
        print(f"  Tolerant (±{self.args.boundary_tolerance}px) — "
              f"F1: {tolerant['f1']:.2f}%   mIOU: {tolerant['iou']:.2f}%")
        print(f"  Pixels excluded by band: {tolerant['pixels_excluded']} "
              f"({tolerant['pct_excluded']:.1f}%)")

        if 'size_stratified' in detailed_results:
            print('\nSize-Stratified (Strict):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in detailed_results['size_stratified']:
                    m = detailed_results['size_stratified'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if 'size_stratified_tolerant' in detailed_results:
            print(f'\nSize-Stratified (Tolerant ±{self.args.boundary_tolerance}px):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in detailed_results['size_stratified_tolerant']:
                    m = detailed_results['size_stratified_tolerant'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if ('fp_fn_analysis' in detailed_results
                and 'fp' in detailed_results['fp_fn_analysis']):
            fp = detailed_results['fp_fn_analysis']['fp']
            print('\nFP Spatial Distribution:')
            print(f"  Within 1px:  {fp['pct_within_1px']:.1f}%")
            print(f"  Within 5px:  {fp['pct_within_5px']:.1f}%")
            print(f"  Within 10px: {fp['pct_within_10px']:.1f}%")

        results_to_save = {'standard': metrics, 'detailed': detailed_results}
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f'\nResults saved → {results_path}')

        # CHANGE: call save_best_worst_visualizations_mcl instead of the
        # visualization.py version, which called model(images) and did
        # torch.argmax(outputs, dim=1) — crashing because MAMNetMCL always
        # returns a dict.
        try:
            print('\nGenerating best/worst prediction visualizations...')
            save_best_worst_visualizations_mcl(
                self.model, self.dataloaders['test'],
                self.device, self.output_dir, num_images=10)
        except Exception as e:
            print(f'Visualization skipped: {e}')

        return metrics


def main():
    args = get_args()
    trainer = MCLTrainer(args)

    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()