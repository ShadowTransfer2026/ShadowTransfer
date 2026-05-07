"""
Training script for OGLANet + DDIB Shadow Detection (Option B).

Option B = full DDIB at bottleneck + lightweight skip domain filters.

Usage examples:

  # Full DDIB + skip filters (C1+C2+C3+SF) -- LOCO fold 0
  python train_oglanet_ddib.py \
      --mode loco --fold_id 0 \
      --base_data_root /path/to/data --resolution highres \
      --use_disentangle --use_vib --use_feat_aug --use_skip_filter \
      --lambda_hsic 0.1 --lambda_domain 0.01 --lambda_kl 0.001

  # Ablation: C3 + skip filters only
  python train_oglanet_ddib.py \
      --mode loco --fold_id 0 \
      --base_data_root /path/to/data --resolution highres \
      --use_feat_aug --use_skip_filter

  # Option A equivalent (no skip filters)
  python train_oglanet_ddib.py \
      --mode loco --fold_id 0 \
      --base_data_root /path/to/data --resolution highres \
      --use_disentangle --use_vib --use_feat_aug

  # No-DDIB baseline
  python train_oglanet_ddib.py \
      --mode loco --fold_id 0 \
      --base_data_root /path/to/data --resolution highres
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
from PIL import Image
import cv2

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.oglanet_ddib import OGLANetDDIB
from data.dataset_ddib import get_dataloaders_ddib
from data.dataset import LOCO_FOLDS
from utils.losses import OGLANetLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.evaluation_detailed import DetailedEvaluator
from utils.visualization import plot_loss_curves, plot_metrics_curves


print("=" * 50)
print("GPU DIAGNOSTICS")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print("=" * 50)


# ======================================================================
# Arguments
# ======================================================================

def get_args():
    p = argparse.ArgumentParser(
        description='Train OGLANet + DDIB (Option B)')

    # -- Data --
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=1)

    # -- Mode --
    p.add_argument('--mode', type=str, default='single',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution', type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id', type=int, default=None, choices=[0, 1, 2])
    p.add_argument('--cities', type=str, nargs='+', default=None)

    # -- Model --
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--use_contrast', action='store_true',
                   help='Use contrast as 4th input channel')

    # -- DDIB component toggles --
    p.add_argument('--use_disentangle', action='store_true', default=False,
                   help='Enable DDIB Component 1 (feature disentanglement)')
    p.add_argument('--use_vib', action='store_true', default=False,
                   help='Enable DDIB Component 2 (VIB)')
    p.add_argument('--use_feat_aug', action='store_true', default=False,
                   help='Enable DDIB Component 3 (feature augmentation)')
    p.add_argument('--use_skip_filter', action='store_true', default=False,
                   help='Enable lightweight skip-connection domain filters (Option B)')

    # -- DDIB loss weights --
    p.add_argument('--lambda_hsic', type=float, default=0.1)
    p.add_argument('--lambda_domain', type=float, default=0.01)
    p.add_argument('--lambda_kl', type=float, default=0.001)

    # -- DDIB hyper-parameters --
    p.add_argument('--hsic_samples', type=int, default=1024)
    p.add_argument('--vib_beta_base', type=float, default=0.001)
    p.add_argument('--vib_beta_scale', type=float, default=0.01)
    p.add_argument('--aug_sigma_style', type=float, default=0.5)
    p.add_argument('--aug_sigma_shift', type=float, default=0.3)
    p.add_argument('--aug_p_aug', type=float, default=0.5)
    p.add_argument('--aug_p_mix', type=float, default=0.3)

    # -- Skip filter hyper-parameters --
    p.add_argument('--skip_reduction', type=int, default=4,
                   help='Channel reduction ratio for skip filters')
    p.add_argument('--skip_kl_weight', type=float, default=0.1,
                   help='Weight for skip VIB KL relative to bottleneck KL')

    # -- Training (OGLANet defaults) --
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--optimizer', type=str, default='adamax',
                   choices=['adam', 'adamax'])

    # -- FDA --
    p.add_argument('--use_fda', action='store_true')
    p.add_argument('--fda_target_root', type=str, default=None)
    p.add_argument('--fda_L', type=float, default=0.01)

    # -- Checkpoint / logging --
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--save_freq', type=int, default=10)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--eval_boundary_tolerant', action='store_true')
    p.add_argument('--early_stopping_patience', type=int, default=15)

    # -- Comparison baselines (passed in from shell scripts) --
    p.add_argument('--comparison_inference_dir', type=str, default=None,
                   help='Directory with existing inference results')
    p.add_argument('--comparison_data_root', type=str, default=None,
                   help='Root directory with ground truth data')

    return p.parse_args()


# ======================================================================
# Per-image metric functions
# ======================================================================

_TOLERANCE_KERNEL_CACHE = {}


def _get_tolerance_kernel(tolerance):
    if tolerance not in _TOLERANCE_KERNEL_CACHE:
        _TOLERANCE_KERNEL_CACHE[tolerance] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (tolerance * 2 + 1, tolerance * 2 + 1))
    return _TOLERANCE_KERNEL_CACHE[tolerance]


def _compute_strict_metrics(pred, gt):
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    shadow_iou    = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou = (shadow_iou + nonshadow_iou) / 2
    oa = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    shadow_err    = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber = (shadow_err + nonshadow_err) / 2

    return {
        'OA': float(oa * 100), 'Precision': float(precision * 100),
        'Recall': float(recall * 100), 'F1': float(f1 * 100),
        'BER': float(ber * 100), 'mIOU': float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100),
    }


def _compute_tolerant_metrics(pred, gt, tolerance=5):
    kernel   = _get_tolerance_kernel(tolerance)
    gt_uint8 = gt.astype(np.uint8)
    eroded   = cv2.erode(gt_uint8, kernel)
    dilated  = cv2.dilate(gt_uint8, kernel)
    band     = (dilated - eroded) > 0
    valid    = ~band

    p, g = pred[valid], gt[valid]
    tp = np.logical_and(p == 1, g == 1).sum()
    fp = np.logical_and(p == 1, g == 0).sum()
    tn = np.logical_and(p == 0, g == 0).sum()
    fn = np.logical_and(p == 0, g == 1).sum()

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    shadow_iou    = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou = (shadow_iou + nonshadow_iou) / 2
    oa = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    shadow_err    = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber = (shadow_err + nonshadow_err) / 2

    return {
        'OA': float(oa * 100), 'Precision': float(precision * 100),
        'Recall': float(recall * 100), 'F1': float(f1 * 100),
        'BER': float(ber * 100), 'mIOU': float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100),
    }


def _average_metrics(metrics_list):
    if not metrics_list:
        return {k: 0.0 for k in
                ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU',
                 'Shadow_IOU']}
    keys = ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


# ======================================================================
# Trainer
# ======================================================================

class TrainerDDIB:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')

        # ---- Output directory naming ----
        ddib_tag = ''
        if args.use_disentangle: ddib_tag += '_C1'
        if args.use_vib:         ddib_tag += '_C2'
        if args.use_feat_aug:    ddib_tag += '_C3'
        if args.use_skip_filter: ddib_tag += '_SF'
        if not ddib_tag:         ddib_tag = '_noDDIB'

        contrast_tag = '_contrast' if args.use_contrast else ''
        fda_tag = '_fda' if args.use_fda else ''

        if args.mode == 'loco':
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = (f'oglanet_ddib{ddib_tag}{contrast_tag}{fda_tag}'
                        f'_loco_holdout_{test_city}_{args.resolution}_1')
        elif args.mode == 'all':
            exp_name = (f'oglanet_ddib{ddib_tag}{contrast_tag}{fda_tag}'
                        f'_all_{args.resolution}_1')
        else:
            city = args.data_root.rstrip('/').split('/')[-2]
            res  = args.data_root.rstrip('/').split('/')[-1]
            exp_name = (f'oglanet_ddib{ddib_tag}{contrast_tag}{fda_tag}'
                        f'_{city}_{res}_1')

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(
            os.path.join(self.output_dir, 'tensorboard'))

        # ---- Data ----
        print('\nLoading datasets …')
        loaders = get_dataloaders_ddib(
            data_root=args.data_root,
            base_data_root=args.base_data_root,
            mode=args.mode, cities=args.cities,
            resolution=args.resolution, fold_id=args.fold_id,
            batch_size=args.batch_size, num_workers=args.num_workers,
            img_size=args.img_size,
            use_fda=args.use_fda,
            fda_target_root=args.fda_target_root, fda_L=args.fda_L,
            use_contrast=args.use_contrast,
        )
        self.dataloaders = loaders
        num_domains = loaders['num_domains']

        print(f'Train: {len(loaders["train"].dataset)}  |  '
              f'Val: {len(loaders["val"].dataset)}  |  '
              f'Test: {len(loaders["test"].dataset)}')

        # ---- Model ----
        print('\nBuilding OGLANet-DDIB (Option B) …')
        print('ASSUMPTION: Using ResNet-34 encoder (not ResNet-101)')
        self.model = OGLANetDDIB(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size,
            use_contrast=args.use_contrast,
            use_disentangle=args.use_disentangle,
            use_vib=args.use_vib,
            use_feat_aug=args.use_feat_aug,
            use_skip_filter=args.use_skip_filter,
            num_domains=num_domains,
            hsic_samples=args.hsic_samples,
            vib_beta_base=args.vib_beta_base,
            vib_beta_scale=args.vib_beta_scale,
            aug_sigma_style=args.aug_sigma_style,
            aug_sigma_shift=args.aug_sigma_shift,
            aug_p_aug=args.aug_p_aug,
            aug_p_mix=args.aug_p_mix,
            skip_reduction=args.skip_reduction,
            skip_kl_weight=args.skip_kl_weight,
        ).to(self.device)

        # ---- Loss / optimiser / scheduler ----
        self.criterion = OGLANetLoss()

        if args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(self.model.parameters(), lr=args.lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # NOTE: scheduler.step() is called with the *decision metric*
        #       (tolerant mIOU when --eval_boundary_tolerant, strict otherwise).
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        # ---- Tracking ----
        self.start_epoch = 0
        self.best_miou = 0.0           # best *decision* mIOU (tolerant or strict)
        self.best_shadow_iou = 0.0
        self.best_f1 = 0.0
        self.epochs_without_improvement = 0

        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [],
            'BER': [], 'mIOU': [], 'Shadow_IOU': []}
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [],
            'BER': [], 'mIOU': [], 'Shadow_IOU': []}

        if args.eval_boundary_tolerant:
            self.detailed_eval_train = DetailedEvaluator()
            self.detailed_eval_val   = DetailedEvaluator()
            print('Boundary-tolerant evaluation enabled')
            print('  -> Decision metric: Tolerant mIOU (±5px boundary excluded)')
        else:
            print('  -> Decision metric: Strict mIOU')
        if args.early_stopping_patience > 0:
            print(f'  -> Early stopping patience: {args.early_stopping_patience} epochs')

        if args.resume:
            self._load_checkpoint(args.resume)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        self.model.train()

        epoch_loss = 0.0
        epoch_seg  = 0.0
        epoch_hsic = 0.0
        epoch_dom  = 0.0
        epoch_kl   = 0.0
        epoch_skip_kl = 0.0
        epoch_component_losses = {
            'loss1': 0.0, 'loss2': 0.0, 'loss3': 0.0,
            'loss4': 0.0, 'loss5': 0.0, 'loss6': 0.0}

        train_metrics = ShadowMetrics()
        n_batches = len(self.dataloaders['train'])
        t0 = time.time()

        for i, batch in enumerate(self.dataloaders['train']):
            images   = batch['image'].to(self.device)
            masks    = batch['mask'].to(self.device)
            int_map  = batch['intensity_map'].to(self.device)
            city_ids = batch['city_id'].to(self.device)

            # Forward (training mode → returns dict of 6 predictions)
            predictions, all_losses = self.model(
                images, int_map, city_ids)

            # Deep supervision loss (P1–P6)
            seg_losses = self.criterion(predictions, masks)
            total = seg_losses['total']

            # DDIB bottleneck losses
            hsic_val    = all_losses.get('hsic_loss', torch.tensor(0.0))
            dom_val     = all_losses.get('domain_loss', torch.tensor(0.0))
            kl_val      = all_losses.get('kl_loss', torch.tensor(0.0))
            skip_kl_val = all_losses.get('skip_kl_loss', torch.tensor(0.0))

            if self.args.use_disentangle:
                total = total + self.args.lambda_hsic * hsic_val
                total = total + self.args.lambda_domain * dom_val
            if self.args.use_vib:
                total = total + self.args.lambda_kl * kl_val
            # Skip KL is already scaled by skip_kl_weight inside model;
            # apply lambda_kl to keep it on the same scale as bottleneck KL
            if self.args.use_skip_filter and self.args.use_vib:
                total = total + self.args.lambda_kl * skip_kl_val

            # Backward
            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()

            # Metrics (on P6 — final prediction)
            filtered = filter_small_predictions(
                predictions['p6'], min_pixels=10)
            train_metrics.update(filtered, masks)

            if self.args.eval_boundary_tolerant:
                preds = torch.argmax(predictions['p6'].detach(), dim=1)
                self.detailed_eval_train.update(preds, masks, images)

            # Track losses
            epoch_loss    += total.item()
            epoch_seg     += seg_losses['total'].item()
            epoch_hsic    += hsic_val.item()
            epoch_dom     += dom_val.item()
            epoch_kl      += kl_val.item()
            epoch_skip_kl += skip_kl_val.item()
            for key in epoch_component_losses:
                epoch_component_losses[key] += seg_losses[key].item()

            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(f'  [{i+1}/{n_batches}]  loss={total.item():.4f}  '
                      f'seg={seg_losses["total"].item():.4f}  '
                      f'hsic={hsic_val.item():.5f}  '
                      f'dom={dom_val.item():.4f}  '
                      f'kl={kl_val.item():.6f}  '
                      f'skip_kl={skip_kl_val.item():.6f}')

        # Averages
        epoch_loss    /= n_batches
        epoch_seg     /= n_batches
        epoch_hsic    /= n_batches
        epoch_dom     /= n_batches
        epoch_kl      /= n_batches
        epoch_skip_kl /= n_batches
        for key in epoch_component_losses:
            epoch_component_losses[key] /= n_batches

        m = train_metrics.compute()
        elapsed = time.time() - t0

        print(f'\nEpoch {epoch} train  ({elapsed:.1f}s)')
        print(f'  loss={epoch_loss:.4f}  seg={epoch_seg:.4f}  '
              f'hsic={epoch_hsic:.5f}  dom={epoch_dom:.4f}  '
              f'kl={epoch_kl:.6f}  skip_kl={epoch_skip_kl:.6f}')
        print(f'  OA={m["OA"]:.2f}  P={m["Precision"]:.2f}  '
              f'F1={m["F1"]:.2f}  BER={m["BER"]:.2f}  '
              f'mIOU={m["mIOU"]:.2f}  ShIOU={m["Shadow_IOU"]:.2f}')

        # Tensorboard
        self.writer.add_scalar('Train/TotalLoss', epoch_loss, epoch)
        self.writer.add_scalar('Train/Seg_Loss', epoch_seg, epoch)
        self.writer.add_scalar('Train/HSIC_Loss', epoch_hsic, epoch)
        self.writer.add_scalar('Train/Domain_Loss', epoch_dom, epoch)
        self.writer.add_scalar('Train/KL_Loss', epoch_kl, epoch)
        self.writer.add_scalar('Train/Skip_KL_Loss', epoch_skip_kl, epoch)
        for key, val in epoch_component_losses.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)
        for key, val in m.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)

        # Log skip gate values
        gate_vals = self.model.get_skip_gate_values()
        for gk, gv in gate_vals.items():
            self.writer.add_scalar(f'SkipGates/{gk}', gv, epoch)
        if gate_vals:
            gates_str = '  '.join(f'{k}={v:.4f}' for k, v in gate_vals.items())
            print(f'  Skip gates: {gates_str}')

        self.train_losses.append(epoch_loss)
        for k in self.train_metrics_history:
            self.train_metrics_history[k].append(m[k])

        if self.args.eval_boundary_tolerant:
            dr = self.detailed_eval_train.compute_metrics()
            t5 = dr['boundary_tolerant']['tolerant_5px']
            self.writer.add_scalar('Train/F1_Tolerant', t5['f1'], epoch)
            self.writer.add_scalar('Train/mIOU_Tolerant', t5['iou'], epoch)
            self.detailed_eval_train.reset()
            print(f'  Boundary-Tolerant: F1={t5["f1"]:.2f}  '
                  f'mIOU={t5["iou"]:.2f}')

        return epoch_loss, m

    # ------------------------------------------------------------------
    def validate(self, epoch):
        """Validate the model.

        Returns:
            val_loss:      average validation loss
            metrics:       strict metrics dict
            decision_miou: the mIOU used for LR scheduler / best-checkpoint /
                           early-stopping decisions.  This is the tolerant mIOU
                           when --eval_boundary_tolerant is set, strict mIOU
                           otherwise.
        """
        print('\nValidating …')
        self.model.eval()
        val_loss = 0.0
        val_metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images  = batch['image'].to(self.device)
                masks   = batch['mask'].to(self.device)
                int_map = batch['intensity_map'].to(self.device)

                # Eval mode → returns P6 tensor
                predictions, _ = self.model(images, int_map)
                val_loss += self.criterion.criterion(
                    predictions, masks).item()

                filtered = filter_small_predictions(
                    predictions, min_pixels=10)
                val_metrics.update(filtered, masks)

                if self.args.eval_boundary_tolerant:
                    preds = torch.argmax(predictions, dim=1)
                    self.detailed_eval_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders['val'])
        m = val_metrics.compute()

        print(f'Val  loss={val_loss:.4f}')
        print(f'  OA={m["OA"]:.2f}  P={m["Precision"]:.2f}  '
              f'F1={m["F1"]:.2f}  BER={m["BER"]:.2f}  '
              f'mIOU={m["mIOU"]:.2f}  ShIOU={m["Shadow_IOU"]:.2f}')

        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        for k, v in m.items():
            self.writer.add_scalar(f'Val/{k}', v, epoch)

        self.val_losses.append(val_loss)
        for k in self.val_metrics_history:
            self.val_metrics_history[k].append(m[k])

        # ---- Decision metric: tolerant mIOU if available, else strict ----
        decision_miou = m['mIOU']   # default: strict

        if self.args.eval_boundary_tolerant:
            dr = self.detailed_eval_val.compute_metrics()
            t5 = dr['boundary_tolerant']['tolerant_5px']
            self.writer.add_scalar('Val/F1_Tolerant', t5['f1'], epoch)
            self.writer.add_scalar('Val/mIOU_Tolerant', t5['iou'], epoch)
            self.detailed_eval_val.reset()
            print(f'  Boundary-Tolerant: F1={t5["f1"]:.2f}  '
                  f'mIOU={t5["iou"]:.2f}')

            # Use tolerant mIOU as the decision metric
            decision_miou = t5['iou']

        return val_loss, m, decision_miou

    # ------------------------------------------------------------------
    def train(self):
        """Main training loop.

        Decision logic (LR scheduler, best checkpoint, early stopping) is
        driven by the *decision metric*:
          - Tolerant mIOU  when --eval_boundary_tolerant is set
          - Strict  mIOU   otherwise
        Both strict and tolerant metrics are always logged / printed.
        """
        metric_label = ("Tolerant mIOU" if self.args.eval_boundary_tolerant
                        else "Strict mIOU")

        print('\n' + '=' * 60)
        print('Starting OGLANet-DDIB (Option B) training')
        print(f'Decision metric: {metric_label}')
        print('=' * 60)

        for epoch in range(self.start_epoch, self.args.epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'\n{"="*60}\nEpoch {epoch+1}/{self.args.epochs}  '
                  f'lr={current_lr:.2e}')

            _, _ = self.train_epoch(epoch + 1)
            _, val_m, decision_miou = self.validate(epoch + 1)

            # ---- All decisions keyed off decision_miou ----
            # Scheduler step
            self.scheduler.step(decision_miou)

            # Best model tracking
            is_best = False
            if decision_miou > self.best_miou:
                self.best_miou = decision_miou
                is_best = True
                self.epochs_without_improvement = 0
                print(f'  ★ New best {metric_label}: {self.best_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1

            if val_m['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_m['Shadow_IOU']
            if val_m['F1'] > self.best_f1:
                self.best_f1 = val_m['F1']

            self._save_checkpoint(epoch + 1, is_best)
            self.writer.add_scalar('Train/LR', current_lr, epoch + 1)

            # ---- Early stopping ----
            if (self.args.early_stopping_patience > 0
                    and self.epochs_without_improvement
                    >= self.args.early_stopping_patience):
                print(f'\nEarly stopping — no improvement in {metric_label} '
                      f'for {self.args.early_stopping_patience} epochs.')
                break

        print(f'\nTraining finished.  Best {metric_label}={self.best_miou:.2f}  '
              f'Shadow_IOU={self.best_shadow_iou:.2f}  '
              f'F1={self.best_f1:.2f}')

        plot_loss_curves(self.train_losses, self.val_losses,
                         os.path.join(self.output_dir, 'loss_curves.png'))
        plot_metrics_curves(self.train_metrics_history,
                            self.val_metrics_history,
                            os.path.join(self.output_dir,
                                         'metrics_curves.png'))
        self.writer.close()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self):
        print('\n' + '=' * 70)
        print('TESTING')
        print('=' * 70)

        best_ckpt = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_ckpt):
            self._load_checkpoint(best_ckpt)
        else:
            print('Warning: best checkpoint not found, using current weights')

        self.model.eval()

        pred_save_dir = os.path.join(self.output_dir, 'predictions')
        os.makedirs(pred_save_dir, exist_ok=True)

        ddib_strict_list   = []
        ddib_tolerant_list = []
        all_filenames      = []

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images  = batch['image'].to(self.device)
                masks   = batch['mask'].to(self.device)
                int_map = batch['intensity_map'].to(self.device)

                outputs, _ = self.model(images, int_map)
                preds = torch.argmax(outputs.detach(), dim=1)

                for i, fname in enumerate(batch['filename']):
                    pred_np = preds[i].cpu().numpy().astype(np.uint8)
                    gt_np   = masks[i].cpu().numpy().astype(np.uint8)

                    Image.fromarray(pred_np * 255).save(
                        os.path.join(pred_save_dir, fname))

                    ddib_strict_list.append(
                        _compute_strict_metrics(pred_np, gt_np))
                    ddib_tolerant_list.append(
                        _compute_tolerant_metrics(pred_np, gt_np,
                                                  tolerance=5))
                    all_filenames.append(fname)

        ddib_strict   = _average_metrics(ddib_strict_list)
        ddib_tolerant = _average_metrics(ddib_tolerant_list)

        print(f'\nOGLANet-DDIB Results ({len(all_filenames)} images):')
        print(f'  Strict:   OA={ddib_strict["OA"]:.2f}  '
              f'P={ddib_strict["Precision"]:.2f}  '
              f'R={ddib_strict["Recall"]:.2f}  '
              f'F1={ddib_strict["F1"]:.2f}  '
              f'BER={ddib_strict["BER"]:.2f}  '
              f'mIOU={ddib_strict["mIOU"]:.2f}  '
              f'ShIOU={ddib_strict["Shadow_IOU"]:.2f}')
        print(f'  Tolerant: OA={ddib_tolerant["OA"]:.2f}  '
              f'P={ddib_tolerant["Precision"]:.2f}  '
              f'R={ddib_tolerant["Recall"]:.2f}  '
              f'F1={ddib_tolerant["F1"]:.2f}  '
              f'BER={ddib_tolerant["BER"]:.2f}  '
              f'mIOU={ddib_tolerant["mIOU"]:.2f}  '
              f'ShIOU={ddib_tolerant["Shadow_IOU"]:.2f}')

        # Log final skip gate values
        gate_vals = self.model.get_skip_gate_values()
        if gate_vals:
            gates_str = '  '.join(f'{k}={v:.4f}' for k, v in gate_vals.items())
            print(f'  Final skip gates: {gates_str}')

        results = {
            'num_images': len(all_filenames),
            'strict': ddib_strict,
            'tolerant_5px': ddib_tolerant,
            'skip_gates': gate_vals,
        }
        with open(os.path.join(self.output_dir, 'test_results.json'),
                  'w') as f:
            json.dump(results, f, indent=4)

        # Baseline comparison (LOCO mode)
        if self.args.mode == 'loco' and self.args.comparison_inference_dir:
            self._compare_with_baselines(
                ddib_strict, ddib_tolerant,
                ddib_strict_list, ddib_tolerant_list,
                all_filenames)

        return ddib_strict

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------

    def _compare_with_baselines(self, ddib_strict, ddib_tolerant,
                                ddib_strict_list, ddib_tolerant_list,
                                filenames):
        fold_id   = self.args.fold_id
        test_city = LOCO_FOLDS[fold_id]['test']
        res       = self.args.resolution
        inf_dir   = self.args.comparison_inference_dir
        data_root = self.args.comparison_data_root
        img_size  = self.args.img_size

        gt_dir = os.path.join(data_root, test_city, res, 'test', 'masks')

        baseline_dirs = [
            ('Upper Bound',
             os.path.join(inf_dir, 'upper', test_city, res,
                          'oglanet', 'base')),
            ('LOCO Vanilla',
             os.path.join(inf_dir, 'loco', test_city, res,
                          'oglanet', 'vanilla')),
            ('LOCO FDA',
             os.path.join(inf_dir, 'loco', test_city, res,
                          'oglanet', 'fda')),
            ('LOCO SegDesic',
             os.path.join(inf_dir, 'loco', test_city, res,
                          'oglanet', 'segdesic')),
            ('LOCO mCL-LC',
             os.path.join(inf_dir, 'loco', test_city, res,
                          'oglanet', 'mcl')),
        ]

        print('\n' + '=' * 70)
        print('BASELINE COMPARISON')
        print(f'  Test city: {test_city}  |  Resolution: {res}')
        print('=' * 70)

        baseline_results = {}
        for label, pred_dir in baseline_dirs:
            if not os.path.isdir(pred_dir):
                print(f'\n  Warning: {label} not found: {pred_dir}')
                continue

            strict_list, tolerant_list = [], []
            n_matched = 0

            for fname in filenames:
                pred_path = os.path.join(pred_dir, fname)
                gt_path   = os.path.join(gt_dir, fname)
                if not os.path.exists(pred_path) or \
                   not os.path.exists(gt_path):
                    continue

                pred_np = np.array(
                    Image.open(pred_path).convert('L').resize(
                        (img_size, img_size), Image.NEAREST))
                pred_bin = (pred_np > 127).astype(np.uint8)

                gt_np = np.array(
                    Image.open(gt_path).convert('L').resize(
                        (img_size, img_size), Image.NEAREST))
                gt_bin = (gt_np > 127).astype(np.uint8)

                strict_list.append(
                    _compute_strict_metrics(pred_bin, gt_bin))
                tolerant_list.append(
                    _compute_tolerant_metrics(pred_bin, gt_bin, tolerance=5))
                n_matched += 1

            if n_matched == 0:
                print(f'\n  Warning: {label}: no matching images found')
                continue

            baseline_results[label] = {
                'strict': _average_metrics(strict_list),
                'tolerant': _average_metrics(tolerant_list),
                'strict_list': strict_list,
                'tolerant_list': tolerant_list,
                'n_images': n_matched,
            }
            print(f'  Found {label}: {n_matched} images')

        if not baseline_results:
            print('\n  No baselines found for comparison.')
            return

        # Print tables
        self._print_comparison_table(
            'STRICT METRICS',
            baseline_results, ddib_strict, 'strict')
        self._print_comparison_table(
            'TOLERANT METRICS (±5px)',
            baseline_results, ddib_tolerant, 'tolerant')
        self._print_recovery_ratios(
            baseline_results, ddib_strict, ddib_tolerant)

        # Bootstrap vs each LOCO baseline
        for bl in ['LOCO Vanilla', 'LOCO FDA',
                    'LOCO SegDesic', 'LOCO mCL-LC']:
            if bl in baseline_results:
                self._print_bootstrap_comparison(
                    baseline_results[bl],
                    ddib_strict_list, ddib_tolerant_list,
                    baseline_label=bl)

        # Save
        comp = {
            'test_city': test_city, 'resolution': res,
            'eval_size': img_size,
            'ddib': {'strict': ddib_strict,
                     'tolerant_5px': ddib_tolerant},
            'baselines': {l: {'strict': r['strict'],
                              'tolerant_5px': r['tolerant'],
                              'n_images': r['n_images']}
                          for l, r in baseline_results.items()},
        }
        comp_path = os.path.join(self.output_dir,
                                 'comparison_results.json')
        with open(comp_path, 'w') as f:
            json.dump(comp, f, indent=4)
        print(f'\nComparison saved to {comp_path}')

    # ------------------------------------------------------------------
    def _print_comparison_table(self, title, baseline_results,
                                ddib_metrics, metric_type='strict'):
        print('\n' + '-' * 70)
        print(f'{title:^70}')
        print('-' * 70)
        header = (f'  {"Method":<20} {"OA":>6} {"Prec":>6} {"Rec":>6} '
                  f'{"F1":>6} {"BER":>6} {"mIOU":>6} {"ShIOU":>6}')
        print(header)
        print('  ' + '-' * 62)

        for label in ['Upper Bound', 'LOCO Vanilla', 'LOCO FDA',
                       'LOCO SegDesic', 'LOCO mCL-LC']:
            if label in baseline_results:
                m = baseline_results[label][metric_type]
                print(f'  {label:<20} {m["OA"]:6.2f} '
                      f'{m["Precision"]:6.2f} {m["Recall"]:6.2f} '
                      f'{m["F1"]:6.2f} {m["BER"]:6.2f} '
                      f'{m["mIOU"]:6.2f} {m["Shadow_IOU"]:6.2f}')

        d = ddib_metrics
        print(f'  {"DDIB (ours)":<20} {d["OA"]:6.2f} '
              f'{d["Precision"]:6.2f} {d["Recall"]:6.2f} '
              f'{d["F1"]:6.2f} {d["BER"]:6.2f} '
              f'{d["mIOU"]:6.2f} {d["Shadow_IOU"]:6.2f}')

    # ------------------------------------------------------------------
    def _print_recovery_ratios(self, baseline_results,
                               ddib_strict, ddib_tolerant):
        if ('Upper Bound' not in baseline_results
                or 'LOCO Vanilla' not in baseline_results):
            return

        print('\n' + '-' * 70)
        print(f'{"RECOVERY RATIOS":^70}')
        print(f'  R = (DDIB - LOCO_Vanilla) / (Upper - LOCO_Vanilla)')
        print('-' * 70)

        key_metrics = ['F1', 'mIOU', 'Shadow_IOU', 'BER']

        for eval_type, ddib_m, label in [
                ('strict', ddib_strict, 'Strict'),
                ('tolerant', ddib_tolerant, 'Tolerant')]:
            ub = baseline_results['Upper Bound'][eval_type]
            lv = baseline_results['LOCO Vanilla'][eval_type]
            parts = []
            for k in key_metrics:
                gap = ub[k] - lv[k]
                rec = ddib_m[k] - lv[k]
                if k == 'BER':
                    gap, rec = -gap, -rec
                R = rec / gap if abs(gap) > 0.01 else float('nan')
                parts.append(f'{k}={R:.3f}')
            print(f'  {label:<10}  ' + '  '.join(parts))

        # Deltas vs adaptation methods
        adapt_methods = ['LOCO FDA', 'LOCO SegDesic', 'LOCO mCL-LC']
        if not any(m in baseline_results for m in adapt_methods):
            return

        print('\n' + '-' * 70)
        print(f'{"DDIB IMPROVEMENT OVER ADAPTATION METHODS (delta)":^70}')
        print('-' * 70)

        for eval_type, ddib_m, label in [
                ('strict', ddib_strict, 'Strict'),
                ('tolerant', ddib_tolerant, 'Tolerant')]:
            print(f'\n  {label}:')
            print(f'    {"Method":<16} {"dF1":>7} {"dmIOU":>7} '
                  f'{"dShIOU":>7} {"dBER":>7}')
            print(f'    ' + '-' * 40)
            for bl in adapt_methods:
                if bl not in baseline_results:
                    continue
                bm = baseline_results[bl][eval_type]
                parts = []
                for k in key_metrics:
                    delta = ddib_m[k] - bm[k]
                    parts.append(f'{delta:+7.2f}')
                print(f'    {bl:<16} ' + ' '.join(parts))

    # ------------------------------------------------------------------
    def _print_bootstrap_comparison(self, loco_baseline,
                                    ddib_strict_list, ddib_tolerant_list,
                                    baseline_label='LOCO Vanilla',
                                    n_bootstrap=5000):
        print('\n' + '-' * 70)
        print(f'{"BOOTSTRAP: DDIB vs " + baseline_label + " (n=5000)":^70}')
        print('-' * 70)

        np.random.seed(42)
        key_metrics = ['F1', 'mIOU', 'Shadow_IOU']

        for eval_type, ddib_list, label in [
                ('strict_list', ddib_strict_list, 'Strict'),
                ('tolerant_list', ddib_tolerant_list, 'Tolerant')]:
            loco_list = loco_baseline[eval_type]
            n = min(len(loco_list), len(ddib_list))
            if n == 0:
                continue
            print(f'\n  {label}:')
            for k in key_metrics:
                loco_vals = np.array([m[k] for m in loco_list[:n]])
                ddib_vals = np.array([m[k] for m in ddib_list[:n]])
                diff = ddib_vals - loco_vals
                obs_mean = np.mean(diff)
                boot_means = np.array([
                    np.mean(diff[np.random.choice(n, n, replace=True)])
                    for _ in range(n_bootstrap)])
                ci_lo = np.percentile(boot_means, 2.5)
                ci_hi = np.percentile(boot_means, 97.5)
                if obs_mean >= 0:
                    p = 2 * max(np.mean(boot_means <= 0),
                                1.0 / n_bootstrap)
                else:
                    p = 2 * max(np.mean(boot_means >= 0),
                                1.0 / n_bootstrap)
                p = min(p, 1.0)
                sig = ''
                if p < 0.001:  sig = ' ***'
                elif p < 0.01: sig = ' **'
                elif p < 0.05: sig = ' *'
                print(f'    {k:<12} delta={obs_mean:+.2f}  '
                      f'95%CI=[{ci_lo:+.2f}, {ci_hi:+.2f}]  '
                      f'p={p:.4f}{sig}')
        print()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'best_shadow_iou': self.best_shadow_iou,
            'best_f1': self.best_f1,
            'epochs_without_improvement': self.epochs_without_improvement,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'args': vars(self.args),
        }
        if is_best:
            torch.save(ckpt,
                        os.path.join(self.output_dir, 'checkpoint_best.pth'))
            print(f'  Best checkpoint saved.')
        if epoch % self.args.save_freq == 0:
            torch.save(ckpt, os.path.join(
                self.output_dir, f'checkpoint_epoch_{epoch}.pth'))

    def _load_checkpoint(self, path):
        print(f'Loading checkpoint: {path}')
        ckpt = torch.load(path, map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_miou = ckpt.get('best_miou', 0.0)
        self.best_shadow_iou = ckpt.get('best_shadow_iou', 0.0)
        self.best_f1 = ckpt.get('best_f1', 0.0)
        self.epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses = ckpt.get('val_losses', [])
        self.train_metrics_history = ckpt.get(
            'train_metrics_history',
            {k: [] for k in self.train_metrics_history})
        self.val_metrics_history = ckpt.get(
            'val_metrics_history',
            {k: [] for k in self.val_metrics_history})
        metric_label = ("Tolerant mIOU" if self.args.eval_boundary_tolerant
                        else "Strict mIOU")
        print(f'  Resumed from epoch {ckpt["epoch"]}  '
              f'(best {metric_label}={self.best_miou:.2f}%)')


# ======================================================================
# Main
# ======================================================================

def main():
    args = get_args()

    active = []
    if args.use_disentangle: active.append('C1')
    if args.use_vib:         active.append('C2')
    if args.use_feat_aug:    active.append('C3')
    if args.use_skip_filter: active.append('SF')
    print(f'\nDDIB components: {", ".join(active) if active else "NONE"}')
    if args.use_disentangle:
        print(f'  lambda_hsic={args.lambda_hsic}  '
              f'lambda_domain={args.lambda_domain}')
    if args.use_vib:
        print(f'  lambda_kl={args.lambda_kl}')
    if args.use_skip_filter:
        print(f'  skip_reduction={args.skip_reduction}  '
              f'skip_kl_weight={args.skip_kl_weight}')

    trainer = TrainerDDIB(args)

    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()