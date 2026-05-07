"""
Training script for DINOv3 + DDIB Shadow Detection.

Decision metrics (best checkpoint, early stopping) are based on
Tolerant mIOU when --eval_boundary_tolerant is enabled, so noisy
boundary pixels don't cause premature stopping or bad checkpoint picks.

Usage examples:

  # Full DDIB (all 3 components) — LOCO fold 0
  python train_dinov3_ddib.py \
      --mode loco --fold_id 0 \
      --base_data_root /path/to/data --resolution highres \
      --weights_path /path/to/dinov3_vits16.pth \
      --use_disentangle --use_vib --use_feat_aug \
      --lambda_hsic 0.1 --lambda_domain 0.01 --lambda_kl 0.001

  # Ablation: C3 only (feature augmentation)
  python train_dinov3_ddib.py \
      --mode loco --fold_id 0 \
      --base_data_root /path/to/data --resolution highres \
      --weights_path /path/to/dinov3_vits16.pth \
      --use_feat_aug

  # Ablation: C1 + C2 (no feature augmentation)
  python train_dinov3_ddib.py \
      --mode loco --fold_id 0 \
      --base_data_root /path/to/data --resolution highres \
      --weights_path /path/to/dinov3_vits16.pth \
      --use_disentangle --use_vib \
      --lambda_hsic 0.1 --lambda_domain 0.01 --lambda_kl 0.001
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

from dinov3_model_ddib import DINOv3ShadowDetectorDDIB
from data.dataset_ddib import get_dataloaders_ddib
from utils.losses import CrossEntropyLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization import (
    plot_loss_curves,
    plot_metrics_curves,
)
from utils.evaluation_detailed import DetailedEvaluator


# ======================================================================
# Arguments
# ======================================================================

def get_args():
    p = argparse.ArgumentParser(description='Train DINOv3 + DDIB')

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

    # -- Backbone --
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--model_name', type=str, default='dinov3_vits16',
                   choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'])
    p.add_argument('--weights_path', type=str, default=None)
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--frozen_stages', type=int, default=-1)

    # -- DDIB component toggles --
    p.add_argument('--use_disentangle', action='store_true', default=False,
                   help='Enable DDIB Component 1 (feature disentanglement)')
    p.add_argument('--use_vib', action='store_true', default=False,
                   help='Enable DDIB Component 2 (VIB)')
    p.add_argument('--use_feat_aug', action='store_true', default=False,
                   help='Enable DDIB Component 3 (feature augmentation)')

    # -- DDIB loss weights --
    p.add_argument('--lambda_hsic', type=float, default=0.1,
                   help='Weight for HSIC independence loss (C1)')
    p.add_argument('--lambda_domain', type=float, default=0.01,
                   help='Weight for domain classification loss (C1)')
    p.add_argument('--lambda_kl', type=float, default=0.001,
                   help='Weight for VIB KL divergence loss (C2)')

    # -- DDIB hyper-parameters --
    p.add_argument('--hsic_samples', type=int, default=1024)
    p.add_argument('--vib_beta_base', type=float, default=0.001)
    p.add_argument('--vib_beta_scale', type=float, default=0.01)
    p.add_argument('--aug_sigma_style', type=float, default=0.5)
    p.add_argument('--aug_sigma_shift', type=float, default=0.3)
    p.add_argument('--aug_p_aug', type=float, default=0.5)
    p.add_argument('--aug_p_mix', type=float, default=0.3)

    # -- Training --
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--min_lr', type=float, default=1e-6)

    # -- FDA --
    p.add_argument('--use_fda', action='store_true')
    p.add_argument('--fda_target_root', type=str, default=None)
    p.add_argument('--fda_L', type=float, default=0.01)

    # -- Checkpoint / logging --
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--save_freq', type=int, default=5)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--eval_boundary_tolerant', action='store_true')
    p.add_argument('--early_stopping_patience', type=int, default=15)

    # -- Comparison baselines (no hardcoded defaults — passed from shell) --
    p.add_argument('--comparison_inference_dir', type=str, default=None,
                   help='Directory with existing inference results (upper/loco)')
    p.add_argument('--comparison_data_root', type=str, default=None,
                   help='Root directory with ground truth data')

    return p.parse_args()


# ======================================================================
# LR schedule
# ======================================================================

class CosineWarmupScheduler:
    """Cosine LR with linear warmup. Purely epoch-based (not metric-gated)."""
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
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ======================================================================
# Per-image metric functions
# (same methodology as analyze_inference_results.py / statistical_analysis.py)
# ======================================================================

_TOLERANCE_KERNEL_CACHE = {}


def _get_tolerance_kernel(tolerance):
    """Cached morphological kernel."""
    if tolerance not in _TOLERANCE_KERNEL_CACHE:
        _TOLERANCE_KERNEL_CACHE[tolerance] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (tolerance * 2 + 1, tolerance * 2 + 1))
    return _TOLERANCE_KERNEL_CACHE[tolerance]


def _compute_strict_metrics(pred, gt):
    """
    Strict per-pixel metrics for a single image.
    Args: pred, gt — uint8 arrays {0,1} of shape [H,W].
    """
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
        'OA':         float(oa * 100),
        'Precision':  float(precision * 100),
        'Recall':     float(recall * 100),
        'F1':         float(f1 * 100),
        'BER':        float(ber * 100),
        'mIOU':       float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100),
    }


def _compute_tolerant_metrics(pred, gt, tolerance=5):
    """
    Boundary-tolerant metrics: ±K px don't-care zone around GT boundaries.
    Pixels in the band are excluded from TP/FP/TN/FN counts entirely.
    """
    kernel   = _get_tolerance_kernel(tolerance)
    gt_uint8 = gt.astype(np.uint8)

    eroded  = cv2.erode(gt_uint8, kernel)
    dilated = cv2.dilate(gt_uint8, kernel)
    band    = (dilated - eroded) > 0
    valid   = ~band

    p = pred[valid]
    g = gt[valid]

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

    total = tp + tn + fp + fn
    oa = (tp + tn) / (total + 1e-10)

    shadow_err    = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber = (shadow_err + nonshadow_err) / 2

    return {
        'OA':         float(oa * 100),
        'Precision':  float(precision * 100),
        'Recall':     float(recall * 100),
        'F1':         float(f1 * 100),
        'BER':        float(ber * 100),
        'mIOU':       float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100),
    }


def _average_metrics(metrics_list):
    """Macro-average a list of per-image metric dicts."""
    if not metrics_list:
        return {k: 0.0 for k in
                ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']}
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

        # ---- Output dir ----
        ddib_tag = ''
        if args.use_disentangle: ddib_tag += '_C1'
        if args.use_vib:         ddib_tag += '_C2'
        if args.use_feat_aug:    ddib_tag += '_C3'
        if not ddib_tag:         ddib_tag = '_noDDIB'

        if args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            fda_suffix = '_fda' if args.use_fda else ''
            exp_name = (f'dinov3_ddib{ddib_tag}{fda_suffix}'
                        f'_loco_holdout_{test_city}_{args.resolution}_1')
        elif args.mode == 'all':
            fda_suffix = '_fda' if args.use_fda else ''
            exp_name = f'dinov3_ddib{ddib_tag}{fda_suffix}_all_{args.resolution}_1'
        else:
            city = args.data_root.rstrip('/').split('/')[-2]
            res  = args.data_root.rstrip('/').split('/')[-1]
            fda_suffix = '_fda' if args.use_fda else ''
            exp_name = f'dinov3_ddib{ddib_tag}{fda_suffix}_{city}_{res}_1'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

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
            fda_target_root=args.fda_target_root, fda_L=args.fda_L)
        self.dataloaders = loaders
        num_domains = loaders['num_domains']
        print(f'Train: {len(loaders["train"].dataset)}  |  '
              f'Val: {len(loaders["val"].dataset)}  |  '
              f'Test: {len(loaders["test"].dataset)}')

        # ---- Model ----
        print('\nBuilding model …')
        self.model = DINOv3ShadowDetectorDDIB(
            num_classes=args.num_classes,
            model_name=args.model_name,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            use_disentangle=args.use_disentangle,
            use_vib=args.use_vib,
            use_feat_aug=args.use_feat_aug,
            num_domains=num_domains,
            hsic_samples=args.hsic_samples,
            vib_beta_base=args.vib_beta_base,
            vib_beta_scale=args.vib_beta_scale,
            aug_sigma_style=args.aug_sigma_style,
            aug_sigma_shift=args.aug_sigma_shift,
            aug_p_aug=args.aug_p_aug,
            aug_p_mix=args.aug_p_mix,
        ).to(self.device)

        # ---- Loss, optimiser, scheduler ----
        self.criterion = CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, betas=(0.9, 0.999))
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, args.warmup_epochs, args.epochs,
            args.lr, args.min_lr)

        # ------------------------------------------------------------------
        # Decision metric: tolerant mIOU if boundary-tolerant eval is on,
        # otherwise fall back to strict mIOU.
        # ------------------------------------------------------------------
        self.use_tolerant_for_decisions = args.eval_boundary_tolerant
        if self.use_tolerant_for_decisions:
            print(">>> Decision metric: Tolerant mIOU (±5px boundary excluded)")
        else:
            print(">>> Decision metric: Strict mIOU")

        # ---- Tracking ----
        self.start_epoch = 0
        self.best_miou = 0.0            # decision metric (tolerant or strict)
        self.best_strict_miou = 0.0      # always strict, for logging
        self.best_shadow_iou = 0.0
        self.best_f1 = 0.0
        self.epochs_without_improvement = 0

        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []}
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []}

        if args.eval_boundary_tolerant:
            self.detailed_eval_train = DetailedEvaluator()
            self.detailed_eval_val   = DetailedEvaluator()

        if args.resume:
            self._load_checkpoint(args.resume)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_seg  = 0.0
        epoch_hsic = 0.0
        epoch_dom  = 0.0
        epoch_kl   = 0.0
        metrics = ShadowMetrics()
        n_batches = len(self.dataloaders['train'])
        t0 = time.time()

        for i, batch in enumerate(self.dataloaders['train']):
            images   = batch['image'].to(self.device)
            masks    = batch['mask'].to(self.device)
            int_map  = batch['intensity_map'].to(self.device)
            city_ids = batch['city_id'].to(self.device)

            # Forward
            outputs, ddib_losses = self.model(images, int_map, city_ids)

            # Losses
            seg_loss = self.criterion(outputs, masks)
            total = seg_loss

            hsic_val = ddib_losses.get('hsic_loss', torch.tensor(0.0))
            dom_val  = ddib_losses.get('domain_loss', torch.tensor(0.0))
            kl_val   = ddib_losses.get('kl_loss', torch.tensor(0.0))

            if self.args.use_disentangle:
                total = total + self.args.lambda_hsic * hsic_val
                total = total + self.args.lambda_domain * dom_val
            if self.args.use_vib:
                total = total + self.args.lambda_kl * kl_val

            # Backward
            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()

            # Metrics
            filtered = filter_small_predictions(outputs.detach(), min_pixels=10)
            metrics.update(filtered, masks)

            if self.args.eval_boundary_tolerant:
                preds = torch.argmax(outputs.detach(), dim=1)
                self.detailed_eval_train.update(preds, masks, images)

            epoch_loss += total.item()
            epoch_seg  += seg_loss.item()
            epoch_hsic += hsic_val.item()
            epoch_dom  += dom_val.item()
            epoch_kl   += kl_val.item()

            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(f'  [{i+1}/{n_batches}]  loss={total.item():.4f}  '
                      f'seg={seg_loss.item():.4f}  '
                      f'hsic={hsic_val.item():.5f}  '
                      f'dom={dom_val.item():.4f}  '
                      f'kl={kl_val.item():.6f}')

        epoch_loss /= n_batches
        epoch_seg  /= n_batches
        epoch_hsic /= n_batches
        epoch_dom  /= n_batches
        epoch_kl   /= n_batches

        m = metrics.compute()
        print(f'\nEpoch {epoch} train  ({time.time()-t0:.1f}s)')
        print(f'  loss={epoch_loss:.4f}  seg={epoch_seg:.4f}  '
              f'hsic={epoch_hsic:.5f}  dom={epoch_dom:.4f}  kl={epoch_kl:.6f}')
        print(f'  OA={m["OA"]:.2f}  P={m["Precision"]:.2f}  F1={m["F1"]:.2f}  '
              f'BER={m["BER"]:.2f}  mIOU={m["mIOU"]:.2f}  ShIOU={m["Shadow_IOU"]:.2f}')

        # Tensorboard
        self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
        self.writer.add_scalar('Train/Seg_Loss', epoch_seg, epoch)
        self.writer.add_scalar('Train/HSIC_Loss', epoch_hsic, epoch)
        self.writer.add_scalar('Train/Domain_Loss', epoch_dom, epoch)
        self.writer.add_scalar('Train/KL_Loss', epoch_kl, epoch)
        for k in m:
            self.writer.add_scalar(f'Train/{k}', m[k], epoch)

        self.train_losses.append(epoch_loss)
        for k in self.train_metrics_history:
            self.train_metrics_history[k].append(m[k])

        if self.args.eval_boundary_tolerant:
            dr = self.detailed_eval_train.compute_metrics()
            t5 = dr['boundary_tolerant']['tolerant_5px']
            self.writer.add_scalar('Train/F1_Tolerant', t5['f1'], epoch)
            self.writer.add_scalar('Train/mIOU_Tolerant', t5['iou'], epoch)
            self.detailed_eval_train.reset()
            print(f'  Boundary-Tolerant: F1={t5["f1"]:.2f}  mIOU={t5["iou"]:.2f}')

        return epoch_loss, m

    # ------------------------------------------------------------------
    def validate(self, epoch):
        """
        Validate the model.

        Returns
        -------
        val_loss      : float
        metrics       : dict       (strict metrics from ShadowMetrics)
        decision_miou : float      Tolerant mIOU if boundary-tolerant eval is on,
                                   otherwise strict mIOU.  Used by train() for
                                   best-checkpoint / early-stopping decisions.
        """
        print('\nValidating …')
        self.model.eval()
        val_loss = 0.0
        metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)
                int_map = batch['intensity_map'].to(self.device)

                outputs, _ = self.model(images, int_map)
                val_loss += self.criterion(outputs, masks).item()

                filtered = filter_small_predictions(outputs, min_pixels=10)
                metrics.update(filtered, masks)

                if self.args.eval_boundary_tolerant:
                    preds = torch.argmax(outputs, dim=1)
                    self.detailed_eval_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders['val'])
        m = metrics.compute()
        print(f'Val  loss={val_loss:.4f}')
        print(f'  OA={m["OA"]:.2f}  P={m["Precision"]:.2f}  F1={m["F1"]:.2f}  '
              f'BER={m["BER"]:.2f}  mIOU={m["mIOU"]:.2f}  ShIOU={m["Shadow_IOU"]:.2f}')

        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        for k in m:
            self.writer.add_scalar(f'Val/{k}', m[k], epoch)

        self.val_losses.append(val_loss)
        for k in self.val_metrics_history:
            self.val_metrics_history[k].append(m[k])

        # ------------------------------------------------------------------
        # Determine the decision mIOU (tolerant when available, else strict)
        # ------------------------------------------------------------------
        decision_miou = m['mIOU']               # default: strict

        if self.args.eval_boundary_tolerant:
            dr = self.detailed_eval_val.compute_metrics()
            t5 = dr['boundary_tolerant']['tolerant_5px']
            self.writer.add_scalar('Val/F1_Tolerant', t5['f1'], epoch)
            self.writer.add_scalar('Val/mIOU_Tolerant', t5['iou'], epoch)
            self.detailed_eval_val.reset()
            print(f'  Boundary-Tolerant: F1={t5["f1"]:.2f}  mIOU={t5["iou"]:.2f}')

            # Override decision metric with tolerant value
            decision_miou = t5['iou']

        return val_loss, m, decision_miou

    # ------------------------------------------------------------------
    def train(self):
        print('\n' + '=' * 60)
        print('Starting DDIB training')
        print('=' * 60)

        metric_label = "Tolerant mIOU" if self.use_tolerant_for_decisions else "Strict mIOU"

        for epoch in range(self.start_epoch, self.args.epochs):
            lr = self.scheduler.step(epoch)
            print(f'\n{"="*60}\nEpoch {epoch+1}/{self.args.epochs}  lr={lr:.2e}')

            _, _ = self.train_epoch(epoch + 1)
            _, val_m, decision_miou = self.validate(epoch + 1)

            # ----------------------------------------------------------
            # Best checkpoint & early stopping keyed on decision_miou
            # (tolerant when --eval_boundary_tolerant, else strict)
            # ----------------------------------------------------------
            is_best = False
            if decision_miou > self.best_miou:
                self.best_miou = decision_miou
                is_best = True
                self.epochs_without_improvement = 0
                print(f'  ★ New best {metric_label}: {self.best_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1

            # Also track strict mIOU for logging (always available)
            if val_m['mIOU'] > self.best_strict_miou:
                self.best_strict_miou = val_m['mIOU']
                print(f'  New best Strict mIOU: {self.best_strict_miou:.2f}%')

            if val_m['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_m['Shadow_IOU']
            if val_m['F1'] > self.best_f1:
                self.best_f1 = val_m['F1']

            self._save_checkpoint(epoch + 1, is_best)
            self.writer.add_scalar('Train/LR', lr, epoch + 1)

            # Early stopping check (keyed on decision_miou)
            if (self.args.early_stopping_patience > 0
                    and self.epochs_without_improvement
                    >= self.args.early_stopping_patience):
                print(f'\nEarly stopping — no {metric_label} improvement for '
                      f'{self.args.early_stopping_patience} epochs.')
                break

        print(f'\nTraining finished.')
        print(f'  Best {metric_label}: {self.best_miou:.2f}%')
        print(f'  Best Strict mIOU: {self.best_strict_miou:.2f}%')
        print(f'  Best Shadow_IOU: {self.best_shadow_iou:.2f}%')
        print(f'  Best F1: {self.best_f1:.2f}%')

        plot_loss_curves(self.train_losses, self.val_losses,
                         os.path.join(self.output_dir, 'loss_curves.png'))
        plot_metrics_curves(self.train_metrics_history,
                            self.val_metrics_history,
                            os.path.join(self.output_dir, 'metrics_curves.png'))
        self.writer.close()

    # ------------------------------------------------------------------
    def test(self):
        """
        Test with comparison against upper-bound and LOCO-vanilla baselines.

        Evaluation methodology matches analyze_inference_results.py exactly:
          - Per-image strict metrics (all pixels)
          - Per-image ±5px don't-care zone tolerant metrics
          - Macro-averaged across images
        This ensures DDIB numbers are directly comparable to existing results.
        """
        print('\n' + '=' * 70)
        print('TESTING')
        print('=' * 70)

        best_ckpt = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_ckpt):
            self._load_checkpoint(best_ckpt)
        else:
            print('Warning: best checkpoint not found, using current weights')

        self.model.eval()

        # --- 1. Run model, save predictions, collect per-image metrics ---
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
                preds = torch.argmax(outputs.detach(), dim=1)   # [B, H, W]

                for i, fname in enumerate(batch['filename']):
                    pred_np = preds[i].cpu().numpy().astype(np.uint8)
                    gt_np   = masks[i].cpu().numpy().astype(np.uint8)

                    # Save prediction (0/255 to match existing convention)
                    Image.fromarray(pred_np * 255).save(
                        os.path.join(pred_save_dir, fname))

                    # Per-image metrics
                    ddib_strict_list.append(
                        _compute_strict_metrics(pred_np, gt_np))
                    ddib_tolerant_list.append(
                        _compute_tolerant_metrics(pred_np, gt_np, tolerance=5))

                    all_filenames.append(fname)

        # Macro-average
        ddib_strict   = _average_metrics(ddib_strict_list)
        ddib_tolerant = _average_metrics(ddib_tolerant_list)

        print(f'\nDDIB Results ({len(all_filenames)} images):')
        print(f'  Strict:   OA={ddib_strict["OA"]:.2f}  P={ddib_strict["Precision"]:.2f}  '
              f'R={ddib_strict["Recall"]:.2f}  F1={ddib_strict["F1"]:.2f}  '
              f'BER={ddib_strict["BER"]:.2f}  mIOU={ddib_strict["mIOU"]:.2f}  '
              f'ShIOU={ddib_strict["Shadow_IOU"]:.2f}')
        print(f'  Tolerant: OA={ddib_tolerant["OA"]:.2f}  P={ddib_tolerant["Precision"]:.2f}  '
              f'R={ddib_tolerant["Recall"]:.2f}  F1={ddib_tolerant["F1"]:.2f}  '
              f'BER={ddib_tolerant["BER"]:.2f}  mIOU={ddib_tolerant["mIOU"]:.2f}  '
              f'ShIOU={ddib_tolerant["Shadow_IOU"]:.2f}')

        results = {
            'num_images': len(all_filenames),
            'strict': ddib_strict,
            'tolerant_5px': ddib_tolerant,
        }

        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        # --- 2. Baseline comparison (LOCO mode only) ---
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
        """
        Load upper-bound and LOCO-vanilla predictions, evaluate using
        the same per-image methodology, and print a comparison table
        with recovery ratios and bootstrap significance tests.
        """
        from data.dataset import LOCO_FOLDS

        fold_id   = self.args.fold_id
        test_city = LOCO_FOLDS[fold_id]['test']
        res       = self.args.resolution
        inf_dir   = self.args.comparison_inference_dir
        data_root = self.args.comparison_data_root
        img_size  = self.args.img_size

        gt_dir = os.path.join(data_root, test_city, res, 'test', 'masks')

        # All baselines to compare against — order matters for table display
        baseline_dirs = [
            ('Upper Bound',  os.path.join(inf_dir, 'upper', test_city, res, 'dinov3', 'base')),
            ('LOCO Vanilla', os.path.join(inf_dir, 'loco',  test_city, res, 'dinov3', 'vanilla')),
            ('LOCO FDA',     os.path.join(inf_dir, 'loco',  test_city, res, 'dinov3', 'fda')),
            ('LOCO SegDesic',os.path.join(inf_dir, 'loco',  test_city, res, 'dinov3', 'segdesic')),
            ('LOCO mCL-LC',  os.path.join(inf_dir, 'loco',  test_city, res, 'dinov3', 'mcl')),
        ]

        print('\n' + '=' * 70)
        print('BASELINE COMPARISON')
        print(f'  Test city: {test_city}  |  Resolution: {res}')
        print(f'  Evaluation at {img_size}x{img_size}  |  ±5px dont-care zone')
        print('=' * 70)

        # --- Evaluate baselines ---
        baseline_results = {}
        for label, pred_dir in baseline_dirs:
            if not os.path.isdir(pred_dir):
                print(f'\n  Warning: {label} not found: {pred_dir}')
                continue

            strict_list   = []
            tolerant_list = []
            n_matched = 0

            for fname in filenames:
                pred_path = os.path.join(pred_dir, fname)
                gt_path   = os.path.join(gt_dir, fname)

                if not os.path.exists(pred_path) or not os.path.exists(gt_path):
                    continue

                # Load and resize to model resolution for fair comparison
                pred_np = np.array(
                    Image.open(pred_path).convert('L').resize(
                        (img_size, img_size), Image.NEAREST))
                pred_bin = (pred_np > 127).astype(np.uint8)

                gt_np = np.array(
                    Image.open(gt_path).convert('L').resize(
                        (img_size, img_size), Image.NEAREST))
                gt_bin = (gt_np > 127).astype(np.uint8)

                strict_list.append(_compute_strict_metrics(pred_bin, gt_bin))
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

        # ---- Comparison table: STRICT ----
        self._print_comparison_table(
            'STRICT METRICS (all pixels)',
            baseline_results, ddib_strict,
            metric_type='strict')

        # ---- Comparison table: TOLERANT ----
        self._print_comparison_table(
            'TOLERANT METRICS (±5px dont-care zone)',
            baseline_results, ddib_tolerant,
            metric_type='tolerant')

        # ---- Recovery ratios for key metrics ----
        self._print_recovery_ratios(baseline_results, ddib_strict, ddib_tolerant)

        # ---- Bootstrap significance: DDIB vs each LOCO baseline ----
        for bl_label in ['LOCO Vanilla', 'LOCO FDA', 'LOCO SegDesic', 'LOCO mCL-LC']:
            if bl_label in baseline_results:
                self._print_bootstrap_comparison(
                    baseline_results[bl_label],
                    ddib_strict_list, ddib_tolerant_list,
                    baseline_label=bl_label)

        # ---- Save to JSON ----
        comp = {
            'test_city': test_city,
            'resolution': res,
            'eval_size': img_size,
            'ddib': {
                'strict': ddib_strict,
                'tolerant_5px': ddib_tolerant,
            },
            'baselines': {},
        }
        for label, br in baseline_results.items():
            comp['baselines'][label] = {
                'strict': br['strict'],
                'tolerant_5px': br['tolerant'],
                'n_images': br['n_images'],
            }

        comp_path = os.path.join(self.output_dir, 'comparison_results.json')
        with open(comp_path, 'w') as f:
            json.dump(comp, f, indent=4)
        print(f'\nComparison saved to {comp_path}')

    # ------------------------------------------------------------------
    def _print_comparison_table(self, title, baseline_results,
                                ddib_metrics, metric_type='strict'):
        """Print a formatted comparison table."""
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
                print(f'  {label:<20} {m["OA"]:6.2f} {m["Precision"]:6.2f} '
                      f'{m["Recall"]:6.2f} {m["F1"]:6.2f} {m["BER"]:6.2f} '
                      f'{m["mIOU"]:6.2f} {m["Shadow_IOU"]:6.2f}')

        d = ddib_metrics
        print(f'  {"DDIB (ours)":<20} {d["OA"]:6.2f} {d["Precision"]:6.2f} '
              f'{d["Recall"]:6.2f} {d["F1"]:6.2f} {d["BER"]:6.2f} '
              f'{d["mIOU"]:6.2f} {d["Shadow_IOU"]:6.2f}')

    # ------------------------------------------------------------------
    def _print_recovery_ratios(self, baseline_results,
                               ddib_strict, ddib_tolerant):
        """Print recovery ratios and deltas vs each baseline."""
        if ('Upper Bound' not in baseline_results
                or 'LOCO Vanilla' not in baseline_results):
            return

        print('\n' + '-' * 70)
        print(f'{"RECOVERY RATIOS":^70}')
        print(f'  R = (DDIB - LOCO_Vanilla) / (Upper - LOCO_Vanilla)')
        print(f'  0 = no help, 1 = gap fully closed')
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
                    gap = -gap
                    rec = -rec
                R = rec / gap if abs(gap) > 0.01 else float('nan')
                parts.append(f'{k}={R:.3f}')

            print(f'  {label:<10}  ' + '  '.join(parts))

        # ---- Deltas vs each adaptation method ----
        adapt_methods = ['LOCO FDA', 'LOCO SegDesic', 'LOCO mCL-LC']
        has_any = any(m in baseline_results for m in adapt_methods)
        if not has_any:
            return

        print('\n' + '-' * 70)
        print(f'{"DDIB IMPROVEMENT OVER ADAPTATION METHODS (delta)":^70}')
        print(f'  Positive = DDIB better.  For BER, negative = DDIB better.')
        print('-' * 70)

        for eval_type, ddib_m, label in [
                ('strict', ddib_strict, 'Strict'),
                ('tolerant', ddib_tolerant, 'Tolerant')]:

            print(f'\n  {label}:')
            print(f'    {"Method":<16} {"dF1":>7} {"dmIOU":>7} {"dShIOU":>7} {"dBER":>7}')
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
        """
        Paired bootstrap test: DDIB vs a baseline.
        Quick version (5000 samples) for inline reporting.
        """
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

                # Two-sided p-value with floor to avoid p=0.0000
                if obs_mean >= 0:
                    p = 2 * max(np.mean(boot_means <= 0), 1.0 / n_bootstrap)
                else:
                    p = 2 * max(np.mean(boot_means >= 0), 1.0 / n_bootstrap)
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
            'best_miou': self.best_miou,
            'best_strict_miou': self.best_strict_miou,
            'best_shadow_iou': self.best_shadow_iou,
            'best_f1': self.best_f1,
            'use_tolerant_for_decisions': self.use_tolerant_for_decisions,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'args': vars(self.args),
        }
        torch.save(ckpt, os.path.join(self.output_dir, 'checkpoint_latest.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(self.output_dir, 'checkpoint_best.pth'))
        if epoch % self.args.save_freq == 0:
            torch.save(ckpt, os.path.join(
                self.output_dir, f'checkpoint_epoch_{epoch}.pth'))

    def _load_checkpoint(self, path):
        print(f'Loading checkpoint: {path}')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_miou = ckpt.get('best_miou', 0.0)
        self.best_strict_miou = ckpt.get('best_strict_miou', 0.0)
        self.best_shadow_iou = ckpt.get('best_shadow_iou', 0.0)
        self.best_f1 = ckpt.get('best_f1', 0.0)
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses = ckpt.get('val_losses', [])
        self.train_metrics_history = ckpt.get('train_metrics_history',
            {k: [] for k in self.train_metrics_history})
        self.val_metrics_history = ckpt.get('val_metrics_history',
            {k: [] for k in self.val_metrics_history})

        metric_label = "Tolerant" if self.use_tolerant_for_decisions else "Strict"
        print(f'  Resumed from epoch {ckpt["epoch"]}  '
              f'(best {metric_label} mIOU={self.best_miou:.2f}%, '
              f'best Strict mIOU={self.best_strict_miou:.2f}%)')


# ======================================================================
# Main
# ======================================================================

def main():
    args = get_args()

    # Print active DDIB components
    active = []
    if args.use_disentangle: active.append('C1')
    if args.use_vib:         active.append('C2')
    if args.use_feat_aug:    active.append('C3')
    print(f'\nDDIB components: {", ".join(active) if active else "NONE"}')
    if args.use_disentangle:
        print(f'  lambda_hsic={args.lambda_hsic}  lambda_domain={args.lambda_domain}')
    if args.use_vib:
        print(f'  lambda_kl={args.lambda_kl}  beta_base={args.vib_beta_base}  '
              f'beta_scale={args.vib_beta_scale}')

    trainer = TrainerDDIB(args)

    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()