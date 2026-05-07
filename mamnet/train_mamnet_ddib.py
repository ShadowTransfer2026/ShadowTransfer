"""
Training script for MAMNet + SIB (Spectral Information Bottleneck).

Supports three modes:
  loco:   Leave-One-City-Out (train on 2 cities, test on holdout)
  all:    Train on all cities of a given resolution
  single: Train on a single data_root

Features:
  - All SIB components independently toggleable for ablation (M1-M10)
  - VIB warmup: linear ramp from 0→1 over first 10% of epochs
  - Boundary-tolerant evaluation (±5px don't-care zone)
  - Per-band KL loss tracking (LL / LH / HL / HH from Haar decomp)
  - Comprehensive loss curves: total + task + per-band KL
  - Prediction image saving + per-image strict/tolerant metrics
  - Baseline comparison → comparison_results.json
  - Best/Worst prediction visualizations
  - Early stopping with patience

Usage:
    python train_mamnet_sib.py \
        --mode loco \
        --base_data_root /path/to/data \
        --resolution highres \
        --fold_id 0 \
        --use_haar --use_vib --use_content_aug --adaptive_beta \
        --use_fda --fda_L 0.005 \
        --use_contrast --use_sag \
        --output_dir /path/to/output \
        --eval_boundary_tolerant
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')   # headless — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet_sib import build_mamnet_sib
from data.dataset_sib import ShadowDatasetSIB, get_dataloaders_sib
from data.dataset import LOCO_FOLDS
from utils.losses import MAMNetLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions


# ════════════════════════════════════════════════════════════════════════════
# GPU diagnostics
# ════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Train MAMNet + SIB')

    # Mode
    p.add_argument('--mode', type=str, default='loco',
                   choices=['loco', 'all', 'single'])
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution', type=str, default='highres',
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id', type=int, default=0)

    # Training
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--early_stopping_patience', type=int, default=15)
    p.add_argument('--device', type=str, default='cuda')

    # SIB components
    p.add_argument('--use_haar', action='store_true')
    p.add_argument('--use_vib', action='store_true')
    p.add_argument('--use_content_aug', action='store_true')
    p.add_argument('--adaptive_beta', action='store_true')
    p.add_argument('--use_sag', action='store_true')
    p.add_argument('--use_multiscale_sib', action='store_true')

    # SIB hyperparameters
    p.add_argument('--beta_content', type=float, default=1e-3)
    p.add_argument('--beta_edge', type=float, default=1e-5)
    p.add_argument('--noise_scale', type=float, default=0.1)
    p.add_argument('--beta_max_multiplier', type=float, default=3.0)
    p.add_argument('--multiscale_beta_base', type=float, default=1e-4)
    p.add_argument('--vib_warmup_fraction', type=float, default=0.1)

    # Data options
    p.add_argument('--use_contrast', action='store_true')
    p.add_argument('--use_fda', action='store_true')
    p.add_argument('--fda_L', type=float, default=0.01)
    p.add_argument('--fda_target_root', type=str, default=None)

    # Evaluation
    p.add_argument('--eval_boundary_tolerant', action='store_true')
    p.add_argument('--comparison_inference_dir', type=str, default=None)
    p.add_argument('--comparison_data_root', type=str, default=None)

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Per-image metric functions
# ════════════════════════════════════════════════════════════════════════════

_KERNEL_CACHE = {}


def _tolerance_kernel(tol):
    if tol not in _KERNEL_CACHE:
        _KERNEL_CACHE[tol] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    return _KERNEL_CACHE[tol]


def _compute_strict_metrics(pred, gt):
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    precision     = tp / (tp + fp + 1e-10)
    recall        = tp / (tp + fn + 1e-10)
    f1            = 2 * precision * recall / (precision + recall + 1e-10)
    shadow_iou    = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou          = (shadow_iou + nonshadow_iou) / 2
    oa            = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    shadow_err    = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber           = (shadow_err + nonshadow_err) / 2

    return {
        'OA': float(oa * 100), 'Precision': float(precision * 100),
        'Recall': float(recall * 100), 'F1': float(f1 * 100),
        'BER': float(ber * 100), 'mIOU': float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100),
    }


def _compute_tolerant_metrics(pred, gt, tolerance=5):
    kernel   = _tolerance_kernel(tolerance)
    gt_u8    = gt.astype(np.uint8)
    eroded   = cv2.erode(gt_u8, kernel)
    dilated  = cv2.dilate(gt_u8, kernel)
    valid    = ~((dilated - eroded) > 0)

    p  = pred[valid]
    g  = gt[valid]
    tp = np.logical_and(p == 1, g == 1).sum()
    fp = np.logical_and(p == 1, g == 0).sum()
    tn = np.logical_and(p == 0, g == 0).sum()
    fn = np.logical_and(p == 0, g == 1).sum()

    precision     = tp / (tp + fp + 1e-10)
    recall        = tp / (tp + fn + 1e-10)
    f1            = 2 * precision * recall / (precision + recall + 1e-10)
    shadow_iou    = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou          = (shadow_iou + nonshadow_iou) / 2
    oa            = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    shadow_err    = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber           = (shadow_err + nonshadow_err) / 2

    return {
        'OA': float(oa * 100), 'Precision': float(precision * 100),
        'Recall': float(recall * 100), 'F1': float(f1 * 100),
        'BER': float(ber * 100), 'mIOU': float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100),
    }


def _average_metrics(lst):
    if not lst:
        return {k: 0.0 for k in
                ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']}
    keys = ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
    return {k: float(np.mean([m[k] for m in lst])) for k in keys}


# ════════════════════════════════════════════════════════════════════════════
# VIB warmup
# ════════════════════════════════════════════════════════════════════════════

def vib_warmup_weight(epoch, total_epochs, warmup_fraction=0.1):
    warmup_epochs = max(1, int(total_epochs * warmup_fraction))
    if epoch < warmup_epochs:
        return float(epoch) / float(warmup_epochs)
    return 1.0


# ════════════════════════════════════════════════════════════════════════════
# Training loop  — tracks per-band KL losses
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    epoch, total_epochs, vib_warmup_frac=0.1):
    model.train()

    total_loss     = 0.0
    total_task_loss = 0.0
    total_kl_loss  = 0.0
    band_kl_accum  = defaultdict(float)   # per-band KL accumulator
    n_batches      = 0
    metrics        = ShadowMetrics()

    vib_w = vib_warmup_weight(epoch, total_epochs, vib_warmup_frac)

    for batch in dataloader:
        images        = batch['image'].to(device)
        labels        = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)

        optimizer.zero_grad()

        outputs, sib_losses = model(images, intensity_map=intensity_map)

        # Task loss
        task_losses = criterion(outputs, labels)
        task_loss   = task_losses['total']

        # Sum all KL terms, accumulate per-band
        kl_loss = torch.tensor(0.0, device=device)
        for band_key, band_val in sib_losses.items():
            if isinstance(band_val, torch.Tensor):
                kl_loss = kl_loss + band_val
                band_kl_accum[band_key] += band_val.item()

        loss = task_loss + vib_w * kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            filtered = filter_small_predictions(outputs['main'], min_pixels=10)
            metrics.update(filtered, labels)

        total_loss      += loss.item()
        total_task_loss += task_loss.item()
        total_kl_loss   += kl_loss.item()
        n_batches       += 1

    m   = metrics.compute()
    nb  = max(n_batches, 1)
    avg_band_kl = {k: v / nb for k, v in band_kl_accum.items()}

    return {
        'total':      total_loss / nb,
        'task':       total_task_loss / nb,
        'kl':         total_kl_loss / nb,
        'band_kl':    avg_band_kl,       # dict: band → float
        'vib_weight': vib_w,
        'metrics':    m,
    }


# ════════════════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, dataloader, criterion, device, boundary_tolerant=False):
    model.eval()
    metrics         = ShadowMetrics()
    all_preds_np    = []
    all_labels_np   = []
    total_loss      = 0.0
    n_batches       = 0

    for batch in dataloader:
        images        = batch['image'].to(device)
        labels        = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)

        outputs, _ = model(images, intensity_map=intensity_map)
        logits     = outputs if isinstance(outputs, torch.Tensor) else outputs['main']

        total_loss += criterion.criterion(logits, labels).item()

        filtered = filter_small_predictions(logits, min_pixels=10)
        metrics.update(filtered, labels)

        preds_np = logits.argmax(dim=1).cpu().numpy()
        for i in range(preds_np.shape[0]):
            all_preds_np.append(preds_np[i])
            all_labels_np.append(labels[i].cpu().numpy())

        n_batches += 1

    val_loss = total_loss / max(n_batches, 1)
    m        = metrics.compute()

    all_p    = np.concatenate([p.flatten() for p in all_preds_np])
    all_l    = np.concatenate([l.flatten() for l in all_labels_np])
    strict   = _compute_strict_metrics(all_p, all_l)

    result = {
        'loss':           val_loss,
        'shadow_metrics': m,
        'strict':         strict,
    }

    if boundary_tolerant:
        tol_list = [
            _compute_tolerant_metrics(p, g, tolerance=5)
            for p, g in zip(all_preds_np, all_labels_np)
        ]
        result['tolerant_5px'] = _average_metrics(tol_list)

    return result


# ════════════════════════════════════════════════════════════════════════════
# Test + save predictions
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_and_save_predictions(model, loader, device, args, output_dir):
    """
    Run test inference, save prediction PNGs, compute per-image metrics.
    Returns averaged strict + tolerant metrics plus per-image lists.
    """
    model.eval()
    pred_save_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_save_dir, exist_ok=True)

    strict_list   = []
    tolerant_list = []
    all_filenames = []

    for batch_idx, batch in enumerate(loader):
        images        = batch['image'].to(device)
        masks         = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)

        outputs, _ = model(images, intensity_map=intensity_map)
        logits     = outputs if isinstance(outputs, torch.Tensor) else outputs['main']
        preds      = logits.argmax(dim=1)   # (B, H, W)

        for i, fname in enumerate(batch['filename']):
            pred_np = preds[i].cpu().numpy().astype(np.uint8)
            gt_np   = masks[i].cpu().numpy().astype(np.uint8)

            Image.fromarray(pred_np * 255).save(
                os.path.join(pred_save_dir, fname))

            strict_list.append(_compute_strict_metrics(pred_np, gt_np))
            tolerant_list.append(
                _compute_tolerant_metrics(pred_np, gt_np, tolerance=5))
            all_filenames.append(fname)

        if (batch_idx + 1) % 20 == 0:
            print(f'  [test] {len(all_filenames)} images done...')

    strict   = _average_metrics(strict_list)
    tolerant = _average_metrics(tolerant_list)

    print(f'\nTest Results ({len(all_filenames)} images):')
    print(f'  Strict  : OA={strict["OA"]:.2f}  P={strict["Precision"]:.2f}  '
          f'R={strict["Recall"]:.2f}  F1={strict["F1"]:.2f}  '
          f'BER={strict["BER"]:.2f}  mIOU={strict["mIOU"]:.2f}  '
          f'ShIOU={strict["Shadow_IOU"]:.2f}')
    print(f'  Tolerant: OA={tolerant["OA"]:.2f}  P={tolerant["Precision"]:.2f}  '
          f'R={tolerant["Recall"]:.2f}  F1={tolerant["F1"]:.2f}  '
          f'BER={tolerant["BER"]:.2f}  mIOU={tolerant["mIOU"]:.2f}  '
          f'ShIOU={tolerant["Shadow_IOU"]:.2f}')

    test_results = {
        'num_images':   len(all_filenames),
        'strict':       strict,
        'tolerant_5px': tolerant,
    }
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

    return (strict, tolerant, strict_list, tolerant_list, all_filenames)


# ════════════════════════════════════════════════════════════════════════════
# Baseline discovery + comparison
#
# Three-strategy cascade (tried in order, first hit wins per label):
#
#   Strategy 1 — DDIB original structure (mirrors train_mamnet_ddib.py)
#     {comparison_inference_dir}/loco/{city}/{res}/mamnet/{vanilla,fda,...}
#     {comparison_inference_dir}/upper/{city}/{res}/mamnet/base
#   This is the primary source: the raw inference dirs produced by the
#   original LOCO baselines, regardless of whether a DDIB run exists.
#
#   Strategy 2 — SIB / DDIB outputs dir scan
#     Scans the parent of comparison_inference_dir for experiment output
#     dirs (e.g. mamnet_loco_holdout_{city}_{res}_1) that contain a
#     predictions/ folder or a test_results.json.
#
#   Strategy 3 — Donor comparison_results.json
#     Any sibling DDIB experiment whose comparison_results.json already
#     contains populated baselines (piggybacked metric lookup).
# ════════════════════════════════════════════════════════════════════════════

def _find_baseline_experiments(args):
    if not args.comparison_inference_dir:
        return []

    inf_dir   = args.comparison_inference_dir.rstrip('/')
    test_city = getattr(args, 'test_city', None)
    res       = args.resolution
    if test_city is None:
        return []

    # ── Strategy 1: DDIB-style original inference directories ────────────
    # These are the raw prediction folders produced by vanilla / FDA /
    # SegDesic / mCL-LC LOCO baselines — the same paths train_mamnet_ddib
    # uses in _compare_with_baselines.
    ddib_style = [
        ('Upper Bound',   os.path.join(inf_dir, 'upper', test_city, res, 'mamnet', 'base')),
        ('LOCO Vanilla',  os.path.join(inf_dir, 'loco',  test_city, res, 'mamnet', 'vanilla')),
        ('LOCO FDA',      os.path.join(inf_dir, 'loco',  test_city, res, 'mamnet', 'fda')),
        ('LOCO SegDesic', os.path.join(inf_dir, 'loco',  test_city, res, 'mamnet', 'segdesic')),
        ('LOCO mCL-LC',   os.path.join(inf_dir, 'loco',  test_city, res, 'mamnet', 'mcl')),
    ]

    found      = {}   # label → (path, source_type) — dict so later strategies don't overwrite
    found_order = []  # insertion order

    print('  [Strategy 1] Checking DDIB-style original inference paths...')
    for label, pred_dir in ddib_style:
        if os.path.isdir(pred_dir):
            # Check it actually contains image files
            imgs = [f for f in os.listdir(pred_dir)
                    if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))]
            if imgs:
                found[label] = (pred_dir, 'predictions')
                found_order.append(label)
                print(f'    ✓ {label}: {pred_dir} ({len(imgs)} images)')
            else:
                print(f'    ~ {label}: dir exists but empty — {pred_dir}')
        else:
            print(f'    ✗ {label}: not found — {pred_dir}')

    # ── Strategy 2: Outputs-dir scan for experiment subdirs ───────────────
    # Works when comparison_inference_dir is the mamnet outputs root
    # (e.g.  .../data/mamnet/outputs)  or its parent.
    missing = [l for l in
               ['Upper Bound', 'LOCO Vanilla', 'LOCO FDA', 'LOCO SegDesic', 'LOCO mCL-LC']
               if l not in found]

    if missing:
        # Try both inf_dir itself and its parent as the outputs root
        scan_roots = [inf_dir]
        parent_dir = os.path.dirname(inf_dir)
        if os.path.isdir(parent_dir):
            scan_roots.append(parent_dir)

        output_patterns = {
            'Upper Bound':   [f'mamnet_upper_{test_city}_{res}_1',
                              f'mamnet_all_{res}_1'],
            'LOCO Vanilla':  [f'mamnet_loco_holdout_{test_city}_{res}_1',
                              f'mamnet_vanilla_loco_holdout_{test_city}_{res}_1'],
            'LOCO FDA':      [f'mamnet_fda_loco_holdout_{test_city}_{res}_1',
                              f'mamnet_loco_fda_holdout_{test_city}_{res}_1'],
            'LOCO SegDesic': [f'mamnet_segdesic_loco_holdout_{test_city}_{res}_1',
                              f'mamnet_loco_segdesic_holdout_{test_city}_{res}_1'],
            'LOCO mCL-LC':   [f'mamnet_mcl_loco_holdout_{test_city}_{res}_1',
                              f'mamnet_loco_mcl_holdout_{test_city}_{res}_1',
                              f'mamnet_mclc_loco_holdout_{test_city}_{res}_1'],
        }

        print('  [Strategy 2] Scanning outputs dirs for experiment subdirs...')
        for scan_root in scan_roots:
            if not os.path.isdir(scan_root):
                continue
            for label in list(missing):  # copy so we can remove while iterating
                if label not in output_patterns:
                    continue
                for pat in output_patterns[label]:
                    candidate = os.path.join(scan_root, pat)
                    if not os.path.isdir(candidate):
                        continue
                    pred_dir  = os.path.join(candidate, 'predictions')
                    test_json = os.path.join(candidate, 'test_results.json')
                    comp_json = os.path.join(candidate, 'comparison_results.json')
                    if os.path.isdir(pred_dir):
                        found[label] = (pred_dir, 'predictions')
                        found_order.append(label)
                        missing.remove(label)
                        print(f'    ✓ {label}: {pred_dir}')
                        break
                    elif os.path.isfile(test_json):
                        found[label] = (test_json, 'test_results')
                        found_order.append(label)
                        missing.remove(label)
                        print(f'    ✓ {label}: {test_json} (pre-computed)')
                        break
                    elif os.path.isfile(comp_json):
                        found[label] = (comp_json, 'comparison_self')
                        found_order.append(label)
                        missing.remove(label)
                        print(f'    ✓ {label}: {comp_json} (self metrics)')
                        break

    # ── Strategy 3: Donor comparison_results.json ─────────────────────────
    # If we still have missing labels, piggyback from a sibling experiment
    # that already ran comparison and populated baselines{}.
    if missing:
        donor_roots = [inf_dir, os.path.dirname(inf_dir)]
        print('  [Strategy 3] Scanning for donor comparison_results.json...')
        for scan_root in donor_roots:
            if not os.path.isdir(scan_root):
                continue
            for entry in sorted(os.listdir(scan_root)):
                # Prefer DDIB experiments; skip SIB (they're incomplete too)
                if 'sib' in entry.lower() and 'ddib' not in entry.lower():
                    continue
                if test_city not in entry.lower() or res not in entry.lower():
                    continue
                comp_path = os.path.join(scan_root, entry, 'comparison_results.json')
                if not os.path.isfile(comp_path):
                    continue
                try:
                    with open(comp_path) as f:
                        data = json.load(f)
                    baselines = data.get('baselines', {})
                    if baselines:
                        found['_donor_' + entry] = (comp_path, 'donor')
                        found_order.append('_donor_' + entry)
                        print(f'    ✓ Donor {entry}: {len(baselines)} baselines')
                        break
                except (json.JSONDecodeError, OSError):
                    continue
            else:
                continue
            break   # found a donor in this root; stop

    if missing:
        print(f'  Still missing after all strategies: {missing}')

    # Return as ordered list of (label, path, source_type)
    return [(label, found[label][0], found[label][1]) for label in found_order]


def _baseline_metrics_from_predictions(pred_dir, gt_dir, filenames, img_size):
    strict_list   = []
    tolerant_list = []
    n_matched     = 0

    for fname in filenames:
        pred_path = os.path.join(pred_dir, fname)
        gt_path   = os.path.join(gt_dir,   fname)
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue

        pred_np  = np.array(Image.open(pred_path).convert('L').resize(
            (img_size, img_size), Image.NEAREST))
        pred_bin = (pred_np > 127).astype(np.uint8)
        gt_np    = np.array(Image.open(gt_path).convert('L').resize(
            (img_size, img_size), Image.NEAREST))
        gt_bin   = (gt_np > 127).astype(np.uint8)

        strict_list.append(_compute_strict_metrics(pred_bin, gt_bin))
        tolerant_list.append(
            _compute_tolerant_metrics(pred_bin, gt_bin, tolerance=5))
        n_matched += 1

    if n_matched == 0:
        return None
    return {
        'strict':        _average_metrics(strict_list),
        'tolerant_5px':  _average_metrics(tolerant_list),
        'strict_list':   strict_list,
        'tolerant_list': tolerant_list,
        'n_images':      n_matched,
    }


def compare_with_baselines(strict, tolerant, strict_list, tolerant_list,
                           filenames, args, output_dir):
    test_city = getattr(args, 'test_city', 'unknown')
    res       = args.resolution
    img_size  = args.img_size

    gt_dir = None
    if args.comparison_data_root:
        # Try from most-specific to least-specific.
        # Strategy 1 (DDIB style): comparison_data_root is base_data_root,
        #   so city/res are subdirs.
        # Strategy 2 (SIB style): comparison_data_root is already {city}/{res}.
        gt_candidates = [
            os.path.join(args.comparison_data_root, test_city, res, 'test', 'masks'),
            os.path.join(args.comparison_data_root, test_city, res, 'masks'),
            os.path.join(args.comparison_data_root, 'test', 'masks'),
            os.path.join(args.comparison_data_root, 'masks'),
        ]
        for candidate in gt_candidates:
            if os.path.isdir(candidate):
                gt_dir = candidate
                break
        if gt_dir is None:
            print(f'  Warning: GT mask dir not found. Tried:')
            for c in gt_candidates:
                print(f'    {c}')

    print(f'\n{"="*70}')
    print(f'BASELINE COMPARISON')
    print(f'  Test city: {test_city}  |  Resolution: {res}')
    print(f'  GT masks:  {gt_dir}')
    print(f'{"="*70}')

    found_baselines = _find_baseline_experiments(args)
    baseline_results = {}
    donor_baselines  = {}

    for label, path, source_type in found_baselines:
        if source_type == 'predictions' and gt_dir:
            bl = _baseline_metrics_from_predictions(
                path, gt_dir, filenames, img_size)
            if bl:
                baseline_results[label] = bl
                print(f'  {label}: computed from {bl["n_images"]} images')

        elif source_type == 'test_results':
            try:
                with open(path) as f:
                    data = json.load(f)
                bl_entry = {k: data[k] for k in ('strict', 'tolerant_5px')
                             if k in data}
                if bl_entry:
                    baseline_results[label] = bl_entry
                    print(f'  {label}: loaded from test_results.json')
            except (json.JSONDecodeError, OSError) as e:
                print(f'  {label}: failed to load: {e}')

        elif source_type == 'comparison_self':
            try:
                with open(path) as f:
                    data = json.load(f)
                for key in ['sib', 'ddib']:
                    if key in data and 'strict' in data[key]:
                        baseline_results[label] = {
                            'strict':       data[key]['strict'],
                            'tolerant_5px': data[key].get('tolerant_5px',
                                            data[key].get('tolerant', {})),
                        }
                        print(f'  {label}: loaded self metrics')
                        break
            except (json.JSONDecodeError, OSError) as e:
                print(f'  {label}: failed: {e}')

        elif source_type == 'donor':
            try:
                with open(path) as f:
                    data = json.load(f)
                donor_baselines = data.get('baselines', {})
                print(f'  Donor: loaded {len(donor_baselines)} baselines')
            except (json.JSONDecodeError, OSError):
                pass

    for bl_label, bl_data in donor_baselines.items():
        if bl_label not in baseline_results:
            baseline_results[bl_label] = bl_data
            print(f'  {bl_label}: from donor experiment')

    # Print comparison tables
    if baseline_results:
        _print_comparison_table(
            'STRICT METRICS (all pixels)',
            baseline_results, strict, 'strict')
        _print_comparison_table(
            'TOLERANT METRICS (±5 px dont-care zone)',
            baseline_results, tolerant, 'tolerant_5px')
        _print_recovery_ratios(baseline_results, strict, tolerant)
        for bl_label in ['LOCO Vanilla', 'LOCO FDA',
                         'LOCO SegDesic', 'LOCO mCL-LC']:
            if (bl_label in baseline_results
                    and 'strict_list' in baseline_results[bl_label]):
                _print_bootstrap_comparison(
                    baseline_results[bl_label],
                    strict_list, tolerant_list, baseline_label=bl_label)
    else:
        print('\n  No baselines found for comparison.')

    # Save comparison_results.json
    comp = {
        'test_city':  test_city,
        'resolution': res,
        'eval_size':  img_size,
        'sib':  {'strict': strict, 'tolerant_5px': tolerant},
        'ddib': {'strict': strict, 'tolerant_5px': tolerant},
        'baselines': {},
    }
    for label, br in baseline_results.items():
        comp['baselines'][label] = {
            'strict':       br.get('strict', {}),
            'tolerant_5px': br.get('tolerant_5px', br.get('tolerant', {})),
        }
        if 'n_images' in br:
            comp['baselines'][label]['n_images'] = br['n_images']

    comp_path = os.path.join(output_dir, 'comparison_results.json')
    with open(comp_path, 'w') as f:
        json.dump(comp, f, indent=4)
    print(f'\nComparison saved to {comp_path}')
    return comp


def _print_comparison_table(title, baseline_results, sib_metrics, metric_type):
    print(f'\n{"-"*70}')
    print(f'{title:^70}')
    print(f'{"-"*70}')
    print(f'  {"Method":<20} {"OA":>6} {"Prec":>6} {"Rec":>6} '
          f'{"F1":>6} {"BER":>6} {"mIOU":>6} {"ShIOU":>6}')
    print('  ' + '-' * 62)
    for label in ['Upper Bound', 'LOCO Vanilla', 'LOCO FDA',
                  'LOCO SegDesic', 'LOCO mCL-LC']:
        if label not in baseline_results:
            continue
        m = baseline_results[label].get(metric_type, {})
        if not m:
            continue
        print(f'  {label:<20} {m.get("OA",0):6.2f} {m.get("Precision",0):6.2f} '
              f'{m.get("Recall",0):6.2f} {m.get("F1",0):6.2f} '
              f'{m.get("BER",0):6.2f} {m.get("mIOU",0):6.2f} '
              f'{m.get("Shadow_IOU",0):6.2f}')
    d = sib_metrics
    print(f'  {"SIB (ours)":<20} {d["OA"]:6.2f} {d["Precision"]:6.2f} '
          f'{d["Recall"]:6.2f} {d["F1"]:6.2f} {d["BER"]:6.2f} '
          f'{d["mIOU"]:6.2f} {d["Shadow_IOU"]:6.2f}')


def _print_recovery_ratios(baseline_results, sib_strict, sib_tolerant):
    if ('Upper Bound' not in baseline_results
            or 'LOCO Vanilla' not in baseline_results):
        return
    print(f'\n{"-"*70}')
    print(f'{"RECOVERY RATIOS":^70}')
    print(f'  R = (SIB − LOCO_Vanilla) / (Upper − LOCO_Vanilla)')
    print(f'{"-"*70}')
    for eval_type, sib_m, label in [
            ('strict', sib_strict, 'Strict'),
            ('tolerant_5px', sib_tolerant, 'Tolerant')]:
        ub = baseline_results['Upper Bound'].get(eval_type, {})
        lv = baseline_results['LOCO Vanilla'].get(eval_type, {})
        if not ub or not lv:
            continue
        parts = []
        for k in ['F1', 'mIOU', 'Shadow_IOU', 'BER']:
            if k not in ub or k not in lv:
                continue
            gap = ub[k] - lv[k]
            rec = sib_m[k] - lv[k]
            if k == 'BER':
                gap, rec = -gap, -rec
            R = rec / gap if abs(gap) > 0.01 else float('nan')
            parts.append(f'{k}={R:.3f}')
        print(f'  {label:<10}  ' + '  '.join(parts))


def _print_bootstrap_comparison(loco_bl, sib_sl, sib_tl,
                                 baseline_label, n_bootstrap=5000):
    print(f'\n{"-"*70}')
    print(f'{"BOOTSTRAP: SIB vs " + baseline_label + " (n=5000)":^70}')
    print(f'{"-"*70}')
    np.random.seed(42)
    for eval_type, sib_list, label in [
            ('strict_list', sib_sl, 'Strict'),
            ('tolerant_list', sib_tl, 'Tolerant')]:
        loco_list = loco_bl.get(eval_type, [])
        n = min(len(loco_list), len(sib_list))
        if n == 0:
            continue
        print(f'\n  {label}:')
        for k in ['F1', 'mIOU', 'Shadow_IOU']:
            loco_vals  = np.array([m[k] for m in loco_list[:n]])
            sib_vals   = np.array([m[k] for m in sib_list[:n]])
            diff       = sib_vals - loco_vals
            obs_mean   = np.mean(diff)
            boot_means = np.array([
                np.mean(diff[np.random.choice(n, n, replace=True)])
                for _ in range(n_bootstrap)])
            ci_lo = np.percentile(boot_means, 2.5)
            ci_hi = np.percentile(boot_means, 97.5)
            if obs_mean >= 0:
                p_val = 2 * max(np.mean(boot_means <= 0), 1.0 / n_bootstrap)
            else:
                p_val = 2 * max(np.mean(boot_means >= 0), 1.0 / n_bootstrap)
            p_val = min(p_val, 1.0)
            sig = (' ***' if p_val < 0.001 else ' **' if p_val < 0.01
                   else ' *' if p_val < 0.05 else '')
            print(f'    {k:<12} delta={obs_mean:+.2f}  '
                  f'95%CI=[{ci_lo:+.2f}, {ci_hi:+.2f}]  p={p_val:.4f}{sig}')
    print('')


# ════════════════════════════════════════════════════════════════════════════
# Comprehensive loss plotting
# All losses on one figure: total, task, kl_total, + per-band KL from Haar
# ════════════════════════════════════════════════════════════════════════════

# Publication-quality matplotlib defaults
matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman'] + matplotlib.rcParams['font.serif'],
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'figure.titlesize':   13,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# Band display names (best-effort mapping; falls back to raw key)
_BAND_DISPLAY = {
    'kl_ll':           'KL — LL (content)',
    'kl_lh':           'KL — LH (h-edge)',
    'kl_hl':           'KL — HL (v-edge)',
    'kl_hh':           'KL — HH (noise)',
    'kl_content':      'KL — Content (LL)',
    'kl_edge_lh':      'KL — Edge LH',
    'kl_edge_hl':      'KL — Edge HL',
    'kl_noise':        'KL — Noise (HH)',
    'kl_multiscale':   'KL — MultiScale',
    'kl_total':        'KL — total',
}


def _band_label(key):
    return _BAND_DISPLAY.get(key, key.replace('_', ' ').title())


def plot_all_losses(history, output_dir):
    """
    Three-panel loss figure:
      Panel 1: Total loss  (train + val on one axis)
      Panel 2: Task (CE) loss  (train + val)
      Panel 3: All KL components from Haar decomposition (train only,
               one line per band + a dashed 'KL total' line)

    Saves:
      loss_curves.png         — all panels
      loss_kl_detail.png      — Panel 3 only, larger for paper use
    """
    epochs = [h['epoch'] for h in history]

    train_total = [h['train_loss']      for h in history]
    train_task  = [h['train_task_loss'] for h in history]
    train_kl    = [h['train_kl_loss']   for h in history]
    val_total   = [h['val_loss']        for h in history]
    val_strict  = [h.get('val_mIOU', 0) for h in history]

    # Collect all band keys across all epochs
    all_band_keys = sorted({
        k for h in history
        for k in h.get('band_kl', {}).keys()
    })
    band_series = {
        k: [h.get('band_kl', {}).get(k, 0.0) for h in history]
        for k in all_band_keys
    }

    # ── Colour palette ──────────────────────────────────────────────────────
    TRAIN_COL  = '#2E86AB'
    VAL_COL    = '#A23B72'
    TASK_COL   = '#F18F01'
    TASK_V_COL = '#C73E1D'

    BAND_COLS = [
        '#6A994E', '#BC4749', '#FF6B35', '#8338EC',
        '#3A86FF', '#FFBE0B', '#FB5607', '#06D6A0',
    ]

    # ── 3-panel figure ───────────────────────────────────────────────────────
    has_bands = len(all_band_keys) > 0
    n_panels  = 3 if has_bands else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 2:
        axes = list(axes) + [None]

    # Panel 1 — Total loss
    ax0 = axes[0]
    ax0.plot(epochs, train_total, 'o-', lw=2, ms=4, color=TRAIN_COL,
             label='Train total', mfc='white', mew=1.5)
    ax0.plot(epochs, val_total, 's--', lw=2, ms=4, color=VAL_COL,
             label='Val total', mfc='white', mew=1.5)
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Loss')
    ax0.set_title('Total Loss')
    ax0.legend()
    ax0.grid(True, alpha=0.25, ls='--', lw=0.6)

    # Panel 2 — Task (CE) loss
    ax1 = axes[1]
    ax1.plot(epochs, train_task, 'o-', lw=2, ms=4, color=TASK_COL,
             label='Train task (CE)', mfc='white', mew=1.5)
    # Val loss is total (no task/kl split) — use val_total as proxy
    ax1.plot(epochs, val_total, 's--', lw=2, ms=4, color=TASK_V_COL,
             label='Val loss', mfc='white', mew=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Task (CE) Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.25, ls='--', lw=0.6)

    # Panel 3 — Per-band KL losses
    if has_bands and axes[2] is not None:
        ax2 = axes[2]

        # Draw KL total first as a dashed grey reference
        ax2.plot(epochs, train_kl, 'k--', lw=1.5, alpha=0.5,
                 label='KL total', zorder=1)

        for idx, bk in enumerate(all_band_keys):
            col  = BAND_COLS[idx % len(BAND_COLS)]
            vals = band_series[bk]
            ax2.plot(epochs, vals, '-', lw=1.8, ms=3.5, color=col,
                     label=_band_label(bk), alpha=0.9, zorder=2)

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('KL Loss')
        ax2.set_title('Per-Band KL Losses (Haar Decomp.)')
        ax2.legend(fontsize=8, loc='upper right', framealpha=0.8)
        ax2.grid(True, alpha=0.25, ls='--', lw=0.6)

    plt.suptitle('MAMNet+SIB — Training Losses', fontweight='bold', y=1.01)
    plt.tight_layout()
    path_main = os.path.join(output_dir, 'loss_curves.png')
    fig.savefig(path_main, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved loss curves → {path_main}')

    # ── Standalone KL-detail figure (paper-ready) ───────────────────────────
    if has_bands:
        fig2, ax2b = plt.subplots(figsize=(7, 4))
        ax2b.plot(epochs, train_kl, 'k--', lw=2, alpha=0.55,
                  label='KL total (sum)', zorder=1)
        for idx, bk in enumerate(all_band_keys):
            col  = BAND_COLS[idx % len(BAND_COLS)]
            vals = band_series[bk]
            ax2b.plot(epochs, vals, '-o', lw=1.8, ms=3, color=col,
                      label=_band_label(bk), alpha=0.9, mfc='white',
                      mew=1.2, zorder=2)

        ax2b.set_xlabel('Epoch', fontweight='bold')
        ax2b.set_ylabel('KL Loss', fontweight='bold')
        ax2b.set_title('Per-Band KL Losses — Haar Wavelet Decomposition',
                       fontweight='bold')
        ax2b.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax2b.grid(True, alpha=0.25, ls='--', lw=0.6)
        plt.tight_layout()
        path_kl = os.path.join(output_dir, 'loss_kl_detail.png')
        fig2.savefig(path_kl, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f'  Saved KL detail  → {path_kl}')

    # ── Val mIOU curve ───────────────────────────────────────────────────────
    if any(v > 0 for v in val_strict):
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(epochs, val_strict, 'D-', lw=2, ms=5, color='#6A994E',
                 label='Val mIOU (strict)', mfc='white', mew=1.5)
        # Mark best epoch
        best_e = epochs[int(np.argmax(val_strict))]
        best_v = max(val_strict)
        ax3.axvline(x=best_e, color='red', ls=':', lw=1.2, alpha=0.7)
        ax3.scatter([best_e], [best_v], color='red', zorder=5, s=60,
                    label=f'Best (epoch {best_e}, mIOU={best_v:.2f})')
        ax3.set_xlabel('Epoch', fontweight='bold')
        ax3.set_ylabel('mIOU (%)', fontweight='bold')
        ax3.set_title('Validation mIOU', fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.25, ls='--', lw=0.6)
        plt.tight_layout()
        path_miou = os.path.join(output_dir, 'val_miou.png')
        fig3.savefig(path_miou, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f'  Saved val mIOU   → {path_miou}')


# ════════════════════════════════════════════════════════════════════════════
# Best/Worst prediction visualizations
# ════════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _denorm(img_tensor):
    """CHW tensor → HW3 float [0,1] numpy (handles 3-ch or 4-ch)."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img[:, :, :3] * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img, 0, 1)


def save_best_worst_predictions(model, test_loader, device, output_dir,
                                 n_display=10):
    """
    Save best/worst PNG grids and an iou_statistics.json.

    Grid columns: Input | GT overlay | Pred overlay | Probability map
    """
    model.eval()
    all_images    = []
    all_masks_gt  = []
    all_masks_pred = []
    all_probs     = []
    all_ious      = []
    all_filenames = []

    print('\nGenerating best/worst visualizations...')

    with torch.no_grad():
        for batch in test_loader:
            images        = batch['image'].to(device)
            masks         = batch['mask'].to(device)
            intensity_map = batch['intensity_map'].to(device)
            filenames     = batch['filename']

            outputs, _ = model(images, intensity_map=intensity_map)
            logits     = outputs if isinstance(outputs, torch.Tensor) else outputs['main']

            probs_batch  = F.softmax(logits, dim=1)[:, 1, :]
            filtered     = filter_small_predictions(logits, min_pixels=10)
            preds_batch  = filtered.argmax(dim=1)

            for i in range(images.size(0)):
                p  = preds_batch[i]
                g  = masks[i]
                tp = ((p == 1) & (g == 1)).sum().float()
                fp = ((p == 1) & (g == 0)).sum().float()
                fn = ((p == 0) & (g == 1)).sum().float()
                iou = (tp / (tp + fp + fn + 1e-10) * 100).item()

                all_images.append(images[i].cpu())
                all_masks_gt.append(g.cpu())
                all_masks_pred.append(p.cpu())
                all_probs.append(probs_batch[i].cpu())
                all_ious.append(iou)
                all_filenames.append(filenames[i])

    # Filter to images that actually contain shadow
    shadow_idx = [i for i in range(len(all_masks_gt))
                  if all_masks_gt[i].sum() > 0]
    if not shadow_idx:
        print('  Warning: no images with shadow in test set; skipping viz.')
        return

    shadow_ious = [all_ious[i] for i in shadow_idx]
    sorted_pos  = np.argsort(shadow_ious)

    worst_idx = [shadow_idx[p] for p in sorted_pos[:n_display]]
    best_idx  = [shadow_idx[p] for p in sorted_pos[-n_display:][::-1]]

    def _grid(indices, title, fname):
        n   = len(indices)
        fig = plt.figure(figsize=(16, 4 * n))
        gs  = gridspec.GridSpec(n, 4, figure=fig, hspace=0.3, wspace=0.08)

        for row, idx in enumerate(indices):
            img_np    = _denorm(all_images[idx])
            gt_np     = all_masks_gt[idx].numpy()
            pred_np   = all_masks_pred[idx].numpy()
            prob_np   = all_probs[idx].numpy()
            iou_val   = all_ious[idx]

            ax0 = fig.add_subplot(gs[row, 0])
            ax0.imshow(img_np)
            ax0.set_title(f'Image {row+1} | ShIOU={iou_val:.1f}%',
                          fontsize=9, fontweight='bold')
            ax0.axis('off')

            ax1 = fig.add_subplot(gs[row, 1])
            ax1.imshow(img_np)
            ov = np.zeros((*gt_np.shape, 4))
            ov[gt_np == 1] = [0, 1, 0, 0.42]
            ax1.imshow(ov)
            ax1.set_title('GT overlay', fontsize=9, fontweight='bold')
            ax1.axis('off')

            ax2 = fig.add_subplot(gs[row, 2])
            ax2.imshow(img_np)
            ov2 = np.zeros((*pred_np.shape, 4))
            ov2[pred_np == 1] = [1, 0, 0, 0.42]
            ax2.imshow(ov2)
            ax2.set_title('Pred overlay', fontsize=9, fontweight='bold')
            ax2.axis('off')

            ax3 = fig.add_subplot(gs[row, 3])
            im  = ax3.imshow(prob_np, cmap='jet', vmin=0, vmax=1)
            ax3.set_title('Shadow prob', fontsize=9, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=13, fontweight='bold', y=1.002)
        plt.tight_layout()
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {fname}')

    _grid(best_idx,  f'Top {n_display} Best  Predictions (Shadow IoU)',
          'best_predictions.png')
    _grid(worst_idx, f'Top {n_display} Worst Predictions (Shadow IoU)',
          'worst_predictions.png')

    # IoU statistics JSON
    stats = {
        'mean_iou':   float(np.mean(all_ious)),
        'std_iou':    float(np.std(all_ious)),
        'min_iou':    float(np.min(all_ious)),
        'max_iou':    float(np.max(all_ious)),
        'median_iou': float(np.median(all_ious)),
        'best_files':  [all_filenames[i] for i in best_idx],
        'best_ious':   [all_ious[i]      for i in best_idx],
        'worst_files': [all_filenames[i] for i in worst_idx],
        'worst_ious':  [all_ious[i]      for i in worst_idx],
    }
    with open(os.path.join(output_dir, 'iou_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    print(f'  IoU stats  → iou_statistics.json  '
          f'(mean={stats["mean_iou"]:.2f}%  median={stats["median_iou"]:.2f}%)')


# ════════════════════════════════════════════════════════════════════════════
# LOCO fold mapping
# ════════════════════════════════════════════════════════════════════════════

CITY_FOLDS = {
    0: {'holdout': 'phoenix'},
    1: {'holdout': 'miami'},
    2: {'holdout': 'chicago'},
}


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def run_training(args):
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')
    print(f'SIB config: haar={args.use_haar}, vib={args.use_vib}, '
          f'aug={args.use_content_aug}, adaptive_beta={args.adaptive_beta}, '
          f'sag={args.use_sag}, multiscale={args.use_multiscale_sib}')
    print(f'Data: contrast={args.use_contrast}, fda={args.use_fda} '
          f'(L={args.fda_L})')

    os.makedirs(args.output_dir, exist_ok=True)

    # Persist config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Resolve test_city for LOCO mode
    if args.mode == 'loco':
        args.test_city = CITY_FOLDS[args.fold_id]['holdout']
    else:
        args.test_city = None

    # Auto-resolve FDA target root
    fda_target_root = getattr(args, 'fda_target_root', None)
    if args.mode == 'loco' and args.use_fda and not fda_target_root:
        fda_target_root = os.path.join(
            args.base_data_root, args.test_city, args.resolution)
        print(f'FDA target auto-resolved: {fda_target_root}')

    # Build dataloaders
    dataloaders = get_dataloaders_sib(
        data_root=args.data_root,
        base_data_root=args.base_data_root,
        mode=args.mode,
        resolution=args.resolution,
        fold_id=args.fold_id if args.mode == 'loco' else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_fda=args.use_fda,
        fda_target_root=fda_target_root if args.use_fda else None,
        fda_L=args.fda_L,
        use_contrast=args.use_contrast,
    )

    train_loader = dataloaders['train']
    val_loader   = dataloaders['val']
    test_loader  = dataloaders['test']

    print(f'Train: {len(train_loader)} batches | '
          f'Val: {len(val_loader)} | Test: {len(test_loader)}')

    # Model, loss, optimizer, scheduler
    model     = build_mamnet_sib(args).to(device)
    criterion = MAMNetLoss(aux_weight=0.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ──────────────────────────────────────────────────────
    best_metric     = -float('inf')
    patience_counter = 0
    history         = []

    for epoch in range(args.epochs):
        t0 = time.time()

        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args.epochs, args.vib_warmup_fraction)

        val_stats = validate(model, val_loader, criterion, device,
                              boundary_tolerant=args.eval_boundary_tolerant)

        scheduler.step()
        elapsed = time.time() - t0

        # Pick tracking metric
        if args.eval_boundary_tolerant and 'tolerant_5px' in val_stats:
            current_metric = val_stats['tolerant_5px'].get('mIOU', 0.0)
        else:
            current_metric = val_stats['strict'].get('mIOU', 0.0)

        # Logging
        tm = train_stats['metrics']
        bk = train_stats['band_kl']
        print(f'\nEpoch {epoch+1}/{args.epochs} ({elapsed:.1f}s)')
        print(f'  Loss  total={train_stats["total"]:.4f}  '
              f'task={train_stats["task"]:.4f}  '
              f'kl={train_stats["kl"]:.6f}  '
              f'vib_w={train_stats["vib_weight"]:.3f}')
        if bk:
            band_str = '  '.join(
                f'{_band_label(k)}={v:.6f}' for k, v in sorted(bk.items()))
            print(f'  KL bands: {band_str}')
        vs = val_stats['strict']
        print(f'  Val strict:   F1={vs["F1"]:.2f}  '
              f'mIOU={vs["mIOU"]:.2f}  ShIOU={vs["Shadow_IOU"]:.2f}  '
              f'BER={vs["BER"]:.2f}')
        if 'tolerant_5px' in val_stats:
            vt = val_stats['tolerant_5px']
            print(f'  Val tolerant: F1={vt["F1"]:.2f}  '
                  f'mIOU={vt["mIOU"]:.2f}  ShIOU={vt["Shadow_IOU"]:.2f}')
        print(f'  Tracking mIOU: {current_metric:.4f}')

        history.append({
            'epoch':           epoch + 1,
            'train_loss':      train_stats['total'],
            'train_task_loss': train_stats['task'],
            'train_kl_loss':   train_stats['kl'],
            'band_kl':         {k: float(v) for k, v in bk.items()},
            'vib_warmup_weight': train_stats['vib_weight'],
            'val_loss':        val_stats['loss'],
            'val_metrics_strict':   val_stats['strict'],
            'val_metrics_tolerant': val_stats.get('tolerant_5px', {}),
            'val_mIOU':        current_metric,
            'lr':              optimizer.param_groups[0]['lr'],
        })

        # Save best checkpoint
        if current_metric > best_metric:
            best_metric      = current_metric
            patience_counter = 0
            ckpt = {
                'epoch':             epoch + 1,
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric':       best_metric,
                'args':              vars(args),
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'  ★ New best: mIOU={best_metric:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # ── Save training history ──────────────────────────────────────────────
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Save loss plots ────────────────────────────────────────────────────
    print('\nGenerating loss plots...')
    plot_all_losses(history, args.output_dir)

    # ══════════════════════════════════════════════════════════════════════
    # Final test evaluation with best model
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    if args.mode == 'loco':
        print(f'Final Test on {args.test_city} (0-shot LOCO)')
    else:
        print(f'Final Test (mode={args.mode})')
    print(f'{"="*70}')

    ckpt_path = os.path.join(args.output_dir, 'best_model.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'Loaded best model from epoch {ckpt["epoch"]}')

    # ── Predictions + per-image metrics ───────────────────────────────────
    (strict, tolerant,
     strict_list, tolerant_list,
     all_filenames) = test_and_save_predictions(
        model, test_loader, device, args, args.output_dir)

    # ── Baseline comparison ────────────────────────────────────────────────
    if args.mode == 'loco':
        compare_with_baselines(
            strict, tolerant,
            strict_list, tolerant_list,
            all_filenames, args, args.output_dir)
    else:
        comp = {
            'sib':      {'strict': strict, 'tolerant_5px': tolerant},
            'ddib':     {'strict': strict, 'tolerant_5px': tolerant},
            'baselines': {},
        }
        with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(comp, f, indent=4)

    # ── Best / Worst visualizations ────────────────────────────────────────
    save_best_worst_predictions(model, test_loader, device,
                                 args.output_dir, n_display=10)

    print(f'\nDone! Output: {args.output_dir}')
    return {'strict': strict, 'tolerant_5px': tolerant}


if __name__ == '__main__':
    args = parse_args()
    run_training(args)