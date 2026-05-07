"""
Visualization utilities for MAMNet MCL training.

Extends utils/visualization.py to handle:
  - MCL-specific contrastive loss components (FL, SL, LC)
  - MAMNetMCL dict model output in best/worst visualizations

plot_loss_curves_mcl
    Overview panel (all available losses, shared y-axis) +
    individual component panels (each at its own y-axis scale so
    small contrastive losses are readable).
    Total loss is shown ONLY in the overview panel.
    Val loss is shown in the overview and in the Main CE panel (val is
    main CE in eval mode; there is no val counterpart for Aux/FL/SL/LC).

save_best_worst_visualizations_mcl
    Same logic as visualization.py's save_best_worst_visualizations but
    calls model(images, return_features=False) and reads outputs['main'],
    since MAMNetMCL always returns a dict.

plot_metrics_curves is imported unchanged from visualization.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# Re-export plot_metrics_curves unchanged
from utils.visualization import plot_metrics_curves  # noqa: F401

# --------------------------------------------------------------------------
# Colour palette (shared with visualization.py for consistency)
# --------------------------------------------------------------------------
_C_TRAIN_TOTAL = '#2E86AB'
_C_VAL_TOTAL   = '#A23B72'
_C_SEG_TRAIN   = '#F18F01'
_C_SEG_VAL     = '#C73E1D'
_C_AUX_TRAIN   = '#6A994E'
_C_FL_TRAIN    = '#8338EC'
_C_SL_TRAIN    = '#3A86FF'
_C_LC_TRAIN    = '#FF006E'

MAX_COLS = 3   # max individual-panel columns

matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.titlesize':  13,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


def _style_ax(ax, title, ylabel):
    ax.set_title(title, fontweight='bold', pad=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls='--', lw=0.6)
    ax.legend(fontsize=8, framealpha=0.85)


def _any_nonzero(lst):
    """Return True if the list is non-empty and contains at least one > 1e-9."""
    return lst is not None and len(lst) > 0 and any(v > 1e-9 for v in lst)


# --------------------------------------------------------------------------
# Loss curves — MCL version
# --------------------------------------------------------------------------

def plot_loss_curves_mcl(
    train_losses,
    val_losses,
    save_path,
    train_seg_losses=None,
    train_aux_losses=None,
    train_fl_losses=None,
    train_sl_losses=None,
    train_lc_losses=None,
):
    """
    Generate and save a comprehensive loss figure for MCL training.

    Layout
    ------
    Row 0 — Full-width overview
        All available loss series plotted on a *shared* y-axis so their
        relative magnitudes are visible at a glance.  Total train loss and
        val loss are always shown.

    Row 1+ — Individual component panels (one per component)
        Each panel shows a *single* loss component on its *own* y-axis so
        fine-grained decreases (e.g. tiny LC values) are clearly visible.
        **Total loss is NOT shown in any individual panel.**

        Panels generated (when data is non-zero):
          • Main (CE) Seg Loss   — train + val (val = main CE in eval mode)
          • Auxiliary Loss       — train only
          • Feature-Level (FL)   — train only
          • Semantic-Level (SL)  — train only
          • Local Consistency (LC) — train only

    Args
    ----
    train_losses      : list[float]  total training loss per epoch (required)
    val_losses        : list[float]  validation (main CE) loss per epoch (required)
    save_path         : str
    train_seg_losses  : list[float]  main CE train loss per epoch (optional)
    train_aux_losses  : list[float]  weighted aux train loss per epoch (optional)
    train_fl_losses   : list[float]  feature-level contrastive loss (optional)
    train_sl_losses   : list[float]  semantic-level contrastive loss (optional)
    train_lc_losses   : list[float]  local-consistency loss (optional)
    """
    epochs = list(range(1, len(train_losses) + 1))

    # ---- Decide which individual panels to generate --------------------
    # Panel spec: (title, ylabel, [(values, label, colour, linestyle, marker), ...])
    # Only panels with at least one non-zero series are included.
    individual_panels = []

    # Main CE — always show if data available (val_losses is its val proxy)
    seg_series = []
    if _any_nonzero(train_seg_losses) and len(train_seg_losses) == len(epochs):
        seg_series.append((train_seg_losses, 'Train Seg CE', _C_SEG_TRAIN, '-', 'o'))
    if _any_nonzero(val_losses) and len(val_losses) == len(epochs):
        seg_series.append((val_losses, 'Val (CE)', _C_SEG_VAL, '--', 's'))
    if seg_series:
        individual_panels.append({
            'title': 'Main (CE) Seg Loss',
            'ylabel': 'Loss',
            'series': seg_series,
        })

    # Auxiliary
    if _any_nonzero(train_aux_losses) and len(train_aux_losses) == len(epochs):
        individual_panels.append({
            'title': 'Auxiliary Loss',
            'ylabel': 'Loss',
            'series': [(train_aux_losses, 'Train Aux', _C_AUX_TRAIN, '-', '^')],
        })

    # Feature-level contrastive
    if _any_nonzero(train_fl_losses) and len(train_fl_losses) == len(epochs):
        individual_panels.append({
            'title': 'Feature-Level Contrastive (FL)',
            'ylabel': 'Loss',
            'series': [(train_fl_losses, 'Train FL', _C_FL_TRAIN, '-', 'D')],
        })

    # Semantic-level contrastive
    if _any_nonzero(train_sl_losses) and len(train_sl_losses) == len(epochs):
        individual_panels.append({
            'title': 'Semantic-Level Contrastive (SL)',
            'ylabel': 'Loss',
            'series': [(train_sl_losses, 'Train SL', _C_SL_TRAIN, '-', 'v')],
        })

    # Local consistency
    if _any_nonzero(train_lc_losses) and len(train_lc_losses) == len(epochs):
        individual_panels.append({
            'title': 'Local Consistency (LC)',
            'ylabel': 'Loss',
            'series': [(train_lc_losses, 'Train LC', _C_LC_TRAIN, '-', 'P')],
        })

    # ---- Figure layout -------------------------------------------------
    n_ind = len(individual_panels)
    n_rows_ind = max(1, (n_ind + MAX_COLS - 1) // MAX_COLS) if n_ind > 0 else 0

    fig_w = max(10, min(5.5 * max(n_ind, 1), 20))
    fig_h = 4.2 + 3.8 * n_rows_ind

    if n_ind > 0:
        fig   = plt.figure(figsize=(fig_w, fig_h))
        outer = gridspec.GridSpec(
            2, 1, figure=fig, hspace=0.55,
            height_ratios=[3.8, 3.8 * n_rows_ind])
        ax_ov = fig.add_subplot(outer[0])
    else:
        fig, ax_ov = plt.subplots(figsize=(10, 4.2))

    # ---- Row 0: Overview -----------------------------------------------
    ax_ov.plot(epochs, train_losses, '-', lw=2, color=_C_TRAIN_TOTAL,
               label='Train total', marker='o', ms=3, mfc='white', mew=1.2)
    ax_ov.plot(epochs, val_losses, '--', lw=1.8, color=_C_VAL_TOTAL,
               label='Val (main CE)', marker='s', ms=3, mfc='white', mew=1.2)
    if _any_nonzero(train_seg_losses) and len(train_seg_losses) == len(epochs):
        ax_ov.plot(epochs, train_seg_losses, '-', lw=1.4, color=_C_SEG_TRAIN,
                   alpha=0.75, label='Train seg CE')
    if _any_nonzero(train_aux_losses) and len(train_aux_losses) == len(epochs):
        ax_ov.plot(epochs, train_aux_losses, '-', lw=1.4, color=_C_AUX_TRAIN,
                   alpha=0.75, label='Train aux')
    if _any_nonzero(train_fl_losses) and len(train_fl_losses) == len(epochs):
        ax_ov.plot(epochs, train_fl_losses, '-', lw=1.4, color=_C_FL_TRAIN,
                   alpha=0.75, label='Train FL')
    if _any_nonzero(train_sl_losses) and len(train_sl_losses) == len(epochs):
        ax_ov.plot(epochs, train_sl_losses, '-', lw=1.4, color=_C_SL_TRAIN,
                   alpha=0.75, label='Train SL')
    if _any_nonzero(train_lc_losses) and len(train_lc_losses) == len(epochs):
        ax_ov.plot(epochs, train_lc_losses, '-', lw=1.4, color=_C_LC_TRAIN,
                   alpha=0.75, label='Train LC')

    ax_ov.set_title('Overview — All Losses (shared y-axis)', fontweight='bold', pad=6)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss')
    ax_ov.legend(fontsize=8, framealpha=0.88,
                 ncol=min(4, 2 + n_ind))
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ---- Rows 1+: Individual component panels --------------------------
    if n_ind > 0:
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows_ind, MAX_COLS, subplot_spec=outer[1],
            hspace=0.55, wspace=0.45)

        for pidx, panel in enumerate(individual_panels):
            r, c = divmod(pidx, MAX_COLS)
            ax   = fig.add_subplot(inner[r, c])

            for vals, lbl, col, ls, mk in panel['series']:
                ep = list(range(1, len(vals) + 1))
                ax.plot(ep, vals, ls=ls, lw=1.8, color=col, label=lbl,
                        marker=mk, ms=3.5, mfc='white', mew=1.2, alpha=0.9)

            _style_ax(ax, panel['title'], panel['ylabel'])

        # Hide unused subplot slots
        for pidx in range(n_ind, n_rows_ind * MAX_COLS):
            r, c = divmod(pidx, MAX_COLS)
            fig.add_subplot(inner[r, c]).set_visible(False)

    fig.suptitle('MAMNet-MCL — Training Loss Curves',
                 fontweight='bold', fontsize=13, y=1.005)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curves saved → {save_path}')


# --------------------------------------------------------------------------
# Best / worst visualizations — MCL version (handles dict model output)
# --------------------------------------------------------------------------

def save_best_worst_visualizations_mcl(model, dataloader, device, output_dir,
                                        num_images=10):
    """
    Save best and worst predicted samples ranked by per-image shadow IoU.

    Identical to visualization.py's save_best_worst_visualizations EXCEPT:
      - Calls model(images, return_features=False) instead of model(images)
      - Accesses outputs['main'] for predictions, since MAMNetMCL always
        returns a dict.  The original function called torch.argmax on the raw
        dict, which raises a TypeError.

    Each row: Input | GT mask | Predicted mask | Overlay (G=TP R=FP B=FN)

    Args
    ----
    model       : trained MAMNetMCL (set to eval mode inside)
    dataloader  : test DataLoader
    device      : torch.device
    output_dir  : directory to save images
    num_images  : how many best + worst samples to save (each)
    """
    model.eval()
    results = []  # list of (iou, image_np, gt_np, pred_np)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)

            # CHANGE: MAMNetMCL always returns a dict — access ['main']
            outputs = model(images, return_features=False)
            preds   = torch.argmax(outputs['main'], dim=1)

            for i in range(images.shape[0]):
                pred_np = preds[i].cpu().numpy().astype(np.uint8)
                gt_np   = masks[i].cpu().numpy().astype(np.uint8)

                tp  = float(np.logical_and(pred_np == 1, gt_np == 1).sum())
                fp  = float(np.logical_and(pred_np == 1, gt_np == 0).sum())
                fn  = float(np.logical_and(pred_np == 0, gt_np == 1).sum())
                iou = tp / (tp + fp + fn + 1e-7)

                # Denormalise — handle 3-channel or 4-channel (RGB + contrast)
                img_t = images[i].cpu().numpy()         # [C, H, W]
                img_t = np.transpose(img_t, (1, 2, 0))  # [H, W, C]
                if img_t.shape[2] >= 3:
                    img_rgb = img_t[:, :, :3] * std + mean
                    img_rgb = np.clip(img_rgb, 0, 1)
                else:
                    img_rgb = np.repeat(img_t[:, :, :1], 3, axis=2)

                results.append((iou, img_rgb, gt_np, pred_np))

    if not results:
        print('No results to visualize.')
        return

    results.sort(key=lambda x: x[0])
    worst = results[:num_images]
    best  = results[-num_images:]

    for label, subset in [('worst', worst), ('best', best)]:
        n = len(subset)
        fig, axes = plt.subplots(n, 4, figsize=(16, 3.5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, (iou, img_rgb, gt_np, pred_np) in enumerate(subset):
            # Overlay: green=TP, red=FP, blue=FN
            overlay  = img_rgb.copy()
            tp_mask  = np.logical_and(pred_np == 1, gt_np == 1)
            fp_mask  = np.logical_and(pred_np == 1, gt_np == 0)
            fn_mask  = np.logical_and(pred_np == 0, gt_np == 1)
            overlay[tp_mask] = overlay[tp_mask] * 0.4 + np.array([0,   0.8, 0  ]) * 0.6
            overlay[fp_mask] = overlay[fp_mask] * 0.4 + np.array([0.9, 0,   0  ]) * 0.6
            overlay[fn_mask] = overlay[fn_mask] * 0.4 + np.array([0,   0,   0.9]) * 0.6

            axes[row, 0].imshow(img_rgb)
            axes[row, 0].set_title(f'Input  (IoU={iou:.3f})', fontsize=9)
            axes[row, 1].imshow(gt_np,   cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title('GT Mask', fontsize=9)
            axes[row, 2].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
            axes[row, 2].set_title('Predicted', fontsize=9)
            axes[row, 3].imshow(np.clip(overlay, 0, 1))
            axes[row, 3].set_title('Overlay (G=TP R=FP B=FN)', fontsize=9)

            for ax in axes[row]:
                ax.axis('off')

        fig.suptitle(f'MAMNet-MCL — {label.capitalize()} Predictions',
                     fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'{label}_predictions.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'{label.capitalize()} predictions saved → {out_path}')