"""
SIB-Extended Shadow Dataset for OGLANet cross-location transfer learning.

Extends the base ShadowDataset to add:
  - city_id extraction (chicago=0, miami=1, phoenix=2)
  - intensity_map computation (grayscale 0-1, BEFORE ImageNet normalization)
  - Optional 4th contrast channel (RMS contrast)
  - FDA (Fourier Domain Adaptation) at data level
  - LOCO (Leave-One-City-Out) fold management
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import cv2
from PIL import Image

# ── Base dataset import (adjust path to your project structure) ──────────────
from data.dataset import ShadowDataset


# ── Constants ────────────────────────────────────────────────────────────────
CITY_MAP = {"chicago": 0, "miami": 1, "phoenix": 2}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_city_id(root_dir: str) -> int:
    """Extract city id from directory path. E.g. '.../chicago/...' → 0."""
    root_lower = root_dir.lower()
    for city, cid in CITY_MAP.items():
        if city in root_lower:
            return cid
    raise ValueError(
        f"Cannot determine city from path '{root_dir}'. "
        f"Expected one of {list(CITY_MAP.keys())} in the path."
    )


def compute_intensity_map(image_rgb: np.ndarray) -> torch.Tensor:
    """
    Compute grayscale intensity map in [0, 1] from uint8 RGB image.
    Must be called BEFORE ImageNet normalization.

    Args:
        image_rgb: np.ndarray of shape (H, W, 3), dtype uint8, range [0, 255]

    Returns:
        torch.Tensor of shape (1, H, W), float32, range [0, 1]
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)  # (H, W), uint8
    intensity = gray.astype(np.float32) / 255.0
    return torch.from_numpy(intensity).unsqueeze(0)  # (1, H, W)


def compute_rms_contrast(image_rgb: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Compute local RMS contrast channel.

    Args:
        image_rgb: (H, W, 3) uint8
        window_size: local window for variance computation

    Returns:
        contrast: (H, W) float32 in [0, 1]
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ksize = (window_size, window_size)
    mu = cv2.blur(gray, ksize)
    mu_sq = cv2.blur(gray ** 2, ksize)
    variance = np.clip(mu_sq - mu ** 2, 0, None)
    contrast = np.sqrt(variance)
    # Normalize to [0, 1]
    cmax = contrast.max()
    if cmax > 0:
        contrast = contrast / cmax
    return contrast


def fda_transfer(source_img: np.ndarray, target_img: np.ndarray,
                 L: float = 0.01) -> np.ndarray:
    """
    Fourier Domain Adaptation: swap low-frequency amplitude of source with target.

    Args:
        source_img: (H, W, 3) float32 [0, 1]
        target_img: (H, W, 3) float32 [0, 1]
        L: fraction of low-frequency band to swap (0.005–0.01 typical)

    Returns:
        adapted: (H, W, 3) float32 [0, 1]
    """
    h, w, c = source_img.shape
    # Determine low-frequency window size
    b_h = int(np.floor(h * L))
    b_w = int(np.floor(w * L))
    if b_h == 0 or b_w == 0:
        return source_img  # L too small for this resolution

    adapted = np.zeros_like(source_img)
    for ch in range(c):
        # FFT
        f_src = np.fft.fft2(source_img[:, :, ch])
        f_tgt = np.fft.fft2(target_img[:, :, ch])
        # Shift zero-frequency to center
        f_src_shift = np.fft.fftshift(f_src)
        f_tgt_shift = np.fft.fftshift(f_tgt)
        # Extract amplitudes and phases
        amp_src = np.abs(f_src_shift)
        pha_src = np.angle(f_src_shift)
        amp_tgt = np.abs(f_tgt_shift)
        # Swap low-freq amplitude
        cy, cx = h // 2, w // 2
        amp_src[cy - b_h:cy + b_h, cx - b_w:cx + b_w] = \
            amp_tgt[cy - b_h:cy + b_h, cx - b_w:cx + b_w]
        # Reconstruct
        f_new = amp_src * np.exp(1j * pha_src)
        f_new = np.fft.ifftshift(f_new)
        adapted[:, :, ch] = np.real(np.fft.ifft2(f_new))

    return np.clip(adapted, 0, 1)


# ── SIB Dataset ──────────────────────────────────────────────────────────────

class ShadowDatasetSIB(ShadowDataset):
    """
    Extends ShadowDataset with SIB-specific returns.

    Inherits the base class's proper split-aware path handling:
        root_dir / {train,val,test} / images / *.png
        root_dir / {train,val,test} / masks  / *.png

    Additional returns per sample:
        - city_id:       int tensor
        - intensity_map: (1, H, W) float tensor in [0, 1]
        - contrast:      (1, H, W) float tensor in [0, 1]  (if use_contrast=True)

    Optionally applies FDA at data level.
    """

    def __init__(self, root_dirs, split='train', img_size=384,
                 use_contrast=True, contrast_window=15,
                 fda_target_dataset=None, fda_L=0.01,
                 **kwargs):
        """
        Args:
            root_dir:    City root, e.g. '.../chicago'
            split:       'train', 'val', or 'test'
            resolution:  'highres' or 'midres'
            use_contrast: Whether to compute 4th RMS contrast channel
            contrast_window: Window size for RMS contrast
            fda_target_dataset: Another ShadowDatasetSIB to draw FDA targets from
            fda_L:       FDA low-frequency band fraction
            **kwargs:    Passed to base ShadowDataset
        """
        super().__init__(root_dirs, split=split, img_size=img_size,
                         **kwargs)
        # root_dirs can be str or list; extract city from first path
        _path = root_dirs[0] if isinstance(root_dirs, list) else root_dirs
        self.city_id = _extract_city_id(_path)
        self.use_contrast = use_contrast
        self.contrast_window = contrast_window
        self.fda_target_dataset = fda_target_dataset
        self.fda_L = fda_L

    def __getitem__(self, idx):
        """
        Returns:
            sample dict with keys:
                'image':         (C, H, W) tensor, ImageNet-normalized (C=3 or 4)
                'mask':          (H, W) long tensor, class labels
                'city_id':       scalar long tensor
                'intensity_map': (1, H, W) float tensor [0, 1]
                'filename':      str
        """
        # Get base sample — this handles split paths correctly
        sample = super().__getitem__(idx)
        # sample['image'] is already ImageNet-normalized (3, H, W) tensor
        # sample['mask'] is (H, W) long tensor

        # We need the RAW image (before normalization) for intensity + contrast
        # Re-read the image file
        img_path = self.img_paths[idx]
        raw_img = np.array(Image.open(img_path).convert("RGB"))  # (H, W, 3) uint8

        # Resize to match the base transform output size
        h, w = sample['mask'].shape[-2:]
        if raw_img.shape[0] != h or raw_img.shape[1] != w:
            raw_img = cv2.resize(raw_img, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Intensity map (before normalization) ──
        intensity_map = compute_intensity_map(raw_img)  # (1, H, W)

        # ── FDA (optional, applied to raw image before normalization) ──
        if self.fda_target_dataset is not None and self.training:
            tgt_idx = np.random.randint(len(self.fda_target_dataset))
            tgt_path = self.fda_target_dataset.img_paths[tgt_idx]
            tgt_img = np.array(Image.open(tgt_path).convert("RGB"))
            if tgt_img.shape[0] != h or tgt_img.shape[1] != w:
                tgt_img = cv2.resize(tgt_img, (w, h), interpolation=cv2.INTER_LINEAR)

            # FDA on float [0,1]
            src_float = raw_img.astype(np.float32) / 255.0
            tgt_float = tgt_img.astype(np.float32) / 255.0
            adapted = fda_transfer(src_float, tgt_float, L=self.fda_L)

            # Re-apply ImageNet normalization to adapted image
            adapted_tensor = torch.from_numpy(adapted).permute(2, 0, 1).float()
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            sample['image'] = (adapted_tensor - mean) / std

        # ── Contrast channel (optional) ──
        if self.use_contrast:
            contrast = compute_rms_contrast(raw_img, self.contrast_window)
            contrast_tensor = torch.from_numpy(contrast).unsqueeze(0).float()
            # Append as 4th channel to image
            sample['image'] = torch.cat([sample['image'], contrast_tensor], dim=0)

        # ── Add SIB-specific fields ──
        sample['city_id'] = torch.tensor(self.city_id, dtype=torch.long)
        sample['intensity_map'] = intensity_map
        if 'filename' not in sample:
            sample['filename'] = os.path.basename(img_path)

        return sample

    @property
    def training(self):
        """Check if dataset is in training split."""
        return self.split == 'train'


# ── LOCO Fold Factory ────────────────────────────────────────────────────────

def get_dataloaders_sib(
    data_root: str,
    test_city: str,
    resolution: str = "highres",
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 384,
    use_contrast: bool = True,
    use_fda: bool = False,
    fda_L: float = 0.01,
    **dataset_kwargs
) -> dict:
    """
    Create LOCO dataloaders for a given test city.

    In LOCO (Leave-One-City-Out):
        - Train on all splits from other cities
        - Val on val split from other cities
        - Test on all splits from the held-out city

    Args:
        data_root:   Root containing city dirs, e.g. '.../shadow_data'
        test_city:   Held-out city name ('chicago', 'miami', 'phoenix')
        resolution:  'highres' or 'midres'
        batch_size:  Batch size for all loaders
        num_workers: DataLoader workers
        use_contrast: 4th contrast channel
        use_fda:     Enable FDA augmentation
        fda_L:       FDA band fraction
        **dataset_kwargs: Extra args for ShadowDatasetSIB

    Returns:
        dict with keys 'train_loader', 'val_loader', 'test_loader',
        'train_dataset', 'val_dataset', 'test_dataset'
    """
    all_cities = ["chicago", "miami", "phoenix"]
    train_cities = [c for c in all_cities if c != test_city]

    # ── Build training datasets (train split from non-test cities) ──
    train_datasets = []
    for city in train_cities:
        city_root = os.path.join(data_root, city, resolution)
        ds = ShadowDatasetSIB(
            city_root, split="train", img_size=img_size,
            use_contrast=use_contrast, fda_L=fda_L,
            **dataset_kwargs
        )
        train_datasets.append(ds)

    # ── FDA: set target datasets (each trains on the other's style) ──
    if use_fda and len(train_datasets) >= 2:
        # Cross-city FDA: city A targets city B and vice versa
        train_datasets[0].fda_target_dataset = train_datasets[1]
        train_datasets[1].fda_target_dataset = train_datasets[0]

    train_dataset = ConcatDataset(train_datasets)

    # ── Validation datasets (val split from non-test cities) ──
    val_datasets = []
    for city in train_cities:
        city_root = os.path.join(data_root, city, resolution)
        ds = ShadowDatasetSIB(
            city_root, split="val", img_size=img_size,
            use_contrast=use_contrast,
            **dataset_kwargs
        )
        val_datasets.append(ds)
    val_dataset = ConcatDataset(val_datasets)

    # ── Test dataset (test split from held-out city, 0-shot) ──
    test_root = os.path.join(data_root, test_city, resolution)
    test_dataset = ShadowDatasetSIB(
        test_root, split="test", img_size=img_size,
        use_contrast=use_contrast,
        **dataset_kwargs
    )

    # ── DataLoaders ──
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }