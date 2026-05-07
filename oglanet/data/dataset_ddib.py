"""
Dataset loader for OGLANet + DDIB training.

Extends ShadowDatasetEnhanced to additionally return:
  - city_id:       integer domain label derived from the source directory path
  - intensity_map: single-channel grayscale intensity computed BEFORE
                   ImageNet normalisation (range [0, 1])

These are consumed by the DDIB module:
  • city_id       → C1 domain classifier, C3 cross-domain mixing
  • intensity_map → C2 intensity-adaptive VIB beta

Supports use_contrast (4-channel RGBC) via ShadowDatasetEnhanced.
"""

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from data.dataset_enhanced import ShadowDatasetEnhanced
from data.dataset import LOCO_FOLDS

# Cities that appear in directory paths — used for auto-detection
_KNOWN_CITIES = ['chicago', 'miami', 'phoenix']


def _city_from_path(path):
    """Extract city name from a directory path (case-insensitive)."""
    path_lower = path.lower()
    for city in _KNOWN_CITIES:
        if city in path_lower:
            return city
    return 'unknown'


# ======================================================================
# Dataset class
# ======================================================================

class ShadowDatasetDDIB(ShadowDatasetEnhanced):
    """
    Shadow dataset that additionally yields *city_id* and *intensity_map*.

    Inherits from ShadowDatasetEnhanced so that use_contrast (4-ch RGBC)
    is handled transparently.  City IDs are assigned in the order that
    distinct cities appear across the supplied root_dirs list.
    """

    def __init__(self, root_dirs, split='train', img_size=384, augment=False,
                 samples_per_dir=None, random_seed=42, selected_filenames=None,
                 use_fda=False, fda_target_root=None, fda_L=0.01,
                 geo_metadata_path=None, use_contrast=False):

        # Let the base class handle everything (images, masks, contrast, FDA…)
        super().__init__(
            root_dirs, split, img_size, augment,
            samples_per_dir, random_seed, selected_filenames,
            use_fda, fda_target_root, fda_L,
            geo_metadata_path, use_contrast,
        )

        # Normalise root_dirs to list
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        # ----- Build per-image city_id mapping -----
        self._city_name_to_id = {}
        for rd in root_dirs:
            cname = _city_from_path(rd)
            if cname not in self._city_name_to_id:
                self._city_name_to_id[cname] = len(self._city_name_to_id)

        # Build  full_img_path → city_id  lookup
        self._path_to_city = {}
        for rd in root_dirs:
            img_dir = os.path.join(rd, split, 'images')
            if not os.path.exists(img_dir):
                continue
            files = sorted([f for f in os.listdir(img_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg',
                                          '.tif', '.tiff'))])
            cname = _city_from_path(rd)
            cid   = self._city_name_to_id[cname]
            for f in files:
                self._path_to_city[os.path.join(img_dir, f)] = cid

        # Build the final per-index city_id array aligned to self.img_paths
        # (which may have been filtered / sampled by the base class)
        self._city_id_array = []
        for p in self.img_paths:
            self._city_id_array.append(self._path_to_city.get(p, 0))

        # Resize transform for intensity map
        self._intensity_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BILINEAR)

        print(f'  DDIB dataset — city mapping: {self._city_name_to_id}')
        print(f'  DDIB dataset — num_domains = {len(self._city_name_to_id)}')

    # ------------------------------------------------------------------
    @property
    def num_domains(self):
        """Number of distinct cities / domains."""
        return len(self._city_name_to_id)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Returns everything from the base class **plus**:
            city_id:       int64 scalar tensor
            intensity_map: float32 tensor [1, H, W] in [0, 1]
        """
        # ---- Intensity map (BEFORE any augmentation / FDA / normalisation) ---
        img_path = self.img_paths[idx]
        raw_image = Image.open(img_path).convert('RGB')

        # Grayscale via BT.601 luminance
        raw_np = np.array(raw_image).astype(np.float32)
        intensity = (0.299 * raw_np[:, :, 0]
                     + 0.587 * raw_np[:, :, 1]
                     + 0.114 * raw_np[:, :, 2])
        intensity = intensity / 255.0

        # Convert to PIL for resizing, then to tensor
        intensity_pil = Image.fromarray(
            (intensity * 255).astype(np.uint8), mode='L')
        intensity_pil = self._intensity_resize(intensity_pil)
        intensity_tensor = transforms.ToTensor()(intensity_pil)  # [1, H, W]

        # ---- Base class item (image, mask, filename, geo, …) ----
        item = super().__getitem__(idx)

        # ---- Additions ----
        item['city_id']       = torch.tensor(self._city_id_array[idx],
                                             dtype=torch.long)
        item['intensity_map'] = intensity_tensor

        return item


# ======================================================================
# Dataloader factory
# ======================================================================

def get_dataloaders_ddib(
        data_root=None, base_data_root=None, mode='single',
        cities=None, resolution=None, fold_id=None,
        batch_size=8, num_workers=1, img_size=384,
        use_fda=False, fda_target_root=None, fda_L=0.01,
        geo_metadata_path=None,
        use_contrast=False):
    """
    Create train / val / test dataloaders for OGLANet + DDIB.

    Mirrors ``data.dataset.get_dataloaders`` but uses ShadowDatasetDDIB
    which returns city_id and intensity_map in addition to the usual
    image / mask / filename / geo fields.

    Returns:
        dict  {'train', 'val', 'test', 'num_domains'}
    """

    # ---- Determine paths ----
    samples_per_dir = None

    if mode == 'single':
        if data_root is None:
            raise ValueError('data_root required for single mode')
        train_paths = [data_root]
        val_paths   = [data_root]
        test_paths  = [data_root]

    elif mode == 'all':
        if base_data_root is None or resolution is None:
            raise ValueError('base_data_root and resolution required')
        if cities is None:
            cities = ['chicago', 'miami', 'phoenix']
        train_paths = [os.path.join(base_data_root, c, resolution)
                       for c in cities]
        val_paths   = train_paths
        test_paths  = train_paths
        # Balance across cities
        sample_dir = os.path.join(train_paths[0], 'train', 'images')
        if os.path.exists(sample_dir):
            n = len([f for f in os.listdir(sample_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg',
                                    '.tif', '.tiff'))])
            samples_per_dir = n // len(train_paths)

    elif mode == 'loco':
        if base_data_root is None or resolution is None or fold_id is None:
            raise ValueError('base_data_root, resolution, fold_id required')
        fold = LOCO_FOLDS[fold_id]
        train_cities = fold['train']
        test_city    = fold['test']
        train_paths = [os.path.join(base_data_root, c, resolution)
                       for c in train_cities]
        val_paths   = train_paths
        test_paths  = [os.path.join(base_data_root, test_city, resolution)]
        # Balance across cities
        sample_dir = os.path.join(train_paths[0], 'train', 'images')
        if os.path.exists(sample_dir):
            n = len([f for f in os.listdir(sample_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg',
                                    '.tif', '.tiff'))])
            samples_per_dir = n // len(train_cities)
    else:
        raise ValueError(f'Invalid mode: {mode}')

    # ---- Datasets (all use ShadowDatasetDDIB) ----
    train_ds = ShadowDatasetDDIB(
        root_dirs=train_paths, split='train', img_size=img_size,
        augment=True,
        samples_per_dir=samples_per_dir if mode in ('loco', 'all') else None,
        use_fda=use_fda, fda_target_root=fda_target_root, fda_L=fda_L,
        geo_metadata_path=geo_metadata_path,
        use_contrast=use_contrast,
    )

    val_ds = ShadowDatasetDDIB(
        root_dirs=val_paths, split='val', img_size=img_size,
        augment=False,
        samples_per_dir=samples_per_dir if mode in ('loco', 'all') else None,
        geo_metadata_path=geo_metadata_path,
        use_contrast=use_contrast,
    )

    test_ds = ShadowDatasetDDIB(
        root_dirs=test_paths, split='test', img_size=img_size,
        augment=False,
        geo_metadata_path=geo_metadata_path,
        use_contrast=use_contrast,
    )

    num_domains = train_ds.num_domains

    # ---- Loaders ----
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    print(f'\nDDIB dataloaders created (OGLANet):')
    print(f'  Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  '
          f'Test: {len(test_ds)}')
    print(f'  num_domains = {num_domains}')
    print(f'  use_contrast = {use_contrast}')

    return {
        'train':       train_loader,
        'val':         val_loader,
        'test':        test_loader,
        'num_domains': num_domains,
    }