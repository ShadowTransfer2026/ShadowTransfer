"""
Dataset loader for DDIB training.

Extends the base ShadowDataset to additionally return:
  - city_id:       integer domain label derived from the source directory path
  - intensity_map: single-channel grayscale intensity computed BEFORE
                   ImageNet normalisation (range [0, 1])

These are consumed by the DDIB module:
  • city_id       → C1 domain classifier, C3 cross-domain mixing
  • intensity_map → C2 intensity-adaptive VIB beta

All other behaviour (augmentation, FDA, geo-metadata) is inherited
from the original ShadowDataset.
"""

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Re-use the base classes and constants from the existing dataset module
from data.dataset import (
    ShadowDataset,
    UnlabeledDataset,
    LOCO_FOLDS,
)

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

class ShadowDatasetDDIB(ShadowDataset):
    """
    Shadow dataset that additionally yields *city_id* and *intensity_map*.

    City IDs are assigned in the order that distinct cities appear across
    the supplied ``root_dir`` list (e.g. first city → 0, second → 1).
    """

    def __init__(self, root_dir, split='train', img_size=384, augment=False,
                 samples_per_dir=None, random_seed=42, selected_filenames=None,
                 use_fda=False, fda_target_root=None, fda_L=0.01,
                 geo_metadata_path=None):
        # Let the base class do all the heavy lifting
        super().__init__(
            root_dir, split, img_size, augment,
            samples_per_dir, random_seed, selected_filenames,
            use_fda, fda_target_root, fda_L,
            geo_metadata_path,
        )

        # ----- build per-image city_id mapping -----
        if isinstance(root_dir, str):
            root_dir = [root_dir]

        # Assign a numeric id to each unique city encountered
        self._city_name_to_id = {}
        for rd in root_dir:
            cname = _city_from_path(rd)
            if cname not in self._city_name_to_id:
                self._city_name_to_id[cname] = len(self._city_name_to_id)

        # Map every image index → city_id by checking which root_dir it
        # came from (mirrors the ordering in ShadowDataset.__init__).
        self._city_ids = []
        idx = 0
        for rd in root_dir:
            img_dir = os.path.join(rd, split, 'images')
            if not os.path.exists(img_dir):
                continue
            n_files = len([f for f in os.listdir(img_dir)
                           if f.endswith(('.png', '.jpg', '.jpeg',
                                         '.tif', '.tiff'))])
            cname = _city_from_path(rd)
            cid   = self._city_name_to_id[cname]
            self._city_ids.extend([cid] * n_files)
            idx += n_files

        # If sampling was applied in the base class, the stored list of
        # img_files may be shorter than the full list.  We need the same
        # filtering here.  The simplest robust approach: match on the
        # stored img_paths rather than relying on indices.
        #
        # Build a lookup  full_img_path → city_id
        self._path_to_city = {}
        idx = 0
        for rd in root_dir:
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
        self._city_id_array = []
        for p in self.img_paths:
            self._city_id_array.append(self._path_to_city.get(p, 0))

        # Resize transform for intensity map (to match img_size)
        self._intensity_resize = transforms.Resize(
            (img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR)

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
        # ---- Intensity map (BEFORE any augmentation / FDA / normalisation) ----
        img_path = self.img_paths[idx]
        raw_image = Image.open(img_path).convert('RGB')

        # Grayscale: 0.299 R + 0.587 G + 0.114 B  (BT.601 luminance)
        raw_np = np.array(raw_image).astype(np.float32)        # [H_orig, W_orig, 3]
        intensity = (0.299 * raw_np[:, :, 0]
                     + 0.587 * raw_np[:, :, 1]
                     + 0.114 * raw_np[:, :, 2])                 # [H_orig, W_orig]
        # Scale to [0, 1]
        intensity = intensity / 255.0

        # Convert to PIL for resizing, then to tensor
        intensity_pil = Image.fromarray(
            (intensity * 255).astype(np.uint8), mode='L')
        intensity_pil = self._intensity_resize(intensity_pil)
        intensity_tensor = transforms.ToTensor()(intensity_pil)  # [1, H, W] in [0, 1]

        # ---- Base class item (image, mask, filename, …) ----
        item = super().__getitem__(idx)

        # ---- City id ----
        item['city_id'] = torch.tensor(
            self._city_id_array[idx], dtype=torch.long)
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
        geo_metadata_path=None):
    """
    Create train / val / test dataloaders for DDIB training.

    Mirrors the interface of ``data.dataset.get_dataloaders`` but uses
    ``ShadowDatasetDDIB`` for train and val sets (which return city_id
    and intensity_map).  The test set uses the base ``ShadowDataset``
    because city_id is not needed at inference (intensity_map is still
    available via the DDIB dataset for diagnostic use).

    Returns:
        dict  {'train': …, 'val': …, 'test': …}
        plus  'num_domains': int  (number of training cities)
    """
    from data.dataset import Dataset as BaseDataset   # plain wrapper

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
        train_paths = [os.path.join(base_data_root, c, resolution) for c in cities]
        val_paths   = train_paths
        test_paths  = train_paths
        # Balance across cities
        sample_dir = os.path.join(train_paths[0], 'train', 'images')
        if os.path.exists(sample_dir):
            n = len([f for f in os.listdir(sample_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            samples_per_dir = n // len(train_paths)

    elif mode == 'loco':
        if base_data_root is None or resolution is None or fold_id is None:
            raise ValueError('base_data_root, resolution, fold_id required')
        fold = LOCO_FOLDS[fold_id]
        train_cities = fold['train']
        test_city    = fold['test']
        train_paths = [os.path.join(base_data_root, c, resolution) for c in train_cities]
        val_paths   = train_paths
        test_paths  = [os.path.join(base_data_root, test_city, resolution)]
        # Balance across cities
        sample_dir = os.path.join(train_paths[0], 'train', 'images')
        if os.path.exists(sample_dir):
            n = len([f for f in os.listdir(sample_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            samples_per_dir = n // len(train_cities)
    else:
        raise ValueError(f'Invalid mode: {mode}')

    # ---- Datasets ----
    train_ds = ShadowDatasetDDIB(
        root_dir=train_paths, split='train', img_size=img_size,
        augment=True,
        samples_per_dir=samples_per_dir if mode in ('loco', 'all') else None,
        use_fda=use_fda, fda_target_root=fda_target_root, fda_L=fda_L,
        geo_metadata_path=geo_metadata_path,
    )

    val_ds = ShadowDatasetDDIB(
        root_dir=val_paths, split='val', img_size=img_size,
        augment=False,
        samples_per_dir=samples_per_dir if mode in ('loco', 'all') else None,
        geo_metadata_path=geo_metadata_path,
    )

    # Test set — DDIB dataset so intensity_map is available for diagnostics
    test_ds = ShadowDatasetDDIB(
        root_dir=test_paths, split='test', img_size=img_size,
        augment=False,
        geo_metadata_path=geo_metadata_path,
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

    print(f'\nDDIB dataloaders created:')
    print(f'  Train: {len(train_ds)} samples  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}')
    print(f'  num_domains = {num_domains}')

    return {
        'train': train_loader,
        'val':   val_loader,
        'test':  test_loader,
        'num_domains': num_domains,
    }