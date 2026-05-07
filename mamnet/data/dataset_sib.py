"""
Dataset loader for SIB training.

Extends the base ShadowDataset to additionally return:
  - city_id:       integer domain label derived from the source directory path
  - intensity_map: single-channel grayscale intensity computed BEFORE
                   ImageNet normalisation (range [0, 1])

These are consumed by the SIB module:
  • city_id       → ContentAugmentation cross-domain mixing
  • intensity_map → ContentVIB / UniformVIB intensity-adaptive beta

Optionally adds a 4th contrast channel (RGBC) when use_contrast=True.

All other behaviour (augmentation, FDA, geo-metadata) is inherited
from the original ShadowDataset.
"""

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from data.dataset import (
    ShadowDataset,
    UnlabeledDataset,
    LOCO_FOLDS,
)
from data.contrast_utils import add_contrast_channel

_KNOWN_CITIES = ['chicago', 'miami', 'phoenix']


def _city_from_path(path):
    """Extract city name from a directory path (case-insensitive)."""
    path_lower = path.lower()
    for city in _KNOWN_CITIES:
        if city in path_lower:
            return city
    return 'unknown'


class ShadowDatasetSIB(ShadowDataset):
    """
    Shadow dataset that additionally yields *city_id*, *intensity_map*,
    and optionally a 4th contrast channel.

    City IDs are assigned in the order that distinct cities appear across
    the supplied ``root_dir`` list.

    Args (beyond ShadowDataset):
        use_contrast: If True, appends a contrast channel → 4-ch RGBC.
    """

    def __init__(self, root_dir, split='train', img_size=384, augment=False,
                 samples_per_dir=None, random_seed=42, selected_filenames=None,
                 use_fda=False, fda_target_root=None, fda_L=0.01,
                 geo_metadata_path=None,
                 use_contrast=False):
        super().__init__(
            root_dir, split, img_size, augment,
            samples_per_dir, random_seed, selected_filenames,
            use_fda, fda_target_root, fda_L,
            geo_metadata_path,
        )

        self.use_contrast = use_contrast

        if isinstance(root_dir, str):
            root_dir = [root_dir]

        # Build city name → numeric id mapping
        self._city_name_to_id = {}
        for rd in root_dir:
            cname = _city_from_path(rd)
            if cname not in self._city_name_to_id:
                self._city_name_to_id[cname] = len(self._city_name_to_id)

        # Build per-image-path → city_id lookup
        self._path_to_city = {}
        for rd in root_dir:
            img_dir = os.path.join(rd, split, 'images')
            if not os.path.exists(img_dir):
                continue
            files = sorted([f for f in os.listdir(img_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg',
                                          '.tif', '.tiff'))])
            cname = _city_from_path(rd)
            cid = self._city_name_to_id[cname]
            for f in files:
                self._path_to_city[os.path.join(img_dir, f)] = cid

        # Align to self.img_paths (which may be subsampled)
        self._city_id_array = []
        for p in self.img_paths:
            self._city_id_array.append(self._path_to_city.get(p, 0))

        self._intensity_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BILINEAR)

        print(f'  SIB dataset — city mapping: {self._city_name_to_id}')
        print(f'  SIB dataset — num_domains = {len(self._city_name_to_id)}')
        if use_contrast:
            print(f'  SIB dataset — contrast channel: ENABLED (4-ch RGBC)')

    @property
    def num_domains(self):
        return len(self._city_name_to_id)

    def __getitem__(self, idx):
        # ---- Load raw image for intensity & contrast (before any transforms) ----
        img_path = self.img_paths[idx]
        raw_image = Image.open(img_path).convert('RGB')

        # ---- Intensity map (luminance, 0-1) ----
        raw_np = np.array(raw_image).astype(np.float32)
        intensity = (0.299 * raw_np[:, :, 0]
                     + 0.587 * raw_np[:, :, 1]
                     + 0.114 * raw_np[:, :, 2])
        intensity = intensity / 255.0

        intensity_pil = Image.fromarray(
            (intensity * 255).astype(np.uint8), mode='L')
        intensity_pil = self._intensity_resize(intensity_pil)
        intensity_tensor = transforms.ToTensor()(intensity_pil)  # [1, H, W]

        # ---- Get base item from ShadowDataset ----
        # item['image'] is [3, H, W] ImageNet-normalised tensor
        # item['mask']  is [H, W] long tensor
        item = super().__getitem__(idx)

        item['city_id'] = torch.tensor(
            self._city_id_array[idx], dtype=torch.long)
        item['intensity_map'] = intensity_tensor

        # ---- Add contrast channel if requested ----
        # add_contrast_channel() expects HWC uint8 numpy [H, W, 3] and
        # returns [H, W, 4] (RGBC).  item['image'] is already a CHW
        # normalised tensor so we CANNOT pass it directly.
        #
        # Instead: resize the raw image to match tensor spatial dims,
        # compute RGBC from that, extract just the 4th channel, convert
        # to a [1, H, W] float tensor, and concatenate.
        if self.use_contrast:
            raw_resized = raw_image.resize(
                (self.img_size, self.img_size), Image.BILINEAR)
            raw_uint8 = np.array(raw_resized)                   # [H, W, 3] uint8

            rgbc_np = add_contrast_channel(raw_uint8)            # [H, W, 4]
            contrast_ch = rgbc_np[:, :, 3:4].astype(np.float32) / 255.0  # [H, W, 1]

            contrast_tensor = torch.from_numpy(
                contrast_ch).permute(2, 0, 1)                    # [1, H, W]
            item['image'] = torch.cat(
                [item['image'], contrast_tensor], dim=0)         # [4, H, W]

        return item


# ======================================================================
# Dataloader factory
# ======================================================================

def get_dataloaders_sib(
        data_root=None, base_data_root=None, mode='single',
        cities=None, resolution=None, fold_id=None,
        batch_size=8, num_workers=1, img_size=384,
        use_fda=False, fda_target_root=None, fda_L=0.01,
        geo_metadata_path=None,
        use_contrast=False):
    """
    Create train / val / test dataloaders for SIB training.

    Returns:
        dict  {'train': ..., 'val': ..., 'test': ..., 'num_domains': int}
    """
    samples_per_dir = None

    if mode == 'single':
        if data_root is None:
            raise ValueError('data_root required for single mode')
        train_paths = [data_root]
        val_paths = [data_root]
        test_paths = [data_root]

    elif mode == 'all':
        if base_data_root is None or resolution is None:
            raise ValueError('base_data_root and resolution required')
        if cities is None:
            cities = ['chicago', 'miami', 'phoenix']
        train_paths = [os.path.join(base_data_root, c, resolution)
                       for c in cities]
        val_paths = train_paths
        test_paths = train_paths
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
        test_city = fold['test']
        train_paths = [os.path.join(base_data_root, c, resolution)
                       for c in train_cities]
        val_paths = train_paths
        test_paths = [os.path.join(base_data_root, test_city, resolution)]
        sample_dir = os.path.join(train_paths[0], 'train', 'images')
        if os.path.exists(sample_dir):
            n = len([f for f in os.listdir(sample_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg',
                                    '.tif', '.tiff'))])
            samples_per_dir = n // len(train_cities)
    else:
        raise ValueError(f'Invalid mode: {mode}')

    train_ds = ShadowDatasetSIB(
        root_dir=train_paths, split='train', img_size=img_size,
        augment=True,
        samples_per_dir=samples_per_dir if mode in ('loco', 'all') else None,
        use_fda=use_fda, fda_target_root=fda_target_root, fda_L=fda_L,
        geo_metadata_path=geo_metadata_path,
        use_contrast=use_contrast,
    )

    val_ds = ShadowDatasetSIB(
        root_dir=val_paths, split='val', img_size=img_size,
        augment=False,
        samples_per_dir=samples_per_dir if mode in ('loco', 'all') else None,
        geo_metadata_path=geo_metadata_path,
        use_contrast=use_contrast,
    )

    test_ds = ShadowDatasetSIB(
        root_dir=test_paths, split='test', img_size=img_size,
        augment=False,
        geo_metadata_path=geo_metadata_path,
        use_contrast=use_contrast,
    )

    num_domains = train_ds.num_domains

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    print(f'\nSIB dataloaders created:')
    print(f'  Train: {len(train_ds)} samples  |  Val: {len(val_ds)}'
          f'  |  Test: {len(test_ds)}')
    print(f'  num_domains = {num_domains}')

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'num_domains': num_domains,
    }