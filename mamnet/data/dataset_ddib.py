"""
Dataset loader for DDIB training (MAMNet).

Extends ShadowDataset to additionally return:
  - city_id:       integer domain label derived from the source directory path
  - intensity_map: single-channel grayscale intensity computed BEFORE
                   ImageNet normalisation (range [0, 1])
  - (optional) 4-channel RGBC image when use_contrast=True

These are consumed by the DDIB module:
  * city_id       -> C1 domain classifier, C3 cross-domain mixing
  * intensity_map -> C2 intensity-adaptive VIB beta
"""

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from data.dataset import ShadowDataset, LOCO_FOLDS

# Cities that appear in directory paths -- used for auto-detection
_KNOWN_CITIES = ['chicago', 'miami', 'phoenix']


def _city_from_path(path):
    """Extract city name from a directory path (case-insensitive)."""
    path_lower = path.lower()
    for city in _KNOWN_CITIES:
        if city in path_lower:
            return city
    return 'unknown'


class ShadowDatasetDDIB(ShadowDataset):
    """
    Shadow dataset that additionally yields *city_id* and *intensity_map*.
    Optionally adds a contrast channel (4th channel) when use_contrast=True.

    City IDs are assigned in the order that distinct cities appear across
    the supplied ``root_dir`` list (e.g. first city -> 0, second -> 1).
    """

    def __init__(self, root_dir, split='train', img_size=384, augment=False,
                 samples_per_dir=None, random_seed=42, selected_filenames=None,
                 use_fda=False, fda_target_root=None, fda_L=0.01,
                 geo_metadata_path=None, use_contrast=False):
        # Let the base class handle file discovery, sampling, transforms, FDA
        super().__init__(
            root_dir, split, img_size, augment,
            samples_per_dir, random_seed, selected_filenames,
            use_fda, fda_target_root, fda_L,
            geo_metadata_path,
        )

        self.use_contrast = use_contrast

        # ----- Build per-image city_id mapping -----
        if isinstance(root_dir, str):
            root_dir = [root_dir]

        self._city_name_to_id = {}
        for rd in root_dir:
            cname = _city_from_path(rd)
            if cname not in self._city_name_to_id:
                self._city_name_to_id[cname] = len(self._city_name_to_id)

        # Build a lookup: full_img_path -> city_id
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

        # Build final per-index city_id array aligned to self.img_paths
        self._city_id_array = []
        for p in self.img_paths:
            self._city_id_array.append(self._path_to_city.get(p, 0))

        # Resize transform for intensity map
        self._intensity_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BILINEAR)

        # For contrast: separate resize + toTensor (no normalize yet)
        self._resize_totensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self._rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        print(f'  DDIB dataset -- city mapping: {self._city_name_to_id}')
        print(f'  DDIB dataset -- num_domains = {len(self._city_name_to_id)}')
        if use_contrast:
            print(f'  DDIB dataset -- contrast channel enabled (4ch output)')

    @property
    def num_domains(self):
        """Number of distinct cities / domains."""
        return len(self._city_name_to_id)

    def __getitem__(self, idx):
        """
        Returns everything from the base class PLUS:
            city_id:       int64 scalar tensor
            intensity_map: float32 tensor [1, H, W] in [0, 1]

        When use_contrast=True, image tensor is [4, H, W] (RGBC).
        """
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img_name = self.img_files[idx]

        # ---- 1. Intensity map (from raw image, before any processing) ----
        raw_image = Image.open(img_path).convert('RGB')
        raw_np = np.array(raw_image).astype(np.float32)
        intensity = (0.299 * raw_np[:, :, 0]
                     + 0.587 * raw_np[:, :, 1]
                     + 0.114 * raw_np[:, :, 2]) / 255.0
        intensity_pil = Image.fromarray(
            (intensity * 255).astype(np.uint8), mode='L')
        intensity_pil = self._intensity_resize(intensity_pil)
        intensity_tensor = transforms.ToTensor()(intensity_pil)  # [1, H, W]

        # ---- 2. Main image processing ----
        image = Image.open(img_path).convert('RGB')

        # FDA (training only, handled by base class init)
        if self.fda_transform is not None:
            image_np = np.array(image)
            image_np = self.fda_transform(image_np)
            image = Image.fromarray(image_np)

        # Mask
        mask = Image.open(mask_path).convert('L')

        # Augmentation (consistent for image + mask)
        if self.augment:
            combined = Image.fromarray(
                np.concatenate([np.array(image),
                                np.array(mask)[:, :, None]], axis=2))
            combined = self.aug_transform(combined)
            combined_np = np.array(combined)
            image = Image.fromarray(combined_np[:, :, :3])
            mask = Image.fromarray(combined_np[:, :, 3])

        # ---- 3. Transform image (with or without contrast) ----
        if self.use_contrast:
            from data.contrast_utils import add_contrast_channel
            image_rgbc = add_contrast_channel(image)  # numpy [H, W, 4]
            image_pil_4ch = Image.fromarray(image_rgbc)
            image_tensor = self._resize_totensor(image_pil_4ch)  # [4, H, W]
            # Normalize RGB channels with ImageNet stats
            image_tensor[:3] = self._rgb_normalize(image_tensor[:3])
            # Contrast channel already in [0, 1]
            image_tensor[3:4] = image_tensor[3:4].clamp(0, 1)
        else:
            image_tensor = self.img_transform(image)  # [3, H, W]

        # ---- 4. Transform mask ----
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float().squeeze(0)  # [H, W]

        # ---- 5. Assemble result ----
        result = {
            'image': image_tensor,
            'mask': mask_tensor.long(),
            'filename': img_name,
            'city_id': torch.tensor(self._city_id_array[idx], dtype=torch.long),
            'intensity_map': intensity_tensor,
        }

        # Geocoordinates
        if self.geo_metadata is not None:
            if img_name in self.geo_metadata:
                coords = self.geo_metadata[img_name]
                result['lat'] = torch.tensor(coords['lat'], dtype=torch.float32)
                result['lon'] = torch.tensor(coords['lon'], dtype=torch.float32)
            else:
                result['lat'] = torch.tensor(39.8283, dtype=torch.float32)
                result['lon'] = torch.tensor(-98.5795, dtype=torch.float32)

        return result


# ======================================================================
# Dataloader factory
# ======================================================================

def get_dataloaders_ddib(
        data_root=None, base_data_root=None, mode='single',
        cities=None, resolution=None, fold_id=None,
        batch_size=8, num_workers=1, img_size=384,
        use_fda=False, fda_target_root=None, fda_L=0.01,
        geo_metadata_path=None, use_contrast=False):
    """
    Create train / val / test dataloaders for MAMNet DDIB training.

    Returns:
        dict  {'train': ..., 'val': ..., 'test': ...}
        plus  'num_domains': int
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
        train_paths = [os.path.join(base_data_root, c, resolution) for c in cities]
        val_paths = train_paths
        test_paths = train_paths
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
        test_city = fold['test']
        train_paths = [os.path.join(base_data_root, c, resolution) for c in train_cities]
        val_paths = train_paths
        test_paths = [os.path.join(base_data_root, test_city, resolution)]
        sample_dir = os.path.join(train_paths[0], 'train', 'images')
        if os.path.exists(sample_dir):
            n = len([f for f in os.listdir(sample_dir)
                     if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            samples_per_dir = n // len(train_cities)
    else:
        raise ValueError(f'Invalid mode: {mode}')

    # ---- Datasets ----
    common_kwargs = dict(
        img_size=img_size,
        geo_metadata_path=geo_metadata_path,
        use_contrast=use_contrast,
    )

    train_ds = ShadowDatasetDDIB(
        root_dir=train_paths, split='train', augment=True,
        samples_per_dir=samples_per_dir if mode in ('loco', 'all') else None,
        use_fda=use_fda, fda_target_root=fda_target_root, fda_L=fda_L,
        **common_kwargs,
    )

    val_ds = ShadowDatasetDDIB(
        root_dir=val_paths, split='val', augment=False,
        samples_per_dir=samples_per_dir if mode in ('loco', 'all') else None,
        **common_kwargs,
    )

    test_ds = ShadowDatasetDDIB(
        root_dir=test_paths, split='test', augment=False,
        **common_kwargs,
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
    print(f'  Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}')
    print(f'  num_domains = {num_domains}  |  contrast = {use_contrast}')

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'num_domains': num_domains,
    }