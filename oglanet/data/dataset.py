"""
Dataset loader for shadow detection with geocoordinate support.
Expects standard PyTorch segmentation format.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .fda_transform import load_target_images, FDATransform

# LOCO fold configurations
LOCO_FOLDS = {
    0: {'train': ['chicago', 'miami'], 'test': 'phoenix'},
    1: {'train': ['chicago', 'phoenix'], 'test': 'miami'},
    2: {'train': ['miami', 'phoenix'], 'test': 'chicago'}
}

class ShadowDataset(Dataset):
    """
    Shadow Detection Dataset with Geocoordinate Support
    
    Expected directory structure:
        dataset/
        ├── train/
        │   ├── images/          # RGB .png files
        │   └── masks/           # Binary .png (0=background, 255=shadow)
        ├── val/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
    """
    
    def __init__(self, root_dirs, split='train', img_size=384, augment=False, 
                 samples_per_dir=None, random_seed=42, selected_filenames=None,
                 use_fda=False, fda_target_root=None, fda_L=0.01,
                 geo_metadata_path=None):
        """
        Args:
            root_dirs: Single path string OR list of paths to dataset directories
            split: 'train', 'val', or 'test'
            img_size: Size to resize images (default 384x384)
            augment: Whether to apply data augmentation (only for training)
            samples_per_dir: Number of samples to take from each directory (for LOCO/all modes)
            random_seed: Random seed for sampling
            selected_filenames: List of specific filenames to use (optional)
            use_fda: Whether to apply FDA augmentation
            fda_target_root: Path to target domain images for FDA
            fda_L: Low-frequency ratio for FDA (beta parameter)
            geo_metadata_path: Path to JSON file with geocoordinate metadata
                          Format: {"image_001.png": {"lat": 33.4484, "lon": -112.0740}, ...}
        """
        # Handle both single path (backward compatibility) and list of paths
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        
        self.root_dirs = root_dirs
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.samples_per_dir = samples_per_dir
        self.random_seed = random_seed
        self.selected_filenames = selected_filenames
        self.geo_metadata_path = geo_metadata_path
        self.geo_metadata = None
        
        # Load geocoordinate metadata if provided
        if geo_metadata_path is not None:
            import json
            with open(geo_metadata_path, 'r') as f:
                self.geo_metadata = json.load(f)
            print(f"Loaded geocoordinate metadata from {geo_metadata_path}")
            print(f"  Total entries: {len(self.geo_metadata)}")
        
        # Collect all image files from all root directories
        self.img_files = []
        self.img_paths = []
        self.mask_paths = []
        
        for root_dir in root_dirs:
            img_dir = os.path.join(root_dir, split, 'images')
            mask_dir = os.path.join(root_dir, split, 'masks')
            
            if not os.path.exists(img_dir):
                print(f"Warning: {img_dir} does not exist, skipping...")
                continue
                
            files = sorted([f for f in os.listdir(img_dir) 
                        if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            
            for f in files:
                self.img_files.append(f)
                self.img_paths.append(os.path.join(img_dir, f))
                self.mask_paths.append(os.path.join(mask_dir, f))
                
        # Filter by selected filenames if provided
        if self.selected_filenames is not None:
            selected_set = set(self.selected_filenames)
            
            # Filter to keep only selected files
            filtered_files = []
            filtered_img_paths = []
            filtered_mask_paths = []
            
            for i, filename in enumerate(self.img_files):
                if filename in selected_set:
                    filtered_files.append(filename)
                    filtered_img_paths.append(self.img_paths[i])
                    filtered_mask_paths.append(self.mask_paths[i])
            
            # Update lists
            self.img_files = filtered_files
            self.img_paths = filtered_img_paths
            self.mask_paths = filtered_mask_paths
            
            if len(self.img_files) == 0:
                print(f"Warning: No matching files found after filtering with selected_filenames")

        # Apply random sampling if specified
        if self.samples_per_dir is not None and len(self.img_files) > 0:
            # Group files by their source directory
            files_per_dir = {}
            current_idx = 0
            for root_dir in root_dirs:
                img_dir = os.path.join(root_dir, split, 'images')
                if os.path.exists(img_dir):
                    num_files = len([f for f in os.listdir(img_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
                    files_per_dir[root_dir] = list(range(current_idx, current_idx + num_files))
                    current_idx += num_files
            
            # Sample from each directory
            np.random.seed(self.random_seed)
            sampled_indices = []
            for root_dir, indices in files_per_dir.items():
                n_to_sample = min(self.samples_per_dir, len(indices))
                sampled = np.random.choice(indices, size=n_to_sample, replace=False)
                sampled_indices.extend(sampled)
            
            # Keep only sampled files
            sampled_indices = sorted(sampled_indices)
            self.img_files = [self.img_files[i] for i in sampled_indices]
            self.img_paths = [self.img_paths[i] for i in sampled_indices]
            self.mask_paths = [self.mask_paths[i] for i in sampled_indices]
        
        # Transforms for images (normalization according to ImageNet stats)
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        # Transforms for masks
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # Data augmentation (only for training)
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        
        # Setup FDA if requested
        self.use_fda = use_fda
        self.fda_transform = None
        
        if self.use_fda and split == 'train':
            if fda_target_root is None:
                raise ValueError("fda_target_root must be provided when use_fda=True")
            
            print(f"Loading FDA target images from {fda_target_root}...")
            target_images = load_target_images(fda_target_root, max_images=100)
            self.fda_transform = FDATransform(target_images, L=fda_L)
            print(f"FDA initialized with L={fda_L}")
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor [3, H, W]
            mask: Tensor [H, W] with values {0, 1}
            filename: str
            lat: Latitude (if metadata available)
            lon: Longitude (if metadata available)
        """
        # Get pre-computed paths
        img_name = self.img_files[idx]
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read image (RGB)
        image = Image.open(img_path).convert('RGB')
        
        # Apply FDA BEFORE other transforms (on PIL image)
        if self.fda_transform is not None:
            image_np = np.array(image)  # Convert to numpy
            image_np = self.fda_transform(image_np)  # Apply FDA
            image = Image.fromarray(image_np)  # Convert back to PIL
        
        # Read mask (grayscale)
        mask = Image.open(mask_path).convert('L')
        
        # Apply augmentation if enabled
        if self.augment:
            # Concatenate for consistent augmentation
            combined = Image.fromarray(
                np.concatenate([np.array(image), np.array(mask)[:, :, None]], axis=2)
            )
            combined = self.aug_transform(combined)
            combined = np.array(combined)
            image = Image.fromarray(combined[:, :, :3])
            mask = Image.fromarray(combined[:, :, 3])
        
        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        
        # Convert mask: 255 -> 1 (shadow), 0 -> 0 (background)
        mask = (mask > 0.5).float().squeeze(0)  # [H, W]
        
        result = {
            'image': image,
            'mask': mask.long(),  # [H, W] with values {0, 1}
            'filename': img_name
        }
        
        # Add geocoordinates if available
        if self.geo_metadata is not None:
            if img_name in self.geo_metadata:
                coords = self.geo_metadata[img_name]
                result['lat'] = torch.tensor(coords['lat'], dtype=torch.float32)
                result['lon'] = torch.tensor(coords['lon'], dtype=torch.float32)
            else:
                # Default coordinates if not found (center of US)
                result['lat'] = torch.tensor(39.8283, dtype=torch.float32)
                result['lon'] = torch.tensor(-98.5795, dtype=torch.float32)
                print(f"Warning: No geocoordinates found for {img_name}, using default")
        
        return result


class UnlabeledDataset(ShadowDataset):
    """
    Unlabeled dataset for target domain in UDA.
    Returns images and coordinates only (masks are dummy).
    """
    
    def __init__(self, root_dirs, split='train', img_size=384, augment=False,
                 samples_per_dir=None, random_seed=42, selected_filenames=None,
                 use_fda=False, fda_target_root=None, fda_L=0.01,
                 geo_metadata_path=None):
        """Initialize unlabeled dataset - same as ShadowDataset but no masks needed"""
        super().__init__(
            root_dirs, split, img_size, augment,
            samples_per_dir, random_seed, selected_filenames,
            use_fda, fda_target_root, fda_L,
            geo_metadata_path
        )
    
    def __getitem__(self, idx):
        """Returns image and coordinates only"""
        # Get pre-computed paths
        img_name = self.img_files[idx]
        img_path = self.img_paths[idx]
        
        # Read image only
        image = Image.open(img_path).convert('RGB')
        
        # Apply FDA if enabled
        if self.fda_transform is not None:
            image_np = np.array(image)
            image_np = self.fda_transform(image_np)
            image = Image.fromarray(image_np)
        
        # Apply transforms
        image = self.img_transform(image)
        
        result = {
            'image': image,
            'filename': img_name
        }
        
        # Add geocoordinates if available
        if self.geo_metadata is not None:
            if img_name in self.geo_metadata:
                coords = self.geo_metadata[img_name]
                result['lat'] = torch.tensor(coords['lat'], dtype=torch.float32)
                result['lon'] = torch.tensor(coords['lon'], dtype=torch.float32)
            else:
                result['lat'] = torch.tensor(39.8283, dtype=torch.float32)
                result['lon'] = torch.tensor(-98.5795, dtype=torch.float32)
        
        return result


class Dataset(ShadowDataset):
    """
    From paper:
    - Images are cropped into 384x384 patches with stride of 64
    - Pixel values normalized to [0, 1]
    - Generated 11,836 patches from 412 training images
    """
    
    def __init__(self, root_dir, split='train', img_size=384, augment=False, 
                 samples_per_dir=None, random_seed=42,
                 use_fda=False, fda_target_root=None, fda_L=0.01,
                 geo_metadata_path=None):
        super().__init__(root_dir, split, img_size, augment, samples_per_dir, random_seed,
                        selected_filenames=None, use_fda=use_fda, fda_target_root=fda_target_root, 
                        fda_L=fda_L, geo_metadata_path=geo_metadata_path)
        
        print(f"OGLANet {split} dataset initialized:")
        print(f"  Images: {len(self.img_files)}")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Augmentation: {augment}")

class ShadowDatasetMCL(ShadowDataset):
    """
    Shadow Dataset for Multi-level Contrastive Learning.
    Returns two augmented views of each image for contrastive learning.
    """
    
    def __init__(self, root_dirs, split='train', img_size=384, augment=False,
                 samples_per_dir=None, random_seed=42, selected_filenames=None,
                 use_fda=False, fda_target_root=None, fda_L=0.01,
                 geo_metadata_path=None):
        super().__init__(
            root_dirs, split, img_size, augment,
            samples_per_dir, random_seed, selected_filenames,
            use_fda, fda_target_root, fda_L,
            geo_metadata_path
        )
        
        # For contrastive learning, we need stronger augmentations
        if augment and split == 'train':
            self.aug_transform_strong = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
    
    def __getitem__(self, idx):
        """
        Returns two augmented views for contrastive learning.
        
        Returns:
            Dictionary with:
                - 'image': Original transformed image
                - 'image_aug1': First augmented view
                - 'image_aug2': Second augmented view
                - 'mask': Ground truth mask (same for both views)
                - 'filename': Image filename
                - 'lat', 'lon': Geographic coordinates (if available)
        """
        # Get paths
        img_name = self.img_files[idx]
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read image and mask
        image_orig = Image.open(img_path).convert('RGB')
        mask_orig = Image.open(mask_path).convert('L')
        
        # Apply FDA if enabled
        if self.fda_transform is not None:
            image_np = np.array(image_orig)
            image_np = self.fda_transform(image_np)
            image_orig = Image.fromarray(image_np)
        
        # For MCL: create THREE versions with consistent augmentation
        if self.augment:
            # Define geometric transform (same for all views)
            geo_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
            ])
            
            # View 0: Main image-mask pair (augmented together)
            seed0 = np.random.randint(2147483647)
            combined0 = Image.fromarray(
                np.concatenate([np.array(image_orig), np.array(mask_orig)[:, :, None]], axis=2)
            )
            torch.manual_seed(seed0)
            np.random.seed(seed0)
            combined0 = geo_transform(combined0)
            combined0_np = np.array(combined0)
            image_main = Image.fromarray(combined0_np[:, :, :3])
            mask_main = Image.fromarray(combined0_np[:, :, 3])
            
            # View 1: For contrastive (augmented, no mask needed)
            seed1 = np.random.randint(2147483647)
            torch.manual_seed(seed1)
            np.random.seed(seed1)
            image_aug1 = geo_transform(image_orig.copy())
            
            # View 2: For contrastive (different augmentation)
            seed2 = np.random.randint(2147483647)
            torch.manual_seed(seed2)
            np.random.seed(seed2)
            image_aug2 = geo_transform(image_orig.copy())
        else:
            image_main = image_orig
            mask_main = mask_orig
            image_aug1 = image_orig
            image_aug2 = image_orig
        
        # Apply transforms
        image_tensor = self.img_transform(image_main)
        image_aug1_tensor = self.img_transform(image_aug1)
        image_aug2_tensor = self.img_transform(image_aug2)
        mask_tensor = self.mask_transform(mask_main)
        mask_tensor = (mask_tensor > 0.5).float().squeeze(0)
        
        result = {
            'image': image_tensor,
            'image_aug1': image_aug1_tensor,
            'image_aug2': image_aug2_tensor,
            'mask': mask_tensor.long(),
            'filename': img_name
        }
        
        # Add geocoordinates if available
        if self.geo_metadata is not None:
            if img_name in self.geo_metadata:
                coords = self.geo_metadata[img_name]
                result['lat'] = torch.tensor(coords['lat'], dtype=torch.float32)
                result['lon'] = torch.tensor(coords['lon'], dtype=torch.float32)
        
        return result

def get_dataloaders(data_root=None, base_data_root=None, mode='single', 
                   cities=None, resolution=None, fold_id=None,
                   batch_size=8, num_workers=1, img_size=384,
                   use_fda=False, fda_target_root=None, fda_L=0.01,
                   geo_metadata_path=None,
                   use_mcl=False, use_contrast=False):
    """
    Create train, validation, and test dataloaders.
    For LOCO mode, also creates unlabeled target domain loader for UDA.
    
    Args:
        data_root: Path for single city mode (backward compatibility)
        base_data_root: Base directory for all/loco modes
        mode: 'single', 'all', or 'loco'
        cities: List of city names (for 'all' mode) or None
        resolution: 'highres' or 'midres' (for all/loco modes)
        fold_id: Fold ID for LOCO (0, 1, 2)
        batch_size: Batch size (default 4 as per paper)
        num_workers: Number of data loading workers
        img_size: Image size (default 384x384)
        use_fda: Whether to apply FDA augmentation
        fda_target_root: Path to target domain images for FDA
        fda_L: Low-frequency ratio for FDA (beta parameter)
        geo_metadata_path: Path to geocoordinate metadata JSON
        
    Returns:
        Dictionary of dataloaders
    """
    if use_mcl:
        DatasetClass = ShadowDatasetMCL
    else:
        DatasetClass = Dataset

    if mode == 'single':
        # Backward compatibility: single city mode
        if data_root is None:
            raise ValueError("data_root must be provided for single city mode")
        
        train_paths = [data_root]
        val_paths = [data_root]
        test_paths = [data_root]
        target_paths = None  # No separate target domain
        
    elif mode == 'all':
        # Train on all cities
        if base_data_root is None or resolution is None:
            raise ValueError("base_data_root and resolution must be provided for 'all' mode")
        
        if cities is None:
            cities = ['chicago', 'miami', 'phoenix']
        
        train_paths = [os.path.join(base_data_root, city, resolution) for city in cities]
        val_paths = train_paths
        test_paths = train_paths
        target_paths = None
        
        # Calculate samples per directory for all mode
        samples_per_dir = None
        sample_city_path = train_paths[0]
        sample_img_dir = os.path.join(sample_city_path, 'train', 'images')
        if os.path.exists(sample_img_dir):
            single_city_size = len([f for f in os.listdir(sample_img_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            samples_per_dir = single_city_size // len(train_paths)
        
    elif mode == 'loco':
        # Leave-One-City-Out mode
        if base_data_root is None or resolution is None or fold_id is None:
            raise ValueError("base_data_root, resolution, and fold_id must be provided for LOCO mode")
        
        if fold_id not in LOCO_FOLDS:
            raise ValueError(f"fold_id must be 0, 1, or 2. Got {fold_id}")
        
        fold_config = LOCO_FOLDS[fold_id]
        train_cities = fold_config['train']
        test_city = fold_config['test']
        
        train_paths = [os.path.join(base_data_root, city, resolution) for city in train_cities]
        val_paths = train_paths  # Validation from training cities
        test_paths = [os.path.join(base_data_root, test_city, resolution)]
        target_paths = test_paths  # Unlabeled target domain for UDA

        # Calculate samples per directory for LOCO
        samples_per_dir = None
        sample_city_path = train_paths[0]
        sample_img_dir = os.path.join(sample_city_path, 'train', 'images')
        if os.path.exists(sample_img_dir):
            single_city_size = len([f for f in os.listdir(sample_img_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            samples_per_dir = single_city_size // len(train_cities)
        
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'single', 'all', or 'loco'")
    
    # Import enhanced dataset at top of file
    from data.dataset_enhanced import ShadowDatasetEnhanced

    # Then use enhanced dataset conditionally
    if use_contrast:
        train_dataset = ShadowDatasetEnhanced(
            train_paths, 
            split='train', 
            img_size=img_size, 
            augment=True, 
            samples_per_dir=samples_per_dir if mode in ['loco', 'all'] else None,
            use_fda=use_fda,
            fda_target_root=fda_target_root,
            fda_L=fda_L,
            geo_metadata_path=geo_metadata_path,
            use_contrast=True
        )
    
        val_dataset = ShadowDatasetEnhanced(
            val_paths, 
            split='val', 
            img_size=img_size, 
            augment=False, 
            samples_per_dir=samples_per_dir if mode in ['loco', 'all'] else None,
            geo_metadata_path=geo_metadata_path,
            use_contrast=True
        )
        
        test_dataset = ShadowDatasetEnhanced(
            test_paths, 
            split='test', 
            img_size=img_size, 
            augment=False,
            geo_metadata_path=geo_metadata_path,
            use_contrast=True
        )
    else:
        train_dataset = Dataset(
            train_paths, 
            split='train', 
            img_size=img_size, 
            augment=True, 
            samples_per_dir=samples_per_dir if mode in ['loco', 'all'] else None,
            use_fda=use_fda,
            fda_target_root=fda_target_root,
            fda_L=fda_L,
            geo_metadata_path=geo_metadata_path
        )
    
        val_dataset = Dataset(
            val_paths, 
            split='val', 
            img_size=img_size, 
            augment=False, 
            samples_per_dir=samples_per_dir if mode in ['loco', 'all'] else None,
            geo_metadata_path=geo_metadata_path
        )
        
        test_dataset = Dataset(
            test_paths, 
            split='test', 
            img_size=img_size, 
            augment=False,
            geo_metadata_path=geo_metadata_path
        )
    
    # Create unlabeled target domain dataset for UDA (only in LOCO mode)
    target_dataset = None
    if target_paths is not None:
        target_dataset = UnlabeledDataset(
            target_paths,
            split='train',  # Use train split from target domain
            img_size=img_size,
            augment=False,
            geo_metadata_path=geo_metadata_path
        )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    result = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    # Add target loader if available
    if target_dataset is not None:
        target_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        result['target'] = target_loader
    
    return result


if __name__ == "__main__":
    # Test dataset
    dataset = Dataset(root_dir='./dataset', split='train', augment=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Mask unique values: {torch.unique(sample['mask'])}")
    print(f"Filename: {sample['filename']}")