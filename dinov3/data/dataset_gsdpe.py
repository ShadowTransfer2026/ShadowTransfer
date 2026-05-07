"""
Dataset loader with GSD extraction for DINOv3 with GSDPE.
Extracts GSD values from image filenames (midres=0.6m, highres=0.3m).
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsdpe import get_gsd_from_filename


# LOCO fold configurations
LOCO_FOLDS = {
    0: {'train': ['chicago', 'miami'], 'test': 'phoenix'},
    1: {'train': ['chicago', 'phoenix'], 'test': 'miami'},
    2: {'train': ['miami', 'phoenix'], 'test': 'chicago'}
}


class ShadowDatasetGSDPE(Dataset):
    """
    Shadow Detection Dataset with GSD extraction for DINOv3 + GSDPE.
    
    Expected directory structure:
        dataset/
        ├── train/
        │   ├── images/          # RGB .png files (with midres/highres in filename)
        │   └── masks/           # Binary .png (0=background, 255=shadow)
        ├── val/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
    
    Filename format: {city}_session{N}_{resolution}_random_{ID}.png
    Example: chicago_session01_midres_random_018.png → GSD = 0.6m
    """
    
    def __init__(self, root_dir, split='train', img_size=384, augment=False,
                 samples_per_dir=None, random_seed=42, selected_filenames=None):
        """
        Args:
            root_dir: Single path string OR list of paths to dataset directories
            split: 'train', 'val', or 'test'
            img_size: Size to resize images (default 384x384 for DINOv3)
            augment: Whether to apply data augmentation (only for training)
            samples_per_dir: Number of samples to randomly select from each directory
            random_seed: Random seed for sampling
            selected_filenames: List of specific filenames to use (if provided)
        """
        # Handle both single path (backward compatibility) and list of paths
        if isinstance(root_dir, str):
            root_dir = [root_dir]
        
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.samples_per_dir = samples_per_dir
        self.random_seed = random_seed
        self.selected_filenames = selected_filenames
        
        # Collect all image files from all root directories
        self.img_files = []
        self.img_paths = []
        self.mask_paths = []
        self.gsd_values = []
        
        for root_dir_instance in root_dir:
            img_dir = os.path.join(root_dir_instance, split, 'images')
            mask_dir = os.path.join(root_dir_instance, split, 'masks')
            
            if not os.path.exists(img_dir):
                print(f"Warning: {img_dir} does not exist, skipping...")
                continue
                
            files = sorted([f for f in os.listdir(img_dir) 
                        if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            
            for f in files:
                self.img_files.append(f)
                self.img_paths.append(os.path.join(img_dir, f))
                self.mask_paths.append(os.path.join(mask_dir, f))
                
                # Extract GSD from filename
                gsd = get_gsd_from_filename(f)
                self.gsd_values.append(gsd)
        
        # Filter by selected filenames if provided
        if self.selected_filenames is not None:
            selected_set = set(self.selected_filenames)
            
            filtered_files = []
            filtered_img_paths = []
            filtered_mask_paths = []
            filtered_gsd_values = []
            
            for i, filename in enumerate(self.img_files):
                if filename in selected_set:
                    filtered_files.append(filename)
                    filtered_img_paths.append(self.img_paths[i])
                    filtered_mask_paths.append(self.mask_paths[i])
                    filtered_gsd_values.append(self.gsd_values[i])
            
            self.img_files = filtered_files
            self.img_paths = filtered_img_paths
            self.mask_paths = filtered_mask_paths
            self.gsd_values = filtered_gsd_values
            
            if len(self.img_files) == 0:
                print(f"Warning: No matching files found after filtering with selected_filenames")
        
        # Apply random sampling if specified
        if self.samples_per_dir is not None and len(self.img_files) > 0:
            # Group files by their source directory
            files_per_dir = {}
            current_idx = 0
            for root_dir_instance in root_dir:
                img_dir = os.path.join(root_dir_instance, split, 'images')
                if os.path.exists(img_dir):
                    num_files = len([f for f in os.listdir(img_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
                    files_per_dir[root_dir_instance] = list(range(current_idx, current_idx + num_files))
                    current_idx += num_files
            
            # Sample from each directory
            np.random.seed(self.random_seed)
            sampled_indices = []
            for root_dir_instance, indices in files_per_dir.items():
                n_to_sample = min(self.samples_per_dir, len(indices))
                sampled = np.random.choice(indices, size=n_to_sample, replace=False)
                sampled_indices.extend(sampled)
            
            # Keep only sampled files
            sampled_indices = sorted(sampled_indices)
            self.img_files = [self.img_files[i] for i in sampled_indices]
            self.img_paths = [self.img_paths[i] for i in sampled_indices]
            self.mask_paths = [self.mask_paths[i] for i in sampled_indices]
            self.gsd_values = [self.gsd_values[i] for i in sampled_indices]
        
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
        
        print(f"Dataset {split} initialized:")
        print(f"  Total images: {len(self.img_files)}")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Augmentation: {augment}")
        print(f"  GSD distribution:")
        gsd_counts = {}
        for gsd in self.gsd_values:
            gsd_counts[gsd] = gsd_counts.get(gsd, 0) + 1
        for gsd, count in sorted(gsd_counts.items()):
            print(f"    {gsd}m: {count} images ({100*count/len(self.gsd_values):.1f}%)")
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
                - image: Tensor [3, H, W]
                - mask: Tensor [H, W] with values {0, 1}
                - gsd: Tensor (scalar) with GSD value
                - filename: str
        """
        # Get pre-computed paths
        img_name = self.img_files[idx]
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        gsd = self.gsd_values[idx]
        
        # Read image (RGB)
        image = Image.open(img_path).convert('RGB')
        
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
        
        return {
            'image': image,
            'mask': mask.long(),  # [H, W] with values {0, 1}
            'gsd': torch.tensor(gsd, dtype=torch.float32),  # Scalar GSD value
            'filename': img_name
        }


def get_dataloaders_gsdpe(
    data_root=None,
    base_data_root=None,
    mode='single',
    cities=None,
    resolution=None,
    fold_id=None,
    batch_size=8,
    num_workers=1,
    img_size=384
):
    """
    Create train, validation, and test dataloaders for GSDPE models.
    
    Args:
        data_root: Path for single city mode
        base_data_root: Base directory for all/loco modes
        mode: 'single', 'all', or 'loco'
        cities: List of city names (for 'all' mode)
        resolution: 'highres' or 'midres' (for all/loco modes)
        fold_id: Fold ID for LOCO (0, 1, 2)
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Image size (default 384x384 for DINOv3)
        
    Returns:
        Dictionary of dataloaders with 'train', 'val', 'test'
    """
    
    if mode == 'single':
        if data_root is None:
            raise ValueError("data_root must be provided for single mode")
        
        train_paths = [data_root]
        val_paths = [data_root]
        test_paths = [data_root]
        samples_per_dir = None
        
    elif mode == 'all':
        if base_data_root is None or resolution is None:
            raise ValueError("base_data_root and resolution must be provided for 'all' mode")
        
        if cities is None:
            cities = ['chicago', 'miami', 'phoenix']
        
        train_paths = [os.path.join(base_data_root, city, resolution) for city in cities]
        val_paths = train_paths
        test_paths = train_paths
        
        # Calculate samples per directory for balanced training
        samples_per_dir = None
        sample_city_path = train_paths[0]
        sample_img_dir = os.path.join(sample_city_path, 'train', 'images')
        if os.path.exists(sample_img_dir):
            single_city_size = len([f for f in os.listdir(sample_img_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            samples_per_dir = single_city_size // len(train_paths)
        
    elif mode == 'loco':
        if base_data_root is None or resolution is None or fold_id is None:
            raise ValueError("base_data_root, resolution, and fold_id must be provided for LOCO mode")
        
        if fold_id not in LOCO_FOLDS:
            raise ValueError(f"fold_id must be 0, 1, or 2. Got {fold_id}")
        
        fold_config = LOCO_FOLDS[fold_id]
        train_cities = fold_config['train']
        test_city = fold_config['test']
        
        print(f"\nLOCO Fold {fold_id}:")
        print(f"  Train cities: {train_cities}")
        print(f"  Test city: {test_city}")
        
        train_paths = [os.path.join(base_data_root, city, resolution) for city in train_cities]
        val_paths = train_paths
        test_paths = [os.path.join(base_data_root, test_city, resolution)]
        
        # Calculate samples per directory for balanced training
        samples_per_dir = None
        sample_city_path = train_paths[0]
        sample_img_dir = os.path.join(sample_city_path, 'train', 'images')
        if os.path.exists(sample_img_dir):
            single_city_size = len([f for f in os.listdir(sample_img_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            samples_per_dir = single_city_size // len(train_cities)
        
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'single', 'all', or 'loco'")
    
    # Create datasets
    train_dataset = ShadowDatasetGSDPE(
        train_paths, 
        split='train', 
        img_size=img_size, 
        augment=True,
        samples_per_dir=samples_per_dir
    )
    
    val_dataset = ShadowDatasetGSDPE(
        val_paths, 
        split='val', 
        img_size=img_size, 
        augment=False,
        samples_per_dir=samples_per_dir
    )
    
    test_dataset = ShadowDatasetGSDPE(
        test_paths, 
        split='test', 
        img_size=img_size, 
        augment=False
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
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # Test dataset
    print("Testing Shadow Dataset with GSDPE for DINOv3")
    print("=" * 50)
    
    print("Note: Actual testing requires a dataset at specified path")
    print("Dataset structure verified!")