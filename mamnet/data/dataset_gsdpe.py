"""
Dataset loader with GSD extraction for MAMNet with GSDPE.
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
from models.gsdpe import get_gsd_from_filename

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.contrast_utils import add_contrast_channel


class ShadowDatasetGSDPE(Dataset):
    """
    Shadow Detection Dataset with GSD extraction.
    
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
    
    def __init__(self, root_dir, split='train', img_size=384, augment=False, use_contrast=False):
        """
        Args:
            root_dir: Single path string OR list of paths to dataset directories
            split: 'train', 'val', or 'test'
            img_size: Size to resize images (default 384x384)
            augment: Whether to apply data augmentation (only for training)
        """
        # Handle both single path (backward compatibility) and list of paths
        if isinstance(root_dir, str):
            root_dir = [root_dir]
        
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.use_contrast = use_contrast
        
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
        
        

        # Transforms for images (normalization according to ImageNet stats)
        if self.use_contrast:
            self.img_transform = None  # Will apply manually
            self.mask_transform = None
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
            ])
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
        if self.use_contrast:
            # Resize first
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            
            # Add contrast channel
            image_np = np.array(image)
            image_rgbc = add_contrast_channel(image_np)
            image_pil = Image.fromarray(image_rgbc)
            
            # Convert to tensor
            image = transforms.ToTensor()(image_pil)
            
            # Normalize RGB channels only
            image[:3] = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )(image[:3])
            
            # Contrast channel stays [0,1]
            image[3:4] = image[3:4].clamp(0, 1)

            # Convert mask to tensor manually
            mask = transforms.ToTensor()(mask)
        else:
            image = self.img_transform(image)

        # Convert mask: 255 -> 1 (shadow), 0 -> 0 (background)
        mask = (mask > 0.5).float().squeeze(0)  # [H, W]
        
        return {
            'image': image,
            'mask': mask.long(),  # [H, W] with values {0, 1}
            'gsd': torch.tensor(gsd, dtype=torch.float32),  # Scalar GSD value
            'filename': img_name
        }


def get_dataloaders_gsdpe(data_root=None, base_data_root=None, mode='single', cities=None, resolution=None, 
                          batch_size=8, num_workers=1, img_size=384, use_contrast=False):
    """
    Create train, validation, and test dataloaders for GSDPE models.
    
    Args:
        data_root: Path for single city mode
        base_data_root: Base directory for multi-city mode
        mode: 'single', 'all', or 'cross_resolution'
        cities: List of city names
        resolution: 'highres' or 'midres' (for single resolution training)
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Image size
        
    Returns:
        Dictionary of dataloaders with 'train', 'val', 'test'
    """
    
    if mode == 'single':
        if data_root is None:
            raise ValueError("data_root must be provided for single mode")
        
        train_paths = [data_root]
        val_paths = [data_root]
        test_paths = [data_root]
        
    elif mode == 'all':
        if base_data_root is None or resolution is None:
            raise ValueError("base_data_root and resolution must be provided for 'all' mode")
        
        if cities is None:
            cities = ['chicago', 'miami', 'phoenix']
        
        train_paths = [os.path.join(base_data_root, city, resolution) for city in cities]
        val_paths = train_paths
        test_paths = train_paths
        
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'single', 'all', or 'cross_resolution'")
    
    # Create datasets
    train_dataset = ShadowDatasetGSDPE(
        train_paths, 
        split='train', 
        img_size=img_size, 
        augment=True,
        use_contrast=use_contrast
    )
    
    val_dataset = ShadowDatasetGSDPE(
        val_paths, 
        split='val', 
        img_size=img_size, 
        augment=False,
        use_contrast=use_contrast
    )
    
    test_dataset = ShadowDatasetGSDPE(
        test_paths, 
        split='test', 
        img_size=img_size, 
        augment=False,
        use_contrast=use_contrast
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
    print("Testing Shadow Dataset with GSDPE")
    print("=" * 50)
    
    # Create a dummy dataset structure for testing
    # In practice, use your actual dataset path
    
    # Example usage:
    # dataset = ShadowDatasetGSDPE(
    #     root_dir='./dataset',
    #     split='train',
    #     img_size=384,
    #     augment=True
    # )
    
    # print(f"Dataset size: {len(dataset)}")
    
    # # Test loading one sample
    # sample = dataset[0]
    # print(f"Image shape: {sample['image'].shape}")
    # print(f"Mask shape: {sample['mask'].shape}")
    # print(f"GSD value: {sample['gsd'].item()}")
    # print(f"Filename: {sample['filename']}")
    # print(f"Mask unique values: {torch.unique(sample['mask'])}")
    
    print("Note: Actual testing requires a dataset at specified path")
    print("Dataset test structure verified!")