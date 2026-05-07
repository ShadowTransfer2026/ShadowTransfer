"""
Fine-tuning dataset loader with filename filtering
Based on dataset.py but adds support for selecting specific files
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Optional


class ShadowDatasetFinetune(Dataset):
    """
    Shadow Detection Dataset with filename filtering for fine-tuning
    
    Supports selecting specific files for training/validation based on
    spatial sampling strategies.
    """
    
    def __init__(self, root_dir, split='train', img_size=384, augment=False,
             selected_filenames: Optional[List[str]] = None,
             check_val_if_missing: bool = False,
             check_train_if_missing: bool = False):
        """
        Args:
            root_dir: Path to dataset directory (e.g., /path/to/chicago/highres/)
            split: 'train', 'val', or 'test'
            img_size: Size to resize images (default 384x384)
            augment: Whether to apply data augmentation (only for training)
            selected_filenames: Optional list of specific filenames to use
            check_val_if_missing: If True, look in val/ directory for missing files (when split='train')
            check_train_if_missing: If True, look in train/ directory for missing files (when split='val')
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.selected_filenames = selected_filenames
        
        # Paths to images and masks
        img_dir = os.path.join(root_dir, split, 'images')
        mask_dir = os.path.join(root_dir, split, 'masks')
        
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory does not exist: {img_dir}")
        
        # Get all available files from primary split
        all_files = sorted([f for f in os.listdir(img_dir) 
                        if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        # Initialize lists
        self.img_files = []
        self.img_paths = []
        self.mask_paths = []
        
        # Build full paths for all files
        for f in all_files:
            self.img_files.append(f)
            self.img_paths.append(os.path.join(img_dir, f))
            self.mask_paths.append(os.path.join(mask_dir, f))
        
        # Filter by selected filenames if provided
        if selected_filenames is not None:
            selected_set = set(selected_filenames)
            
            # Filter to keep only selected files
            filtered_files = []
            filtered_img_paths = []
            filtered_mask_paths = []
            
            for i, filename in enumerate(self.img_files):
                if filename in selected_set:
                    filtered_files.append(filename)
                    filtered_img_paths.append(self.img_paths[i])
                    filtered_mask_paths.append(self.mask_paths[i])
            
            # Check if we need to look in the other split for missing files
            missing = selected_set - set(filtered_files)
            
            if len(missing) > 0:
                other_split = 'val' if split == 'train' else 'train'
                check_other = check_val_if_missing if split == 'train' else check_train_if_missing
                
                if check_other:
                    other_img_dir = os.path.join(root_dir, other_split, 'images')
                    other_mask_dir = os.path.join(root_dir, other_split, 'masks')
                    
                    if os.path.exists(other_img_dir):
                        other_files = [f for f in os.listdir(other_img_dir)
                                    if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
                        
                        # Find missing files in other split
                        found_in_other = 0
                        for filename in missing:
                            if filename in other_files:
                                filtered_files.append(filename)
                                filtered_img_paths.append(os.path.join(other_img_dir, filename))
                                filtered_mask_paths.append(os.path.join(other_mask_dir, filename))
                                found_in_other += 1
                        
                        if found_in_other > 0:
                            print(f"  Note: Found {found_in_other} files in {other_split}/ directory")
            
            # Update the lists
            self.img_files = filtered_files
            self.img_paths = filtered_img_paths
            self.mask_paths = filtered_mask_paths
            
            if len(self.img_files) == 0:
                raise ValueError(f"No matching files found in {split}/ or fallback directory. "
                            f"Selected {len(selected_filenames)} files, found 0 matches.")
            
            # Report if some files still missing
            still_missing = selected_set - set(self.img_files)
            if len(still_missing) > 0:
                print(f"Warning: {len(still_missing)} selected files not found in either split")
        
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
        
        print(f"Finetune {split} dataset initialized:")
        print(f"  Images: {len(self.img_files)}")
        if selected_filenames:
            print(f"  (Selected from {len(selected_filenames)} filenames)")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Augmentation: {augment}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor [3, H, W]
            mask: Tensor [H, W] with values {0, 1}
            filename: str
        """
        img_name = self.img_files[idx]
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
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
            'filename': img_name
        }


def get_finetune_dataloaders(
    data_root: str,
    train_filenames: List[str],
    val_filenames: List[str],
    batch_size: int = 8,
    num_workers: int = 1,
    img_size: int = 384
):
    """
    Create train and validation dataloaders for fine-tuning
    
    Args:
        data_root: Path to city/resolution data (e.g., /path/to/chicago/midres/)
        train_filenames: List of training image filenames
        val_filenames: List of validation image filenames
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Image size
    
    Returns:
        Dictionary with train and val dataloaders
    """
    # Create datasets
    train_dataset = ShadowDatasetFinetune(
        data_root, 
        split='train', 
        img_size=img_size, 
        augment=True,
        selected_filenames=train_filenames,
        check_val_if_missing=True
    )
    
    val_dataset = ShadowDatasetFinetune(
        data_root, 
        split='val', 
        img_size=img_size, 
        augment=False,
        selected_filenames=val_filenames,
        check_train_if_missing=True
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
    
    return {
        'train': train_loader,
        'val': val_loader
    }


if __name__ == '__main__':
    # Test the dataset
    print("Testing ShadowDatasetFinetune...")
    
    # Example usage
    test_filenames = ['chicago_session01_midres_random_001.png',
                     'chicago_session01_midres_random_002.png']
    
    dataset = ShadowDatasetFinetune(
        root_dir='/path/to/chicago/midres',
        split='train',
        augment=True,
        selected_filenames=test_filenames
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Filename: {sample['filename']}")