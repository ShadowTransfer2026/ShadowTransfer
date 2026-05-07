"""
HRDA Dataset for Shadow Detection
Generates LR context crops and HR detail crops for multi-resolution training.

Based on HRDA (ECCV 2022): https://arxiv.org/abs/2204.13132
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.contrast_utils import add_contrast_channel


class HRDAShadowDataset(Dataset):
    """
    Shadow Detection Dataset with HRDA multi-resolution cropping.
    
    Generates:
    - LR context crop: Downsampled to capture large context
    - HR detail crop: Full resolution, nested within context crop
    
    Expected directory structure:
        dataset/
        ├── train/
        │   ├── images/
        │   └── masks/
        ├── val/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
    
    Args:
        root_dir: Path to dataset directory
        split: 'train', 'val', or 'test'
        is_source: Whether this is source domain (has labels) or target domain (for pseudo-labels)
        img_size: Base image size (default: 384)
        context_size: Context crop size (default: 384, then downsampled)
        detail_size: Detail crop size (default: 192)
        scale_factor: LR downsampling factor (default: 0.5)
        output_stride: Model output stride for alignment (default: 8)
        augment: Whether to apply augmentation
    """
    
    def __init__(self, root_dir, split='train', is_source=True,
                 img_size=384, context_size=384, detail_size=192,
                 scale_factor=0.5, output_stride=8, augment=False, use_contrast=False):
        
        self.root_dir = root_dir
        self.split = split
        self.is_source = is_source
        self.img_size = img_size
        self.context_size = context_size
        self.detail_size = detail_size
        self.scale_factor = scale_factor
        self.output_stride = output_stride
        self.augment = augment and (split == 'train')
        self.use_contrast = use_contrast
        
        # Load file paths
        img_dir = os.path.join(root_dir, split, 'images')
        mask_dir = os.path.join(root_dir, split, 'masks')
        
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.img_paths = [os.path.join(img_dir, f) for f in self.img_files]
        self.mask_paths = [os.path.join(mask_dir, f) for f in self.img_files]
        
        
        
        # Image transforms (normalization) - will handle 3 or 4 channels
        if self.use_contrast:
            # For 4-channel images, create custom transform
            self.img_transform = None  # Will apply manually
        else:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
            ])
            
        # Mask transform
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Augmentation
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        
        print(f"HRDA Dataset ({split}, {'source' if is_source else 'target'}):")
        print(f"  Images: {len(self.img_files)}")
        print(f"  Context size: {context_size} (→ {int(context_size*scale_factor)} after downsample)")
        print(f"  Detail size: {detail_size}")
    
    def __len__(self):
        return len(self.img_files)
    
    def _get_crop_coordinates(self, img_size, crop_size):
        """
        Generate random crop coordinates aligned to output stride.
        
        Args:
            img_size: Size of image to crop from
            crop_size: Size of crop
            
        Returns:
            (top, left): Crop coordinates
        """
        k = self.output_stride
        
        # Ensure crop fits within image
        max_top = max(0, img_size - crop_size)
        max_left = max(0, img_size - crop_size)
        
        # Random position aligned to output stride
        if max_top > 0:
            top = random.randint(0, max_top // k) * k
        else:
            top = 0
            
        if max_left > 0:
            left = random.randint(0, max_left // k) * k
        else:
            left = 0
        
        return top, left
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
            - 'image_context': LR context crop [3, H_c, W_c]
            - 'image_detail': HR detail crop [3, H_d, W_d]
            - 'mask_context': Context mask [H_c, W_c] (if source)
            - 'mask_detail': Detail mask [H_d, W_d] (if source)
            - 'detail_coords': Coordinates of detail in context (b1, b2, b3, b4)
            - 'is_source': Boolean flag
            - 'filename': Image filename
        """
        # Load image
        img_name = self.img_files[idx]
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('RGB')

        # Load mask only if source domain (labeled)
        if self.is_source:
            mask = Image.open(mask_path).convert('L')
        else:
            # Create dummy mask for target domain (unlabeled)
            mask = Image.new('L', image.size, 0)
        
        # Resize to base size if needed
        if image.size != (self.img_size, self.img_size):
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
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
        
        # Generate context crop coordinates
        context_top, context_left = self._get_crop_coordinates(
            self.img_size, self.context_size
        )
        
        # Crop context
        image_context = image.crop((
            context_left,
            context_top,
            context_left + self.context_size,
            context_top + self.context_size
        ))
        mask_context = mask.crop((
            context_left,
            context_top,
            context_left + self.context_size,
            context_top + self.context_size
        ))
        
        # Downsample context to LR
        context_lr_size = int(self.context_size * self.scale_factor)
        image_context_lr = image_context.resize(
            (context_lr_size, context_lr_size), 
            Image.BILINEAR
        )
        mask_context_lr = mask_context.resize(
            (context_lr_size, context_lr_size), 
            Image.NEAREST
        )
        
        # Generate detail crop coordinates (within context crop)
        detail_top, detail_left = self._get_crop_coordinates(
            self.context_size, self.detail_size
        )
        
        # Crop detail from original context (before downsampling)
        image_detail = image_context.crop((
            detail_left,
            detail_top,
            detail_left + self.detail_size,
            detail_top + self.detail_size
        ))
        mask_detail = mask_context.crop((
            detail_left,
            detail_top,
            detail_left + self.detail_size,
            detail_top + self.detail_size
        ))
        
        # Apply transforms
        if self.use_contrast:
            # Add contrast channel to both context and detail BEFORE transforms
            # import numpy as np
            
            # Context LR
            image_context_lr_np = np.array(image_context_lr)
            image_context_lr_rgbc = add_contrast_channel(image_context_lr_np)
            image_context_lr_pil = Image.fromarray(image_context_lr_rgbc)
            image_context_tensor = transforms.ToTensor()(image_context_lr_pil)
            # Normalize RGB channels only
            image_context_tensor[:3] = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )(image_context_tensor[:3])
            # Contrast channel stays [0,1]
            image_context_tensor[3:4] = image_context_tensor[3:4].clamp(0, 1)
            
            # Detail HR
            image_detail_np = np.array(image_detail)
            image_detail_rgbc = add_contrast_channel(image_detail_np)
            image_detail_pil = Image.fromarray(image_detail_rgbc)
            image_detail_tensor = transforms.ToTensor()(image_detail_pil)
            # Normalize RGB channels only
            image_detail_tensor[:3] = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )(image_detail_tensor[:3])
            # Contrast channel stays [0,1]
            image_detail_tensor[3:4] = image_detail_tensor[3:4].clamp(0, 1)
        else:
            # Standard 3-channel
            image_context_tensor = self.img_transform(image_context_lr)
            image_detail_tensor = self.img_transform(image_detail)

        mask_context_tensor = self.mask_transform(mask_context_lr)
        mask_detail_tensor = self.mask_transform(mask_detail)
        
        # Convert masks: 255 -> 1
        mask_context_tensor = (mask_context_tensor > 0.5).float().squeeze(0)
        mask_detail_tensor = (mask_detail_tensor > 0.5).float().squeeze(0)
        
        # Detail coordinates in context (for fusion)
        # Coordinates are in the downsampled context space
        detail_coords_lr = (
            int(detail_top * self.scale_factor),
            int((detail_top + self.detail_size) * self.scale_factor),
            int(detail_left * self.scale_factor),
            int((detail_left + self.detail_size) * self.scale_factor)
        )
        
        result = {
            'image_context': image_context_tensor,
            'image_detail': image_detail_tensor,
            'mask_context': mask_context_tensor.long(),
            'mask_detail': mask_detail_tensor.long(),
            'detail_coords': detail_coords_lr,
            'is_source': self.is_source,
            'filename': img_name
        }
        
        return result


def get_hrda_dataloaders(source_root, target_root,
                         batch_size=2, num_workers=1,
                         img_size=384, context_size=384, detail_size=192):
    """
    Create HRDA dataloaders for source and target domains.
    
    Args:
        source_root: Path to source domain data
        target_root: Path to target domain data (for adaptation)
        batch_size: Batch size (default: 2, HRDA is memory-intensive)
        num_workers: Number of workers
        img_size: Base image size
        context_size: Context crop size
        detail_size: Detail crop size
        
    Returns:
        Dictionary with dataloaders
    """
    
    # Source domain (labeled)
    source_train = HRDAShadowDataset(
        source_root, split='train', is_source=True,
        img_size=img_size, context_size=context_size, detail_size=detail_size,
        augment=True
    )
    
    source_val = HRDAShadowDataset(
        source_root, split='val', is_source=True,
        img_size=img_size, context_size=context_size, detail_size=detail_size,
        augment=False
    )
    
    # Target domain (unlabeled for adaptation)
    target_train = HRDAShadowDataset(
        target_root, split='train', is_source=False,
        img_size=img_size, context_size=context_size, detail_size=detail_size,
        augment=True
    )
    
    target_val = HRDAShadowDataset(
        target_root, split='val', is_source=False,
        img_size=img_size, context_size=context_size, detail_size=detail_size,
        augment=False
    )
    
    # Test on target domain
    target_test = HRDAShadowDataset(
        target_root, split='test', is_source=False,
        img_size=img_size, context_size=context_size, detail_size=detail_size,
        augment=False
    )
    
    # Create dataloaders
    source_train_loader = torch.utils.data.DataLoader(
        source_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    target_train_loader = torch.utils.data.DataLoader(
        target_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    source_val_loader = torch.utils.data.DataLoader(
        source_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    target_val_loader = torch.utils.data.DataLoader(
        target_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    target_test_loader = torch.utils.data.DataLoader(
        target_test, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return {
        'source_train': source_train_loader,
        'target_train': target_train_loader,
        'source_val': source_val_loader,
        'target_val': target_val_loader,
        'target_test': target_test_loader
    }


if __name__ == "__main__":
    # Test HRDA dataset
    dataset = HRDAShadowDataset(
        root_dir='./dataset',
        split='train',
        is_source=True,
        img_size=384,
        context_size=384,
        detail_size=192,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    sample = dataset[0]
    
    print(f"\nSample keys: {sample.keys()}")
    print(f"Context image: {sample['image_context'].shape}")
    print(f"Detail image: {sample['image_detail'].shape}")
    print(f"Context mask: {sample['mask_context'].shape}")
    print(f"Detail mask: {sample['mask_detail'].shape}")
    print(f"Detail coords: {sample['detail_coords']}")
    print(f"Is source: {sample['is_source']}")
    print(f"Filename: {sample['filename']}")