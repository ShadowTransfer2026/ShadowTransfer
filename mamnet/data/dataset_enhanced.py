"""
Enhanced datasets for shadow detection.
Implements Tasks 1-5 (data-level improvements).
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import sys
# Add parent directory (mamnet_enhanced/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add data directory itself
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from augmentation_enhanced import (
    RandomCropAugmentation,
    GlobalBrightnessAugmentation,
    LocalBrightnessAugmentation,
    MultiCropSampler,
    compute_shadow_centers
)
from contrast_utils import add_contrast_channel


class ShadowDatasetEnhanced(Dataset):
    """
    Enhanced shadow detection dataset with task-specific augmentations.
    
    Supports:
    - Task 1: Random crop augmentation
    - Task 2: Contrast as 4th channel
    - Task 3: Size-balanced sampling (handled by get_dataloader)
    - Task 4: Global brightness augmentation
    - Task 5: Local brightness augmentation
    """
    
    def __init__(self, root_dir, split='train', img_size=384, 
                 task_id=0, augment=False,
                 shadow_size_info=None,
                 geo_metadata_path=None, use_fda=False, fda_target_root=None, fda_L=0.01,
                 use_mcl=False):
        """
        Args:
            root_dir: Single path string OR list of paths to dataset directories
            split: 'train', 'val', or 'test'
            img_size: Size to resize images (default 384x384)
            task_id: Task ID for augmentation selection
            augment: Whether to apply data augmentation (only for training)
            shadow_size_info: Shadow size analysis results (for Task 3)
            geo_metadata_path: Path to JSON file with geocoordinate metadata
        """
        # Handle both single path and list of paths
        if isinstance(root_dir, str):
            root_dir = [root_dir]
        
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.task_id = task_id
        self.augment = augment and (split == 'train')
        self.shadow_size_info = shadow_size_info
        self.geo_metadata_path = geo_metadata_path
        self.geo_metadata = None
        
        # Load geocoordinate metadata if provided
        if geo_metadata_path is not None:
            with open(geo_metadata_path, 'r') as f:
                self.geo_metadata = json.load(f)
            print(f"Loaded geocoordinate metadata from {geo_metadata_path}")

        # Setup FDA if requested
        self.fda_transform = None
        if use_fda and split == 'train':
            if fda_target_root is None:
                raise ValueError("fda_target_root must be provided when use_fda=True")
            
            from .fda_transform import load_target_images, FDATransform
            print(f"Loading FDA target images from {fda_target_root}...")
            target_images = load_target_images(fda_target_root, max_images=100)
            self.fda_transform = FDATransform(target_images, L=fda_L)
            print(f"FDA initialized with L={fda_L}")
        
        # Collect all image files
        self.use_mcl = use_mcl
        self.img_files = []
        self.img_paths = []
        self.mask_paths = []
        
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
        
        # Setup task-specific augmentations
        self._setup_augmentations()
        
        # Setup base transforms
        self._setup_base_transforms()
        
        print(f"Enhanced Dataset (Task {task_id}) initialized:")
        print(f"  Split: {split}")
        print(f"  Images: {len(self.img_files)}")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Augmentation: {self.augment}")
    
    def _setup_augmentations(self):
        """Setup task-specific augmentations"""
        self.random_crop_aug = None
        self.global_brightness_aug = None
        self.local_brightness_aug = None
        
        if not self.augment:
            return
        
        # Task 1: Random Crop Augmentation
        if self.task_id == 1:
            self.random_crop_aug = RandomCropAugmentation(
                crop_size=self.img_size,
                resize_size=512
            )
            print("  Using: Random Crop Augmentation (512→384)")
        
        # Task 4: Global Brightness Augmentation
        elif self.task_id == 4:
            self.global_brightness_aug = GlobalBrightnessAugmentation(
                brightness_factor=0.3
            )
            print("  Using: Global Brightness Augmentation")
        
        # Task 5: Local Brightness Augmentation
        elif self.task_id == 5:
            self.local_brightness_aug = LocalBrightnessAugmentation(
                brightness_factor=0.3,
                boundary_width=10
            )
            print("  Using: Local Brightness Augmentation")
    
    def _setup_base_transforms(self):
        """Setup base transforms for images and masks"""
        # Task 2: Use contrast as 4th channel (handled in __getitem__)
        self.use_contrast_channel = (self.task_id == 2)
        
        # Image normalization
        if self.use_contrast_channel:
            # 4 channels: RGBC
            # Normalize RGB with ImageNet stats, contrast with [0,1]
            self.img_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                # Custom normalization will be applied in __getitem__
            ])
        else:
            # Standard 3 channels: RGB
            self.img_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Mask transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), 
                            interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # Standard augmentation (flip)
        if self.augment and self.task_id not in [1, 5]:  # Task 1 and 5 handle augmentation differently
            self.standard_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        else:
            self.standard_aug = None
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
                - 'image': Tensor [3, H, W] or [4, H, W] for Task 2
                - 'mask': Tensor [H, W] with values {0, 1}
                - 'filename': str
                - 'lat', 'lon': Geographic coordinates (if available)
        """
        # Get paths
        img_name = self.img_files[idx]
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Apply FDA BEFORE other transforms (on PIL image)
        if self.fda_transform is not None:
            image_np = np.array(image)  # Convert to numpy
            image_np = self.fda_transform(image_np)  # Apply FDA
            image = Image.fromarray(image_np)  # Convert back to PIL
        
        # Apply task-specific augmentations (before other transforms)
        if self.augment:
            # Task 1: Random Crop
            if self.random_crop_aug is not None:
                image, mask = self.random_crop_aug(image, mask)
            
            # Task 4: Global Brightness
            if self.global_brightness_aug is not None:
                image = self.global_brightness_aug(image)
            
            # Task 5: Local Brightness
            if self.local_brightness_aug is not None:
                image, mask = self.local_brightness_aug(image, mask)
            
            # Standard augmentation (flip) for tasks other than 1, 5
            if self.standard_aug is not None:
                # Concatenate for consistent augmentation
                combined = Image.fromarray(
                    np.concatenate([np.array(image), np.array(mask)[:, :, None]], axis=2)
                )
                combined = self.standard_aug(combined)
                combined_np = np.array(combined)
                image = Image.fromarray(combined_np[:, :, :3])
                mask = Image.fromarray(combined_np[:, :, 3])
        
        # Task 2: Add contrast as 4th channel
        if self.use_contrast_channel:
            # Add contrast channel before transforms
            image_rgbc = add_contrast_channel(image)
            image_pil = Image.fromarray(image_rgbc)
            
            # Apply transforms
            image_tensor = self.img_transform(image_pil)  # [4, H, W]
            
            # Normalize RGB channels with ImageNet stats
            image_tensor[:3] = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )(image_tensor[:3])
            
            # Normalize contrast channel to [0, 1] (already done in add_contrast_channel)
            # Just ensure it's in right range
            image_tensor[3:4] = image_tensor[3:4].clamp(0, 1)
        else:
            # Standard RGB transform
            image_tensor = self.img_transform(image)
        
        # Transform mask
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float().squeeze(0)  # [H, W]
        
        # Prepare result
        result = {
            'image': image_tensor,
            'mask': mask_tensor.long(),
            'filename': img_name
        }

        if self.use_mcl and self.augment:
            # Create two additional augmented views
            seed1 = np.random.randint(2147483647)
            seed2 = np.random.randint(2147483647)
            
            # View 1
            image_aug1 = image  # Already PIL Image before transforms
            if self.random_crop_aug:
                image_aug1, _ = self.random_crop_aug(image_aug1, mask)
            if self.global_brightness_aug:
                image_aug1 = self.global_brightness_aug(image_aug1)
            
            torch.manual_seed(seed1)
            np.random.seed(seed1)
            if self.standard_aug:
                image_aug1 = self.standard_aug(image_aug1)
            
            # Apply contrast channel if needed
            if self.use_contrast_channel:
                image_aug1_rgbc = add_contrast_channel(image_aug1)
                image_aug1_pil = Image.fromarray(image_aug1_rgbc)
                image_aug1_tensor = self.img_transform(image_aug1_pil)
                image_aug1_tensor[:3] = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )(image_aug1_tensor[:3])
                image_aug1_tensor[3:4] = image_aug1_tensor[3:4].clamp(0, 1)
            else:
                image_aug1_tensor = self.img_transform(image_aug1)
            
            # View 2 (similar process with seed2)
            image_aug2 = image
            if self.random_crop_aug:
                image_aug2, _ = self.random_crop_aug(image_aug2, mask)
            if self.global_brightness_aug:
                image_aug2 = self.global_brightness_aug(image_aug2)
            
            torch.manual_seed(seed2)
            np.random.seed(seed2)
            if self.standard_aug:
                image_aug2 = self.standard_aug(image_aug2)
            
            if self.use_contrast_channel:
                image_aug2_rgbc = add_contrast_channel(image_aug2)
                image_aug2_pil = Image.fromarray(image_aug2_rgbc)
                image_aug2_tensor = self.img_transform(image_aug2_pil)
                image_aug2_tensor[:3] = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )(image_aug2_tensor[:3])
                image_aug2_tensor[3:4] = image_aug2_tensor[3:4].clamp(0, 1)
            else:
                image_aug2_tensor = self.img_transform(image_aug2)
            
            result['image_aug1'] = image_aug1_tensor
            result['image_aug2'] = image_aug2_tensor
        
        # Add geocoordinates if available
        if self.geo_metadata is not None:
            if img_name in self.geo_metadata:
                coords = self.geo_metadata[img_name]
                result['lat'] = torch.tensor(coords['lat'], dtype=torch.float32)
                result['lon'] = torch.tensor(coords['lon'], dtype=torch.float32)
            else:
                # Default coordinates
                result['lat'] = torch.tensor(39.8283, dtype=torch.float32)
                result['lon'] = torch.tensor(-98.5795, dtype=torch.float32)
        
        return result


class ShadowDatasetSizeBalanced(Dataset):
    """
    Size-balanced dataset for Task 3.
    
    Generates multiple crops from images with small shadows to oversample them.
    """
    
    def __init__(self, root_dir, split='train', img_size=384,
                 shadow_size_json=None,
                 geo_metadata_path=None):
        """
        Args:
            root_dir: Single path string OR list of paths
            split: 'train', 'val', or 'test'
            img_size: Size of crops (default 384)
            shadow_size_json: Path to shadow size analysis JSON
            geo_metadata_path: Path to geocoordinate metadata
        """
        # Handle both single path and list
        if isinstance(root_dir, str):
            root_dir = [root_dir]
        
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # Load shadow size info
        if shadow_size_json and os.path.exists(shadow_size_json):
            with open(shadow_size_json, 'r') as f:
                data = json.load(f)
                self.shadow_size_info = data['per_image']
                print(f"Loaded shadow size info from {shadow_size_json}")
        else:
            self.shadow_size_info = None
            print("Warning: No shadow size info provided, using uniform sampling")
        
        # Collect all image files with oversampling
        self.samples = []  # List of (img_path, mask_path, filename, crop_id)
        
        for root_dir_instance in root_dir:
            img_dir = os.path.join(root_dir_instance, split, 'images')
            mask_dir = os.path.join(root_dir_instance, split, 'masks')
            
            if not os.path.exists(img_dir):
                continue
            
            files = sorted([f for f in os.listdir(img_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            
            for f in files:
                img_path = os.path.join(img_dir, f)
                mask_path = os.path.join(mask_dir, f)
                
                # Determine number of crops based on shadow size
                num_crops = self._get_num_crops(f)
                
                # Add multiple entries for oversampling
                for crop_id in range(num_crops):
                    self.samples.append((img_path, mask_path, f, crop_id))
        
        # Setup transforms
        self.multi_crop = MultiCropSampler(crop_size=img_size, num_crops=1)
        
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
        
        # Standard augmentation
        self.standard_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        
        print(f"Size-Balanced Dataset (Task 3) initialized:")
        print(f"  Split: {split}")
        print(f"  Base images: {len(set([s[2] for s in self.samples]))}")
        print(f"  Total samples (with oversampling): {len(self.samples)}")
    
    def _get_num_crops(self, filename):
        """Determine number of crops based on shadow size"""
        if self.shadow_size_info is None:
            return 1
        
        if filename not in self.shadow_size_info:
            return 1
        
        info = self.shadow_size_info[filename]
        categories = info.get('categories', [])
        
        # Oversample based on smallest shadow
        if 'tiny' in categories:
            return 5  # 5x oversampling for tiny shadows
        elif 'small' in categories:
            return 3  # 3x oversampling for small shadows
        else:
            return 1  # Normal sampling
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with image, mask, filename
        """
        img_path, mask_path, filename, crop_id = self.samples[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # If crop_id > 0, generate centered crop around shadow
        if crop_id > 0 and self.shadow_size_info is not None:
            # Get shadow centers
            centers = compute_shadow_centers(mask)
            
            if len(centers) > 0:
                # Generate crop
                crops = self.multi_crop(image, mask, centers)
                if len(crops) > 0:
                    image, mask = crops[0]
        
        # Apply standard augmentation
        combined = Image.fromarray(
            np.concatenate([np.array(image), np.array(mask)[:, :, None]], axis=2)
        )
        combined = self.standard_aug(combined)
        combined_np = np.array(combined)
        image = Image.fromarray(combined_np[:, :, :3])
        mask = Image.fromarray(combined_np[:, :, 3])
        
        # Apply transforms
        image_tensor = self.img_transform(image)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float().squeeze(0)
        
        result = {
            'image': image_tensor,
            'mask': mask_tensor.long(),
            'filename': filename
        }
        
        return result


if __name__ == "__main__":
    # Test enhanced dataset
    print("Testing Enhanced Dataset...")
    
    # Test Task 0 (baseline)
    dataset_t0 = ShadowDatasetEnhanced(
        root_dir='./dataset',
        split='train',
        task_id=0,
        augment=True
    )
    print(f"\nTask 0 dataset size: {len(dataset_t0)}")
    
    # Test Task 1 (random crop)
    dataset_t1 = ShadowDatasetEnhanced(
        root_dir='./dataset',
        split='train',
        task_id=1,
        augment=True
    )
    print(f"\nTask 1 dataset size: {len(dataset_t1)}")
    
    # Test Task 2 (contrast channel)
    dataset_t2 = ShadowDatasetEnhanced(
        root_dir='./dataset',
        split='train',
        task_id=2,
        augment=True
    )
    print(f"\nTask 2 dataset size: {len(dataset_t2)}")
    if len(dataset_t2) > 0:
        sample = dataset_t2[0]
        print(f"Image shape (with contrast): {sample['image'].shape}")