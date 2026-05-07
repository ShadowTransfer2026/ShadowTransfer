"""
Enhanced dataset for OGLANet with contrast channel support.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fda_transform import load_target_images, FDATransform

# Import contrast utilities
try:
    from utils.contrast_utils import add_contrast_channel
except ImportError:
    print("Warning: contrast_utils not found. Make sure to copy it from MAMNet.")
    add_contrast_channel = None


class ShadowDatasetEnhanced(Dataset):
    """
    Shadow detection dataset with optional contrast channel.
    
    When use_contrast=True, returns 4-channel images (RGBC).
    Otherwise identical to base dataset.
    """
    
    def __init__(self, root_dirs, split='train', img_size=384, augment=False,
                 samples_per_dir=None, random_seed=42, selected_filenames=None,
                 use_fda=False, fda_target_root=None, fda_L=0.01,
                 geo_metadata_path=None, use_contrast=False, use_mcl=False):
        """
        Args:
            root_dirs: Single path string OR list of paths to dataset directories
            split: 'train', 'val', or 'test'
            img_size: Size to resize images (default 384x384)
            augment: Whether to apply data augmentation (only for training)
            samples_per_dir: Number of samples to take from each directory
            random_seed: Random seed for sampling
            selected_filenames: List of specific filenames to use (optional)
            use_fda: Whether to apply FDA augmentation
            fda_target_root: Path to target domain images for FDA
            fda_L: Low-frequency ratio for FDA (beta parameter)
            geo_metadata_path: Path to JSON file with geocoordinate metadata
            use_contrast: Whether to add contrast as 4th channel
        """
        # Handle both single path and list of paths
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
        self.use_contrast = use_contrast
        self.use_mcl = use_mcl
        
        # Validate contrast utils availability
        if self.use_contrast and add_contrast_channel is None:
            raise ImportError("use_contrast=True but contrast_utils not available. Copy from MAMNet.")
        
        # Load geocoordinate metadata if provided
        if geo_metadata_path is not None:
            with open(geo_metadata_path, 'r') as f:
                self.geo_metadata = json.load(f)
            print(f"Loaded geocoordinate metadata from {geo_metadata_path}")
        
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
            filtered_files = []
            filtered_img_paths = []
            filtered_mask_paths = []
            
            for i, filename in enumerate(self.img_files):
                if filename in selected_set:
                    filtered_files.append(filename)
                    filtered_img_paths.append(self.img_paths[i])
                    filtered_mask_paths.append(self.mask_paths[i])
            
            self.img_files = filtered_files
            self.img_paths = filtered_img_paths
            self.mask_paths = filtered_mask_paths
            
            if len(self.img_files) == 0:
                print(f"Warning: No matching files found after filtering")
        
        # Apply random sampling if specified
        if self.samples_per_dir is not None and len(self.img_files) > 0:
            files_per_dir = {}
            current_idx = 0
            for root_dir in root_dirs:
                img_dir = os.path.join(root_dir, split, 'images')
                if os.path.exists(img_dir):
                    num_files = len([f for f in os.listdir(img_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
                    files_per_dir[root_dir] = list(range(current_idx, current_idx + num_files))
                    current_idx += num_files
            
            np.random.seed(self.random_seed)
            sampled_indices = []
            for root_dir, indices in files_per_dir.items():
                n_to_sample = min(self.samples_per_dir, len(indices))
                sampled = np.random.choice(indices, size=n_to_sample, replace=False)
                sampled_indices.extend(sampled)
            
            sampled_indices = sorted(sampled_indices)
            self.img_files = [self.img_files[i] for i in sampled_indices]
            self.img_paths = [self.img_paths[i] for i in sampled_indices]
            self.mask_paths = [self.mask_paths[i] for i in sampled_indices]
        
        # Setup transforms
        if self.use_contrast:
            # 4 channels: RGBC
            self.img_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                # Custom normalization applied in __getitem__
            ])
        else:
            # Standard 3 channels: RGB
            self.img_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
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
        
        print(f"Enhanced Dataset initialized:")
        print(f"  Images: {len(self.img_files)}")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Augmentation: {augment}")
        print(f"  Contrast channel: {use_contrast}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor [3, H, W] or [4, H, W] if use_contrast=True
            mask: Tensor [H, W] with values {0, 1}
            filename: str
            lat: Latitude (if metadata available)
            lon: Longitude (if metadata available)
        """
        # Get paths
        img_name = self.img_files[idx]
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read image (RGB)
        image = Image.open(img_path).convert('RGB')
        
        # Apply FDA BEFORE other transforms (on PIL image)
        if self.fda_transform is not None:
            image_np = np.array(image)
            image_np = self.fda_transform(image_np)
            image = Image.fromarray(image_np)
        
        # Read mask (grayscale)
        mask = Image.open(mask_path).convert('L')
        
        # Apply augmentation if enabled
        if self.augment:
            combined = Image.fromarray(
                np.concatenate([np.array(image), np.array(mask)[:, :, None]], axis=2)
            )
            combined = self.aug_transform(combined)
            combined = np.array(combined)
            image = Image.fromarray(combined[:, :, :3])
            mask = Image.fromarray(combined[:, :, 3])
        
        # Add contrast channel if requested
        if self.use_contrast:
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
            
            # Normalize contrast channel to [0, 1]
            image_tensor[3:4] = image_tensor[3:4].clamp(0, 1)
        else:
            # Standard RGB transform
            image_tensor = self.img_transform(image)
        
        # Transform mask
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float().squeeze(0)  # [H, W]

        # MCL: Generate two augmented views if requested
        if self.use_mcl and self.augment:
            # Create two additional augmented views
            seed1 = np.random.randint(2147483647)
            seed2 = np.random.randint(2147483647)
            
            # Reload original image for independent augmentation
            image_orig = Image.open(img_path).convert('RGB')
            
            # Apply FDA if enabled
            if self.fda_transform is not None:
                image_np = np.array(image_orig)
                image_np = self.fda_transform(image_np)
                image_orig = Image.fromarray(image_np)
            
            # View 1
            torch.manual_seed(seed1)
            np.random.seed(seed1)
            image_aug1 = image_orig.copy()
            if self.aug_transform:
                image_aug1 = self.aug_transform(image_aug1)
            
            # View 2
            torch.manual_seed(seed2)
            np.random.seed(seed2)
            image_aug2 = image_orig.copy()
            if self.aug_transform:
                image_aug2 = self.aug_transform(image_aug2)
            
            # Apply contrast channel if requested
            if self.use_contrast:
                # Add contrast channel to augmented views
                image_aug1_rgbc = add_contrast_channel(image_aug1)
                image_aug1_pil = Image.fromarray(image_aug1_rgbc)
                image_aug1_tensor = self.img_transform(image_aug1_pil)
                image_aug1_tensor[:3] = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )(image_aug1_tensor[:3])
                image_aug1_tensor[3:4] = image_aug1_tensor[3:4].clamp(0, 1)
                
                image_aug2_rgbc = add_contrast_channel(image_aug2)
                image_aug2_pil = Image.fromarray(image_aug2_rgbc)
                image_aug2_tensor = self.img_transform(image_aug2_pil)
                image_aug2_tensor[:3] = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )(image_aug2_tensor[:3])
                image_aug2_tensor[3:4] = image_aug2_tensor[3:4].clamp(0, 1)
            else:
                # Standard RGB transform
                image_aug1_tensor = self.img_transform(image_aug1)
                image_aug2_tensor = self.img_transform(image_aug2)
        
        result = {
            'image': image_tensor,
            'mask': mask_tensor.long(),
            'filename': img_name
        }

        # Add augmented views if MCL is enabled
        if self.use_mcl and self.augment:
            result['image_aug1'] = image_aug1_tensor
            result['image_aug2'] = image_aug2_tensor
        
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