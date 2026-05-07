"""
Enhanced augmentations for shadow detection.
Implements:
- Task 1: Random crop augmentation
- Task 4: Global brightness augmentation
- Task 5: Local/boundary brightness augmentation
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch


class RandomCropAugmentation:
    """
    Random crop augmentation (Task 1).
    
    Strategy:
    1. Resize image to larger size (512x512)
    2. Extract random 384x384 crop
    3. Provides more spatial variations
    """
    
    def __init__(self, crop_size=384, resize_size=512):
        """
        Args:
            crop_size: Size of final crop (default: 384)
            resize_size: Size to resize before cropping (default: 512)
        """
        self.crop_size = crop_size
        self.resize_size = resize_size
        
        # Resize transform
        self.resize = transforms.Resize((resize_size, resize_size))
        
        # Random crop
        self.random_crop = transforms.RandomCrop(crop_size)
    
    def __call__(self, image, mask):
        """
        Apply random crop augmentation.
        
        Args:
            image: PIL Image
            mask: PIL Image (grayscale mask)
            
        Returns:
            Cropped image and mask (PIL Images)
        """
        # Resize both to larger size
        image = self.resize(image)
        mask = self.resize(mask)
        
        # Get random crop parameters (same for both)
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.crop_size, self.crop_size)
        )
        
        # Apply same crop to both
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        
        return image, mask


class GlobalBrightnessAugmentation:
    """
    Global brightness augmentation (Task 4).
    
    Applies uniform brightness adjustment to entire image.
    """
    
    def __init__(self, brightness_factor=0.3):
        """
        Args:
            brightness_factor: Range of brightness adjustment (default: 0.3)
                              Final brightness will be in [1-factor, 1+factor]
        """
        self.brightness_factor = brightness_factor
    
    def __call__(self, image):
        """
        Apply global brightness augmentation.
        
        Args:
            image: PIL Image
            
        Returns:
            Augmented image (PIL Image)
        """
        # Random brightness factor
        factor = 1.0 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
        
        # Apply brightness adjustment
        enhancer = ImageEnhance.Brightness(image)
        image_aug = enhancer.enhance(factor)
        
        return image_aug


class LocalBrightnessAugmentation:
    """
    Local/boundary brightness augmentation (Task 5).
    
    Applies brightness adjustment only near shadow boundaries.
    This specifically targets the boundary FP oversegmentation issue.
    """
    
    def __init__(self, brightness_factor=0.3, boundary_width=10):
        """
        Args:
            brightness_factor: Range of brightness adjustment (default: 0.3)
            boundary_width: Width of boundary region in pixels (default: 10)
        """
        self.brightness_factor = brightness_factor
        self.boundary_width = boundary_width
    
    def __call__(self, image, mask):
        """
        Apply local brightness augmentation near boundaries.
        
        Args:
            image: PIL Image
            mask: PIL Image (grayscale mask)
            
        Returns:
            Augmented image (PIL Image), mask unchanged
        """
        # Convert to numpy
        image_np = np.array(image).astype(np.float32)
        mask_np = np.array(mask)
        
        # Create boundary mask
        boundary_mask = self._create_boundary_mask(mask_np)
        
        # Random brightness factor
        factor = 1.0 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
        
        # Apply brightness only in boundary region
        image_aug = image_np.copy()
        for c in range(3):  # RGB channels
            image_aug[:, :, c] = (
                image_np[:, :, c] * (1 - boundary_mask) + 
                image_np[:, :, c] * factor * boundary_mask
            )
        
        # Clip and convert back
        image_aug = np.clip(image_aug, 0, 255).astype(np.uint8)
        image_aug = Image.fromarray(image_aug)
        
        return image_aug, mask
    
    def _create_boundary_mask(self, mask):
        """
        Create boundary mask using morphological operations.
        
        Args:
            mask: Binary mask [H, W]
            
        Returns:
            Boundary mask [H, W] with values [0, 1]
        """
        # Binarize
        mask_binary = (mask > 127).astype(np.uint8)
        
        # Dilate and erode
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.boundary_width * 2 + 1, self.boundary_width * 2 + 1)
        )
        dilated = cv2.dilate(mask_binary, kernel, iterations=1)
        eroded = cv2.erode(mask_binary, kernel, iterations=1)
        
        # Boundary = dilated - eroded
        boundary = dilated - eroded
        
        # Normalize to [0, 1]
        boundary = boundary.astype(np.float32)
        
        return boundary


class MultiCropSampler:
    """
    Generate multiple crops from a single image for size-balanced sampling (Task 3).
    
    For images with small shadows, extract multiple crops containing the shadow.
    """
    
    def __init__(self, crop_size=384, num_crops=3):
        """
        Args:
            crop_size: Size of each crop (default: 384)
            num_crops: Number of crops to generate (default: 3)
        """
        self.crop_size = crop_size
        self.num_crops = num_crops
    
    def __call__(self, image, mask, shadow_centers):
        """
        Generate multiple crops centered around shadows.
        
        Args:
            image: PIL Image or numpy array
            mask: PIL Image or numpy array
            shadow_centers: List of (y, x) coordinates of shadow centroids
            
        Returns:
            List of (image_crop, mask_crop) tuples
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            mask_np = np.array(mask)
        else:
            image_np = image
            mask_np = mask
        
        H, W = image_np.shape[:2]
        crops = []
        
        # Generate crops
        for i in range(min(self.num_crops, len(shadow_centers))):
            # Get shadow center
            cy, cx = shadow_centers[i]
            
            # Add random offset
            offset_y = np.random.randint(-self.crop_size // 4, self.crop_size // 4)
            offset_x = np.random.randint(-self.crop_size // 4, self.crop_size // 4)
            cy += offset_y
            cx += offset_x
            
            # Compute crop boundaries
            top = max(0, cy - self.crop_size // 2)
            left = max(0, cx - self.crop_size // 2)
            bottom = min(H, top + self.crop_size)
            right = min(W, left + self.crop_size)
            
            # Adjust if crop extends beyond image
            if bottom - top < self.crop_size:
                top = max(0, bottom - self.crop_size)
            if right - left < self.crop_size:
                left = max(0, right - self.crop_size)
            
            # Extract crop
            image_crop = image_np[top:bottom, left:right]
            mask_crop = mask_np[top:bottom, left:right]
            
            # Pad if necessary
            if image_crop.shape[0] < self.crop_size or image_crop.shape[1] < self.crop_size:
                image_crop = self._pad_to_size(image_crop, self.crop_size)
                mask_crop = self._pad_to_size(mask_crop, self.crop_size)
            
            # Convert back to PIL
            image_crop = Image.fromarray(image_crop)
            mask_crop = Image.fromarray(mask_crop)
            
            crops.append((image_crop, mask_crop))
        
        return crops
    
    def _pad_to_size(self, img, target_size):
        """Pad image to target size"""
        if img.ndim == 3:
            padded = np.zeros((target_size, target_size, img.shape[2]), dtype=img.dtype)
            padded[:img.shape[0], :img.shape[1], :] = img
        else:
            padded = np.zeros((target_size, target_size), dtype=img.dtype)
            padded[:img.shape[0], :img.shape[1]] = img
        return padded


def compute_shadow_centers(mask):
    """
    Compute centroids of all shadow regions.
    
    Args:
        mask: Binary mask [H, W] or PIL Image
        
    Returns:
        List of (y, x) centroid coordinates
    """
    # Convert to numpy if needed
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    else:
        mask_np = mask
    
    # Binarize
    mask_binary = (mask_np > 127).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask_binary)
    
    # Compute centroids
    centers = []
    for label_id in range(1, num_labels):
        component = (labels == label_id)
        if component.sum() > 0:
            y_coords, x_coords = np.where(component)
            cy = int(y_coords.mean())
            cx = int(x_coords.mean())
            centers.append((cy, cx))
    
    return centers


if __name__ == "__main__":
    # Test augmentations
    import matplotlib.pyplot as plt
    
    # Create test image and mask
    test_img = np.ones((384, 384, 3), dtype=np.uint8) * 150
    test_mask = np.zeros((384, 384), dtype=np.uint8)
    test_mask[150:250, 150:250] = 255  # Shadow region
    
    test_img_pil = Image.fromarray(test_img)
    test_mask_pil = Image.fromarray(test_mask)
    
    # Test Task 1: Random Crop
    print("Testing Random Crop Augmentation...")
    aug1 = RandomCropAugmentation(crop_size=384, resize_size=512)
    img_crop, mask_crop = aug1(test_img_pil, test_mask_pil)
    print(f"  Output size: {np.array(img_crop).shape}")
    
    # Test Task 4: Global Brightness
    print("Testing Global Brightness Augmentation...")
    aug4 = GlobalBrightnessAugmentation(brightness_factor=0.3)
    img_bright = aug4(test_img_pil)
    print(f"  Mean brightness change: {np.array(img_bright).mean() - test_img.mean():.1f}")
    
    # Test Task 5: Local Brightness
    print("Testing Local Brightness Augmentation...")
    aug5 = LocalBrightnessAugmentation(brightness_factor=0.3, boundary_width=10)
    img_local, _ = aug5(test_img_pil, test_mask_pil)
    print(f"  Boundary augmentation applied")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(test_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_crop)
    axes[0, 1].set_title('Random Crop')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_bright)
    axes[0, 2].set_title('Global Brightness')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_local)
    axes[1, 0].set_title('Local Brightness')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(test_mask, cmap='gray')
    axes[1, 1].set_title('Mask')
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/augmentation_test.png', dpi=150, bbox_inches='tight')
    print("\nTest visualization saved to /home/claude/augmentation_test.png")