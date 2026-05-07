"""
FDA: Fourier Domain Adaptation for Semantic Segmentation
CVPR 2020 - Yang & Soatto

Swaps low-frequency amplitude spectrum between source and target images.
Model-agnostic preprocessing method for domain adaptation.
"""

import numpy as np
import torch
import cv2
from typing import Union, Optional


def extract_amplitude_spectrum(img_np: np.ndarray) -> np.ndarray:
    """
    Extract amplitude spectrum from image using FFT.
    
    Args:
        img_np: Image array [H, W, C] in range [0, 255]
    
    Returns:
        Amplitude spectrum [H, W, C]
    """
    # Apply FFT to each channel
    fft = np.fft.fft2(img_np, axes=(0, 1))
    amplitude = np.abs(fft)
    return amplitude


def extract_phase_spectrum(img_np: np.ndarray) -> np.ndarray:
    """
    Extract phase spectrum from image using FFT.
    
    Args:
        img_np: Image array [H, W, C] in range [0, 255]
    
    Returns:
        Phase spectrum [H, W, C]
    """
    fft = np.fft.fft2(img_np, axes=(0, 1))
    phase = np.angle(fft)
    return phase


def low_freq_mutate(amp_src: np.ndarray, amp_trg: np.ndarray, L: float = 0.01) -> np.ndarray:
    """
    Swap low-frequency components of amplitude spectrum.
    
    Args:
        amp_src: Source amplitude spectrum [H, W, C]
        amp_trg: Target amplitude spectrum [H, W, C]
        L: Ratio for low-frequency components (default 0.01 = 1%)
    
    Returns:
        Mutated amplitude spectrum [H, W, C]
    """
    h, w = amp_src.shape[:2]
    
    # Calculate boundary for low-frequency region
    b = int(np.floor(min(h, w) * L))
    
    # Create mask for center (low-frequency) region
    # Shift so that low frequencies are at center
    amp_src_shifted = np.fft.fftshift(amp_src, axes=(0, 1))
    amp_trg_shifted = np.fft.fftshift(amp_trg, axes=(0, 1))
    
    # Replace low-frequency region of source with target
    c_h, c_w = h // 2, w // 2
    amp_src_shifted[c_h - b:c_h + b, c_w - b:c_w + b] = \
        amp_trg_shifted[c_h - b:c_h + b, c_w - b:c_w + b]
    
    # Shift back
    amp_mutated = np.fft.ifftshift(amp_src_shifted, axes=(0, 1))
    
    return amp_mutated


def fda_source_to_target(src_img: np.ndarray, trg_img: np.ndarray, L: float = 0.01) -> np.ndarray:
    """
    Apply FDA: transfer target style to source image.
    
    Args:
        src_img: Source image [H, W, C] in range [0, 255], uint8
        trg_img: Target image [H, W, C] in range [0, 255], uint8
        L: Low-frequency ratio (beta in paper), typically 0.01-0.09
    
    Returns:
        Adapted image [H, W, C] in range [0, 255], uint8
    """
    # Convert to float
    src_img = src_img.astype(np.float32)
    trg_img = trg_img.astype(np.float32)
    
    # Resize target to match source if needed
    if src_img.shape[:2] != trg_img.shape[:2]:
        trg_img = cv2.resize(trg_img, (src_img.shape[1], src_img.shape[0]))
    
    # Extract amplitude and phase from source
    fft_src = np.fft.fft2(src_img, axes=(0, 1))
    amp_src = np.abs(fft_src)
    pha_src = np.angle(fft_src)
    
    # Extract amplitude from target
    fft_trg = np.fft.fft2(trg_img, axes=(0, 1))
    amp_trg = np.abs(fft_trg)
    
    # Mutate amplitude: swap low-frequency components
    amp_mutated = low_freq_mutate(amp_src, amp_trg, L=L)
    
    # Combine mutated amplitude with source phase
    fft_mutated = amp_mutated * np.exp(1j * pha_src)
    
    # Inverse FFT to get adapted image
    img_adapted = np.fft.ifft2(fft_mutated, axes=(0, 1))
    img_adapted = np.real(img_adapted)
    
    # Clip to valid range
    img_adapted = np.clip(img_adapted, 0, 255).astype(np.uint8)
    
    return img_adapted


class FDATransform:
    """
    FDA transformation as a callable class for integration with PyTorch Dataset.
    
    Usage:
        fda_transform = FDATransform(target_images, L=0.01)
        adapted_img = fda_transform(source_img)
    """
    
    def __init__(self, target_images: list, L: float = 0.01, target_selection: str = 'random'):
        """
        Args:
            target_images: List of target domain images (numpy arrays [H, W, C])
            L: Low-frequency ratio (beta parameter)
            target_selection: How to select target image ('random' or 'fixed')
        """
        self.target_images = target_images
        self.L = L
        self.target_selection = target_selection
        self.fixed_target_idx = 0
        
        if len(target_images) == 0:
            raise ValueError("target_images list cannot be empty")
    
    def __call__(self, src_img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply FDA to source image.
        
        Args:
            src_img: Source image, numpy [H, W, C] or torch tensor [C, H, W]
        
        Returns:
            Adapted image as numpy array [H, W, C]
        """
        # Convert torch tensor to numpy if needed
        if isinstance(src_img, torch.Tensor):
            src_img = src_img.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
            src_img = (src_img * 255).astype(np.uint8)
        
        # Select target image
        if self.target_selection == 'random':
            trg_idx = np.random.randint(0, len(self.target_images))
        else:  # fixed
            trg_idx = self.fixed_target_idx
            self.fixed_target_idx = (self.fixed_target_idx + 1) % len(self.target_images)
        
        trg_img = self.target_images[trg_idx]
        
        # Apply FDA
        adapted_img = fda_source_to_target(src_img, trg_img, L=self.L)
        
        return adapted_img


def load_target_images(target_root: str, max_images: int = 100) -> list:
    """
    Load target domain images for FDA.
    
    Args:
        target_root: Path to target domain images directory
        max_images: Maximum number of images to load (for memory efficiency)
    
    Returns:
        List of numpy arrays [H, W, C]
    """
    import os
    from PIL import Image
    
    target_images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    # Find all image files
    img_files = []
    if os.path.isdir(target_root):
        for root, dirs, files in os.walk(target_root):
            for f in files:
                if f.lower().endswith(valid_extensions):
                    img_files.append(os.path.join(root, f))
    else:
        raise ValueError(f"target_root {target_root} is not a directory")
    
    # Randomly sample if too many images
    if len(img_files) > max_images:
        img_files = np.random.choice(img_files, max_images, replace=False)
    
    # Load images
    for img_path in img_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            target_images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            continue
    
    print(f"Loaded {len(target_images)} target domain images from {target_root}")
    return target_images


if __name__ == "__main__":
    # Test FDA
    import matplotlib.pyplot as plt
    
    # Create synthetic source and target
    src = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    trg = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    
    # Apply FDA with different L values
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(src)
    axes[0].set_title('Source')
    axes[0].axis('off')
    
    axes[1].imshow(trg)
    axes[1].set_title('Target')
    axes[1].axis('off')
    
    for idx, L in enumerate([0.01, 0.05, 0.09]):
        adapted = fda_source_to_target(src, trg, L=L)
        axes[idx + 2].imshow(adapted)
        axes[idx + 2].set_title(f'FDA (L={L})')
        axes[idx + 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('fda_test.png', dpi=150, bbox_inches='tight')
    print("FDA test saved to fda_test.png")