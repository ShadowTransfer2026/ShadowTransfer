"""
GSDPE: Ground Sample Distance Positional Encoding
Original formula from Scale-MAE (ICCV 2023) for Vision Transformers.

Reference:
Reed et al. "Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale 
Geospatial Representation Learning" ICCV 2023

Original GSDPE formula:
    v_gsd,x(pos, 2i) = sin((g/G) * pos / 10000^(2i/D))
    v_gsd,y(pos, 2i+1) = cos((g/G) * pos / 10000^(2i/D))
    
where g = image GSD, G = reference GSD (1.0m)

This implementation is designed for ViT architectures where positional encoding
is added to patch embeddings.
"""

import torch
import torch.nn as nn
import numpy as np


class GSDPE(nn.Module):
    """
    Ground Sample Distance Positional Encoding for Vision Transformers.
    
    Applies scale-aware positional encoding to patch embeddings based on
    the Ground Sample Distance (GSD) of the input image.
    
    Args:
        embed_dim: Embedding dimension of patches (e.g., 384 for ViT-S)
        reference_gsd: Reference GSD value (default: 1.0m as in Scale-MAE)
        num_patches_h: Number of patches in height direction
        num_patches_w: Number of patches in width direction
    """
    
    def __init__(self, embed_dim=384, reference_gsd=1.0, num_patches_h=24, num_patches_w=24):
        super(GSDPE, self).__init__()
        
        self.embed_dim = embed_dim
        self.reference_gsd = reference_gsd
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        # Pre-compute frequency bands for sinusoidal encoding
        # freq = 1 / (10000^(2i/D)) where i is the feature index
        self.register_buffer(
            'freq_bands',
            self._get_frequency_bands(embed_dim)
        )
        
    def _get_frequency_bands(self, embed_dim):
        """
        Compute frequency bands for sinusoidal encoding.
        freq = 1 / (10000^(2i/D)) where i is the feature index
        """
        # For half the channels (sin) and half (cos)
        num_bands = embed_dim // 2
        freq_bands = 1.0 / (10000 ** (torch.arange(0, num_bands, dtype=torch.float32) / num_bands))
        return freq_bands
    
    def _create_gsd_scaled_encoding(self, height, width, gsd_ratio, device):
        """
        Create GSD-scaled positional encoding using original Scale-MAE formula.
        
        Args:
            height, width: Number of patches in each dimension
            gsd_ratio: g/G where g=image GSD, G=reference GSD
            device: Device to create tensors on
            
        Returns:
            encoding: [embed_dim, height, width]
        """
        # Create position grids for patches
        # pos ranges from 0 to height-1 and 0 to width-1
        y_pos = torch.arange(height, dtype=torch.float32, device=device)
        x_pos = torch.arange(width, dtype=torch.float32, device=device)
        
        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
        
        # Scale positions by GSD ratio (g/G)
        # This is the key insight: positions are scaled by the area covered
        scaled_y = y_grid * gsd_ratio  # [H, W]
        scaled_x = x_grid * gsd_ratio  # [H, W]
        
        # Apply sinusoidal encoding to scaled positions
        num_bands = self.embed_dim // 2
        
        # Y-direction encoding (sin for even indices)
        # v_gsd,x(pos, 2i) = sin((g/G) * pos / 10000^(2i/D))
        y_encoding_sin = torch.sin(
            scaled_y.unsqueeze(0) * self.freq_bands.view(-1, 1, 1)
        )  # [num_bands, H, W]
        
        # X-direction encoding (cos for odd indices)
        # v_gsd,y(pos, 2i+1) = cos((g/G) * pos / 10000^(2i/D))
        x_encoding_cos = torch.cos(
            scaled_x.unsqueeze(0) * self.freq_bands.view(-1, 1, 1)
        )  # [num_bands, H, W]
        
        # Interleave sin and cos
        encoding = torch.zeros(self.embed_dim, height, width, device=device)
        encoding[0::2] = y_encoding_sin  # Even indices: sin(y)
        encoding[1::2] = x_encoding_cos  # Odd indices: cos(x)
        
        return encoding
    
    def forward(self, patch_embeddings, gsd):
        """
        Add GSD-aware positional encoding to patch embeddings.
        
        Args:
            patch_embeddings: Patch embeddings [B, H, W, D] or [B, N, D]
            gsd: GSD values for each sample [B] or [B, 1]
            
        Returns:
            encoded_embeddings: Patch embeddings with GSDPE added (same shape as input)
        """
        # Handle both [B, H, W, D] and [B, N, D] formats
        original_shape = patch_embeddings.shape
        
        if patch_embeddings.dim() == 4:
            # [B, H, W, D] format - reshape to [B, N, D]
            B, H, W, D = patch_embeddings.shape
            patch_embeddings = patch_embeddings.reshape(B, H * W, D)
        elif patch_embeddings.dim() == 3:
            B, N, D = patch_embeddings.shape
            H = W = int(N ** 0.5)
        else:
            raise ValueError(f"Unexpected patch_embeddings shape: {patch_embeddings.shape}")
        
        # Ensure gsd is the right shape
        if gsd.dim() == 1:
            gsd = gsd.unsqueeze(1)  # [B, 1]
        
        # Compute GSD ratio: g/G
        gsd_ratio = gsd / self.reference_gsd  # [B, 1]
        
        # Create encoding for each sample
        encodings = []
        for i in range(B):
            encoding = self._create_gsd_scaled_encoding(
                H, W,
                gsd_ratio[i].item(),
                patch_embeddings.device
            )  # [D, H, W]
            
            # Reshape to [N, D] where N = H*W
            encoding = encoding.flatten(1).transpose(0, 1)  # [N, D]
            encodings.append(encoding)
        
        encodings = torch.stack(encodings, dim=0)  # [B, N, D]
        
        # Add encoding to patch embeddings
        encoded_embeddings = patch_embeddings + encodings
        
        # Restore original shape if needed
        if len(original_shape) == 4:
            encoded_embeddings = encoded_embeddings.reshape(B, H, W, D)
        
        return encoded_embeddings


def get_gsd_from_filename(filename):
    """
    Extract GSD value from filename.
    
    Args:
        filename: Image filename containing 'midres' or 'highres'
        
    Returns:
        gsd: GSD value in meters (0.6 for midres, 0.3 for highres)
    """
    if 'midres' in filename.lower():
        return 0.6
    elif 'highres' in filename.lower():
        return 0.3
    else:
        # Default to midres if not specified
        print(f"Warning: Could not determine GSD from filename {filename}, defaulting to 0.6m")
        return 0.6


if __name__ == "__main__":
    # Test GSDPE module
    print("Testing GSDPE Module (Original Formula)")
    print("=" * 50)
    
    # Create GSDPE module for ViT-S (384 dim, 24x24 patches)
    gsdpe = GSDPE(embed_dim=384, reference_gsd=1.0, num_patches_h=24, num_patches_w=24)
    
    # Test with different GSDs
    batch_size = 4
    num_patches = 24 * 24  # 576 patches
    embed_dim = 384
    
    # Simulate patch embeddings
    patch_embeddings = torch.randn(batch_size, num_patches, embed_dim)
    
    # Mix of midres (0.6m) and highres (0.3m)
    gsd_values = torch.tensor([0.6, 0.3, 0.6, 0.3])
    
    print(f"Input patch embeddings shape: {patch_embeddings.shape}")
    print(f"GSD values: {gsd_values.tolist()}")
    
    # Apply GSDPE
    encoded_embeddings = gsdpe(patch_embeddings, gsd_values)
    
    print(f"Output embeddings shape: {encoded_embeddings.shape}")
    print(f"Output statistics:")
    print(f"  Mean: {encoded_embeddings.mean().item():.4f}")
    print(f"  Std: {encoded_embeddings.std().item():.4f}")
    
    # Test that different GSDs produce different encodings
    single_patch = patch_embeddings[0:1]  # Take first sample
    
    encoded_midres = gsdpe(single_patch, torch.tensor([0.6]))
    encoded_highres = gsdpe(single_patch, torch.tensor([0.3]))
    
    diff = (encoded_midres - encoded_highres).abs().mean()
    print(f"\nDifference between midres and highres encoding: {diff.item():.4f}")
    print("  (Should be non-zero, confirming GSD-awareness)")
    
    print("\n" + "=" * 50)
    print("GSDPE module test completed successfully!")