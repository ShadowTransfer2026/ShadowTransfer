"""
GSDPE: Ground Sample Distance Positional Encoding
Adapted from Scale-MAE (ICCV 2023) for CNN architectures.

Reference:
Reed et al. "Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale 
Geospatial Representation Learning" ICCV 2023

Original GSDPE (for ViTs):
    v_gsd,x(pos, 2i) = sin((g/G) * pos / 10000^(2i/D))
    v_gsd,y(pos, 2i+1) = cos((g/G) * pos / 10000^(2i/D))
    
where g = image GSD, G = reference GSD (1m)

Adaptation for CNNs:
    - Create spatial encoding maps that are scaled by GSD ratio
    - Add to feature maps at encoder input (equivalent to adding to patches in ViTs)
"""

import torch
import torch.nn as nn
import numpy as np


class GSDPE(nn.Module):
    """
    Ground Sample Distance Positional Encoding for CNNs.
    
    This module creates GSD-aware spatial encodings that are added to 
    feature maps, similar to how GSDPE is added to patch embeddings in ViTs.
    
    Args:
        channels: Number of channels in the feature map
        reference_gsd: Reference GSD value (default: 1.0m as in Scale-MAE)
        max_spatial_size: Maximum spatial size (H, W) to pre-compute encodings
    """
    
    def __init__(self, channels=64, reference_gsd=1.0, max_spatial_size=384):
        super(GSDPE, self).__init__()
        
        self.channels = channels
        self.reference_gsd = reference_gsd
        self.max_spatial_size = max_spatial_size
        
        # Pre-compute frequency bands for sinusoidal encoding
        # Following the transformer positional encoding formula
        self.register_buffer(
            'freq_bands',
            self._get_frequency_bands(channels)
        )
        
    def _get_frequency_bands(self, channels):
        """
        Compute frequency bands for sinusoidal encoding.
        freq = 1 / (10000^(2i/D)) where i is the feature index
        """
        # For half the channels (sin) and half (cos)
        num_bands = channels // 2
        freq_bands = 1.0 / (10000 ** (torch.arange(0, num_bands, dtype=torch.float32) / num_bands))
        return freq_bands
    
    def _create_gsd_scaled_encoding(self, height, width, gsd_ratio, device):
        """
        Create GSD-scaled positional encoding.
        
        Args:
            height, width: Spatial dimensions of feature map
            gsd_ratio: g/G where g=image GSD, G=reference GSD
            device: Device to create tensors on
            
        Returns:
            encoding: [channels, height, width]
        """
        # Create coordinate grids (normalized to [0, 1])
        y_coords = torch.linspace(0, 1, height, device=device)
        x_coords = torch.linspace(0, 1, width, device=device)
        
        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Scale coordinates by GSD ratio (g/G)
        # This is the key insight from Scale-MAE: positions are scaled by area covered
        scaled_y = y_grid * gsd_ratio * height  # Scale by spatial extent
        scaled_x = x_grid * gsd_ratio * width
        
        # Apply sinusoidal encoding to scaled positions
        num_bands = self.channels // 2
        
        # Y-direction encoding (sin for even indices)
        y_encoding_sin = torch.sin(
            scaled_y.unsqueeze(0) * self.freq_bands.view(-1, 1, 1)
        )  # [num_bands, H, W]
        
        # X-direction encoding (cos for odd indices)
        x_encoding_cos = torch.cos(
            scaled_x.unsqueeze(0) * self.freq_bands.view(-1, 1, 1)
        )  # [num_bands, H, W]
        
        # Interleave sin and cos
        encoding = torch.zeros(self.channels, height, width, device=device)
        encoding[0::2] = y_encoding_sin  # Even indices: sin(y)
        encoding[1::2] = x_encoding_cos  # Odd indices: cos(x)
        
        return encoding
    
    def forward(self, x, gsd):
        """
        Add GSD-aware positional encoding to feature maps.
        
        Args:
            x: Feature map [B, C, H, W]
            gsd: GSD values for each sample [B] or [B, 1]
            
        Returns:
            x_encoded: Feature map with GSD encoding added [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Ensure gsd is the right shape
        if gsd.dim() == 1:
            gsd = gsd.unsqueeze(1)  # [B, 1]
        
        # Compute GSD ratio: g/G
        gsd_ratio = gsd / self.reference_gsd  # [B, 1]
        
        # Create encoding for each sample (since GSD can vary per sample)
        encodings = []
        for i in range(B):
            encoding = self._create_gsd_scaled_encoding(
                H, W, 
                gsd_ratio[i].item(), 
                x.device
            )  # [C, H, W]
            encodings.append(encoding)
        
        encodings = torch.stack(encodings, dim=0)  # [B, C, H, W]
        
        # Add encoding to input features
        # Note: We add rather than concatenate (following Scale-MAE)
        x_encoded = x + encodings
        
        return x_encoded


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
    print("Testing GSDPE Module")
    print("=" * 50)
    
    # Create GSDPE module
    gsdpe = GSDPE(channels=64, reference_gsd=1.0)
    
    # Test with different GSDs
    batch_size = 4
    feature_map = torch.randn(batch_size, 64, 96, 96)
    
    # Mix of midres (0.6m) and highres (0.3m)
    gsd_values = torch.tensor([0.6, 0.3, 0.6, 0.3])
    
    print(f"Input feature map shape: {feature_map.shape}")
    print(f"GSD values: {gsd_values.tolist()}")
    
    # Apply GSDPE
    encoded_features = gsdpe(feature_map, gsd_values)
    
    print(f"Output feature map shape: {encoded_features.shape}")
    print(f"Output statistics:")
    print(f"  Mean: {encoded_features.mean().item():.4f}")
    print(f"  Std: {encoded_features.std().item():.4f}")
    
    # Test that different GSDs produce different encodings
    single_feature = feature_map[0:1]  # Take first sample
    
    encoded_midres = gsdpe(single_feature, torch.tensor([0.6]))
    encoded_highres = gsdpe(single_feature, torch.tensor([0.3]))
    
    diff = (encoded_midres - encoded_highres).abs().mean()
    print(f"\nDifference between midres and highres encoding: {diff.item():.4f}")
    print("  (Should be non-zero, confirming GSD-awareness)")
    
    print("\n" + "=" * 50)
    print("GSDPE module test completed successfully!")