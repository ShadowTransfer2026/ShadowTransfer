"""
Data package for MAMNet
"""

from .dataset import ShadowDataset, AISDataset, get_dataloaders

__all__ = ['ShadowDataset', 'AISDataset', 'get_dataloaders']