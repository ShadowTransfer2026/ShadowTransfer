"""Data module for DINOv3 with GSDPE"""

from .dataset_gsdpe import ShadowDatasetGSDPE, get_dataloaders_gsdpe, LOCO_FOLDS

__all__ = [
    'ShadowDatasetGSDPE',
    'get_dataloaders_gsdpe',
    'LOCO_FOLDS',
]