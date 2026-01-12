"""
PTv3 Encoder Self-Supervised Pretraining Module for Body Point Clouds.

Usage:
    python pretrain.py

After pretraining, encoder weights will be saved to:
    checkpoints/encoder_best.pth
"""

from .config import config, PretrainConfig
from .ptv3_encoder import PTv3EncoderWrapper, MaskedAutoEncoder, build_mae_model
from .pretrain_dataset import BodyPretrainDataset, build_dataloaders
from .utils import random_mask, interpolate_features, normalize_point_cloud

__all__ = [
    'config',
    'PretrainConfig',
    'PTv3EncoderWrapper',
    'MaskedAutoEncoder',
    'build_mae_model',
    'BodyPretrainDataset',
    'build_dataloaders',
    'random_mask',
    'interpolate_features',
    'normalize_point_cloud',
]
