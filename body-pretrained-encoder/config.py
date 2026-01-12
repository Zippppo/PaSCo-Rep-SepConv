"""
Configuration for PTv3 encoder pretraining on body point clouds.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PretrainConfig:
    # Data
    data_root: str = "Dataset/voxel_data"
    num_points: int = 8192
    train_ratio: float = 0.8

    # PTv3 Encoder
    in_channels: int = 6  # xyz + normalized_xyz
    enc_depths: Tuple[int, ...] = (2, 2, 2, 6, 2)
    enc_channels: Tuple[int, ...] = (32, 64, 128, 256, 512)
    enc_num_head: Tuple[int, ...] = (2, 4, 8, 16, 32)
    enc_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024, 1024)
    grid_size: float = 4.0  # mm, match voxel resolution
    order: Tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")

    # Masking
    mask_ratio: float = 0.6
    mask_group_size: int = 32

    # Decoder
    decoder_dims: Tuple[int, ...] = (512, 256, 128, 3)

    # Training
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 100
    warmup_epochs: int = 10

    # Augmentation
    rotate: bool = True
    jitter: float = 0.01
    scale_range: Tuple[float, float] = (0.8, 1.2)

    # System
    num_workers: int = 4
    device: str = "cuda"
    checkpoint_dir: str = "body-pretrained-encoder/checkpoints"
    log_interval: int = 50
    save_interval: int = 10


config = PretrainConfig()
