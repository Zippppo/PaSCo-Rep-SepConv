"""
Configuration for PTv3 encoder pretraining on body point clouds.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PretrainConfig:
    # Data
    data_root: str = "Dataset/voxel_data"
    split_file: Optional[str] = None  # Path to dataset_split.json (auto-detected if None)
    num_points: int = 8192

    # PTv3 Encoder
    in_channels: int = 6  # xyz + normalized_xyz
    enc_depths: Tuple[int, ...] = (2, 2, 2, 6, 2)
    enc_channels: Tuple[int, ...] = (32, 64, 128, 256, 512)
    enc_num_head: Tuple[int, ...] = (2, 4, 8, 16, 32)
    enc_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024, 1024)

    # PTv3 Decoder (for full resolution feature output)
    dec_depths: Tuple[int, ...] = (2, 2, 2, 2)
    dec_channels: Tuple[int, ...] = (64, 64, 128, 256)
    dec_num_head: Tuple[int, ...] = (4, 4, 8, 16)
    dec_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024)

    grid_size: float = 4.0  # mm, match voxel resolution
    order: Tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")

    # Masking
    mask_ratio: float = 0.6
    mask_group_size: int = 32

    # Validation metrics
    fscore_threshold: float = 0.02  # F-Score threshold (adjust based on data scale)

    # MAE Decoder (MLP for coordinate prediction)
    mae_decoder_hidden_dims: Tuple[int, ...] = (256, 128)

    # Training
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 100
    warmup_epochs: int = 10
    grad_clip_norm: float = 1.0  # Gradient clipping max norm

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
    use_amp: bool = True  # Use automatic mixed precision


config = PretrainConfig()
