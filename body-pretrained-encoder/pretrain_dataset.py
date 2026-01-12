"""
Dataset for self-supervised pretraining on body point clouds.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

from utils import normalize_point_cloud, random_rotate, random_scale, random_jitter


class BodyPretrainDataset(Dataset):
    """
    Dataset for loading body surface point clouds for self-supervised pretraining.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_points: int = 8192,
        train_ratio: float = 0.8,
        augment: bool = True,
        rotate: bool = True,
        jitter: float = 0.01,
        scale_range: Tuple[float, float] = (0.8, 1.2),
    ):
        """
        Args:
            data_root: Path to voxel_data directory
            split: 'train' or 'val'
            num_points: Number of points to sample per cloud
            train_ratio: Ratio of samples for training
            augment: Whether to apply augmentation
            rotate: Apply random rotation
            jitter: Jitter noise std
            scale_range: Random scale range
        """
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.augment = augment and (split == 'train')
        self.rotate = rotate
        self.jitter = jitter
        self.scale_range = scale_range

        # Get all npz files
        all_files = sorted(glob.glob(os.path.join(data_root, '*.npz')))
        if len(all_files) == 0:
            raise ValueError(f"No npz files found in {data_root}")

        # Split train/val
        np.random.seed(42)
        indices = np.random.permutation(len(all_files))
        split_idx = int(len(all_files) * train_ratio)

        if split == 'train':
            self.files = [all_files[i] for i in indices[:split_idx]]
        else:
            self.files = [all_files[i] for i in indices[split_idx:]]

        print(f"[{split}] Loaded {len(self.files)} samples from {data_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        # Load point cloud
        data = np.load(self.files[idx])
        points = data['sensor_pc'].astype(np.float32)  # (N, 3)

        # Random subsample
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            # Pad by repeating
            pad_size = self.num_points - len(points)
            pad_indices = np.random.choice(len(points), pad_size, replace=True)
            points = np.concatenate([points, points[pad_indices]], axis=0)

        points = torch.from_numpy(points)

        # Normalize
        normalized, centroid, scale = normalize_point_cloud(points)

        # Augmentation
        if self.augment:
            if self.rotate:
                normalized = random_rotate(normalized)
            if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
                normalized = random_scale(normalized, self.scale_range)
            if self.jitter > 0:
                normalized = random_jitter(normalized, self.jitter)

        # Create features: concat xyz with normalized_xyz
        # Original coords (denormalized for reconstruction target)
        original_coords = normalized * scale + centroid
        # Features: xyz + normalized_xyz
        feat = torch.cat([original_coords, normalized], dim=-1)  # (N, 6)

        return {
            'coord': original_coords,  # (N, 3) - for reconstruction target
            'feat': feat,              # (N, 6) - input features
            'normalized': normalized,  # (N, 3) - normalized coords
            'centroid': centroid,      # (3,)
            'scale': scale,            # scalar
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    Collate function for batching point clouds.
    Creates offset tensor for batch indexing (used by PTv3).
    """
    coords = []
    feats = []
    offset = []

    for i, sample in enumerate(batch):
        coords.append(sample['coord'])
        feats.append(sample['feat'])
        offset.append(sample['coord'].shape[0])

    # Stack
    coord = torch.cat(coords, dim=0)
    feat = torch.cat(feats, dim=0)
    offset = torch.cumsum(torch.tensor(offset), dim=0)

    return {
        'coord': coord,
        'feat': feat,
        'offset': offset,
    }


def build_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders from config."""

    train_dataset = BodyPretrainDataset(
        data_root=config.data_root,
        split='train',
        num_points=config.num_points,
        train_ratio=config.train_ratio,
        augment=True,
        rotate=config.rotate,
        jitter=config.jitter,
        scale_range=config.scale_range,
    )

    val_dataset = BodyPretrainDataset(
        data_root=config.data_root,
        split='val',
        num_points=config.num_points,
        train_ratio=config.train_ratio,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
