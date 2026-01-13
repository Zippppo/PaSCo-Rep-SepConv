"""
Dataset for self-supervised pretraining on body point clouds.
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

try:
    from .utils import normalize_point_cloud, random_rotate, random_scale, random_jitter
except ImportError:
    from utils import normalize_point_cloud, random_rotate, random_scale, random_jitter


class BodyPretrainDataset(Dataset):
    """
    Dataset for loading body surface point clouds for self-supervised pretraining.
    Uses dataset_split.json for train/val/test splits.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        split_file: str = None,
        num_points: int = 8192,
        augment: bool = True,
        rotate: bool = True,
        jitter: float = 0.01,
        scale_range: Tuple[float, float] = (0.8, 1.2),
    ):
        """
        Args:
            data_root: Path to voxel_data directory
            split: 'train', 'val', or 'test'
            split_file: Path to dataset_split.json (if None, looks in project root)
            num_points: Number of points to sample per cloud
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

        # Load split file
        if split_file is None:
            # Look for dataset_split.json in project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            split_file = os.path.join(project_root, 'dataset_split.json')

        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_data = json.load(f)

            sample_ids = split_data['splits'].get(split, [])
            if len(sample_ids) == 0:
                raise ValueError(f"No samples found for split '{split}' in {split_file}")

            # Build file paths from sample IDs
            self.files = []
            for sample_id in sample_ids:
                file_path = os.path.join(data_root, f'{sample_id}.npz')
                if os.path.exists(file_path):
                    self.files.append(file_path)

            if len(self.files) == 0:
                raise ValueError(f"No npz files found in {data_root} for split '{split}'")

            print(f"[{split}] Loaded {len(self.files)} samples from {split_file}")
        else:
            # Fallback: use all files with random split (legacy behavior)
            print(f"Warning: {split_file} not found, falling back to random split")
            all_files = sorted(glob.glob(os.path.join(data_root, '*.npz')))
            if len(all_files) == 0:
                raise ValueError(f"No npz files found in {data_root}")

            np.random.seed(42)
            indices = np.random.permutation(len(all_files))
            train_end = int(len(all_files) * 0.8)
            val_end = int(len(all_files) * 0.9)

            if split == 'train':
                self.files = [all_files[i] for i in indices[:train_end]]
            elif split == 'val':
                self.files = [all_files[i] for i in indices[train_end:val_end]]
            else:  # test
                self.files = [all_files[i] for i in indices[val_end:]]

            print(f"[{split}] Loaded {len(self.files)} samples (random split)")

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

        # Normalize to unit sphere
        normalized, centroid, scale = normalize_point_cloud(points)

        # Apply augmentation (on normalized coordinates)
        augmented = normalized.clone()
        if self.augment:
            if self.rotate:
                augmented = random_rotate(augmented)
            if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
                augmented = random_scale(augmented, self.scale_range)
            if self.jitter > 0:
                augmented = random_jitter(augmented, self.jitter)

        # Denormalize augmented coords for reconstruction target
        # Note: reconstruction target is in augmented space (consistent with input)
        coords = augmented * scale + centroid

        # Features: augmented xyz (absolute) + augmented xyz (normalized)
        feat = torch.cat([coords, augmented], dim=-1)  # (N, 6)

        return {
            'coord': coords,           # (N, 3) - reconstruction target (augmented space)
            'feat': feat,              # (N, 6) - input features
            'normalized': augmented,   # (N, 3) - normalized augmented coords
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
    offset = torch.cumsum(torch.tensor(offset, dtype=torch.long), dim=0)

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
        split_file=getattr(config, 'split_file', None),
        num_points=config.num_points,
        augment=True,
        rotate=config.rotate,
        jitter=config.jitter,
        scale_range=config.scale_range,
    )

    val_dataset = BodyPretrainDataset(
        data_root=config.data_root,
        split='val',
        split_file=getattr(config, 'split_file', None),
        num_points=config.num_points,
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
