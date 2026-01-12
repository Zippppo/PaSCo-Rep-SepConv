"""
Utility functions for masked point modeling.
"""

import torch
import torch.nn as nn
from typing import Tuple


def fps(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Farthest Point Sampling.

    Args:
        points: (N, 3) point coordinates
        num_samples: number of points to sample

    Returns:
        indices: (num_samples,) indices of sampled points
    """
    N = points.shape[0]
    if num_samples >= N:
        return torch.arange(N, device=points.device)

    device = points.device
    indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.ones(N, device=device) * 1e10

    # Start from random point
    farthest = torch.randint(0, N, (1,), device=device).item()

    for i in range(num_samples):
        indices[i] = farthest
        centroid = points[farthest].unsqueeze(0)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances).item()

    return indices


def knn(src: torch.Tensor, dst: torch.Tensor, k: int) -> torch.Tensor:
    """
    K-Nearest Neighbors.

    Args:
        src: (N, 3) source points
        dst: (M, 3) query points
        k: number of neighbors

    Returns:
        indices: (M, k) indices of k nearest neighbors in src for each dst point
    """
    # (M, N)
    dist = torch.cdist(dst, src)
    _, indices = dist.topk(k, dim=-1, largest=False)
    return indices


def random_mask(
    points: torch.Tensor,
    mask_ratio: float = 0.6,
    group_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random group masking using FPS centers.

    Args:
        points: (N, 3) point coordinates
        mask_ratio: ratio of points to mask
        group_size: number of points per group

    Returns:
        visible_points: (N_v, 3) visible points
        visible_idx: (N_v,) indices of visible points
        masked_points: (N_m, 3) masked points
        masked_idx: (N_m,) indices of masked points
    """
    N = points.shape[0]
    num_mask = int(N * mask_ratio)
    num_centers = max(1, num_mask // group_size)

    # FPS to select mask centers
    center_idx = fps(points, num_centers)
    centers = points[center_idx]

    # KNN to find points near each center
    neighbor_idx = knn(points, centers, group_size)  # (num_centers, group_size)
    masked_idx = neighbor_idx.flatten().unique()

    # Limit to desired mask ratio
    if len(masked_idx) > num_mask:
        perm = torch.randperm(len(masked_idx), device=points.device)[:num_mask]
        masked_idx = masked_idx[perm]

    # Get visible indices
    all_idx = torch.arange(N, device=points.device)
    mask = torch.ones(N, dtype=torch.bool, device=points.device)
    mask[masked_idx] = False
    visible_idx = all_idx[mask]

    visible_points = points[visible_idx]
    masked_points = points[masked_idx]

    return visible_points, visible_idx, masked_points, masked_idx


def interpolate_features(
    visible_points: torch.Tensor,
    visible_feats: torch.Tensor,
    query_points: torch.Tensor,
    k: int = 3
) -> torch.Tensor:
    """
    Interpolate features from visible points to query points using inverse distance weighting.

    Args:
        visible_points: (N_v, 3) coordinates of visible points
        visible_feats: (N_v, D) features of visible points
        query_points: (N_q, 3) coordinates of query points
        k: number of neighbors for interpolation

    Returns:
        query_feats: (N_q, D) interpolated features
    """
    # Find k nearest visible points for each query point
    neighbor_idx = knn(visible_points, query_points, k)  # (N_q, k)

    # Get neighbor coordinates and features
    neighbor_coords = visible_points[neighbor_idx]  # (N_q, k, 3)
    neighbor_feats = visible_feats[neighbor_idx]    # (N_q, k, D)

    # Compute distances
    diff = query_points.unsqueeze(1) - neighbor_coords  # (N_q, k, 3)
    dist = torch.norm(diff, dim=-1, keepdim=True)       # (N_q, k, 1)

    # Inverse distance weighting (avoid division by zero)
    weights = 1.0 / (dist + 1e-8)  # (N_q, k, 1)
    weights = weights / weights.sum(dim=1, keepdim=True)  # normalize

    # Weighted sum
    query_feats = (neighbor_feats * weights).sum(dim=1)  # (N_q, D)

    return query_feats


def normalize_point_cloud(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize point cloud to unit sphere.

    Args:
        points: (N, 3) point coordinates

    Returns:
        normalized: (N, 3) normalized points
        centroid: (3,) center of point cloud
        scale: scalar, max distance from center
    """
    centroid = points.mean(dim=0)
    centered = points - centroid
    scale = centered.norm(dim=-1).max()
    normalized = centered / (scale + 1e-8)
    return normalized, centroid, scale


def random_rotate(points: torch.Tensor) -> torch.Tensor:
    """Apply random rotation around z-axis."""
    angle = torch.rand(1, device=points.device) * 2 * 3.14159
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    rotation = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], device=points.device, dtype=points.dtype).squeeze()
    return points @ rotation.T


def random_scale(points: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
    """Apply random scaling."""
    scale = torch.rand(1, device=points.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    return points * scale


def random_jitter(points: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """Add random jitter noise."""
    noise = torch.randn_like(points) * sigma
    return points + noise
