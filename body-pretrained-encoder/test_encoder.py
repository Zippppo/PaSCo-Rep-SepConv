"""
Test script for pretrained encoder.
Loads a test sample, runs inference with masking, and saves input/output point clouds.

Usage:
    cd /home/comp/25481568/code/PaSCo-Rep-SepConv
    python body-pretrained-encoder/test_encoder.py
"""

import os
import sys
import json
import numpy as np
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from ptv3_encoder import build_mae_model
from utils import normalize_point_cloud


def load_test_sample(data_root: str, split_file: str = None, idx: int = 0):
    """Load a test sample from the dataset."""
    # Find split file
    if split_file is None:
        split_file = os.path.join(project_root, 'dataset_split.json')

    with open(split_file, 'r') as f:
        split_data = json.load(f)

    test_ids = split_data['splits']['test']
    print(f"Found {len(test_ids)} test samples")

    # Load sample
    sample_id = test_ids[idx]
    file_path = os.path.join(data_root, f'{sample_id}.npz')

    print(f"Loading sample: {sample_id}")
    data = np.load(file_path)
    points = data['sensor_pc'].astype(np.float32)

    print(f"  Original point cloud shape: {points.shape}")

    return points, sample_id


def prepare_input(points: np.ndarray, num_points: int = 8192):
    """Prepare input for the model."""
    # Subsample if needed
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        pad_size = num_points - len(points)
        pad_indices = np.random.choice(len(points), pad_size, replace=True)
        points = np.concatenate([points, points[pad_indices]], axis=0)

    points = torch.from_numpy(points)

    # Normalize to unit sphere
    normalized, centroid, scale = normalize_point_cloud(points)

    # Features: xyz(absolute) + xyz(normalized)
    coords = normalized * scale + centroid  # Keep in original space
    feat = torch.cat([coords, normalized], dim=-1)  # (N, 6)

    # Create offset for single sample batch
    offset = torch.tensor([len(coords)], dtype=torch.long)

    return {
        'coord': coords,
        'feat': feat,
        'offset': offset,
        'centroid': centroid,
        'scale': scale,
        'normalized': normalized,
    }


def save_point_clouds(output_dir: str, sample_id: str, data: dict):
    """Save point clouds in various formats."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as NPZ
    npz_path = os.path.join(output_dir, f'{sample_id}_results.npz')
    np.savez(
        npz_path,
        input_pc=data['input_pc'],
        visible_pc=data['visible_pc'],
        masked_target_pc=data['masked_target_pc'],
        masked_pred_pc=data['masked_pred_pc'],
        reconstructed_pc=data['reconstructed_pc'],
    )
    print(f"Saved NPZ: {npz_path}")

    # Save individual PLY files for visualization
    for name, pc in data.items():
        if pc is None:
            continue
        ply_path = os.path.join(output_dir, f'{sample_id}_{name}.ply')
        save_ply(ply_path, pc)
        print(f"Saved PLY: {ply_path}")


def save_ply(path: str, points: np.ndarray, colors: np.ndarray = None):
    """Save point cloud as PLY file."""
    n_points = len(points)

    with open(path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # Data
        for i in range(n_points):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--sample_idx', type=int, default=0, help='Test sample index')
    args = parser.parse_args()

    # Configuration
    checkpoint_path = os.path.join(config.checkpoint_dir, 'best.pth')
    output_dir = os.path.join(os.path.dirname(config.checkpoint_dir), 'test')
    data_root = config.data_root

    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load test sample
    print("\n=== Loading Test Sample ===")
    points, sample_id = load_test_sample(data_root, idx=args.sample_idx)

    # Prepare input
    print("\n=== Preparing Input ===")
    input_data = prepare_input(points, num_points=config.num_points)

    coord = input_data['coord'].to(device)
    feat = input_data['feat'].to(device)
    offset = input_data['offset'].to(device)

    print(f"  Input coord shape: {coord.shape}")
    print(f"  Input feat shape: {feat.shape}")
    print(f"  Offset: {offset}")

    # Build model
    print("\n=== Building Model ===")
    model = build_mae_model(config)
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"  Loaded from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f}")

    # Run inference (with masking)
    print("\n=== Running Inference ===")
    model.train()  # Keep train mode for spconv compatibility (use no_grad instead)

    with torch.no_grad():
        outputs = model(coord, feat, offset)

    pred_coords = outputs['pred_coords'].cpu().numpy()
    target_coords = outputs['target_coords'].cpu().numpy()
    visible_idx = outputs['visible_idx'].cpu().numpy()
    masked_idx = outputs['masked_idx'].cpu().numpy()

    print(f"  Visible points: {len(visible_idx)}")
    print(f"  Masked points: {len(masked_idx)}")
    print(f"  Predicted coords shape: {pred_coords.shape}")
    print(f"  Target coords shape: {target_coords.shape}")

    # Compute reconstruction error
    l2_dist = np.sqrt(np.sum((pred_coords - target_coords) ** 2, axis=-1))
    print(f"\n=== Reconstruction Metrics ===")
    print(f"  L2 Mean: {l2_dist.mean():.6f}")
    print(f"  L2 Std: {l2_dist.std():.6f}")
    print(f"  L2 Median: {np.median(l2_dist):.6f}")
    print(f"  L2 Max: {l2_dist.max():.6f}")

    # Prepare output data
    input_pc = coord.cpu().numpy()  # All input points
    visible_pc = input_pc[visible_idx]  # Visible points used by encoder

    # Reconstruct full point cloud: visible + predicted
    reconstructed_pc = np.zeros_like(input_pc)
    reconstructed_pc[visible_idx] = visible_pc
    reconstructed_pc[masked_idx] = pred_coords

    output_data = {
        'input_pc': input_pc,
        'visible_pc': visible_pc,
        'masked_target_pc': target_coords,
        'masked_pred_pc': pred_coords,
        'reconstructed_pc': reconstructed_pc,
    }

    # Save results
    print(f"\n=== Saving Results ===")
    save_point_clouds(output_dir, sample_id, output_data)

    print("\n=== Done ===")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
