#!/usr/bin/env python
"""
Visualization script to validate precomputed multiscale labels.

This script loads both original and precomputed data, performs validation checks,
and creates interactive 3D visualizations using plotly.

Usage:
    python scripts/body/visualize_precomputed.py \
        --sample_id BDMAP_00000001 \
        --original_dir Dataset/voxel_data \
        --precomputed_dir Dataset/voxel_data_precomputed \
        --output_dir visualizations
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_original_data(sample_id, original_dir):
    """Load original npz data."""
    npz_path = os.path.join(original_dir, f"{sample_id}.npz")
    data = np.load(npz_path)
    return {
        'sensor_pc': data['sensor_pc'].astype(np.float32),
        'voxel_labels': data['voxel_labels'].astype(np.uint8),
        'grid_world_min': data['grid_world_min'].astype(np.float32),
        'grid_world_max': data['grid_world_max'].astype(np.float32),
    }


def load_precomputed_data(sample_id, precomputed_dir):
    """Load precomputed npz data."""
    npz_path = os.path.join(precomputed_dir, f"{sample_id}.npz")
    data = np.load(npz_path)
    return {
        'semantic_label': data['semantic_label'],
        'geo_1_1': data['geo_1_1'],
        'geo_1_2': data['geo_1_2'],
        'geo_1_4': data['geo_1_4'],
        'sem_1_1': data['sem_1_1'],
        'sem_1_2': data['sem_1_2'],
        'sem_1_4': data['sem_1_4'],
        'pc_offset': data['pc_offset'],
    }


def normalize_grid_size(labels, grid_world_min, target_size=(128, 128, 256), voxel_size=4.0):
    """Normalize grid size (same as in data_pre_process.py)."""
    D, H, W = labels.shape
    TD, TH, TW = target_size

    result = np.zeros((TD, TH, TW), dtype=labels.dtype)

    d_off = (TD - D) // 2
    h_off = (TH - H) // 2
    w_off = (TW - W) // 2

    src_d_start = max(0, -d_off)
    src_d_end = min(D, TD - d_off)
    tgt_d_start = max(0, d_off)
    tgt_d_end = min(TD, D + d_off)

    src_h_start = max(0, -h_off)
    src_h_end = min(H, TH - h_off)
    tgt_h_start = max(0, h_off)
    tgt_h_end = min(TH, H + h_off)

    src_w_start = max(0, -w_off)
    src_w_end = min(W, TW - w_off)
    tgt_w_start = max(0, w_off)
    tgt_w_end = min(TW, W + w_off)

    result[
        tgt_d_start:tgt_d_end,
        tgt_h_start:tgt_h_end,
        tgt_w_start:tgt_w_end
    ] = labels[
        src_d_start:src_d_end,
        src_h_start:src_h_end,
        src_w_start:src_w_end
    ]

    pc_offset = np.array([d_off, h_off, w_off], dtype=np.float32) * voxel_size
    pc_offset = pc_offset - grid_world_min

    return result, pc_offset


def validate_data(original_data, precomputed_data, target_size=(128, 128, 256)):
    """
    Validate precomputed data against original data.

    Returns:
        dict with validation results
    """
    logger.info("Validating precomputed data...")

    results = {
        'semantic_label_match': False,
        'pc_offset_match': False,
        'geo_labels_valid': False,
        'sem_labels_valid': False,
        'errors': []
    }

    # Step 1: Check semantic_label
    expected_semantic, expected_pc_offset = normalize_grid_size(
        original_data['voxel_labels'],
        original_data['grid_world_min'],
        target_size
    )

    if np.array_equal(expected_semantic, precomputed_data['semantic_label']):
        logger.info("✓ semantic_label matches expected")
        results['semantic_label_match'] = True
    else:
        diff = np.sum(expected_semantic != precomputed_data['semantic_label'])
        logger.error(f"✗ semantic_label mismatch: {diff} voxels differ")
        results['errors'].append(f"semantic_label: {diff} voxels differ")

    # Step 2: Check pc_offset
    if np.allclose(expected_pc_offset, precomputed_data['pc_offset'], atol=1e-5):
        logger.info("✓ pc_offset matches expected")
        results['pc_offset_match'] = True
    else:
        diff = np.abs(expected_pc_offset - precomputed_data['pc_offset'])
        logger.error(f"✗ pc_offset mismatch: max diff = {diff.max()}")
        results['errors'].append(f"pc_offset: max diff = {diff.max()}")

    # Step 3: Validate geo_labels (occupancy)
    semantic = precomputed_data['semantic_label']
    expected_geo = (semantic > 1).astype(np.uint8)

    if np.array_equal(expected_geo, precomputed_data['geo_1_1']):
        logger.info("✓ geo_1_1 is correct (organs = 1, empty = 0)")
        results['geo_labels_valid'] = True
    else:
        diff = np.sum(expected_geo != precomputed_data['geo_1_1'])
        logger.error(f"✗ geo_1_1 mismatch: {diff} voxels differ")
        results['errors'].append(f"geo_1_1: {diff} voxels differ")

    # Step 4: Validate sem_labels (should match semantic_label for scale 1_1)
    if np.array_equal(semantic, precomputed_data['sem_1_1']):
        logger.info("✓ sem_1_1 matches semantic_label")
        results['sem_labels_valid'] = True
    else:
        diff = np.sum(semantic != precomputed_data['sem_1_1'])
        logger.error(f"✗ sem_1_1 mismatch: {diff} voxels differ")
        results['errors'].append(f"sem_1_1: {diff} voxels differ")

    # Step 5: Check shapes of downsampled labels
    expected_shapes = {
        'geo_1_2': (64, 64, 128),
        'geo_1_4': (32, 32, 64),
        'sem_1_2': (64, 64, 128),
        'sem_1_4': (32, 32, 64),
    }

    shapes_ok = True
    for key, expected_shape in expected_shapes.items():
        actual_shape = precomputed_data[key].shape
        if actual_shape != expected_shape:
            logger.error(f"✗ {key} shape mismatch: expected {expected_shape}, got {actual_shape}")
            results['errors'].append(f"{key}: wrong shape {actual_shape}")
            shapes_ok = False
        else:
            logger.info(f"✓ {key} shape is correct: {actual_shape}")

    # Summary
    all_valid = (results['semantic_label_match'] and
                 results['pc_offset_match'] and
                 results['geo_labels_valid'] and
                 results['sem_labels_valid'] and
                 shapes_ok)

    if all_valid:
        logger.info("\n✅ All validation checks passed!")
    else:
        logger.error(f"\n❌ Validation failed with {len(results['errors'])} errors")

    return results


def visualize_point_cloud(sensor_pc, pc_offset, title="Point Cloud"):
    """Visualize point cloud with plotly."""
    # Adjust point cloud with offset
    adjusted_pc = sensor_pc + pc_offset

    # Sample points if too many (for performance)
    max_points = 10000
    if len(adjusted_pc) > max_points:
        indices = np.random.choice(len(adjusted_pc), max_points, replace=False)
        adjusted_pc = adjusted_pc[indices]

    fig = go.Figure(data=[go.Scatter3d(
        x=adjusted_pc[:, 0],
        y=adjusted_pc[:, 1],
        z=adjusted_pc[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=adjusted_pc[:, 2],  # Color by depth
            colorscale='Viridis',
            opacity=0.6
        )
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        height=700
    )

    return fig


def visualize_voxel_slice(voxel_data, slice_idx=None, axis='z', title="Voxel Slice"):
    """Visualize a 2D slice of voxel data."""
    D, H, W = voxel_data.shape

    if slice_idx is None:
        if axis == 'z':
            slice_idx = W // 2
        elif axis == 'y':
            slice_idx = H // 2
        else:  # x
            slice_idx = D // 2

    if axis == 'z':
        slice_data = voxel_data[:, :, slice_idx]
        xlabel, ylabel = 'Width', 'Height'
    elif axis == 'y':
        slice_data = voxel_data[:, slice_idx, :]
        xlabel, ylabel = 'Width', 'Depth'
    else:  # x
        slice_data = voxel_data[slice_idx, :, :]
        xlabel, ylabel = 'Width', 'Height'

    fig = go.Figure(data=go.Heatmap(
        z=slice_data.T,  # Transpose for correct orientation
        colorscale='Viridis',
        colorbar=dict(title="Class ID")
    ))

    fig.update_layout(
        title=f"{title} ({axis}={slice_idx})",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=600,
        width=800
    )

    return fig


def create_voxel_mesh(voxel_data, voxel_size=4.0, downsample=1, is_geo=False):
    """
    Create 3D mesh representation of voxels for plotly.

    Args:
        voxel_data: [D, H, W] voxel labels
        voxel_size: size of each voxel in mm
        downsample: downsample factor for visualization (to reduce points)
        is_geo: if True, treat as binary occupancy; else as semantic labels

    Returns:
        tuple: (x, y, z, colors, labels) for scatter3d
    """
    # Get occupied voxels
    if is_geo:
        occupied_mask = voxel_data > 0
    else:
        occupied_mask = voxel_data > 1  # Exclude outside_body and inside_body_empty

    # Get coordinates
    coords = np.argwhere(occupied_mask)

    # Downsample if needed
    if downsample > 1 and len(coords) > 0:
        coords = coords[::downsample]

    if len(coords) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Convert to world coordinates (in mm)
    x = coords[:, 0] * voxel_size + voxel_size / 2
    y = coords[:, 1] * voxel_size + voxel_size / 2
    z = coords[:, 2] * voxel_size + voxel_size / 2

    # Get labels
    labels = voxel_data[coords[:, 0], coords[:, 1], coords[:, 2]]

    return x, y, z, labels, labels


def visualize_multiscale_3d(sensor_pc, pc_offset, precomputed_data, sample_id, downsample_voxels=2, max_points=5000):
    """
    Create 3D visualization with 3 columns (scales) x 2 rows (geo/sem).

    Args:
        sensor_pc: [N, 3] point cloud
        pc_offset: [3] offset for point cloud
        precomputed_data: dict with precomputed labels
        sample_id: sample ID for title
        downsample_voxels: downsample voxels for visualization
        max_points: maximum point cloud points to show

    Returns:
        plotly figure
    """
    # Adjust point cloud
    adjusted_pc = sensor_pc + pc_offset

    # Downsample point cloud if needed
    if len(adjusted_pc) > max_points:
        indices = np.random.choice(len(adjusted_pc), max_points, replace=False)
        adjusted_pc = adjusted_pc[indices]

    # Create subplots: 2 rows (geo, sem) x 3 cols (coarse, medium, fine)
    specs = [[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
             [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]

    fig = make_subplots(
        rows=2, cols=3,
        specs=specs,
        subplot_titles=(
            'Geo 1:4 (Coarse) [32×32×64]',
            'Geo 1:2 (Medium) [64×64×128]',
            'Geo 1:1 (Fine) [128×128×256]',
            'Sem 1:4 (Coarse) [32×32×64]',
            'Sem 1:2 (Medium) [64×64×128]',
            'Sem 1:1 (Fine) [128×128×256]',
        ),
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    # Row 1: Geometric labels (occupancy)
    scales = [
        ('geo_1_4', 1, 1, 16.0),  # (key, row, col, voxel_size)
        ('geo_1_2', 1, 2, 8.0),
        ('geo_1_1', 1, 3, 4.0),
    ]

    for key, row, col, vsize in scales:
        # Add point cloud (in gray)
        fig.add_trace(
            go.Scatter3d(
                x=adjusted_pc[:, 0],
                y=adjusted_pc[:, 1],
                z=adjusted_pc[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.3),
                name=f'Point Cloud',
                showlegend=False,
            ),
            row=row, col=col
        )

        # Add voxels
        x, y, z, colors, labels = create_voxel_mesh(
            precomputed_data[key],
            voxel_size=vsize,
            downsample=downsample_voxels,
            is_geo=True
        )

        if len(x) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='red',  # All occupied voxels in red
                        opacity=0.6,
                    ),
                    name=f'Occupied',
                    showlegend=False,
                ),
                row=row, col=col
            )

    # Row 2: Semantic labels
    scales_sem = [
        ('sem_1_4', 2, 1, 16.0),
        ('sem_1_2', 2, 2, 8.0),
        ('sem_1_1', 2, 3, 4.0),
    ]

    for key, row, col, vsize in scales_sem:
        # Add point cloud (in gray)
        fig.add_trace(
            go.Scatter3d(
                x=adjusted_pc[:, 0],
                y=adjusted_pc[:, 1],
                z=adjusted_pc[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.3),
                name=f'Point Cloud',
                showlegend=False,
            ),
            row=row, col=col
        )

        # Add voxels with semantic colors
        x, y, z, colors, labels = create_voxel_mesh(
            precomputed_data[key],
            voxel_size=vsize,
            downsample=downsample_voxels,
            is_geo=False
        )

        if len(x) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=colors,
                        colorscale='Viridis',
                        opacity=0.7,
                        colorbar=dict(
                            title="Class ID",
                            x=1.02 if col == 3 else None,
                            len=0.4,
                            y=0.25,
                        ) if col == 3 else None,
                        showscale=(col == 3),
                    ),
                    name=f'Semantic',
                    showlegend=False,
                ),
                row=row, col=col
            )

    # Update layout
    fig.update_layout(
        title_text=f"Multiscale 3D Visualization - {sample_id}<br>"
                   f"<sub>Row 1: Geometric (Occupancy) | Row 2: Semantic (Organ Classes)</sub>",
        height=1000,
        width=1800,
        showlegend=False,
    )

    # Update all 3D scene axes
    for i in range(1, 7):
        scene_name = 'scene' if i == 1 else f'scene{i}'
        fig.update_layout({
            scene_name: dict(
                xaxis=dict(title='X', showticklabels=False),
                yaxis=dict(title='Y', showticklabels=False),
                zaxis=dict(title='Z', showticklabels=False),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        })

    return fig


def visualize_geo_vs_sem(precomputed_data, slice_idx=None):
    """Visualize geometric vs semantic labels."""
    if slice_idx is None:
        slice_idx = 128  # Middle slice

    geo = precomputed_data['geo_1_1'][:, :, slice_idx]
    sem = precomputed_data['sem_1_1'][:, :, slice_idx]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Geometric (Occupancy)', 'Semantic (Class ID)'),
        horizontal_spacing=0.1
    )

    fig.add_trace(
        go.Heatmap(z=geo.T, colorscale='Greys', showscale=True,
                   colorbar=dict(title="Occupied", x=0.45)),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(z=sem.T, colorscale='Viridis', showscale=True,
                   colorbar=dict(title="Class ID", x=1.02)),
        row=1, col=2
    )

    fig.update_layout(
        title_text="Geometric vs Semantic Labels",
        height=600,
        width=1400
    )

    return fig


def compute_statistics(precomputed_data):
    """Compute and print statistics about the data."""
    logger.info("\n" + "="*60)
    logger.info("DATA STATISTICS")
    logger.info("="*60)

    # Semantic label stats
    sem = precomputed_data['semantic_label']
    unique, counts = np.unique(sem, return_counts=True)
    logger.info(f"\nSemantic Label (1:1):")
    logger.info(f"  Shape: {sem.shape}")
    logger.info(f"  Unique classes: {len(unique)}")
    logger.info(f"  Top 5 classes by count:")
    top_indices = np.argsort(counts)[-5:][::-1]
    for idx in top_indices:
        class_id = unique[idx]
        count = counts[idx]
        pct = 100.0 * count / sem.size
        logger.info(f"    Class {class_id}: {count:>10} voxels ({pct:>5.2f}%)")

    # Geometric label stats
    geo = precomputed_data['geo_1_1']
    occupied = np.sum(geo == 1)
    empty = np.sum(geo == 0)
    logger.info(f"\nGeometric Label (Occupancy):")
    logger.info(f"  Occupied voxels: {occupied:>10} ({100.0 * occupied / geo.size:.2f}%)")
    logger.info(f"  Empty voxels:    {empty:>10} ({100.0 * empty / geo.size:.2f}%)")

    # Multiscale sizes
    logger.info(f"\nMultiscale Label Sizes:")
    for scale in ['1_1', '1_2', '1_4']:
        geo_key = f'geo_{scale}'
        sem_key = f'sem_{scale}'
        geo_shape = precomputed_data[geo_key].shape
        sem_shape = precomputed_data[sem_key].shape
        geo_size = precomputed_data[geo_key].nbytes / 1024 / 1024
        sem_size = precomputed_data[sem_key].nbytes / 1024 / 1024
        logger.info(f"  Scale {scale}: {geo_shape} = {geo_size:.2f} MB + {sem_size:.2f} MB")

    logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize and validate precomputed body data")
    parser.add_argument('--sample_id', type=str, required=True, help='Sample ID to visualize (e.g., BDMAP_00000001)')
    parser.add_argument('--original_dir', type=str, default='Dataset/voxel_data', help='Original data directory')
    parser.add_argument('--precomputed_dir', type=str, default='Dataset/voxel_data_precomputed', help='Precomputed data directory')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory for HTML files')
    parser.add_argument('--downsample_voxels', type=int, default=2, help='Downsample voxels for visualization (1=no downsample)')
    parser.add_argument('--max_points', type=int, default=5000, help='Maximum point cloud points to display')
    parser.add_argument('--skip_validation', action='store_true', help='Skip validation checks')
    parser.add_argument('--save_extra', action='store_true', help='Save extra visualizations (slices, etc.)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading data for sample: {args.sample_id}")

    # Load data
    original_data = load_original_data(args.sample_id, args.original_dir)
    precomputed_data = load_precomputed_data(args.sample_id, args.precomputed_dir)

    logger.info(f"Original voxel shape: {original_data['voxel_labels'].shape}")
    logger.info(f"Point cloud size: {original_data['sensor_pc'].shape[0]} points")

    # Validate
    if not args.skip_validation:
        validation_results = validate_data(original_data, precomputed_data)
        if validation_results['errors']:
            logger.warning("Validation found errors, but continuing with visualization...")

    # Compute statistics
    compute_statistics(precomputed_data)

    # Generate main 3D visualization
    logger.info("Generating main 3D multiscale visualization...")
    fig_3d = visualize_multiscale_3d(
        original_data['sensor_pc'],
        precomputed_data['pc_offset'],
        precomputed_data,
        args.sample_id,
        downsample_voxels=args.downsample_voxels,
        max_points=args.max_points
    )
    output_path = os.path.join(args.output_dir, f"{args.sample_id}_multiscale_3d.html")
    fig_3d.write_html(output_path)
    logger.info(f"  ✅ Saved main visualization to: {output_path}")

    # Generate extra visualizations if requested
    if args.save_extra:
        logger.info("Generating extra visualizations...")

        # Point cloud only
        logger.info("  Creating point cloud visualization...")
        fig_pc = visualize_point_cloud(
            original_data['sensor_pc'],
            precomputed_data['pc_offset'],
            title=f"Point Cloud - {args.sample_id}"
        )
        output_path = os.path.join(args.output_dir, f"{args.sample_id}_pointcloud.html")
        fig_pc.write_html(output_path)
        logger.info(f"    Saved to: {output_path}")

        # Voxel slices
        logger.info("  Creating voxel slice visualizations...")
        fig_slice = visualize_voxel_slice(
            precomputed_data['semantic_label'],
            slice_idx=None,
            axis='z',
            title=f"Semantic Label Slice - {args.sample_id}"
        )
        output_path = os.path.join(args.output_dir, f"{args.sample_id}_slice_z.html")
        fig_slice.write_html(output_path)
        logger.info(f"    Saved z-slice to: {output_path}")

        # Geometric vs Semantic
        logger.info("  Creating geometric vs semantic comparison...")
        fig_geo_sem = visualize_geo_vs_sem(precomputed_data, slice_idx=None)
        output_path = os.path.join(args.output_dir, f"{args.sample_id}_geo_vs_sem.html")
        fig_geo_sem.write_html(output_path)
        logger.info(f"    Saved to: {output_path}")

    logger.info(f"\n✅ Visualization complete!")
    logger.info(f"   Main output: {args.output_dir}/{args.sample_id}_multiscale_3d.html")
    logger.info(f"   Open the HTML file in a web browser to view the interactive 3D plot.")
    logger.info(f"\n   💡 Tip: Use --save_extra to generate additional visualizations (slices, etc.)")


if __name__ == "__main__":
    main()
