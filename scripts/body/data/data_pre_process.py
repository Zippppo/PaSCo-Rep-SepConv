#!/usr/bin/env python
"""
Precompute multiscale labels for body scene completion task.

This script precomputes the expensive multiscale label generation to accelerate training.
It processes voxel labels to generate geometric and semantic labels at multiple scales.

NEW: Also generates instance labels for panoptic segmentation (35-class scheme).
- outside_body is mapped to 255 (IGNORE_LABEL)
- Class 0: inside_body_empty
- Classes 1-34: organs

Usage:
    python scripts/body/data/data_pre_process.py \
        --input_dir Dataset/voxel_data \
        --output_dir Dataset/voxel_data_precomputed \
        --split_file dataset_split.json \
        --num_workers 8
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from pasco.data.body.label_mapping import remap_labels, N_CLASSES_NEW, N_CLASSES_OLD, IGNORE_LABEL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_grid_size(labels, grid_world_min, target_size=(128, 128, 256), voxel_size=4.0):
    """
    Normalize voxel grid to target size using center alignment.

    Args:
        labels: [D, H, W] voxel labels
        grid_world_min: [3] original grid minimum
        target_size: tuple (TD, TH, TW)
        voxel_size: float, voxel size in mm

    Returns:
        normalized_labels: [TD, TH, TW] normalized labels
        pc_offset: [3] offset to apply to point cloud (in mm)

    Note:
        Padding is filled with IGNORE_LABEL (255) for regions outside original data.
    """
    D, H, W = labels.shape
    TD, TH, TW = target_size

    # Create result array filled with IGNORE_LABEL (255) for padding
    result = np.full((TD, TH, TW), IGNORE_LABEL, dtype=labels.dtype)

    # Compute center alignment offsets
    d_off = (TD - D) // 2
    h_off = (TH - H) // 2
    w_off = (TW - W) // 2

    # Source and target slices for each dimension
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

    # Copy data
    result[
        tgt_d_start:tgt_d_end,
        tgt_h_start:tgt_h_end,
        tgt_w_start:tgt_w_end
    ] = labels[
        src_d_start:src_d_end,
        src_h_start:src_h_end,
        src_w_start:src_w_end
    ]

    # Compute point cloud offset
    pc_offset = np.array([d_off, h_off, w_off], dtype=np.float32) * voxel_size
    pc_offset = pc_offset - grid_world_min

    return result, pc_offset


def generate_multiscale_labels_optimized(semantic_label, n_classes=35, device='cpu'):
    """
    Generate multi-scale geometric and semantic labels (optimized version).

    Uses GPU acceleration if available and optimizes one-hot encoding to avoid
    creating huge tensors.

    Args:
        semantic_label: [D, H, W] numpy array or torch tensor (values 0-34 or 255)
        n_classes: int, number of classes (default 35)
        device: str or torch.device

    Returns:
        geo_labels: dict with "1_1", "1_2", "1_4" keys
        sem_labels: dict with "1_1", "1_2", "1_4" keys

    Note:
        - Class 255 (IGNORE_LABEL): outside_body/padding, ignored
        - Class 0: inside_body_empty (background)
        - Classes 1-34: organs (occupied)
    """
    if isinstance(semantic_label, np.ndarray):
        semantic_label = torch.from_numpy(semantic_label).long()
    else:
        semantic_label = semantic_label.long()

    # Move to device
    semantic_label = semantic_label.to(device)

    # Compute occupancy: class >= 1 means occupied (organs)
    # Class 0 = inside_body_empty, Class 255 = IGNORE (padding/outside)
    complete_voxel = semantic_label.clone().float()
    complete_voxel[semantic_label >= 1] = 1  # Actual organs are occupied
    complete_voxel[semantic_label == 0] = 0  # inside_body_empty
    complete_voxel[semantic_label == IGNORE_LABEL] = 0  # padding/outside

    scales = [1, 2, 4]
    geo_labels = {}
    sem_labels = {}

    # Create ignore mask for handling IGNORE_LABEL in downsampling
    ignore_mask = (semantic_label == IGNORE_LABEL)

    for scale in scales:
        if scale == 1:
            downscaled_geo = complete_voxel
            downscaled_sem = semantic_label.clone()
        else:
            # Geometric: max pooling (simple and fast)
            downscaled_geo = F.max_pool3d(
                complete_voxel.unsqueeze(0).unsqueeze(0),
                kernel_size=scale,
                stride=scale,
            ).squeeze(0).squeeze(0)

            # Semantic: optimized voting without full one-hot
            # Use mode pooling approximation: most frequent class in each block
            downscaled_sem = semantic_mode_pool3d(semantic_label, scale, n_classes, ignore_mask)

        geo_labels[f"1_{scale}"] = downscaled_geo.cpu().numpy().astype(np.uint8)
        sem_labels[f"1_{scale}"] = downscaled_sem.cpu().numpy().astype(np.uint8)

    return geo_labels, sem_labels


def semantic_mode_pool3d(semantic_label, scale, n_classes, ignore_mask=None):
    """
    Semantic downsampling using mode (most frequent class) pooling.

    This is an optimized version that avoids creating full one-hot tensors.
    Uses a voting mechanism over each downsampled block.

    Args:
        semantic_label: [D, H, W] torch tensor (values 0-34 or 255)
        scale: int, downsampling factor
        n_classes: int, number of classes (35)
        ignore_mask: [D, H, W] bool tensor marking IGNORE_LABEL positions

    Returns:
        downscaled: [D//scale, H//scale, W//scale] torch tensor

    Note:
        - Class 0: inside_body_empty (background)
        - Classes 1-34: organs
        - Class 255: IGNORE_LABEL (padding/outside)
    """
    device = semantic_label.device
    D, H, W = semantic_label.shape

    # Compute output shape
    D_out, H_out, W_out = D // scale, H // scale, W // scale

    # Reshape into blocks
    # [D, H, W] -> [D_out, scale, H_out, scale, W_out, scale]
    reshaped = semantic_label[:D_out*scale, :H_out*scale, :W_out*scale].reshape(
        D_out, scale, H_out, scale, W_out, scale
    )

    # Permute to [D_out, H_out, W_out, scale, scale, scale]
    reshaped = reshaped.permute(0, 2, 4, 1, 3, 5).contiguous()

    # Reshape to [D_out, H_out, W_out, scale^3]
    reshaped = reshaped.view(D_out, H_out, W_out, -1)

    # Also reshape ignore_mask if provided
    if ignore_mask is not None:
        ignore_reshaped = ignore_mask[:D_out*scale, :H_out*scale, :W_out*scale].reshape(
            D_out, scale, H_out, scale, W_out, scale
        ).permute(0, 2, 4, 1, 3, 5).contiguous().view(D_out, H_out, W_out, -1)
    else:
        ignore_reshaped = None

    # For each block, find the most frequent non-empty class
    # Strategy: separate organ classes from empty/ignore classes
    downscaled = torch.zeros(D_out, H_out, W_out, dtype=torch.long, device=device)

    for d in range(D_out):
        for h in range(H_out):
            for w in range(W_out):
                block = reshaped[d, h, w]  # [scale^3]

                # Get valid (non-IGNORE) values
                if ignore_reshaped is not None:
                    valid_mask = ~ignore_reshaped[d, h, w]
                    valid_block = block[valid_mask]
                else:
                    valid_block = block

                if valid_block.numel() == 0:
                    # All values are IGNORE
                    downscaled[d, h, w] = IGNORE_LABEL
                    continue

                # Count organ classes (>= 1)
                organ_mask = valid_block >= 1
                if organ_mask.any():
                    # Use mode of organ classes
                    organ_classes = valid_block[organ_mask]
                    downscaled[d, h, w] = torch.mode(organ_classes).values
                else:
                    # Check if mostly IGNORE or inside_body_empty
                    if ignore_reshaped is not None:
                        ignore_count = ignore_reshaped[d, h, w].sum()
                        total_count = block.numel()
                        if ignore_count > total_count // 2:
                            downscaled[d, h, w] = IGNORE_LABEL
                        else:
                            downscaled[d, h, w] = 0  # inside_body_empty
                    else:
                        downscaled[d, h, w] = torch.mode(valid_block).values

    return downscaled


def generate_multiscale_labels_original(semantic_label, n_classes=35, device='cpu'):
    """
    Generate multi-scale labels using the original one-hot method.

    This is the original implementation from BodyDataset, kept for reference
    and validation purposes.

    Args:
        semantic_label: [D, H, W] numpy array or torch tensor (values 0-34 or 255)
        n_classes: int, number of classes (default 35)
        device: str or torch.device

    Note:
        - Class 255 (IGNORE_LABEL): outside_body/padding, ignored
        - Class 0: inside_body_empty (background)
        - Classes 1-34: organs (occupied)
    """
    if isinstance(semantic_label, np.ndarray):
        semantic_label = torch.from_numpy(semantic_label).long()
    else:
        semantic_label = semantic_label.long()

    semantic_label = semantic_label.to(device)

    # Create ignore mask
    ignore_mask = (semantic_label == IGNORE_LABEL)

    # Compute occupancy: class >= 1 means occupied (organs)
    complete_voxel = semantic_label.clone().float()
    complete_voxel[semantic_label >= 1] = 1
    complete_voxel[semantic_label == 0] = 0
    complete_voxel[ignore_mask] = 0

    scales = [1, 2, 4]
    geo_labels = {}
    sem_labels = {}

    # One-hot encoding for semantic labels
    # Replace IGNORE_LABEL (255) with 0 temporarily for one-hot encoding
    temp = semantic_label.clone().long()
    temp[temp == IGNORE_LABEL] = 0
    sem_label_oh = F.one_hot(temp, num_classes=n_classes).permute(3, 0, 1, 2).float()

    for scale in scales:
        if scale == 1:
            downscaled_geo = complete_voxel
            downscaled_sem = semantic_label.clone()
        else:
            # Geometric: max pooling
            downscaled_geo = F.max_pool3d(
                complete_voxel.unsqueeze(0).unsqueeze(0),
                kernel_size=scale,
                stride=scale,
            ).squeeze(0).squeeze(0)

            # Semantic: average pooling then argmax
            # Only consider organ classes for voting (classes 1-34)
            sem_label_oh_occ = sem_label_oh.clone()
            sem_label_oh_occ[0, :, :, :] = 0  # Exclude inside_body_empty from voting

            downscaled_sem_oh = F.avg_pool3d(
                sem_label_oh_occ.unsqueeze(0),
                kernel_size=scale,
                stride=scale,
            ).squeeze(0)

            downscaled_sem = torch.argmax(downscaled_sem_oh, dim=0)

            # Handle empty voxels
            has_organ = downscaled_sem_oh.sum(dim=0) > 0

            # Downsample ignore mask to determine if region should be IGNORE
            ignore_mask_float = ignore_mask.float().unsqueeze(0).unsqueeze(0)
            downscaled_ignore = F.avg_pool3d(
                ignore_mask_float,
                kernel_size=scale,
                stride=scale,
            ).squeeze(0).squeeze(0)
            mostly_ignore = downscaled_ignore > 0.5

            # Default empty regions to inside_body_empty (0)
            downscaled_sem[~has_organ] = 0
            # Override with IGNORE_LABEL where mostly ignored
            downscaled_sem[~has_organ & mostly_ignore] = IGNORE_LABEL

        geo_labels[f"1_{scale}"] = downscaled_geo.cpu().numpy().astype(np.uint8)
        sem_labels[f"1_{scale}"] = downscaled_sem.cpu().numpy().astype(np.uint8)

    return geo_labels, sem_labels


def process_single_sample(args):
    """
    Process a single sample and save precomputed labels.

    Args:
        args: tuple of (sample_id, input_dir, output_dir, target_size, voxel_size, n_classes, use_gpu, use_optimized)

    Returns:
        tuple: (sample_id, success, message)
    """
    sample_id, input_dir, output_dir, target_size, voxel_size, n_classes, use_gpu, use_optimized = args

    try:
        # Setup device
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Load original data
        input_path = os.path.join(input_dir, f"{sample_id}.npz")
        data = np.load(input_path)

        voxel_labels = data["voxel_labels"].astype(np.uint8)
        grid_world_min = data["grid_world_min"].astype(np.float32)

        # Step 1: Normalize grid size (still 72-class at this point)
        normalized_labels_72, pc_offset = normalize_grid_size(
            voxel_labels, grid_world_min, target_size, voxel_size
        )

        # Step 2: Remap 72-class to 35-class semantic + instance labels
        semantic_label, instance_label = remap_labels(normalized_labels_72)

        # Step 3: Generate multiscale labels (now using 35-class)
        if use_optimized:
            geo_labels, sem_labels = generate_multiscale_labels_optimized(
                semantic_label, n_classes, device
            )
        else:
            geo_labels, sem_labels = generate_multiscale_labels_original(
                semantic_label, n_classes, device
            )

        # Step 4: Save precomputed data (including instance_label for panoptic)
        output_path = os.path.join(output_dir, f"{sample_id}.npz")
        np.savez_compressed(
            output_path,
            semantic_label=semantic_label,
            instance_label=instance_label,
            geo_1_1=geo_labels["1_1"],
            geo_1_2=geo_labels["1_2"],
            geo_1_4=geo_labels["1_4"],
            sem_1_1=sem_labels["1_1"],
            sem_1_2=sem_labels["1_2"],
            sem_1_4=sem_labels["1_4"],
            pc_offset=pc_offset,
        )

        return (sample_id, True, "Success")

    except Exception as e:
        return (sample_id, False, str(e))


def process_sequential(sample_ids, args_dict, device):
    """
    Process samples sequentially (used for GPU processing to avoid CUDA fork issues).

    Args:
        sample_ids: list of sample IDs to process
        args_dict: dict with processing parameters
        device: torch.device

    Returns:
        tuple: (success_count, error_count)
    """
    success_count = 0
    error_count = 0

    with tqdm(total=len(sample_ids), desc="Processing samples (sequential)") as pbar:
        for sample_id in sample_ids:
            try:
                # Load original data
                input_path = os.path.join(args_dict['input_dir'], f"{sample_id}.npz")
                data = np.load(input_path)

                voxel_labels = data["voxel_labels"].astype(np.uint8)
                grid_world_min = data["grid_world_min"].astype(np.float32)

                # Step 1: Normalize grid size (still 72-class at this point)
                normalized_labels_72, pc_offset = normalize_grid_size(
                    voxel_labels, grid_world_min,
                    args_dict['target_size'], args_dict['voxel_size']
                )

                # Step 2: Remap 72-class to 36-class semantic + instance labels
                semantic_label, instance_label = remap_labels(normalized_labels_72)

                # Step 3: Generate multiscale labels (now using 36-class)
                if args_dict['use_optimized']:
                    geo_labels, sem_labels = generate_multiscale_labels_optimized(
                        semantic_label, args_dict['n_classes'], device
                    )
                else:
                    geo_labels, sem_labels = generate_multiscale_labels_original(
                        semantic_label, args_dict['n_classes'], device
                    )

                # Step 4: Save precomputed data (including instance_label for panoptic)
                output_path = os.path.join(args_dict['output_dir'], f"{sample_id}.npz")
                np.savez_compressed(
                    output_path,
                    semantic_label=semantic_label,
                    instance_label=instance_label,
                    geo_1_1=geo_labels["1_1"],
                    geo_1_2=geo_labels["1_2"],
                    geo_1_4=geo_labels["1_4"],
                    sem_1_1=sem_labels["1_1"],
                    sem_1_2=sem_labels["1_2"],
                    sem_1_4=sem_labels["1_4"],
                    pc_offset=pc_offset,
                )

                success_count += 1

            except Exception as e:
                error_count += 1
                logger.error(f"Failed to process {sample_id}: {str(e)}")

            pbar.update(1)
            pbar.set_postfix({'success': success_count, 'errors': error_count})

    return success_count, error_count


def main():
    parser = argparse.ArgumentParser(description="Precompute multiscale labels for body dataset")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with original npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for precomputed npz files')
    parser.add_argument('--split_file', type=str, required=True, help='Dataset split JSON file')
    parser.add_argument('--target_size', type=int, nargs=3, default=[128, 128, 256], help='Target grid size')
    parser.add_argument('--voxel_size', type=float, default=4.0, help='Voxel size in mm')
    parser.add_argument('--n_classes', type=int, default=N_CLASSES_NEW, help='Number of classes (default: 35 for new scheme)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers (only for CPU mode)')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU acceleration (disables multiprocessing)')
    parser.add_argument('--use_optimized', action='store_true', help='Use optimized mode pooling (faster but may differ slightly)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'], help='Splits to process')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load split file
    logger.info(f"Loading split file: {args.split_file}")
    with open(args.split_file, 'r') as f:
        split_data = json.load(f)

    # Handle both formats
    if "splits" in split_data:
        splits = split_data["splits"]
    else:
        splits = split_data

    # Collect all sample IDs to process
    sample_ids = []
    for split in args.splits:
        if split in splits:
            sample_ids.extend(splits[split])
            logger.info(f"Split '{split}': {len(splits[split])} samples")

    # Filter out already processed samples if not overwriting
    if not args.overwrite:
        sample_ids_to_process = []
        for sample_id in sample_ids:
            output_path = os.path.join(args.output_dir, f"{sample_id}.npz")
            if not os.path.exists(output_path):
                sample_ids_to_process.append(sample_id)

        skipped = len(sample_ids) - len(sample_ids_to_process)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed samples")
        sample_ids = sample_ids_to_process

    logger.info(f"Total samples to process: {len(sample_ids)}")

    if len(sample_ids) == 0:
        logger.info("No samples to process. Done!")
        return

    # Check GPU availability and set device
    use_gpu_actual = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu:
        if torch.cuda.is_available():
            logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            logger.info("Note: GPU mode uses sequential processing to avoid CUDA fork issues")
        else:
            logger.warning("GPU requested but not available, falling back to CPU multiprocessing")

    # Prepare arguments dict for processing
    args_dict = {
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'target_size': tuple(args.target_size),
        'voxel_size': args.voxel_size,
        'n_classes': args.n_classes,
        'use_optimized': args.use_optimized,
    }

    if use_gpu_actual:
        # GPU mode: sequential processing to avoid CUDA fork issues
        device = torch.device('cuda')
        success_count, error_count = process_sequential(sample_ids, args_dict, device)
    else:
        # CPU mode: parallel processing with multiprocessing
        logger.info(f"Processing with {args.num_workers} workers (CPU mode)...")

        # Prepare arguments for parallel processing
        process_args = [
            (
                sample_id,
                args.input_dir,
                args.output_dir,
                tuple(args.target_size),
                args.voxel_size,
                args.n_classes,
                False,  # use_gpu = False for multiprocessing
                args.use_optimized,
            )
            for sample_id in sample_ids
        ]

        success_count = 0
        error_count = 0

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_sample, arg): arg[0] for arg in process_args}

            with tqdm(total=len(sample_ids), desc="Processing samples") as pbar:
                for future in as_completed(futures):
                    sample_id, success, message = future.result()

                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        logger.error(f"Failed to process {sample_id}: {message}")

                    pbar.update(1)
                    pbar.set_postfix({'success': success_count, 'errors': error_count})

    logger.info(f"\nProcessing complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
