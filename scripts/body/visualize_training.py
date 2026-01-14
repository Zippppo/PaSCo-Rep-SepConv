#!/usr/bin/env python
"""
Visualize training metrics from the body training log.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_training_log(log_path):
    """Parse training log file and extract metrics."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Pattern to match epoch blocks
    epoch_pattern = r'\[[\d-]+ [\d:]+\] Epoch (\d+) - (TRAIN|VAL) Metrics:'

    train_metrics = {}
    val_metrics = {}

    # Split by epoch blocks
    blocks = re.split(r'\n(?=\[[\d-]+ [\d:]+\] Epoch)', content)

    for block in blocks:
        match = re.match(epoch_pattern, block.strip())
        if not match:
            continue

        epoch = int(match.group(1))
        phase = match.group(2)

        # Extract metrics from block
        metrics = {}
        for line in block.split('\n'):
            # Match metric lines like "    train/loss_ce: 0.099935"
            metric_match = re.match(r'\s+([\w/]+):\s+([-\d.eE+nan]+)', line)
            if metric_match:
                key = metric_match.group(1)
                try:
                    value = float(metric_match.group(2))
                except ValueError:
                    value = float('nan')
                metrics[key] = value

        if phase == 'TRAIN':
            if epoch not in train_metrics:
                train_metrics[epoch] = {}
            train_metrics[epoch].update(metrics)
        else:
            if epoch not in val_metrics:
                val_metrics[epoch] = {}
            val_metrics[epoch].update(metrics)

    return train_metrics, val_metrics


def plot_metrics(train_metrics, val_metrics, output_dir):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = sorted(train_metrics.keys())

    # 1. Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    ax = axes[0, 0]
    total_loss = [train_metrics[e].get('train/total_loss_epoch', np.nan) for e in epochs]
    ax.plot(epochs, total_loss, 'b-', label='Total Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss (Epoch)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Component losses
    ax = axes[0, 1]
    compl_ce = [train_metrics[e].get('train/compl_ce_loss_epoch', np.nan) for e in epochs]
    compl_lovasz = [train_metrics[e].get('train/compl_lovasz_loss_epoch', np.nan) for e in epochs]
    ax.plot(epochs, compl_ce, 'r-', label='Compl CE Loss', linewidth=1.5)
    ax.plot(epochs, compl_lovasz, 'g-', label='Compl Lovasz Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Completion Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mask and Dice losses
    ax = axes[1, 0]
    loss_ce = [train_metrics[e].get('train/loss_ce_epoch', np.nan) for e in epochs]
    loss_dice = [train_metrics[e].get('train/loss_dice_epoch', np.nan) for e in epochs]
    loss_mask = [train_metrics[e].get('train/loss_mask_epoch', np.nan) for e in epochs]
    ax.plot(epochs, loss_ce, 'r-', label='CE Loss', linewidth=1.5)
    ax.plot(epochs, loss_dice, 'g-', label='Dice Loss', linewidth=1.5)
    ax.plot(epochs, loss_mask, 'b-', label='Mask Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Transformer Losses (CE, Dice, Mask)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SSC losses
    ax = axes[1, 1]
    ssc_ce_l0 = [train_metrics[e].get('train/ssc_ce_loss_level0_epoch', np.nan) for e in epochs]
    ssc_lovasz_l0 = [train_metrics[e].get('train/ssc_lovasz_loss_level0_epoch', np.nan) for e in epochs]
    ax.plot(epochs, ssc_ce_l0, 'r-', label='SSC CE Loss L0', linewidth=1.5)
    ax.plot(epochs, ssc_lovasz_l0, 'g-', label='SSC Lovasz Loss L0', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('SSC Losses (Level 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'loss_curves.png'}")

    # 2. Validation metrics
    val_epochs = sorted(val_metrics.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # mIoU
    ax = axes[0, 0]
    val_miou = [val_metrics[e].get('val_subnet0/mIoU', np.nan) for e in val_epochs]
    ax.plot(val_epochs, val_miou, 'b-o', label='Val mIoU', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.set_title('Validation mIoU')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IoU
    ax = axes[0, 1]
    val_iou = [val_metrics[e].get('val_subnet0/IoU', np.nan) for e in val_epochs]
    ax.plot(val_epochs, val_iou, 'r-o', label='Val IoU', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Validation IoU (Note: values > 1 indicate metric bug)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PQ dagger
    ax = axes[1, 0]
    val_pq_dagger = [val_metrics[e].get('val_subnet0/pq_dagger_all', np.nan) for e in val_epochs]
    ax.plot(val_epochs, val_pq_dagger, 'g-o', label='Val PQ Dagger', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PQ Dagger')
    ax.set_title('Validation PQ Dagger All')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # All PQ, SQ, RQ
    ax = axes[1, 1]
    val_pq = [val_metrics[e].get('val_subnet0/All_pq', np.nan) for e in val_epochs]
    val_sq = [val_metrics[e].get('val_subnet0/All_sq', np.nan) for e in val_epochs]
    val_rq = [val_metrics[e].get('val_subnet0/All_rq', np.nan) for e in val_epochs]
    ax.plot(val_epochs, val_pq, 'r-o', label='All PQ', markersize=3)
    ax.plot(val_epochs, val_sq, 'g-o', label='All SQ', markersize=3)
    ax.plot(val_epochs, val_rq, 'b-o', label='All RQ', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Panoptic Quality Metrics (All = 0!)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'validation_metrics.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'validation_metrics.png'}")

    # 3. Loss volatility analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Step vs Epoch loss comparison
    ax = axes[0]
    total_loss_epoch = [train_metrics[e].get('train/total_loss_epoch', np.nan) for e in epochs]
    total_loss_step = [train_metrics[e].get('train/total_loss_step', np.nan) for e in epochs]
    ax.plot(epochs, total_loss_epoch, 'b-', label='Epoch Loss', linewidth=1.5)
    ax.plot(epochs, total_loss_step, 'r--', label='Step Loss', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Volatility: Epoch vs Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss difference (volatility indicator)
    ax = axes[1]
    loss_diff = [abs(e - s) if not (np.isnan(e) or np.isnan(s)) else np.nan
                 for e, s in zip(total_loss_epoch, total_loss_step)]
    ax.bar(epochs, loss_diff, alpha=0.7, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|Epoch Loss - Step Loss|')
    ax.set_title('Loss Volatility (High = Unstable Training)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_volatility.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'loss_volatility.png'}")

    return val_epochs, val_miou, val_pq_dagger


def print_summary(train_metrics, val_metrics):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    epochs = sorted(train_metrics.keys())
    val_epochs = sorted(val_metrics.keys())

    print(f"\nTotal epochs: {len(epochs)}")
    print(f"Validation epochs: {len(val_epochs)}")

    # Final metrics
    if val_epochs:
        last_epoch = val_epochs[-1]
        print(f"\n--- Final Validation Metrics (Epoch {last_epoch}) ---")
        print(f"  mIoU: {val_metrics[last_epoch].get('val_subnet0/mIoU', 'N/A'):.4f}")
        print(f"  IoU: {val_metrics[last_epoch].get('val_subnet0/IoU', 'N/A'):.4f}")
        print(f"  PQ Dagger: {val_metrics[last_epoch].get('val_subnet0/pq_dagger_all', 'N/A'):.4f}")
        print(f"  All PQ: {val_metrics[last_epoch].get('val_subnet0/All_pq', 'N/A'):.4f}")
        print(f"  All SQ: {val_metrics[last_epoch].get('val_subnet0/All_sq', 'N/A'):.4f}")
        print(f"  All RQ: {val_metrics[last_epoch].get('val_subnet0/All_rq', 'N/A'):.4f}")

    # Check for NaN losses
    nan_epochs = []
    for e in epochs:
        total_loss = train_metrics[e].get('train/total_loss_epoch', 0)
        if np.isnan(total_loss):
            nan_epochs.append(e)

    if nan_epochs:
        print(f"\n--- WARNING: NaN Loss Detected ---")
        print(f"  Epochs with NaN total_loss: {nan_epochs}")

    # Loss statistics
    print(f"\n--- Loss Statistics ---")
    total_losses = [train_metrics[e].get('train/total_loss_epoch', np.nan) for e in epochs]
    valid_losses = [l for l in total_losses if not np.isnan(l)]
    if valid_losses:
        print(f"  Total Loss - Min: {min(valid_losses):.4f}, Max: {max(valid_losses):.4f}")
        print(f"  Total Loss - Mean: {np.mean(valid_losses):.4f}, Std: {np.std(valid_losses):.4f}")

    # mIoU progression
    print(f"\n--- mIoU Progression ---")
    val_miou = [val_metrics[e].get('val_subnet0/mIoU', np.nan) for e in val_epochs]
    valid_miou = [(e, m) for e, m in zip(val_epochs, val_miou) if not np.isnan(m)]
    if valid_miou:
        print(f"  Initial (Epoch {valid_miou[0][0]}): {valid_miou[0][1]:.4f}")
        print(f"  Final (Epoch {valid_miou[-1][0]}): {valid_miou[-1][1]:.4f}")
        best_epoch, best_miou = max(valid_miou, key=lambda x: x[1])
        print(f"  Best (Epoch {best_epoch}): {best_miou:.4f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str,
                        default='logs/body_bs32_lr0.0003_wd0.0_queries100_nInfers1_noHeavyDecoder/training_log.txt')
    parser.add_argument('--output_dir', type=str, default='logs/training_analysis')
    args = parser.parse_args()

    print(f"Parsing log: {args.log_path}")
    train_metrics, val_metrics = parse_training_log(args.log_path)

    print(f"Found {len(train_metrics)} training epochs, {len(val_metrics)} validation epochs")

    plot_metrics(train_metrics, val_metrics, args.output_dir)
    print_summary(train_metrics, val_metrics)
