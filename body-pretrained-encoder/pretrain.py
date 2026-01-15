"""
Training script for PTv3 encoder self-supervised pretraining.

Usage:
    python pretrain.py
    python body-pretrained-encoder/pretrain.py --epochs 200 --batch_size 16
"""

import os
import sys
import argparse
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import config
from pretrain_dataset import build_dataloaders
from ptv3_encoder import build_mae_model


def setup_logging(checkpoint_dir: str) -> logging.Logger:
    """Setup logging to both file and console."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('pretrain')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(checkpoint_dir, f'train_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")
    return logger


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2 reconstruction loss."""
    return F.mse_loss(pred, target)


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Chamfer Distance between two point clouds.

    Args:
        pred: (N, 3) predicted points
        target: (M, 3) target points

    Returns:
        CD value (mean of bidirectional nearest neighbor distances)
    """
    # pred -> target
    diff_p2t = pred.unsqueeze(1) - target.unsqueeze(0)  # (N, M, 3)
    dist_p2t = torch.sum(diff_p2t ** 2, dim=-1)  # (N, M)
    min_dist_p2t = torch.min(dist_p2t, dim=1)[0]  # (N,)

    # target -> pred
    min_dist_t2p = torch.min(dist_p2t, dim=0)[0]  # (M,)

    return (min_dist_p2t.mean() + min_dist_t2p.mean()) / 2


def f_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.01) -> tuple:
    """
    Compute F-Score at given threshold.

    Args:
        pred: (N, 3) predicted points
        target: (M, 3) target points
        threshold: distance threshold for considering a match

    Returns:
        (f_score, precision, recall)
    """
    # pred -> target distances
    diff = pred.unsqueeze(1) - target.unsqueeze(0)  # (N, M, 3)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # (N, M)

    # Precision: fraction of pred points that have a target within threshold
    min_dist_p2t = torch.min(dist, dim=1)[0]  # (N,)
    precision = (min_dist_p2t < threshold).float().mean()

    # Recall: fraction of target points that have a pred within threshold
    min_dist_t2p = torch.min(dist, dim=0)[0]  # (M,)
    recall = (min_dist_t2p < threshold).float().mean()

    # F-Score
    if precision + recall > 0:
        f = 2 * precision * recall / (precision + recall)
    else:
        f = torch.tensor(0.0, device=pred.device)

    return f, precision, recall


def point_wise_l2(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute point-wise L2 distance statistics.

    Args:
        pred: (N, 3) predicted points (aligned with target)
        target: (N, 3) target points

    Returns:
        dict with mean, std, median, p90, p95 L2 distances
    """
    l2_dist = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))  # (N,)
    return {
        'mean': l2_dist.mean().item(),
        'std': l2_dist.std().item(),
        'median': l2_dist.median().item(),
        'p90': torch.quantile(l2_dist, 0.9).item(),
        'p95': torch.quantile(l2_dist, 0.95).item(),
    }


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    scaler: GradScaler,
    epoch: int,
    device: str,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False, dynamic_ncols=True)
    for batch in pbar:
        coord = batch['coord'].to(device)
        feat = batch['feat'].to(device)
        offset = batch['offset'].to(device)

        optimizer.zero_grad()

        # Forward with automatic mixed precision
        with autocast(enabled=use_amp):
            outputs = model(coord, feat, offset)
            loss = reconstruction_loss(outputs['pred_coords'], outputs['target_coords'])

        # Backward with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first for proper clipping)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/num_batches:.4f}'})

    return total_loss / num_batches


@torch.no_grad()
def validate(model: nn.Module, val_loader, device: str, use_amp: bool = True,
             compute_metrics: bool = True, fscore_threshold: float = 0.02) -> dict:
    """Validate model.

    Note: We keep model in train() mode because spconv has issues finding
    suitable algorithms in eval() mode for certain tensor shapes.
    Since we use torch.no_grad(), dropout and batchnorm still behave correctly.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        compute_metrics: Whether to compute additional metrics (CD, F-Score, etc.)
        fscore_threshold: Distance threshold for F-Score computation

    Returns:
        dict with 'loss' and optionally other metrics
    """
    # Keep model in train mode to avoid spconv algorithm issues
    # torch.no_grad() already disables gradient computation
    model.train()
    total_loss = 0.0
    num_batches = 0
    skipped = 0

    # Accumulators for additional metrics
    total_cd = 0.0
    total_fscore = 0.0
    total_precision = 0.0
    total_recall = 0.0
    all_l2_mean = []

    pbar = tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True)
    for batch in pbar:
        coord = batch['coord'].to(device)
        feat = batch['feat'].to(device)
        offset = batch['offset'].to(device)

        try:
            with autocast(enabled=use_amp):
                outputs = model(coord, feat, offset)
                loss = reconstruction_loss(outputs['pred_coords'], outputs['target_coords'])

            total_loss += loss.item()
            num_batches += 1

            # Compute additional metrics (in float32 for numerical stability)
            if compute_metrics:
                pred = outputs['pred_coords'].float()
                target = outputs['target_coords'].float()

                # Chamfer Distance (sample if too many points to avoid OOM)
                max_pts = 2000
                if pred.shape[0] > max_pts:
                    idx = torch.randperm(pred.shape[0])[:max_pts]
                    pred_sample, target_sample = pred[idx], target[idx]
                else:
                    pred_sample, target_sample = pred, target

                cd = chamfer_distance(pred_sample, target_sample)
                total_cd += cd.item()

                # F-Score
                fs, prec, rec = f_score(pred_sample, target_sample, threshold=fscore_threshold)
                total_fscore += fs.item()
                total_precision += prec.item()
                total_recall += rec.item()

                # Point-wise L2 (pred and target are aligned)
                l2_stats = point_wise_l2(pred, target)
                all_l2_mean.append(l2_stats['mean'])

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/num_batches:.4f}'})
        except RuntimeError as e:
            # Skip batches that cause spconv algorithm issues
            if "can't find suitable algorithm" in str(e):
                skipped += 1
                continue
            raise

    if skipped > 0:
        tqdm.write(f"  Warning: Skipped {skipped} batches due to spconv issues")

    # Compute averages
    result = {
        'loss': total_loss / num_batches if num_batches > 0 else float('inf'),
    }

    if compute_metrics and num_batches > 0:
        result['chamfer_distance'] = total_cd / num_batches
        result['f_score'] = total_fscore / num_batches
        result['precision'] = total_precision / num_batches
        result['recall'] = total_recall / num_batches
        result['l2_mean'] = sum(all_l2_mean) / len(all_l2_mean) if all_l2_mean else 0.0

    return result


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Save checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'encoder': model.get_encoder_state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'loss': loss,
    }

    # Save latest
    torch.save(state, os.path.join(checkpoint_dir, 'latest.pth'))

    # Save periodic
    if (epoch + 1) % config.save_interval == 0:
        torch.save(state, os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth'))

    # Save best
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'best.pth'))
        # Also save encoder-only weights
        torch.save(
            model.get_encoder_state_dict(),
            os.path.join(checkpoint_dir, 'encoder_best.pth')
        )


def load_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    checkpoint_path: str,
    device: str,
) -> int:
    """Load checkpoint and return start epoch."""
    if not os.path.exists(checkpoint_path):
        return 0

    print(f"Loading checkpoint from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    if 'scaler' in state:
        scaler.load_state_dict(state['scaler'])

    return state['epoch'] + 1


def main():
    parser = argparse.ArgumentParser(description='PTv3 Encoder Pretraining')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--gpuids', type=str, default=None, help='GPU IDs to use (e.g., "0" or "0,1,2")')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    args = parser.parse_args()

    # Override config with args
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.device:
        config.device = args.device
    if args.no_amp:
        config.use_amp = False

    # Handle GPU selection
    if args.gpuids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
        # Use first GPU in the list as primary device
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.device if torch.cuda.is_available() else 'cpu'
    use_amp = config.use_amp and device != 'cpu'

    # Setup logging
    logger = setup_logging(config.checkpoint_dir)
    logger.info("=" * 60)
    logger.info("PTv3 Encoder Pretraining")
    logger.info("=" * 60)
    if args.gpuids is not None:
        logger.info(f"Using GPU(s): {args.gpuids}")
    logger.info(f"Device: {device}, AMP: {use_amp}")

    # Build dataloaders
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    logger.info("Building model...")
    model = build_mae_model(config)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params / 1e6:.2f}M")

    # Log config
    logger.info("-" * 40)
    logger.info("Config:")
    logger.info(f"  epochs: {config.epochs}")
    logger.info(f"  batch_size: {config.batch_size}")
    logger.info(f"  lr: {config.lr}")
    logger.info(f"  weight_decay: {config.weight_decay}")
    logger.info(f"  num_points: {config.num_points}")
    logger.info(f"  mask_ratio: {config.mask_ratio}")
    logger.info("-" * 40)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Scheduler: linear warmup + cosine decay
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=config.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs],
    )

    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, scaler, args.resume, device)
        logger.info(f"Resumed from checkpoint: {args.resume}")
    elif os.path.exists(os.path.join(config.checkpoint_dir, 'latest.pth')):
        start_epoch = load_checkpoint(
            model, optimizer, scheduler, scaler,
            os.path.join(config.checkpoint_dir, 'latest.pth'),
            device
        )
        logger.info(f"Resumed from latest checkpoint")

    # Warmup spconv by running a few batches in train mode first
    # This helps spconv cache the algorithm for similar tensor shapes
    logger.info("Warming up spconv...")
    model.train()
    warmup_iter = iter(train_loader)
    for _ in range(min(3, len(train_loader))):
        try:
            batch = next(warmup_iter)
            coord = batch['coord'].to(device)
            feat = batch['feat'].to(device)
            offset = batch['offset'].to(device)
            with torch.no_grad():
                with autocast(enabled=use_amp):
                    _ = model(coord, feat, offset)
        except StopIteration:
            break
    logger.info("Warmup complete")

    # Training loop
    best_val_loss = float('inf')
    logger.info(f"\nStarting training from epoch {start_epoch + 1}...")

    epoch_pbar = tqdm(range(start_epoch, config.epochs), desc="Training", initial=start_epoch, total=config.epochs)
    for epoch in epoch_pbar:
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, device,
            use_amp=use_amp,
            grad_clip_norm=config.grad_clip_norm,
        )

        # Validate with metrics
        val_metrics = validate(model, val_loader, device, use_amp=use_amp,
                               compute_metrics=True, fscore_threshold=config.fscore_threshold)
        val_loss = val_metrics['loss']

        # Update scheduler
        scheduler.step()

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, val_loss,
            config.checkpoint_dir, is_best
        )

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'lr': f'{lr:.1e}'
        })

        # Log to file
        log_msg = (f"Epoch {epoch+1}/{config.epochs} | "
                   f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                   f"Best: {best_val_loss:.6f} | LR: {lr:.2e} | Time: {epoch_time:.1f}s")
        if is_best:
            log_msg += " *"
        logger.info(log_msg)

        # Log additional metrics
        if 'chamfer_distance' in val_metrics:
            metrics_msg = (f"  Metrics | CD: {val_metrics['chamfer_distance']:.6f} | "
                          f"F-Score: {val_metrics['f_score']:.4f} (P:{val_metrics['precision']:.4f} R:{val_metrics['recall']:.4f}) | "
                          f"L2: {val_metrics['l2_mean']:.6f}")
            logger.info(metrics_msg)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Encoder weights saved to: {os.path.join(config.checkpoint_dir, 'encoder_best.pth')}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
