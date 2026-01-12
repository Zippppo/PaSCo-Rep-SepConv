"""
Training script for PTv3 encoder self-supervised pretraining.

Usage:
    python pretrain.py
    python pretrain.py --epochs 200 --batch_size 16
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import config
from pretrain_dataset import build_dataloaders
from ptv3_encoder import build_mae_model


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2 reconstruction loss."""
    return F.mse_loss(pred, target)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    epoch: int,
    device: str,
    log_interval: int = 50,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        coord = batch['coord'].to(device)
        feat = batch['feat'].to(device)
        offset = batch['offset'].to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(coord, feat, offset)

        # Loss
        loss = reconstruction_loss(outputs['pred_coords'], outputs['target_coords'])

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.6f}")

    return total_loss / num_batches


@torch.no_grad()
def validate(model: nn.Module, val_loader, device: str) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        coord = batch['coord'].to(device)
        feat = batch['feat'].to(device)
        offset = batch['offset'].to(device)

        outputs = model(coord, feat, offset)
        loss = reconstruction_loss(outputs['pred_coords'], outputs['target_coords'])

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
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

    return state['epoch'] + 1


def main():
    parser = argparse.ArgumentParser(description='PTv3 Encoder Pretraining')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None)
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

    device = config.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Build dataloaders
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    print("Building model...")
    model = build_mae_model(config)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

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

    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)
    elif os.path.exists(os.path.join(config.checkpoint_dir, 'latest.pth')):
        start_epoch = load_checkpoint(
            model, optimizer, scheduler,
            os.path.join(config.checkpoint_dir, 'latest.pth'),
            device
        )

    # Training loop
    best_val_loss = float('inf')
    print(f"\nStarting training from epoch {start_epoch}...")

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, epoch, device, config.log_interval
        )

        # Validate
        val_loss = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss,
            config.checkpoint_dir, is_best
        )

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"Best: {best_val_loss:.6f} | LR: {lr:.2e} | Time: {epoch_time:.1f}s")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Encoder weights saved to: {os.path.join(config.checkpoint_dir, 'encoder_best.pth')}")


if __name__ == '__main__':
    main()
