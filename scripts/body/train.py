#!/usr/bin/env python
"""
Training script for body scene completion task.

Usage:
    python scripts/body/train.py --dataset_root Dataset/voxel_data --split_file dataset_split.json --gpuids 1 --use_precomputed --bs 16 --max_epochs 100 --exp_prefix body_full_instance --num_queries 200
"""

import os
import json
import csv
import logging
from datetime import datetime
import click
import torch
import numpy as np
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment

from pasco.data.body.body_dm import BodyDataModule
from pasco.data.body.params import class_names, class_frequencies, thing_ids, n_classes, ignore_label
from pasco.models.net_panoptic_sparse import Net
from pasco.utils.torch_util import set_random_seed


class MetricsLogger(Callback):
    """Custom callback to log metrics to console and file in readable format."""

    def __init__(self, log_dir, exp_name, n_infers=1):
        super().__init__()
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.n_infers = n_infers

        # Create metrics log file
        self.metrics_file = os.path.join(log_dir, exp_name, "metrics.csv")
        self.readable_log_file = os.path.join(log_dir, exp_name, "training_log.txt")

        os.makedirs(os.path.join(log_dir, exp_name), exist_ok=True)

        # Initialize CSV file with headers
        self.csv_initialized = False

        # Write header to readable log
        with open(self.readable_log_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"Training Log - {exp_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")

    def on_train_epoch_end(self, trainer, pl_module):
        """Log training metrics at end of each epoch."""
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        # Extract key training metrics
        train_metrics = {}
        for key, value in metrics.items():
            if key.startswith('train/'):
                train_metrics[key] = value.item() if torch.is_tensor(value) else value

        if train_metrics:
            self._log_to_console_and_file(epoch, "train", train_metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics at end of each validation epoch."""
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        # Extract key validation metrics
        val_metrics = {}
        for key, value in metrics.items():
            if key.startswith('val_'):
                val_metrics[key] = value.item() if torch.is_tensor(value) else value

        if val_metrics:
            self._log_to_console_and_file(epoch, "val", val_metrics)
            self._write_to_csv(epoch, metrics)

    def _log_to_console_and_file(self, epoch, phase, metrics):
        """Log metrics to both console and readable file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Build log message
        lines = []
        lines.append(f"\n[{timestamp}] Epoch {epoch} - {phase.upper()} Metrics:")
        lines.append("-" * 60)

        # Group and sort metrics for better readability
        loss_metrics = {}
        iou_metrics = {}
        pq_metrics = {}
        other_metrics = {}

        for key, value in sorted(metrics.items()):
            if 'loss' in key.lower():
                loss_metrics[key] = value
            elif 'iou' in key.lower():
                iou_metrics[key] = value
            elif 'pq' in key.lower() or 'sq' in key.lower() or 'rq' in key.lower():
                pq_metrics[key] = value
            else:
                other_metrics[key] = value

        # Print losses
        if loss_metrics:
            lines.append("  Losses:")
            for key, value in loss_metrics.items():
                lines.append(f"    {key}: {value:.6f}")

        # Print IoU metrics
        if iou_metrics:
            lines.append("  IoU Metrics:")
            for key, value in iou_metrics.items():
                if 'SemIoU' not in key:  # Skip per-class IoU for console
                    lines.append(f"    {key}: {value:.4f}")

        # Print PQ metrics
        if pq_metrics:
            lines.append("  Panoptic Quality:")
            for key, value in pq_metrics.items():
                lines.append(f"    {key}: {value:.4f}")

        log_message = "\n".join(lines)

        # Print to console
        print(log_message)

        # Write to file
        with open(self.readable_log_file, 'a') as f:
            f.write(log_message + "\n")

    def _write_to_csv(self, epoch, metrics):
        """Write metrics to CSV file."""
        # Prepare row data
        row_data = {'epoch': epoch, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        for key, value in metrics.items():
            row_data[key] = value.item() if torch.is_tensor(value) else value

        # Initialize CSV with headers if needed
        if not self.csv_initialized:
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
                writer.writeheader()
            self.csv_initialized = True

        # Append row
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
            writer.writerow(row_data)


@click.command()
@click.option('--log_dir', default="logs", help='logging directory')
@click.option('--dataset_root', default="Dataset/voxel_data", help='path to voxel_data')
@click.option('--split_file', default="dataset_split.json", help='dataset split JSON file')
@click.option('--n_infers', default=1, help='number of subnets')

@click.option('--lr', default=3e-4, help='learning rate')
@click.option('--wd', default=0.0, help='weight decay')
@click.option('--bs', default=1, help="batch size (default 1 due to memory)")
@click.option('--scale', default=1, help="Scale")
@click.option('--gpuids', default="0", help="GPU IDs to use (e.g., '0' or '0,1')")
@click.option('--n_workers_per_gpu', default=4, help="Number of workers per GPU")
@click.option('--exp_prefix', default="body", help='prefix of logging directory')
@click.option('--enable_log', default=True, help='Enable logging')

@click.option('--transformer_dropout', default=0.2)
@click.option('--net_3d_dropout', default=0.0)
@click.option('--n_dropout_levels', default=3)
@click.option('--point_dropout_ratio', default=0.0, help='point dropout (disabled for body)')

@click.option('--alpha', default=0.0, help='uncertainty weight')

@click.option('--transformer_enc_layers', default=0, help='Transformer encoder layer')
@click.option('--transformer_dec_layers', default=1, help='Transformer decoder layer')

@click.option('--num_queries', default=100, help='Number of queries')
@click.option('--mask_weight', default=20.0, help='mask weight (reduced from 40.0 to balance with ce_weight)')
@click.option('--occ_weight', default=1.0, help='occupancy loss weight')

@click.option('--use_se_layer', default=False, help='use SE layer')
@click.option('--heavy_decoder', default=False, help='use heavy decoder')

@click.option('--use_voxel_query_loss', default=True, help='use voxel query loss')

@click.option('--accum_batch', default=1, help='gradient accumulation')

@click.option('--pretrained_model', default="", help='path to pretrained model')
@click.option('--f', default=64, help='base feature dimension')
@click.option('--seed', default=42, help='random seed')
@click.option('--max_epochs', default=60, help='maximum epochs')
@click.option('--warmup_epochs', default=10, help='Epochs without class-based pruning (warmup)')
@click.option('--check', is_flag=True, help='just check model initialization')
@click.option('--use_precomputed', is_flag=True, help='Use precomputed multiscale labels for faster loading')
@click.option('--precomputed_dir', default="", help='Path to precomputed data directory (default: {dataset_root}_precomputed)')
def main(
    lr, wd,
    bs, scale, alpha,
    n_workers_per_gpu, gpuids,
    exp_prefix, log_dir, enable_log,
    mask_weight, heavy_decoder,
    transformer_dropout, net_3d_dropout, n_dropout_levels,
    point_dropout_ratio, use_voxel_query_loss,
    seed,
    transformer_dec_layers, transformer_enc_layers, n_infers, occ_weight,
    num_queries, use_se_layer, accum_batch, pretrained_model,
    dataset_root, split_file, f, max_epochs, warmup_epochs, check,
    use_precomputed, precomputed_dir
):
    set_random_seed(seed)

    # Setup logging
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_debug.log")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File handler: captures all logs (DEBUG and above)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Console handler: only INFO and above (skip DEBUG to avoid slowdown)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    # Parse GPU IDs
    gpu_list = [int(x.strip()) for x in gpuids.split(',')]
    n_gpus = len(gpu_list)

    # Body-specific parameters (imported from params.py)
    body_n_classes = n_classes  # 35 semantic classes (outside_body removed, uses ignore_label=255)
    body_in_channels = 38  # xyz(3) + xyz_rel(3) + pos_enc(32)
    body_scene_size = (128, 128, 256)
    body_thing_ids = thing_ids  # 34 thing classes for instance segmentation (1-34)

    encoder_dropouts = [point_dropout_ratio, 0.0, 0.0, 0.0, 0.0, 0.0]
    decoder_dropouts = [0.0, 0.0, 0.0, 0.0, 0.0]
    for l in range(n_dropout_levels):
        encoder_dropouts[len(encoder_dropouts) - l - 1] = net_3d_dropout
        decoder_dropouts[l] = net_3d_dropout

    logger.info("log_dir: %s", log_dir)
    exp_name = exp_prefix
    exp_name += f"_bs{bs}_lr{lr}_wd{wd}"
    exp_name += f"_queries{num_queries}_nInfers{n_infers}"
    if not heavy_decoder:
        exp_name += "_noHeavyDecoder"

    logger.info("Experiment: %s", exp_name)
    logger.info("Warmup epochs (no class-based pruning): %d", warmup_epochs)

    query_sample_ratio = 1.0

    # Class weights for All-Instance Strategy (35 classes)
    # - outside_body: mapped to 255 (IGNORE_LABEL), handled by ignore_index in loss
    # - Class 0 (inside_body_empty): Low weight background (like KITTI empty)
    # - Classes 1-34 (organs): Normal weight (all are instances now)
    # - Dustbin class: Moderate weight to provide gradient for unmatched queries
    class_weights = []
    for _ in range(n_infers):
        class_weight = torch.ones(body_n_classes + 1)
        class_weight[0] = 0.1   # inside_body_empty - low weight (similar to KITTI empty)
        class_weight[-1] = 0.5  # dustbin class - increased from 0.1 to 0.5 to break "dead loop"
        # All organ classes (1-34) keep weight 1.0
        class_weights.append(class_weight)

    logger.info("All-Instance Strategy class weights (35 classes):")
    logger.info("  IGNORE_LABEL=255 (outside_body): handled by ignore_index")
    logger.info("  Class 0 (inside_body_empty): %.2f", class_weight[0].item())
    logger.info("  Classes 1-34 (organs): 1.0")
    logger.info("  Dustbin class: %.2f", class_weight[-1].item())

    # Completion label weights (inverse frequency)
    # class_frequencies already excludes outside_body (35 classes now)
    # outside_body (255) is handled by ignore_index in loss function
    complt_num_per_class = class_frequencies["1_1"].copy()
    # Avoid division by zero
    complt_num_per_class[complt_num_per_class == 0] = 1.0
    compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
    compl_labelweights = np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)
    # Set inside_body_empty (class 0) to low weight
    compl_labelweights[0] = compl_labelweights[0] * 0.1
    compl_labelweights = torch.from_numpy(compl_labelweights).float()

    # Determine and validate precomputed directory
    if use_precomputed:
        if not precomputed_dir:
            precomputed_dir = dataset_root + "_precomputed"

        if not os.path.exists(precomputed_dir):
            logger.error(f"Precomputed directory not found: {precomputed_dir}")
            logger.error("Please run scripts/body/data/data_pre_process.py first")
            raise FileNotFoundError(f"Directory does not exist: {precomputed_dir}")

        logger.info(f"Using precomputed labels from: {precomputed_dir}")
    else:
        precomputed_dir = None
        logger.info("Using on-the-fly label generation (original mode)")

    # Data module
    data_module = BodyDataModule(
        root=dataset_root,
        split_file=split_file,
        batch_size=max(1, int(bs / n_gpus)),
        num_workers=int(n_workers_per_gpu),
        target_size=body_scene_size,
        n_subnets=n_infers,
        use_precomputed=use_precomputed,
        precomputed_root=precomputed_dir,
    )

    # Model
    model = Net(
        heavy_decoder=heavy_decoder,
        class_frequencies=class_frequencies,
        n_classes=body_n_classes,
        in_channels=body_in_channels,
        scene_size=body_scene_size,
        thing_ids=body_thing_ids,
        occ_weight=occ_weight,
        class_names=class_names,
        lr=lr,
        weight_decay=wd,
        class_weights=class_weights,
        transformer_dropout=transformer_dropout,
        encoder_dropouts=encoder_dropouts,
        decoder_dropouts=decoder_dropouts,
        dense3d_dropout=net_3d_dropout,
        scale=scale,
        enc_layers=transformer_enc_layers,
        dec_layers=transformer_dec_layers,
        aux_loss=False,
        num_queries=num_queries,
        mask_weight=mask_weight,
        use_se_layer=use_se_layer,
        alpha=alpha,
        query_sample_ratio=query_sample_ratio,
        n_infers=n_infers,
        f=f,
        compl_labelweights=compl_labelweights,
        use_voxel_query_loss=use_voxel_query_loss,
        warmup_epochs=warmup_epochs,
    )

    if check:
        logger.info("Model initialized successfully!")
        logger.info("=== All-Instance Strategy (35 classes) ===")
        logger.info("  n_classes: %d", body_n_classes)
        logger.info("  ignore_label: %d (outside_body)", ignore_label)
        logger.info("  in_channels: %d", body_in_channels)
        logger.info("  scene_size: %s", body_scene_size)
        logger.info("  num_queries: %d", num_queries)
        logger.info("  thing_ids: %d classes (all organs 1-34)", len(body_thing_ids))
        logger.info("  Expected instances/sample: ~69")
        return

    # Logger and callbacks
    if enable_log:
        tb_logger = TensorBoardLogger(save_dir=log_dir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        metrics_logger = MetricsLogger(log_dir=log_dir, exp_name=exp_name, n_infers=n_infers)
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor=f"val_subnet{n_infers}/pq_dagger_all",
                save_top_k=5,
                mode="max",
                filename="{epoch:03d}-{val_subnet" + str(n_infers) + "/pq_dagger_all:.5f}",
            ),
            lr_monitor,
            metrics_logger,
        ]
    else:
        tb_logger = False
        checkpoint_callbacks = False

    # Check for existing checkpoint (for resuming training)
    model_path = os.path.join(log_dir, exp_name, "checkpoints/last.ckpt")

    # Load pretrained model weights if specified
    # This only loads weights, not optimizer state or epoch info
    if pretrained_model:
        assert os.path.isfile(pretrained_model), f"Pretrained model not found: {pretrained_model}"
        logger.info("Loading pretrained weights from %s", pretrained_model)
        checkpoint = torch.load(pretrained_model, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Load weights with strict=False to allow partial loading
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning("Missing keys when loading pretrained model: %s", missing_keys)
        if unexpected_keys:
            logger.warning("Unexpected keys when loading pretrained model: %s", unexpected_keys)
        logger.info("Successfully loaded pretrained weights")

    # Log training configuration to file
    config_file = os.path.join(log_dir, exp_name, "config.json")
    os.makedirs(os.path.join(log_dir, exp_name), exist_ok=True)
    config = {
        'lr': lr, 'wd': wd, 'bs': bs, 'scale': scale, 'alpha': alpha,
        'n_infers': n_infers, 'num_queries': num_queries, 'max_epochs': max_epochs,
        'warmup_epochs': warmup_epochs,
        'transformer_dropout': transformer_dropout, 'net_3d_dropout': net_3d_dropout,
        'mask_weight': mask_weight, 'occ_weight': occ_weight,
        'pretrained_model': pretrained_model, 'seed': seed,
        'dataset_root': dataset_root, 'split_file': split_file,
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info("Saved config to %s", config_file)

    # Trainer configuration
    trainer_kwargs = dict(
        accumulate_grad_batches=accum_batch,
        callbacks=checkpoint_callbacks,
        max_epochs=max_epochs,
        gradient_clip_val=0.5,
        logger=tb_logger,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=gpu_list,
        num_nodes=1,
    )

    # DDP for multi-GPU
    if n_gpus > 1:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)
        trainer_kwargs["sync_batchnorm"] = True
        trainer_kwargs["limit_val_batches"] = 0.25 * accum_batch * n_gpus
        trainer_kwargs["limit_train_batches"] = 0.25 * accum_batch * n_gpus

    # Resume checkpoint path (if exists) - this resumes full training state
    ckpt_path = model_path if os.path.isfile(model_path) else None
    if ckpt_path:
        logger.info("Will resume training from %s", ckpt_path)

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
