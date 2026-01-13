#!/usr/bin/env python
"""
Training script for body scene completion task.

Usage:
    python scripts/body/train.py --dataset_root Dataset/voxel_data --split_file dataset_split.json
"""

import os
import click
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment

from pasco.data.body.body_dm import BodyDataModule
from pasco.data.body.params import class_names, class_frequencies, thing_ids, n_classes
from pasco.models.net_panoptic_sparse import Net
from pasco.utils.torch_util import set_random_seed


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
@click.option('--mask_weight', default=40.0, help='mask weight')
@click.option('--occ_weight', default=1.0, help='occupancy loss weight')

@click.option('--use_se_layer', default=False, help='use SE layer')
@click.option('--heavy_decoder', default=False, help='use heavy decoder')

@click.option('--use_voxel_query_loss', default=True, help='use voxel query loss')

@click.option('--accum_batch', default=1, help='gradient accumulation')

@click.option('--pretrained_model', default="", help='path to pretrained model')
@click.option('--f', default=64, help='base feature dimension')
@click.option('--seed', default=42, help='random seed')
@click.option('--max_epochs', default=60, help='maximum epochs')
@click.option('--check', is_flag=True, help='just check model initialization')
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
    dataset_root, split_file, f, max_epochs, check
):
    set_random_seed(seed)

    # Parse GPU IDs
    gpu_list = [int(x.strip()) for x in gpuids.split(',')]
    n_gpus = len(gpu_list)

    # Body-specific parameters
    body_n_classes = n_classes  # 72
    body_in_channels = 38  # xyz(3) + xyz_rel(3) + pos_enc(32)
    body_scene_size = (128, 128, 256)
    body_thing_ids = thing_ids  # []

    encoder_dropouts = [point_dropout_ratio, 0.0, 0.0, 0.0, 0.0, 0.0]
    decoder_dropouts = [0.0, 0.0, 0.0, 0.0, 0.0]
    for l in range(n_dropout_levels):
        encoder_dropouts[len(encoder_dropouts) - l - 1] = net_3d_dropout
        decoder_dropouts[l] = net_3d_dropout

    print("log_dir", log_dir)
    exp_name = exp_prefix
    exp_name += f"_bs{bs}_lr{lr}_wd{wd}"
    exp_name += f"_queries{num_queries}_nInfers{n_infers}"
    if not heavy_decoder:
        exp_name += "_noHeavyDecoder"

    print(f"Experiment: {exp_name}")

    query_sample_ratio = 1.0

    # Class weights
    class_weights = []
    for _ in range(n_infers):
        class_weight = torch.ones(body_n_classes + 1)
        class_weight[0] = 0.1  # outside_body - reduce weight
        class_weight[1] = 0.1  # inside_body_empty - reduce weight
        class_weight[-1] = 0.1  # dustbin class
        class_weights.append(class_weight)

    # Completion label weights (inverse frequency)
    complt_num_per_class = class_frequencies["1_1"]
    compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
    compl_labelweights = np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)
    compl_labelweights = torch.from_numpy(compl_labelweights).float()

    # Data module
    data_module = BodyDataModule(
        root=dataset_root,
        split_file=split_file,
        batch_size=max(1, int(bs / n_gpus)),
        num_workers=int(n_workers_per_gpu),
        target_size=body_scene_size,
        n_subnets=n_infers,
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
    )

    if check:
        print("Model initialized successfully!")
        print(f"  n_classes: {body_n_classes}")
        print(f"  in_channels: {body_in_channels}")
        print(f"  scene_size: {body_scene_size}")
        print(f"  thing_ids: {body_thing_ids}")
        return

    # Logger and callbacks
    if enable_log:
        logger = TensorBoardLogger(save_dir=log_dir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor=f"val_subnet{n_infers}/pq_dagger_all",
                save_top_k=5,
                mode="max",
                filename="{epoch:03d}-{val_subnet" + str(n_infers) + "/pq_dagger_all:.5f}",
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False

    # Check for existing checkpoint
    model_path = os.path.join(log_dir, exp_name, "checkpoints/last.ckpt")

    if not os.path.isfile(model_path) and pretrained_model != "":
        assert os.path.isfile(pretrained_model), f"Pretrained model not found: {pretrained_model}"
        model = Net.load_from_checkpoint(checkpoint_path=pretrained_model)
        print(f"Loaded pretrained model from {pretrained_model}")

    # Trainer configuration
    trainer_kwargs = dict(
        accumulate_grad_batches=accum_batch,
        callbacks=checkpoint_callbacks,
        max_epochs=max_epochs,
        gradient_clip_val=0.5,
        logger=logger,
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

    # Resume checkpoint path (if exists)
    ckpt_path = model_path if os.path.isfile(model_path) else None
    if ckpt_path:
        print(f"Will resume from {ckpt_path}")

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
