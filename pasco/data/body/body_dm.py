from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from .body_dataset import BodyDataset
from .collate import collate_fn_simple
from pasco.utils.torch_util import worker_init_fn


class BodyDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for body scene completion task.
    """

    def __init__(
        self,
        root,
        split_file,
        batch_size=1,
        num_workers=4,
        target_size=(128, 128, 256),
        n_subnets=1,
        complete_scale=8,
        voxel_size=4.0,
    ):
        """
        Args:
            root: path to voxel_data directory
            split_file: path to dataset_split.json
            batch_size: batch size (default 1 due to memory constraints)
            num_workers: number of data loading workers
            target_size: target grid size [D, H, W]
            n_subnets: number of sub-networks
            complete_scale: scale for scene completion
            voxel_size: voxel size in mm
        """
        super().__init__()
        self.root = root
        self.split_file = split_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.n_subnets = n_subnets
        self.complete_scale = complete_scale
        self.voxel_size = voxel_size

    def setup(self, stage=None):
        """Setup train, val, and test datasets."""
        self.train_ds = BodyDataset(
            split="train",
            root=self.root,
            split_file=self.split_file,
            target_size=self.target_size,
            n_subnets=self.n_subnets,
            data_aug=False,  # No augmentation for body task
            complete_scale=self.complete_scale,
            voxel_size=self.voxel_size,
        )

        self.val_ds = BodyDataset(
            split="val",
            root=self.root,
            split_file=self.split_file,
            target_size=self.target_size,
            n_subnets=self.n_subnets,
            data_aug=False,
            complete_scale=self.complete_scale,
            voxel_size=self.voxel_size,
        )

        self.test_ds = BodyDataset(
            split="test",
            root=self.root,
            split_file=self.split_file,
            target_size=self.target_size,
            n_subnets=self.n_subnets,
            data_aug=False,
            complete_scale=self.complete_scale,
            voxel_size=self.voxel_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn_simple,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn_simple,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn_simple,
        )
