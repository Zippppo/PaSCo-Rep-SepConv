"""
PTv3 Encoder wrapper and Masked AutoEncoder for self-supervised pretraining.
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

# Add reference repo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reference_repo', 'PointTransformerV3-main'))

from model import PointTransformerV3, Point, offset2batch, batch2offset
from utils import random_mask, interpolate_features, normalize_point_cloud


class PTv3EncoderWrapper(nn.Module):
    """
    Wrapper for PointTransformerV3 encoder.
    Outputs point-level features (N, D).
    """

    def __init__(
        self,
        in_channels: int = 6,
        enc_depths: Tuple[int, ...] = (2, 2, 2, 6, 2),
        enc_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        enc_num_head: Tuple[int, ...] = (2, 4, 8, 16, 32),
        enc_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024, 1024),
        grid_size: float = 4.0,
        order: Tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        enable_flash: bool = False,  # Set False for compatibility
    ):
        super().__init__()
        self.grid_size = grid_size
        self.out_channels = enc_channels[-1]

        # Initialize PTv3 in cls_mode (encoder only)
        self.ptv3 = PointTransformerV3(
            in_channels=in_channels,
            order=order,
            stride=(2, 2, 2, 2),
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            cls_mode=True,  # Only use encoder
            enable_flash=enable_flash,
            pdnorm_bn=False,
            pdnorm_ln=False,
        )

    def forward(self, coord: torch.Tensor, feat: torch.Tensor, offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            coord: (N, 3) point coordinates
            feat: (N, C) point features
            offset: (B,) batch offsets

        Returns:
            encoded_feat: (N', D) encoded features (may have different N due to pooling)
            encoded_coord: (N', 3) corresponding coordinates
        """
        data_dict = {
            "coord": coord,
            "feat": feat,
            "offset": offset,
            "grid_size": self.grid_size,
        }

        point = self.ptv3(data_dict)

        return point.feat, point.coord

    @classmethod
    def load_pretrained(cls, checkpoint_path: str, **kwargs) -> 'PTv3EncoderWrapper':
        """Load pretrained encoder weights."""
        model = cls(**kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'encoder' in state_dict:
            state_dict = state_dict['encoder']
        model.load_state_dict(state_dict)
        return model


class MaskedDecoder(nn.Module):
    """
    Simple MLP decoder for predicting masked point coordinates.
    """

    def __init__(self, feat_dim: int = 512, hidden_dims: Tuple[int, ...] = (256, 128)):
        super().__init__()

        layers = []
        in_dim = feat_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 3))  # Output xyz

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N_m, D) interpolated features at masked positions

        Returns:
            pred_coords: (N_m, 3) predicted coordinates
        """
        return self.mlp(features)


class MaskedAutoEncoder(nn.Module):
    """
    Masked AutoEncoder for point cloud self-supervised pretraining.

    Pipeline:
    1. Random mask points
    2. Encode visible points with PTv3
    3. Interpolate features to masked positions
    4. Decode to predict masked point coordinates
    """

    def __init__(
        self,
        encoder_cfg: dict,
        mask_ratio: float = 0.6,
        mask_group_size: int = 32,
        decoder_dims: Tuple[int, ...] = (256, 128),
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.mask_group_size = mask_group_size

        # Encoder
        self.encoder = PTv3EncoderWrapper(**encoder_cfg)

        # Decoder
        self.decoder = MaskedDecoder(
            feat_dim=self.encoder.out_channels,
            hidden_dims=decoder_dims,
        )

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        offset: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with masking.

        Args:
            coord: (N, 3) point coordinates (normalized)
            feat: (N, C) point features
            offset: (B,) batch offsets

        Returns:
            dict with keys:
                - pred_coords: (N_m, 3) predicted masked coordinates
                - target_coords: (N_m, 3) ground truth masked coordinates
                - visible_idx: (N_v,) indices of visible points
                - masked_idx: (N_m,) indices of masked points
        """
        batch = offset2batch(offset)
        device = coord.device
        B = offset.shape[0]

        all_visible_coords = []
        all_visible_feats = []
        all_masked_coords = []
        all_masked_idx = []
        all_visible_idx = []
        new_offset = []

        # Apply masking per sample in batch
        start_idx = 0
        for b in range(B):
            end_idx = offset[b].item()
            sample_coord = coord[start_idx:end_idx]
            sample_feat = feat[start_idx:end_idx]

            # Random masking
            visible_pts, visible_idx, masked_pts, masked_idx = random_mask(
                sample_coord,
                mask_ratio=self.mask_ratio,
                group_size=self.mask_group_size,
            )

            all_visible_coords.append(visible_pts)
            all_visible_feats.append(sample_feat[visible_idx])
            all_masked_coords.append(masked_pts)
            all_masked_idx.append(masked_idx + start_idx)
            all_visible_idx.append(visible_idx + start_idx)
            new_offset.append(len(visible_pts))

            start_idx = end_idx

        # Stack visible points
        visible_coord = torch.cat(all_visible_coords, dim=0)
        visible_feat = torch.cat(all_visible_feats, dim=0)
        visible_offset = torch.cumsum(torch.tensor(new_offset, device=device), dim=0)
        masked_coord = torch.cat(all_masked_coords, dim=0)

        # Encode visible points
        encoded_feat, encoded_coord = self.encoder(visible_coord, visible_feat, visible_offset)

        # Interpolate features to masked positions
        # Note: encoded points may be downsampled, interpolate from encoded to masked
        masked_feat = interpolate_features(encoded_coord, encoded_feat, masked_coord, k=3)

        # Decode
        pred_coords = self.decoder(masked_feat)

        return {
            'pred_coords': pred_coords,
            'target_coords': masked_coord,
            'visible_idx': torch.cat(all_visible_idx),
            'masked_idx': torch.cat(all_masked_idx),
        }

    def encode(self, coord: torch.Tensor, feat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """
        Encode points without masking (for inference).

        Args:
            coord: (N, 3) point coordinates
            feat: (N, C) point features
            offset: (B,) batch offsets

        Returns:
            features: (N', D) encoded point features
        """
        encoded_feat, _ = self.encoder(coord, feat, offset)
        return encoded_feat

    def get_encoder_state_dict(self) -> dict:
        """Get encoder weights for downstream use."""
        return self.encoder.state_dict()


def build_mae_model(config) -> MaskedAutoEncoder:
    """Build MaskedAutoEncoder from config."""
    encoder_cfg = {
        'in_channels': config.in_channels,
        'enc_depths': config.enc_depths,
        'enc_channels': config.enc_channels,
        'enc_num_head': config.enc_num_head,
        'enc_patch_size': config.enc_patch_size,
        'grid_size': config.grid_size,
        'order': config.order,
    }

    return MaskedAutoEncoder(
        encoder_cfg=encoder_cfg,
        mask_ratio=config.mask_ratio,
        mask_group_size=config.mask_group_size,
        decoder_dims=config.decoder_dims[:-1],  # Exclude final 3
    )
