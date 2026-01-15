import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

from .positional_encoding import sinusoidal_positional_encoding, normalize_coordinates
from .params import thing_ids, n_classes
from .label_mapping import remap_labels, N_CLASSES_OLD


class BodyDataset(Dataset):
    """
    Dataset for body scene completion task.

    Loads .npz files containing:
    - sensor_pc: [N, 3] partial body surface point cloud (mm)
    - voxel_labels: [D, H, W] semantic labels (uint8)
    - grid_world_min: [3] bounding box min (mm)
    - grid_world_max: [3] bounding box max (mm)
    - grid_voxel_size: [3] voxel size (always [4, 4, 4] mm)
    """

    def __init__(
        self,
        split,
        root,
        split_file,
        target_size=(128, 128, 256),
        n_subnets=1,
        data_aug=False,
        complete_scale=8,
        voxel_size=4.0,
        use_precomputed=False,
        precomputed_root=None,
    ):
        """
        Args:
            split: "train", "val", or "test"
            root: path to voxel_data directory containing BDMAP_*.npz files
            split_file: path to dataset_split.json
            target_size: target grid size [D, H, W] (default [128, 128, 256])
            n_subnets: number of sub-networks (default 1 for MVP)
            data_aug: data augmentation flag (not used for body task)
            complete_scale: scale for scene completion (default 8)
            voxel_size: voxel size in mm (default 4.0)
        """
        super().__init__()
        self.root = root
        self.split = split
        self.target_size = tuple(target_size)
        self.n_subnets = n_subnets
        self.data_aug = data_aug
        self.complete_scale = complete_scale
        self.voxel_size = voxel_size
        self.n_classes = n_classes
        self.thing_ids = thing_ids

        # Precomputed labels configuration
        self.use_precomputed = use_precomputed
        self.precomputed_root = precomputed_root

        if self.use_precomputed and self.precomputed_root is None:
            raise ValueError("use_precomputed=True requires precomputed_root to be specified")

        # Load split file
        import json
        with open(split_file, 'r') as f:
            split_data = json.load(f)

        # Handle both formats: {"train": [...]} and {"splits": {"train": [...]}}
        if "splits" in split_data:
            self.sample_ids = split_data["splits"][split]
        else:
            self.sample_ids = split_data[split]

        # Build file paths
        self.samples = []
        for sample_id in self.sample_ids:
            npz_path = os.path.join(root, f"{sample_id}.npz")
            if os.path.exists(npz_path):
                sample_dict = {
                    "sample_id": sample_id,
                    "npz_path": npz_path,
                }

                # Add precomputed path if enabled
                if self.use_precomputed:
                    precomputed_path = os.path.join(self.precomputed_root, f"{sample_id}.npz")
                    sample_dict["precomputed_path"] = precomputed_path if os.path.exists(precomputed_path) else None
                else:
                    sample_dict["precomputed_path"] = None

                self.samples.append(sample_dict)

        mode_str = "precomputed" if self.use_precomputed else "on-the-fly"
        print(f"BodyDataset [{split}]: {len(self.samples)} samples (mode: {mode_str})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns collated batch for n_subnets (default 1)."""
        from .collate import collate_fn_body

        if self.split == "val" or self.split == "test":
            idx_list = [idx] * self.n_subnets
        else:
            # Ensure we don't sample more than available
            n_to_sample = min(self.n_subnets - 1, len(self.samples) - 1)
            if n_to_sample > 0:
                idx_list = np.random.choice(
                    len(self.samples), n_to_sample, replace=False
                )
                idx_list = idx_list.tolist() + [idx]
            else:
                idx_list = [idx]

        items = []
        for id in idx_list:
            items.append(self.get_individual(id))

        return collate_fn_body(items, self.complete_scale)

    def get_individual(self, idx):
        """Load and process a single sample."""
        sample = self.samples[idx]
        sample_id = sample["sample_id"]
        npz_path = sample["npz_path"]

        # Load data
        data = np.load(npz_path)
        sensor_pc = data["sensor_pc"].astype(np.float32)  # [N, 3]
        voxel_labels = data["voxel_labels"].astype(np.uint8)  # [D, H, W] - 72 classes
        grid_world_min = data["grid_world_min"].astype(np.float32)  # [3]
        grid_world_max = data["grid_world_max"].astype(np.float32)  # [3]

        # Remap 72-class labels to 36-class semantic + instance labels
        semantic_label_72, instance_label_raw = remap_labels(voxel_labels)

        # Normalize grid to target size (center alignment with crop/pad)
        semantic_label, pc_offset = self._normalize_grid_size(
            semantic_label_72, sensor_pc, grid_world_min
        )
        # Also normalize instance labels with same offset
        instance_label, _ = self._normalize_grid_size(
            instance_label_raw, sensor_pc, grid_world_min
        )

        # Adjust point cloud coordinates
        sensor_pc_adjusted = sensor_pc.copy()
        sensor_pc_adjusted += pc_offset

        # Compute grid bounds for adjusted coordinates
        target_grid_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        target_grid_max = np.array(self.target_size, dtype=np.float32) * self.voxel_size

        # Voxelize point cloud and compute features
        in_feat, in_coord = self._compute_features(
            sensor_pc_adjusted, target_grid_min, target_grid_max
        )

        # Identity transform (no augmentation)
        T = torch.eye(4)

        # Compute min/max coordinates
        min_C = torch.tensor([0, 0, 0], dtype=torch.int32)
        max_C = torch.tensor(self.target_size, dtype=torch.int32) - 1

        # Convert to tensors
        semantic_label_tensor = torch.from_numpy(semantic_label)
        instance_label_tensor = torch.from_numpy(instance_label)

        # Generate or load multi-scale labels (check for precomputed with new format)
        if self.use_precomputed and sample.get("precomputed_path") is not None:
            loaded = self._load_precomputed_labels(
                sample["precomputed_path"], semantic_label_tensor
            )
            if loaded is not None:
                geo_labels, sem_labels, precomputed_instance = loaded
                # Use precomputed instance labels if available
                if precomputed_instance is not None:
                    instance_label_tensor = precomputed_instance
            else:
                geo_labels, sem_labels = self._generate_multiscale_labels(semantic_label_tensor)
        else:
            geo_labels, sem_labels = self._generate_multiscale_labels(semantic_label_tensor)

        # Generate mask labels with panoptic support (thing + stuff)
        mask_label = self._prepare_mask_label(semantic_label_tensor, instance_label_tensor)

        ret_data = {
            "xyz": sensor_pc_adjusted,
            "frame_id": sample_id,
            "sequence": "body",
            "in_feat": in_feat.float(),
            "in_coord": in_coord,
            "T": T,
            "min_C": min_C,
            "max_C": max_C,
            "semantic_label": semantic_label_tensor,
            "instance_label": instance_label_tensor,
            "mask_label": mask_label,
            "geo_labels": geo_labels,
            "sem_labels": sem_labels,
            "semantic_label_origin": semantic_label_tensor.clone(),
            "instance_label_origin": instance_label_tensor.clone(),
            "mask_label_origin": mask_label,
            "input_pcd_instance_label": np.zeros((sensor_pc.shape[0], 1), dtype=np.int32),
        }

        return ret_data

    def _normalize_grid_size(self, labels, sensor_pc, grid_world_min):
        """
        Normalize voxel grid to target size using center alignment.

        Args:
            labels: [D, H, W] voxel labels
            sensor_pc: [N, 3] point cloud in world coordinates
            grid_world_min: [3] original grid minimum

        Returns:
            normalized_labels: [TD, TH, TW] normalized labels
            pc_offset: [3] offset to apply to point cloud (in mm)
        """
        D, H, W = labels.shape
        TD, TH, TW = self.target_size

        # Create result array filled with class 0 (outside_body)
        result = np.zeros((TD, TH, TW), dtype=labels.dtype)

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
        # Original: pc_world = grid_world_min + voxel_idx * voxel_size
        # New: pc_world_new = 0 + (voxel_idx + offset) * voxel_size
        # So: pc_offset = (offset * voxel_size) - grid_world_min
        pc_offset = np.array([d_off, h_off, w_off], dtype=np.float32) * self.voxel_size
        pc_offset = pc_offset - grid_world_min

        return result, pc_offset

    def _compute_features(self, sensor_pc, grid_min, grid_max):
        """
        Compute input features for point cloud.

        Features (38D):
        - xyz (3): absolute position in mm
        - xyz_rel (3): position relative to voxel center
        - pos_enc (32): sinusoidal positional encoding

        Args:
            sensor_pc: [N, 3] point cloud in adjusted coordinates
            grid_min: [3] grid minimum (usually [0, 0, 0])
            grid_max: [3] grid maximum

        Returns:
            in_feat: [N, 38] input features
            in_coord: [N, 3] voxel coordinates (int)
        """
        # Compute voxel coordinates
        coords = ((sensor_pc - grid_min) / self.voxel_size).astype(np.int32)

        # Clamp to valid range
        coords = np.clip(coords, 0, np.array(self.target_size) - 1)

        # Compute voxel centers
        voxel_centers = (coords.astype(np.float32) + 0.5) * self.voxel_size + grid_min

        # xyz_rel: position relative to voxel center
        xyz_rel = sensor_pc - voxel_centers

        # Normalize coordinates for positional encoding
        coords_normalized = normalize_coordinates(sensor_pc, grid_min, grid_max)

        # Compute positional encoding
        pos_enc = sinusoidal_positional_encoding(
            torch.from_numpy(coords_normalized).float(),
            num_freqs=5,
            pad_to=32
        ).numpy()

        # Concatenate features: xyz(3) + xyz_rel(3) + pos_enc(32) = 38D
        in_feat = np.concatenate([sensor_pc, xyz_rel, pos_enc], axis=1)

        in_feat = torch.from_numpy(in_feat).float()
        in_coord = torch.from_numpy(coords).int()

        return in_feat, in_coord

    def _prepare_mask_label(self, semantic_label, instance_label):
        """
        Prepare mask labels for panoptic segmentation.

        For stuff classes (not in thing_ids): one binary mask per unique class
        For thing classes (in thing_ids): one binary mask per unique instance

        Args:
            semantic_label: [D, H, W] semantic labels (0-35)
            instance_label: [D, H, W] instance IDs (0 for stuff, >0 for things)

        Returns:
            dict with "labels" and "masks"
        """
        all_labels = []
        all_masks = []

        # 1. Handle stuff classes (not in thing_ids)
        unique_sem_ids = torch.unique(semantic_label)
        for sem_id in unique_sem_ids:
            sem_id_val = sem_id.item()
            # Skip background (class 0) and thing classes
            if sem_id_val == 0 or sem_id_val in self.thing_ids:
                continue
            mask = semantic_label == sem_id
            all_labels.append(sem_id)
            all_masks.append(mask)

        # 2. Handle thing classes (in thing_ids) - one mask per instance
        unique_inst_ids = torch.unique(instance_label)
        for inst_id in unique_inst_ids:
            inst_id_val = inst_id.item()
            if inst_id_val == 0:  # Skip non-instance voxels
                continue
            inst_mask = instance_label == inst_id
            # Get the semantic class for this instance
            # All voxels with this instance ID should have the same semantic class
            sem_at_inst = semantic_label[inst_mask]
            if len(sem_at_inst) > 0:
                sem_id = sem_at_inst[0]  # Take first (all should be same)
                # Only add if it's actually a thing class
                if sem_id.item() in self.thing_ids:
                    all_labels.append(sem_id)
                    all_masks.append(inst_mask)

        if len(all_masks) == 0:
            # No valid labels, return empty masks
            return {
                "labels": torch.tensor([], dtype=torch.long),
                "masks": torch.zeros((0,) + semantic_label.shape, dtype=torch.bool),
            }

        masks = torch.stack(all_masks)
        labels = torch.tensor([l.item() if torch.is_tensor(l) else l for l in all_labels], dtype=torch.long)

        return {
            "labels": labels,
            "masks": masks,
        }

    @staticmethod
    def prepare_instance_target(semantic_target, instance_target, ignore_label=0):
        """
        Prepare instance targets for thing classes.

        Each unique instance_id becomes a separate mask.

        Args:
            semantic_target: [D, H, W] semantic labels
            instance_target: [D, H, W] instance IDs
            ignore_label: instance ID to ignore (default 0)

        Returns:
            dict with "labels" and "masks", or None if no instances
        """
        unique_instance_ids = torch.unique(instance_target)
        unique_instance_ids = unique_instance_ids[unique_instance_ids != ignore_label]

        if len(unique_instance_ids) == 0:
            return None

        masks = []
        semantic_labels = []

        for inst_id in unique_instance_ids:
            inst_mask = instance_target == inst_id
            masks.append(inst_mask)
            # Get semantic label for this instance
            semantic_labels.append(semantic_target[inst_mask][0])

        return {
            "labels": torch.tensor([l.item() for l in semantic_labels], dtype=torch.long),
            "masks": torch.stack(masks),
        }

    def _generate_multiscale_labels(self, semantic_label):
        """
        Generate multi-scale geometric and semantic labels.

        Args:
            semantic_label: [D, H, W] semantic labels

        Returns:
            geo_labels: dict with "1_1", "1_2", "1_4" keys
            sem_labels: dict with "1_1", "1_2", "1_4" keys
        """
        # Compute occupancy: class > 0 means occupied
        # Class 0 = outside_body, Class 1 = inside_body_empty
        # Both are "empty" for geometric occupancy
        complete_voxel = semantic_label.clone().float()
        complete_voxel[semantic_label > 1] = 1  # Only actual organs are occupied
        complete_voxel[semantic_label <= 1] = 0

        scales = [1, 2, 4]
        geo_labels = {}
        sem_labels = {}

        # One-hot encoding for semantic labels
        # Map labels to range [0, n_classes]
        temp = semantic_label.clone().long()
        sem_label_oh = F.one_hot(temp, num_classes=self.n_classes).permute(3, 0, 1, 2).float()

        for scale in scales:
            if scale == 1:
                downscaled_geo = complete_voxel
                downscaled_sem = semantic_label
            else:
                # Geometric: max pooling
                downscaled_geo = F.max_pool3d(
                    complete_voxel.unsqueeze(0).unsqueeze(0),
                    kernel_size=scale,
                    stride=scale,
                ).squeeze(0).squeeze(0)

                # Semantic: average pooling then argmax
                # Only consider non-empty classes for voting
                sem_label_oh_occ = sem_label_oh.clone()
                sem_label_oh_occ[0, :, :, :] = 0  # Exclude outside_body
                sem_label_oh_occ[1, :, :, :] = 0  # Exclude inside_body_empty

                downscaled_sem_oh = F.avg_pool3d(
                    sem_label_oh_occ.unsqueeze(0),
                    kernel_size=scale,
                    stride=scale,
                ).squeeze(0)

                downscaled_sem = torch.argmax(downscaled_sem_oh, dim=0)

                # Handle empty voxels (where all organ channels were 0)
                # Check if any organ was present
                has_organ = downscaled_sem_oh.sum(dim=0) > 0
                # For empty voxels, use original class 0 or 1
                sem_label_oh_empty = sem_label_oh.clone()
                sem_label_oh_empty[2:, :, :, :] = 0  # Only keep classes 0 and 1
                downscaled_empty = F.avg_pool3d(
                    sem_label_oh_empty.unsqueeze(0),
                    kernel_size=scale,
                    stride=scale,
                ).squeeze(0)
                # Handle case where both class 0 and 1 are 0 (shouldn't happen but be safe)
                # When sum is 0, default to class 0 (outside_body)
                has_empty_class = downscaled_empty.sum(dim=0) > 0
                downscaled_empty_label = torch.argmax(downscaled_empty, dim=0)
                # Default to 0 for completely empty regions
                downscaled_empty_label[~has_empty_class] = 0
                downscaled_sem[~has_organ] = downscaled_empty_label[~has_organ]

            geo_labels[f"1_{scale}"] = downscaled_geo.type(torch.uint8)
            sem_labels[f"1_{scale}"] = downscaled_sem.type(torch.uint8)

        return geo_labels, sem_labels

    def _load_precomputed_labels(self, precomputed_path, semantic_label_tensor):
        """
        Load precomputed multiscale labels from NPZ file with validation and fallback.

        Supports both old format (72-class, no instance) and new format (36-class, with instance).

        Args:
            precomputed_path: str, path to precomputed .npz file
            semantic_label_tensor: torch.Tensor [D, H, W], for validation

        Returns:
            tuple: (geo_labels, sem_labels, instance_label) or None if failed
            - geo_labels: dict with "1_1", "1_2", "1_4" keys (torch.Tensor)
            - sem_labels: dict with "1_1", "1_2", "1_4" keys (torch.Tensor)
            - instance_label: torch.Tensor [D, H, W] or None if not available
        """
        try:
            data = np.load(precomputed_path)

            # Check if this is new format (36-class with instance_label)
            has_instance = 'instance_label' in data

            if has_instance:
                # New format: semantic_label is already 36-class
                # Validate shape matches
                precomputed_sem = data['semantic_label']
                if precomputed_sem.shape != semantic_label_tensor.shape:
                    import warnings
                    warnings.warn(f"Shape mismatch in {precomputed_path}, using on-the-fly generation")
                    return None

                instance_label = torch.from_numpy(data['instance_label'].astype(np.uint8))
            else:
                # Old format: needs remapping (this is fallback for old precomputed data)
                # We don't validate exact match since old format has 72 classes
                instance_label = None

            # Construct dictionaries
            geo_labels = {
                "1_1": torch.from_numpy(data["geo_1_1"]),
                "1_2": torch.from_numpy(data["geo_1_2"]),
                "1_4": torch.from_numpy(data["geo_1_4"]),
            }

            sem_labels = {
                "1_1": torch.from_numpy(data["sem_1_1"]),
                "1_2": torch.from_numpy(data["sem_1_2"]),
                "1_4": torch.from_numpy(data["sem_1_4"]),
            }

            return geo_labels, sem_labels, instance_label

        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load {precomputed_path}: {e}. Using on-the-fly generation.")
            return None
