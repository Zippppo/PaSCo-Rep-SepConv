# PaSCo Body Scene Completion - Implementation Plan

## Overview

Migrate PaSCo from autonomous driving to human body scene completion. Body dataset description in @Dataset/DATASET_README.md

Principle: Reuse existing architecture.

---

## Architecture Mapping

| Component   | KITTI (Original)         | Body (Target)       |
|-------------|--------------------------|---------------------|
| n_classes   | 20                       | 72                  |
| scene_size  | [256, 256, 32]           | [128, 128, 256]     |
| in_channels | 283 (27 + 256 embedding) | 38 (6 + 32 pos_enc) |
| n_infers    | 1-3                      | 1 (MVP)             |
| voxel_size  | 0.2m                     | 4mm                 |
| thing_ids   | [1-8]                    | [] (semantic only)  |

---

## Design Decisions

| Decision        | Choice          | Rationale                                              |
|-----------------|-----------------|--------------------------------------------------------|
| Feature dim     | 38D             | xyz(3) + xyz_rel(3) + pos_enc(32), validate with simple approach first |
| Data augmentation | None          | Human anatomy has fixed orientation, random rotation may break structure |
| num_queries     | 100             | Same as KITTI, more queries may capture more details  |
| Grid alignment  | Center-aligned  | Place different-sized bodies centered in [128,128,256] |
| Coordinate bounds | Fixed [0,0,0] | No dynamic computation needed, simplifies collate logic |

---

## Technical Details

### Input Feature Design (38D)

**KITTI 283D feature composition:**
```
in_feat = [vote(19) + intensity(1) + radius(1) + embedding(256) + xyz_rel(3) + xyz(3)] = 283D
```
The key is that `embedding(256)` comes from a pretrained WaffleIron model.

**Body task 38D feature design:**
```
[0:3]   xyz         - Absolute position (mm)
[3:6]   xyz_rel     - Position relative to voxel center
[6:38]  pos_enc     - Sinusoidal encoding (5 freqs x 3 dims x 2 = 30, pad to 32)
```

Since there's no pretrained encoder, use sinusoidal positional encoding instead:
```python
def sinusoidal_positional_encoding(coords, num_freqs=5):
    """
    coords: [N, 3] - coordinates normalized to [0,1]
    returns: [N, 30] -> pad to [N, 32]

    Formula: [sin(2^0*pi*x), cos(2^0*pi*x), ..., sin(2^(L-1)*pi*x), cos(2^(L-1)*pi*x)]
    Encode xyz three dimensions separately: 3 * 5 * 2 = 30D
    """
    freqs = 2.0 ** torch.arange(num_freqs)  # [1, 2, 4, 8, 16]
    coords_freq = coords.unsqueeze(-1) * freqs * np.pi  # [N, 3, 5]
    sin_enc = torch.sin(coords_freq).reshape(N, -1)    # [N, 15]
    cos_enc = torch.cos(coords_freq).reshape(N, -1)    # [N, 15]
    pos_enc = torch.cat([sin_enc, cos_enc], dim=-1)    # [N, 30]
    pos_enc = F.pad(pos_enc, (0, 2))                   # [N, 32]
    return pos_enc
```

### Instance Segmentation Handling

**PaSCo's instance segmentation mechanism:**

1. **thing_ids definition** (`params.py:21`):
   - KITTI: `thing_ids = [1,2,3,4,5,6,7,8]` represents "things" classes (cars, pedestrians, etc.)
   - Other classes (road, building, etc.) are "stuff", only semantic segmentation

2. **mask_label generation** (`kitti_dataset.py:prepare_mask_label`):
   - Things classes: each instance generates an independent binary mask
   - Stuff classes: each class generates one binary mask

3. **panoptic_inference** (`helper.py`):
   - thing classes: each query may predict an independent instance
   - stuff classes: same-class queries are merged

**Body task simplification:**
- Set `thing_ids = []`
- All 72 classes treated as "stuff", semantic segmentation only
- Transformer still works, but won't produce instance-level output

### Grid Normalization (Center Alignment)

```python
def normalize_grid_size(labels, sensor_pc, target=[128,128,256]):
    D, H, W = labels.shape
    TD, TH, TW = target
    result = np.zeros((TD, TH, TW), dtype=labels.dtype)  # class 0 = outside_body

    # Center alignment: compute offsets
    d_off = (TD - D) // 2
    h_off = (TH - H) // 2
    w_off = (TW - W) // 2

    # Bounds checking + copy
    src_d = slice(max(0, -d_off), min(D, TD - d_off))
    tgt_d = slice(max(0, d_off), min(TD, D + d_off))
    # ... same for H, W

    result[tgt_d, tgt_h, tgt_w] = labels[src_d, src_h, src_w]

    # Transform point cloud coordinates simultaneously
    new_sensor_pc = sensor_pc.copy()
    new_sensor_pc[:, 0] += d_off * voxel_size
    new_sensor_pc[:, 1] += h_off * voxel_size
    new_sensor_pc[:, 2] += w_off * voxel_size

    return result, new_sensor_pc, [d_off, h_off, w_off]
```

**Notes:**
- Point cloud coordinates need synchronized translation
- Offsets should be saved for subsequent coordinate transformations
- class 0 (`outside_body`) fills the boundary padding

### Coordinate System

**KITTI coordinate system:**
- `vox_origin = [0, -25.6, -2]` world coordinate origin
- Needs dynamic boundary computation (data augmentation, multi-frame fusion)

**Body task can be simplified:**
- Each sample normalized to fixed `[128, 128, 256]`
- Fixed `global_min_Cs = [0, 0, 0]`, `global_max_Cs = [127, 127, 255]`
- No dynamic boundary computation needed

```python
def collate_fn_body(batch, complete_scale=8):
    # Fixed bounds
    global_min_Cs = torch.tensor([0, 0, 0], dtype=torch.int32)
    global_max_Cs = torch.tensor([127, 127, 255], dtype=torch.int32)
    # ... rest same as KITTI collate
```

---

## Files to Create

### 1. pasco/data/body/params.py

- 72 class names from dataset_info.json
- class_frequencies from @docs/dataset_statistics.json
- thing_ids = [] (no instance segmentation)

### 2. pasco/data/body/body_dataset.py

```python
class BodyDataset(Dataset):
    def __init__(self, split, root, target_size=[128,128,256],
                 n_subnets=1, data_aug=False, complete_scale=8):
        ...

    def __getitem__(self, idx) -> dict:
        # Returns same structure as KittiDataset
        return {
            "in_feat": [N, 38],      # xyz + xyz_rel + pos_enc
            "in_coord": [N, 3],      # voxel coordinates
            "semantic_label": [D,H,W],
            "geo_labels": {"1_1":..., "1_2":..., "1_4":...},
            "sem_labels": {"1_1":..., "1_2":..., "1_4":...},
            "mask_label": {"labels":..., "masks":...},
            "min_C": [3], "max_C": [3], "T": [4,4],
        }
```

Key methods:
- `_normalize_grid_size()`: Pad/crop to [128, 128, 256], fill with class 0, center alignment
- `_voxelize_sensor_pc()`: Point cloud to sparse voxel coords + features
- `_compute_input_features()`: xyz(3) + xyz_rel(3) + sinusoidal_pos_enc(32)
- `_generate_multiscale_labels()`: Downsample at 1x, 2x, 4x scales

### 3. pasco/data/body/collate.py

```python
def collate_fn_body(batch, complete_scale=8) -> dict:
    # Same structure as semantic_kitti/collate.py
    # Fixed global bounds: [0,0,0] to [127,127,255]
```

### 4. pasco/data/body/body_dm.py

```python
class BodyDataModule(pl.LightningDataModule):
    # Load split from dataset_split.json (generated by scripts/body/split_dataset.py)
    # Split: 8:1:1 (train:val:test), seed=42
    # Data path: Dataset/voxel_data/BDMAP_*.npz
```

### 5. pasco/data/body/positional_encoding.py

```python
def sinusoidal_positional_encoding(coords, num_freqs=5):
    # coords: [N, 3] normalized to [0,1]
    # returns: [N, 32] (30D encoding + 2D padding)
```

### 6. configs/body.yaml

```yaml
dataset: body
n_classes: 72
scene_size: [128, 128, 256]
in_channels: 38
n_infers: 1
num_queries: 100
voxel_size: 4
```

### 7. scripts/body/train.py

```python
@click.command()
@click.option('--dataset_root', default='Dataset/voxel_data')
@click.option('--lr', default=0.001)
@click.option('--bs', default=1)
@click.option('--max_epochs', default=60)
def main(dataset_root, lr, bs, max_epochs, ...):
    from pasco.data.body.body_dm import BodyDataModule
    from pasco.data.body.params import class_names, class_frequencies, thing_ids

    n_classes = 72
    in_channels = 38
    scene_size = (128, 128, 256)
    thing_ids = []

    dm = BodyDataModule(root=dataset_root, batch_size=bs, ...)
    model = Net(n_classes=n_classes, in_channels=in_channels, scene_size=scene_size, ...)
    trainer = pl.Trainer(max_epochs=max_epochs, ...)
    trainer.fit(model, dm)
```

---

## Files to Modify

### 1. pasco/models/net_panoptic_sparse.py

**Line 84**: Make scene_size configurable
```python
# Before:
self.scene_size = (256 // scale, 256 // scale, 32 // scale)

# After:
self.scene_size = scene_size if scene_size else (256 // scale, 256 // scale, 32 // scale)
```

**Line 180**: Make thing_ids configurable
```python
# Before:
self.thing_ids = thing_ids  # imported from semantic_kitti

# After:
self.thing_ids = thing_ids if thing_ids is not None else []
```

---

## Implementation Phases

### Phase 1: Data Pipeline

1. Create pasco/data/body/__init__.py
2. Create params.py with 72 class definitions
3. Create positional_encoding.py
4. Create body_dataset.py:
   - Load .npz files
   - Normalize grid size (center alignment)
   - Voxelize point cloud
   - Generate multi-scale labels
5. Create collate.py
6. Create body_dm.py

**Verification**: Load dataset, print shapes, visualize one sample

### Phase 2: Model Adaptation

1. Modify net_panoptic_sparse.py:
   - Add scene_size parameter
   - Add thing_ids parameter
2. Create configs/body.yaml

**Verification**: Forward pass with dummy data

### Phase 3: Training Integration

1. Create scripts/body/train.py
2. Create scripts/body/split_dataset.py
3. Run training for 1 epoch

**Verification**: Loss decreases, no OOM

---

## File Structure

### New Files
```
pasco/data/body/
├── __init__.py
├── params.py                 # 72 class definitions + class_frequencies
├── positional_encoding.py    # Sinusoidal positional encoding
├── body_dataset.py           # BodyDataset class
├── collate.py                # collate_fn_body
└── body_dm.py                # BodyDataModule

scripts/body/
├── train.py                  # Training script
└── split_dataset.py          # Dataset split script

configs/
└── body.yaml                 # Configuration file
```

### Modified Files
```
pasco/models/net_panoptic_sparse.py
  - Line 84: scene_size parameterization
  - Line 180: thing_ids configurable
```

---

## Verification Checklist

- [ ] BodyDataset loads .npz and returns correct dict structure
- [ ] Grid normalization handles all size variations (center alignment)
- [ ] Voxelization produces valid sparse coordinates
- [ ] Positional encoding has correct shape [N, 32]
- [ ] Collate function returns correct batch format
- [ ] Forward pass completes without error
- [ ] Loss computation produces valid gradients
- [ ] Training loop runs for 1 epoch without OOM

---

## Key Reference Files

| Purpose             | File                                          |
|---------------------|-----------------------------------------------|
| Dataset template    | pasco/data/semantic_kitti/kitti_dataset.py    |
| Collate template    | pasco/data/semantic_kitti/collate.py          |
| DataModule template | pasco/data/semantic_kitti/kitti_dm.py         |
| Params template     | pasco/data/semantic_kitti/params.py           |
| Model entry         | pasco/models/net_panoptic_sparse.py           |
| Feature extraction  | pasco/models/unet3d_sparse_v2.py:CylinderFeat |
| Training script     | scripts/train.py                              |

---

## Risk Mitigation

| Risk                               | Mitigation                                 |
|------------------------------------|--------------------------------------------|
| OOM ([128,128,256] = 4.19M voxels) | batch_size=1, gradient accumulation        |
| Class imbalance (72 classes)       | Weighted CE from class_frequencies         |
| Poor convergence                   | Start with larger lr, add scheduler        |
| Coordinate mismatch                | Unit test: world -> voxel -> world roundtrip |

---

## Verification Commands

1. **Data pipeline verification**
   ```bash
   python -c "from pasco.data.body.body_dm import BodyDataModule; dm = BodyDataModule(...); dm.setup(); batch = next(iter(dm.train_dataloader())); print(batch.keys())"
   ```

2. **Model forward verification**
   ```bash
   python scripts/body/train.py --max_epochs=0 --check
   ```

3. **Training verification**
   ```bash
   python scripts/body/train.py --max_epochs=1
   # Expected: loss decreases, no OOM
   ```
