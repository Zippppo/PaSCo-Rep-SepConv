# PaSCo Human Body Completion Adaptation Plan (v3.0)

## Executive Summary

Adapt PaSCo (Panoptic Scene Completion) from autonomous driving to human body completion.

**Core Principle:** Correct migration first, minimize changes to original PaSCo.

**Core Challenges:** 50x scale difference + simplified input features (3D vs 283D) + extreme class imbalance + dynamic grid sizes.

---

## 1. Data Specification (Verified from Dataset)

```
Dataset: E:\CODE\PaSCo-Rep-SepConv\Dataset\voxel_data
Files: 4028 samples (.npz)

NPZ Structure:
- sensor_pc:      (N, 3) float32    # Surface point cloud, N ~ 16K-131K
- voxel_labels:   (D, H, W) uint8   # Dense label grid, labels 0-70
- grid_world_min: (3,) float32      # World coordinate bounds
- grid_world_max: (3,) float32
- grid_voxel_size: (3,) float32     # Always [4, 4, 4] mm
- grid_occ_size:  (3,) int32        # Grid dimensions
```

### Key Statistics

| Metric              | Original PaSCo        | Human Body                       |
| ------------------- | --------------------- | -------------------------------- |
| Voxel size          | 200mm                 | 4mm (50x smaller)                |
| Grid shape          | Fixed 256x256x32      | Dynamic 75-129 x 50-110 x 37-249 |
| Total voxels/sample | ~2M                   | 220K - 3M                        |
| Classes             | 20                    | 71 (0-70)                        |
| Input features      | 283D (xyz+vote+embed) | 3D (xyz only)                    |
| Points/sample       | ~100K                 | 16K-131K (avg 55K)               |

### Class Distribution (Critical)

```
Label  0 (background): 51% of all voxels
Label  1 (major tissue): 37%
Labels 2-70: Remaining 12% (extremely imbalanced)
```

Top classes: 0, 1, 15, 2, 20, 18, 23, 67, 68, 11, 6, 62, 61, 3, 57, 58, 69, 70, 5, 4

---

## 2. Code Analysis: What Can Be Configured vs What Must Be Modified

### 2.1 Configurable Parameters (NO code changes needed)

| Parameter           | Location     | How to Configure                |
| ------------------- | ------------ | ------------------------------- |
| `n_classes`         | Net.__init__ | CLI `--n_classes` or config     |
| `in_channels`       | Net.__init__ | CLI arg (default 283, set to 3) |
| `class_weights`     | Net.__init__ | Computed from params.py         |
| `class_frequencies` | Net.__init__ | From params.py                  |
| `f` (base channels) | Net.__init__ | CLI `--f`                       |
| `f_maps`            | UNet3DV2     | Derived from `f`                |
| `num_queries`       | Net.__init__ | CLI `--num_queries`             |
| `is_predict_panop`  | forward()    | Runtime flag                    |
| `heavy_decoder`     | Net.__init__ | CLI `--heavy_decoder`           |

**Key Discovery:** `CylinderFeat(fea_dim=in_channels)` - input feature dimension is already parameterized!

### 2.2 Hardcoded Values Requiring Code Changes

| Hardcoded Value                                    | File:Line                 | Impact               | Required Change        |
| -------------------------------------------------- | ------------------------- | -------------------- | ---------------------- |
| `scene_size = (256//scale, 256//scale, 32//scale)` | net_panoptic_sparse.py:84 | Ensembler operations | Make configurable      |
| `from ...params import thing_ids`                  | net_panoptic_sparse.py:20 | Import path          | Pass as parameter      |
| `KittiDataModule` import                           | train.py:2                | Dataset selection    | Add conditional import |
| `class_names, class_frequencies` import            | train.py:9                | Dataset params       | Add conditional import |

### 2.3 Architecture (DO NOT MODIFY in Phase 1)

| Component          | Current Design         | Notes                   |
| ------------------ | ---------------------- | ----------------------- |
| Encoder levels     | 4 levels (s1→s2→s4→s8) | Keep as-is for baseline |
| f_maps             | [f, f*2, f*4, f*4]     | Keep as-is              |
| Decoder structure  | 3-level upsampling     | Keep as-is              |
| Pruning thresholds | Dynamic                | Keep as-is              |

**Rationale:** Verify baseline works correctly before any architectural changes.

---

## 3. Architecture Decisions

### 3.1 Decision: Semantic-Only Mode First

**Rationale:**

1. Dataset has NO instance labels (only semantic voxel_labels)
2. Panoptic adds complexity without ground truth
3. Focus on core SSC (Semantic Scene Completion) task first

**Implementation:**

- Set `is_predict_panop=False` at runtime
- Set `thing_ids=[]` (empty list for no instance classes)
- Panoptic loss terms automatically disabled when thing_ids is empty

### 3.2 Decision: Input Feature Strategy

**Analysis of CylinderFeat (unet3d_sparse_v2.py:15-86):**

```python
class CylinderFeat(nn.Module):
    def __init__(self, fea_dim=3, out_pt_fea_dim=64, ...):
        # MLP: fea_dim → 64 → 128 → 256 → out_pt_fea_dim
        # Uses scatter_max for voxel aggregation
```

**Key Finding:** `fea_dim` is already a parameter! No structural change needed.

**Strategy:**

| Phase   | Features      | Dimension | Config Change     |
| ------- | ------------- | --------- | ----------------- |
| Phase 1 | xyz only      | 3D        | `--in_channels=3` |
| Phase 2 | xyz + density | 4D        | `--in_channels=4` |

### 3.3 Decision: Dynamic Grid Handling

**Problem:** Grid sizes vary (75-129 × 50-110 × 37-249)

**Solution:** MinkowskiEngine handles this natively via sparse tensors.

- No padding needed
- Batch index distinguishes samples
- Labels stored as list (variable sizes)

### 3.4 Receptive Field Analysis (For Future Reference)

Original PaSCo with 4-level encoder (stride 1→2→4→8):

- RF at bottleneck: ~16 voxels × 200mm = 3.2m (sufficient for vehicles)

Human body at 4mm voxel with same architecture:

- RF: ~16 voxels × 4mm = 64mm = 6.4cm

**Note:** This may be insufficient for organ-level reasoning (15-25cm needed).
**Action:** Validate baseline first. If RF is the bottleneck, consider adding encoder level in Phase 2.

---

## 4. Implementation Strategy

### Phase 0: Data Pipeline (MUST DO)

#### 4.1 params.py - Class Definitions

**File:** `pasco/data/human_body/params.py`

```python
n_classes = 71  # 0-70 inclusive
class_names = [...]  # 71 anatomical structure names
thing_ids = []  # Empty for semantic-only mode
class_frequencies = {
    "1_1": [...],  # Scale 1 frequencies
    "1_2": [...],  # Scale 2 frequencies
    "1_4": [...],  # Scale 4 frequencies
}
```

**Prerequisite:** Run frequency analysis script to compute exact counts.

#### 4.2 human_body_dataset.py - Core Dataset

**Required Interface (must match KittiDataset output):**

```python
def __getitem__(self, idx) -> dict:
    return {
        "frame_id": str,
        "sequence": str,
        "in_feat": Tensor[N, feat_dim],  # Point features
        "in_coord": Tensor[N, 3],        # Voxel coordinates (int)
        "T": Tensor[4, 4],               # Transform matrix
        "min_C": Tensor[3],              # Min coord bound
        "max_C": Tensor[3],              # Max coord bound
        "semantic_label": Tensor[...],   # Dense semantic labels
        "instance_label": Tensor[...],   # Can be zeros
        "mask_label": Tensor[...],       # Can be zeros
        "geo_labels": dict,              # Multi-scale geometry
        "sem_labels": dict,              # Multi-scale semantics
        # ... other fields
    }
```

**Data flow:**

1. Load NPZ → extract sensor_pc, voxel_labels, grid metadata
2. Point-to-voxel: `floor((xyz - grid_min) / voxel_size)`
3. Remove duplicates via `np.unique` with scatter_max
4. Features: xyz coordinates (optionally normalized)
5. Generate multi-scale labels via majority-vote downsampling
6. Return dict compatible with existing collate

**Augmentation:**

- Rotation: 360° around vertical axis
- Flip: X and Y axes
- Scale: 0.95-1.05 range

#### 4.3 collate.py - Batch Assembly

Use existing collate pattern from `pasco/data/semantic_kitti/collate.py`.

Key design:

- Concatenate coords with batch index prefix
- Store labels as list (variable sizes)
- Compute global min/max for decoder bounds

#### 4.4 human_body_dm.py - DataModule

**Split strategy (no predefined split):**

- Deterministic split via sorted filenames + modulo
- 80/10/10: 3222 train / 403 val / 403 test
- Fixed random seed for reproducibility

### Phase 1: Minimal Model Adaptation (REQUIRED)

#### 4.5 train.py Modification

**Current (hardcoded):**

```python
from pasco.data.semantic_kitti.kitti_dm import KittiDataModule
from pasco.data.semantic_kitti.params import class_names, class_frequencies
```

**Change to (conditional):**

```python
@click.option('--dataset', default="semantic_kitti", type=click.Choice(["semantic_kitti", "human_body"]))
def main(..., dataset, ...):
    if dataset == "semantic_kitti":
        from pasco.data.semantic_kitti.kitti_dm import KittiDataModule as DataModule
        from pasco.data.semantic_kitti.params import class_names, class_frequencies, thing_ids
    elif dataset == "human_body":
        from pasco.data.human_body.human_body_dm import HumanBodyDataModule as DataModule
        from pasco.data.human_body.params import class_names, class_frequencies, thing_ids
```

#### 4.6 net_panoptic_sparse.py Modification

**Issue 1: thing_ids hardcoded import**

```python
# Current (line 20):
from pasco.data.semantic_kitti.params import thing_ids

# Change to: Pass thing_ids as constructor parameter
def __init__(self, ..., thing_ids=None, ...):
    self.thing_ids = thing_ids if thing_ids is not None else []
```

**Issue 2: scene_size hardcoded**

```python
# Current (line 84):
self.scene_size = (256 // scale, 256 // scale, 32 // scale)

# Change to: Pass as parameter or compute from data
def __init__(self, ..., scene_size=None, ...):
    if scene_size is None:
        scene_size = (256 // scale, 256 // scale, 32 // scale)
    self.scene_size = scene_size
```

**Note:** scene_size is used in ensembler. For human body with dynamic grids, may need per-batch computation.

### Phase 2: Performance Optimization (ONLY IF NEEDED)

**Do NOT implement these until baseline is validated:**

1. Add 5th encoder level (s8→s16) if RF insufficient
2. Dilated convolutions if context still lacking
3. Attention mechanism for global context
4. Pruning threshold tuning

---

## 5. Implementation File Summary (Revised Priority)

### Must Create (P0)

| File                                          | Purpose                             |
| --------------------------------------------- | ----------------------------------- |
| `pasco/data/human_body/__init__.py`           | Module init                         |
| `pasco/data/human_body/params.py`             | Classes, frequencies, thing_ids     |
| `pasco/data/human_body/human_body_dataset.py` | Core dataset                        |
| `pasco/data/human_body/collate.py`            | Batch assembly                      |
| `pasco/data/human_body/human_body_dm.py`      | Lightning DataModule                |
| `configs/human-body.yaml`                     | Data config (class mapping, splits) |
| `scripts/compute_class_freq.py`               | Frequency analysis tool             |

### Must Modify (P0)

| File                                  | Change                             |
| ------------------------------------- | ---------------------------------- |
| `scripts/train.py`                    | Add dataset selection logic        |
| `pasco/models/net_panoptic_sparse.py` | Parameterize thing_ids, scene_size |

### Do NOT Modify (Phase 1)

| File                               | Reason                     |
| ---------------------------------- | -------------------------- |
| `pasco/models/encoder_v2.py`       | Keep original architecture |
| `pasco/models/decoder_v3.py`       | Keep original architecture |
| `pasco/models/unet3d_sparse_v2.py` | Already parameterized      |

---

## 6. Training Configuration

### 6.1 CLI Parameters for Human Body

```bash
python scripts/train.py \
    --dataset human_body \
    --dataset_root "E:/CODE/PaSCo-Rep-SepConv/Dataset/voxel_data" \
    --config_path "configs/human-body.yaml" \
    --in_channels 3 \
    --n_classes 71 \
    --lr 1e-4 \
    --bs 1 \
    --n_infers 1 \
    --heavy_decoder False \
    --data_aug True
```

### 6.2 Key Hyperparameters

| Parameter        | Value | Rationale                         |
| ---------------- | ----- | --------------------------------- |
| in_channels      | 3     | xyz only                          |
| n_classes        | 71    | 0-70 labels                       |
| lr               | 1e-4  | Lower than original, more classes |
| batch_size       | 1     | Memory constraints                |
| n_infers         | 1     | Disable MIMO initially            |
| is_predict_panop | False | Semantic only                     |

### 6.3 Class Imbalance Strategy

1. **Inverse-frequency weighting** with power scaling:
   
   ```
   raw_weight = max_freq / class_freq
   scaled_weight = raw_weight ^ (1/3)
   ```

2. **Lovász-Softmax loss** (already in codebase)

3. **Monitor per-class IoU** during training

---

## 7. Verification Checklist

### Stage 1: Data Pipeline Validation

- [ ] Load single NPZ, verify all fields present
- [ ] Point-to-voxel conversion correct
- [ ] Dataset __getitem__ returns compatible dict
- [ ] Collate produces valid batch structure
- [ ] Multi-scale labels shapes correct (1:1, 1:2, 1:4)
- [ ] Class frequency matches expected distribution

### Stage 2: Model Forward Pass (No Training)

- [ ] CylinderFeat accepts 3D input
- [ ] Encoder produces 4-level features
- [ ] Decoder outputs 71-class logits
- [ ] No runtime errors with single sample
- [ ] Peak GPU memory acceptable

### Stage 3: Training Sanity

- [ ] Single batch overfit: loss decreases
- [ ] Gradient flow: no NaN/Inf
- [ ] All 71 classes have non-zero gradients
- [ ] Validation metrics computed

### Stage 4: Baseline Validation

- [ ] 10 epochs without crash
- [ ] Loss consistently decreasing
- [ ] mIoU > 0 (model learning something)
- [ ] Per-class IoU: check for mode collapse

---

## 8. Risk Analysis & Mitigations

### Risk 1: Memory OOM with Large Grids

**Probability:** High
**Mitigations:**

1. Mixed precision (FP16)
2. Batch size = 1
3. Gradient accumulation
4. Skip samples > 500K voxels initially

### Risk 2: Class Imbalance Mode Collapse

**Probability:** High
**Mitigations:**

1. Power-scaled class weights
2. Lovász-Softmax loss
3. Monitor per-class IoU
4. Early stopping on validation mIoU

### Risk 3: Insufficient Receptive Field

**Probability:** Medium
**Mitigations:**

1. Validate baseline first
2. Add encoder level only if confirmed needed
3. Reserve attention mechanism for future

---

## 9. Decision Log

| Version | Decision                                     | Rationale                               |
| ------- | -------------------------------------------- | --------------------------------------- |
| v2.0    | Semantic-only mode                           | No instance labels in dataset           |
| v2.0    | Reuse CylinderFeat                           | Already parameterized for input dim     |
| v3.0    | **Remove** encoder modification from Phase 1 | Verify baseline first, minimize changes |
| v3.0    | Parameterize thing_ids                       | Avoid hardcoded import                  |
| v3.0    | Parameterize scene_size                      | Support dynamic grids                   |
| v3.0    | Keep original architecture in Phase 1        | Correct migration > optimization        |

---

## 10. Data Flow Diagram

```
NPZ File (human body)
    │
    ├─► sensor_pc (N, 3)           voxel_labels (D, H, W)
    │        │                            │
    │   point_to_voxel()             multi_scale_downsample()
    │   (floor division)                  │
    │        │                            │
    │        ▼                            ▼
    │   coords (M, 3) int         labels_1x, labels_2x, labels_4x
    │   feats (M, 3) float
    │        │
    │        ▼
    │   CylinderFeat(fea_dim=3)    ← No change needed!
    │        │
    │        ▼
    │   ME.SparseTensor
    │        │
    │        ▼
    │   Encoder (4 levels)         ← Keep original
    │        │
    │        ▼
    │   Dense Bottleneck
    │        │
    │        ▼
    │   Decoder                    ← Keep original
    │        │
    │        ▼
    │   71-class logits
```

---

## 11. Quick Start Implementation Order

1. **Create params.py** - Define 71 classes, compute frequencies
2. **Create human_body_dataset.py** - Implement __getitem__
3. **Create collate.py** - Copy from semantic_kitti, adjust
4. **Create human_body_dm.py** - Implement DataModule
5. **Modify train.py** - Add dataset selection
6. **Modify net_panoptic_sparse.py** - Parameterize thing_ids, scene_size
7. **Test forward pass** - Single sample, check shapes
8. **Test training** - Single batch overfit
9. **Full training** - Monitor metrics
