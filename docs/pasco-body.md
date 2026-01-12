# PaSCo Human Body Adaptation Plan (v4.1)

> **Revision History:**
> - v4.1: Added eval.py modification, complete interface contract, feature dimension risk, augmentation risk
> - v4.0: Initial three-layer boundary model

## Executive Summary

**Goal:** Migrate PaSCo from autonomous driving to human body scene completion.

**Core Principle:** Correct migration first, minimize changes to original PaSCo.

**Key Insight:** PaSCo's sparse convolution architecture is highly parameterized. Most adaptation can be achieved through **data adaptation layer** without touching model internals.

---

## 1. Architectural Analysis

### 1.1 Three-Layer Boundary Model

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Data Adaptation Layer  (NEW - must create)            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  params.py          - Class definitions, frequencies    │    │
│  │  human_body_dataset - NPZ loading, voxelization         │    │
│  │  collate.py         - Batch assembly                    │    │
│  │  human_body_dm.py   - Lightning DataModule              │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Interface Layer  (MODIFY - minimal surgery)           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  train.py           - Dataset selection dispatch         │    │
│  │  eval.py            - Dataset selection dispatch         │    │
│  │  net_panoptic.py    - Parameterize thing_ids injection   │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Model Core  (DO NOT TOUCH)                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  encoder_v2.py      - Sparse 3D encoder                  │    │
│  │  decoder_v3.py      - Sparse 3D decoder + transformer    │    │
│  │  unet3d_sparse_v2.py- UNet backbone (already parameterized)│  │
│  │  criterion_sparse.py- Loss computation                   │    │
│  │  helper.py          - Inference utilities                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Discovery: What's Already Parameterized

| Parameter | Location | Status |
|-----------|----------|--------|
| `in_channels` | CylinderFeat | Parameterized via `fea_dim` |
| `n_classes` | Net.__init__ | CLI configurable |
| `class_weights` | Net.__init__ | Computed from frequencies |
| `f` (base channels) | Net.__init__ | CLI configurable |
| `scene_size` (training) | helper.py:432 | Dynamic via `compute_scene_size()` |

### 1.3 What Must Be Modified (Minimal Set)

| Issue | Location | Required Change |
|-------|----------|-----------------|
| `thing_ids` hardcoded import | net_panoptic_sparse.py:20 | Parameterize via constructor |
| `thing_ids` not passed to Net | train.py:146-172 | Add `thing_ids=thing_ids` to Net() |
| `n_classes` hardcoded | train.py:115 | Add CLI param `--n_classes` |
| `in_channels` default 283 | net_panoptic_sparse.py:51 | Add CLI param `--in_channels` |
| `scene_size` hardcoded in ensemble | ensembler.py (multiple) | **Avoided by n_infers=1** |
| Dataset dispatch | train.py, eval.py | Add conditional import |
| Multi-scale label hardcode | human_body_dataset.py | Use `n_classes+1` instead of 21 |

---

## 2. Constraint Analysis

### 2.1 Scale Constraint
- Human body: 4mm voxels (50x smaller than KITTI's 200mm)
- Grid dimensions: Dynamic (75-129 x 50-110 x 37-249)
- **Implication:** MinkowskiEngine handles natively; no architecture change needed

### 2.2 Feature Constraint
- KITTI: 283D features (xyz_offset(3) + xyz(3) + vote(3) + intensity(1) + embedding(256) + ...)
- Human body: 3D features (xyz only)
- **Implication:** `CylinderFeat(fea_dim=3)` - already parameterized
- **Risk:** 283→3 feature compression may cause information loss; monitor baseline performance

### 2.3 Class Constraint
- KITTI: 20 classes, 8 "thing" classes with instances
- Human body: 71 classes (0-70), no instance labels
- **Implication:** `thing_ids=[]` for semantic-only mode

### 2.4 MIMO Constraint (Critical)
- `ensembler.py` hardcodes `(256, 256, 32)` for multi-subnet alignment
- **Implication:** Must use `n_infers=1` to bypass ensemble code path

---

## 3. Migration Strategy

### 3.1 Phase 1: Data-Only Migration (Target State)

**Principle:** All changes confined to Data Adaptation Layer + minimal Interface Layer surgery.

```
Changes Required:
├── CREATE: pasco/data/human_body/
│   ├── __init__.py
│   ├── params.py           # Class definitions
│   ├── human_body_dataset.py
│   ├── collate.py
│   └── human_body_dm.py
├── CREATE: scripts/
│   └── compute_human_body_freq.py  # Utility to compute class frequencies
├── MODIFY: scripts/train.py
│   └── Add dataset dispatch, --n_classes, --in_channels params
├── MODIFY: scripts/eval.py
│   └── Add dataset dispatch (same pattern as train.py)
└── MODIFY: pasco/models/net_panoptic_sparse.py
    └── Line 20: Remove hardcoded import, add thing_ids parameter
```

### 3.2 Interface Contract

**Dataset.__getitem__() must return:**

```python
{
    # Required for forward pass
    "in_feat": Tensor[N, 3],        # Point features (xyz)
    "in_coord": Tensor[N, 3],       # Voxel coordinates (int)
    "min_C": Tensor[3],             # Min coordinate bound
    "max_C": Tensor[3],             # Max coordinate bound
    "T": Tensor[4, 4],              # Augmentation transform
    "xyz": Tensor[N, 3],            # Original point cloud coordinates

    # Required for loss computation
    "semantic_label": Tensor[D, H, W],  # Dense semantic grid (augmented)
    "geo_labels": {"1_1": ..., "1_2": ..., "1_4": ...},
    "sem_labels": {"1_1": ..., "1_2": ..., "1_4": ...},

    # Required for collate (pre-augmentation labels)
    "semantic_label_origin": Tensor[D, H, W],  # Dense semantic grid (original)
    "instance_label_origin": Tensor[D, H, W],  # Dense instance grid (original)
    "mask_label_origin": {"labels": [], "masks": []},  # Original mask labels

    # Instance-related (zeros for semantic-only mode)
    "instance_label": Tensor[D, H, W],  # Can be zeros
    "mask_label": {"labels": [], "masks": []},  # Empty lists for semantic-only
    "input_pcd_instance_label": Tensor[N],  # Can be zeros

    # Metadata
    "frame_id": str,
    "sequence": str,
}
```

**Critical Notes:**
1. Collate function must produce `global_min_Cs`, `global_max_Cs` aligned to `complete_scale=8`
2. Coordinate alignment: `min_C = floor(min_C / complete_scale) * complete_scale`
3. For semantic-only mode: `mask_label["labels"]=[]`, `mask_label["masks"]=[]`

### 3.3 Configuration for Human Body

```bash
python scripts/train.py \
    --dataset human_body \
    --in_channels 3 \
    --n_classes 71 \
    --n_infers 1 \           # CRITICAL: Disable MIMO
    --heavy_decoder False \  # Memory optimization
    --lr 1e-4 \
    --bs 1
```

---

## 4. Implementation Priorities

### P0: Must Do (Correctness)

1. **Data Pipeline** - Create all files in `pasco/data/human_body/`
2. **Interface Surgery** - Parameterize `thing_ids` in Net constructor
3. **Train Dispatch** - Add dataset selection in `train.py`

### P1: Should Do (Robustness)

4. **Class Frequency Analysis** - Compute accurate weights from dataset
5. **Multi-scale Label Generation** - Verify downsampling logic

### P2: Deferred (Optimization)

6. **Receptive Field** - Validate 4-level encoder sufficiency
7. **MIMO Support** - Modify ensembler if needed (unlikely)
8. **Architecture Tuning** - Add encoder levels if RF insufficient

---

## 5. Modification Details

### 5.1 net_panoptic_sparse.py Modification

**Current (line 20):**
```python
from pasco.data.semantic_kitti.params import thing_ids
```

**Target:**
```python
# Remove hardcoded import
# Add thing_ids as constructor parameter with default
def __init__(self, ..., thing_ids=None, ...):
    self.thing_ids = thing_ids if thing_ids is not None else []
```

**Rationale:** Single point of change, backward compatible with default.

### 5.2 train.py Modification

**Current:**
```python
from pasco.data.semantic_kitti.kitti_dm import KittiDataModule
from pasco.data.semantic_kitti.params import class_names, class_frequencies
# ...
n_classes = 20  # Line 115: hardcoded
# ...
model = Net(
    n_classes=n_classes,
    # thing_ids not passed - uses hardcoded import
)
```

**Target:**
```python
# Add new CLI options
@click.option('--dataset', default="semantic_kitti",
              type=click.Choice(["semantic_kitti", "human_body"]))
@click.option('--n_classes', default=20, help='Number of semantic classes')
@click.option('--in_channels', default=283, help='Input feature dimension')
def main(..., dataset, n_classes, in_channels, ...):
    # Dataset dispatch
    if dataset == "semantic_kitti":
        from pasco.data.semantic_kitti.kitti_dm import KittiDataModule as DataModule
        from pasco.data.semantic_kitti.params import (class_names, class_frequencies, thing_ids)
    elif dataset == "human_body":
        from pasco.data.human_body.human_body_dm import HumanBodyDataModule as DataModule
        from pasco.data.human_body.params import (class_names, class_frequencies, thing_ids)

    # Pass thing_ids to Net constructor
    model = Net(
        n_classes=n_classes,
        in_channels=in_channels,
        thing_ids=thing_ids,  # NEW: pass thing_ids
        # ... other params
    )
```

**Rationale:** Clean dispatch pattern, parameterized n_classes/in_channels, thing_ids passed to model.

### 5.3 eval.py Modification

**Current:**
```python
from pasco.data.semantic_kitti.kitti_dm import KittiDataModule
from pasco.data.semantic_kitti.params import class_frequencies
```

**Target:**
```python
@click.option('--dataset', default="semantic_kitti",
              type=click.Choice(["semantic_kitti", "human_body"]))
def main(..., dataset, ...):
    if dataset == "semantic_kitti":
        from pasco.data.semantic_kitti.kitti_dm import KittiDataModule as DataModule
        from pasco.data.semantic_kitti.params import class_frequencies
    elif dataset == "human_body":
        from pasco.data.human_body.human_body_dm import HumanBodyDataModule as DataModule
        from pasco.data.human_body.params import class_frequencies

    data_module = DataModule(...)
```

**Rationale:** Same dispatch pattern as train.py for consistency.

---

## 6. Validation Strategy

### Stage 1: Data Pipeline Isolation Test
- Load single sample, verify all fields present
- Verify coordinate conversion: `floor((xyz - grid_min) / voxel_size)`
- Verify multi-scale labels shapes match expected ratios

### Stage 2: Forward Pass Test (No Training)
- Single sample through model
- Check: CylinderFeat accepts 3D input
- Check: Output logits have 71 channels
- Check: No runtime errors

### Stage 3: Training Sanity Test
- Single batch overfit (loss should decrease)
- Check gradient flow (no NaN/Inf)
- Check per-class loss distribution

### Stage 4: Baseline Validation
- 10 epochs without crash
- mIoU > 0 (model learning)
- Per-class IoU distribution (check for mode collapse)

---

## 7. Risk Mitigation

### Risk 1: Memory OOM
- **Trigger:** Large grids (3M voxels)
- **Mitigation:** `bs=1`, FP16, gradient accumulation, sample filtering

### Risk 2: Class Imbalance Collapse
- **Trigger:** 51% background dominates
- **Mitigation:** Power-scaled weights, Lovasz-Softmax, per-class monitoring

### Risk 3: Insufficient Receptive Field
- **Trigger:** Poor large-structure segmentation
- **Mitigation:** Validate baseline first; defer architecture changes to Phase 2

### Risk 4: Dynamic Grid Incompatibility
- **Trigger:** Ensemble code expects fixed grid
- **Mitigation:** `n_infers=1` bypasses ensemble entirely

### Risk 5: Feature Dimension Information Loss
- **Trigger:** KITTI uses 283D features, human body uses only 3D (xyz)
- **Symptoms:** Model underfits, poor convergence, low mIoU
- **Mitigation:**
  1. Monitor baseline performance carefully
  2. If insufficient, consider adding engineered features (normals, curvature)
  3. Defer to Phase 2 if needed

### Risk 6: Data Augmentation Mismatch
- **Trigger:** KITTI augmentation params (max_angle=5.0) designed for driving scenes
- **Mitigation:** Tune augmentation for human body (larger rotation range, appropriate flip axes)

---

## 8. Decision Record

| Decision | Rationale |
|----------|-----------|
| `n_infers=1` mandatory | Ensembler hardcodes (256,256,32) |
| `thing_ids=[]` | No instance labels in dataset |
| `in_channels=3` | xyz-only features sufficient for baseline |
| Defer architecture changes | Verify correct migration first |
| Parameterize via constructor | Minimal code change, backward compatible |

---

## 9. File Checklist

### Create (Data Adaptation Layer)
- [ ] `pasco/data/human_body/__init__.py`
- [ ] `pasco/data/human_body/params.py`
- [ ] `pasco/data/human_body/human_body_dataset.py`
- [ ] `pasco/data/human_body/collate.py`
- [ ] `pasco/data/human_body/human_body_dm.py`

### Create (Utility Scripts)
- [ ] `scripts/compute_human_body_freq.py` - Compute class frequencies from dataset

### Modify (Interface Layer)
- [ ] `scripts/train.py` - Dataset dispatch, --n_classes, --in_channels, thing_ids pass-through
- [ ] `scripts/eval.py` - Dataset dispatch (same pattern as train.py)
- [ ] `pasco/models/net_panoptic_sparse.py` - thing_ids parameter in constructor

### Do Not Modify
- [ ] `pasco/models/encoder_v2.py`
- [ ] `pasco/models/decoder_v3.py`
- [ ] `pasco/models/unet3d_sparse_v2.py`
- [ ] `pasco/models/ensembler.py`
- [ ] `pasco/loss/*`

---

## 10. Success Criteria

**Phase 1 Complete When:**
1. Training runs without error for 10 epochs
2. Loss consistently decreases
3. mIoU > 0 on validation set
4. No model core files modified

**Migration Validated When:**
1. Per-class IoU shows learning across multiple classes (not just background)
2. Visual inspection of predictions shows reasonable structure
3. Memory usage stable within GPU limits
