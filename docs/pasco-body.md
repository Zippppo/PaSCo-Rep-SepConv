# PaSCo Human Body Adaptation Plan

> **Goal:** Migrate PaSCo from autonomous driving to human body scene completion.
> **Core Principle:** Zero modification to original PaSCo codebase. All new code in isolated directories.
> **Data Description:** See `Dataset/DATASET_README.md`

---

## 1. Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Data Adaptation Layer  (CREATE - isolated)            │
│    pasco/data/human_body/                                       │
│    ├── params.py          - Class definitions, frequencies      │
│    ├── human_body_dataset - NPZ loading, voxelization           │
│    ├── collate.py         - Batch assembly                      │
│    └── human_body_dm.py   - Lightning DataModule                │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Interface Layer  (CREATE - isolated, imports origin)  │
│    scripts/body/                                                │
│    ├── train.py           - Standalone training script          │
│    └── eval.py            - Standalone evaluation script        │
│    pasco/models/body/                                           │
│    └── net_panoptic_sparse.py - Wrapper with thing_ids param    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Model Core  (DO NOT TOUCH - import only)              │
│    pasco/models/encoder_v2.py, decoder_v3.py, unet3d_sparse_v2  │
│    pasco/loss/criterion_sparse.py                               │
└─────────────────────────────────────────────────────────────────┘
```

**What's Already Parameterized (No Changes Needed):**

| Parameter | Location | Status |
|-----------|----------|--------|
| `in_channels` | CylinderFeat | Via `fea_dim` |
| `n_classes` | Net.__init__ | CLI configurable |
| `class_weights` | Net.__init__ | Computed from frequencies |
| `scene_size` | helper.py | Dynamic via `compute_scene_size()` |

---

## 2. Constraints

| Constraint | KITTI (Original) | Human Body | Implication |
|------------|------------------|------------|-------------|
| **Voxel Size** | 200mm | 4mm | MinkowskiEngine handles natively |
| **Features** | 283D | 6D (xyz + xyz_offset) | `CylinderFeat(fea_dim=6)` |
| **Classes** | 20 classes, 8 things | 71 classes, no instances | `thing_ids=[]` |
| **MIMO** | Multi-subnet ensemble | Must disable | `n_infers=1` (bypasses hardcoded 256x256x32) |

**Key Insight:** Only surface point cloud needs input features. Interior voxels are prediction targets (GT labels), not inputs.

---

## 3. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `n_infers=1` mandatory | Ensembler hardcodes (256,256,32) grid size |
| `thing_ids=[]` | No instance labels in dataset |
| `in_channels=6` | xyz(3) + xyz_offset(3) for sub-voxel precision |
| Isolated file structure | Zero modification to original PaSCo |
| Wrapper class for Net | Minimal code, reuses all original logic |
| Defer architecture changes | Verify correct migration first |

---

## 4. File Checklist

### CREATE (Data Adaptation Layer)
- [ ] `pasco/data/human_body/__init__.py`
- [ ] `pasco/data/human_body/params.py` - Class names, frequencies, thing_ids
- [ ] `pasco/data/human_body/human_body_dataset.py` - NPZ loading, voxelization, xyz_offset
- [ ] `pasco/data/human_body/collate.py` - Batch assembly with coordinate alignment
- [ ] `pasco/data/human_body/human_body_dm.py` - Lightning DataModule

### CREATE (Model Wrapper Layer)
- [ ] `pasco/models/body/__init__.py`
- [ ] `pasco/models/body/net_panoptic_sparse.py` - Wrapper with thing_ids parameter

### CREATE (Scripts Layer)
- [ ] `scripts/body/train.py` - Standalone training script
- [ ] `scripts/body/eval.py` - Standalone evaluation script
- [ ] `scripts/compute_human_body_freq.py` - Compute class frequencies

### DO NOT MODIFY
- `scripts/train.py`, `scripts/eval.py`
- `pasco/models/net_panoptic_sparse.py`, `encoder_v2.py`, `decoder_v3.py`
- `pasco/models/ensembler.py`
- `pasco/loss/*`

---

## 5. Interface Contract

**Dataset.__getitem__() must return:**

```python
{
    # Forward pass
    "in_feat": Tensor[N, 6],           # xyz + xyz_offset
    "in_coord": Tensor[N, 3],          # Voxel coordinates (int)
    "min_C": Tensor[3],                # Min coordinate bound
    "max_C": Tensor[3],                # Max coordinate bound
    "T": Tensor[4, 4],                 # Augmentation transform
    "xyz": Tensor[N, 3],               # Original point coordinates

    # Loss computation (multi-scale labels)
    "semantic_label": Tensor[D, H, W],
    "geo_labels": {"1_1": ..., "1_2": ..., "1_4": ...},
    "sem_labels": {"1_1": ..., "1_2": ..., "1_4": ...},

    # Instance-related (zeros/empty for semantic-only)
    "instance_label": Tensor[D, H, W],  # Zeros
    "mask_label": {"labels": [], "masks": []},  # Empty

    # Metadata
    "frame_id": str, "sequence": str,
}
```

**Critical:** Collate must align coordinates to `complete_scale=8`: `min_C = floor(min_C / 8) * 8`

---

## 6. Validation & Success Criteria

### Validation Stages
1. **Data Pipeline** - Load single sample, verify all fields, check coordinate conversion
2. **Forward Pass** - Single sample through model, verify 71-channel output
3. **Training Sanity** - Single batch overfit, check gradient flow
4. **Baseline** - 10 epochs, mIoU > 0

### Success Criteria
- [ ] Training runs 10 epochs without error
- [ ] Loss consistently decreases
- [ ] mIoU > 0 on validation set
- [ ] **No original PaSCo files modified**
- [ ] Per-class IoU shows learning across multiple classes

---

## 7. Risks

| Risk | Trigger | Mitigation |
|------|---------|------------|
| Memory OOM | Large grids (3M voxels) | `bs=1`, FP16, gradient accumulation |
| Class Imbalance | 51% background | Power-scaled weights, Lovasz-Softmax |
| Insufficient RF | Poor large-structure segmentation | Defer to Phase 2 |
| Feature Dimension Gap | 283D → 6D | xyz_offset helps; fallback: add normals (9D) |

---

## 8. CLI Configuration

```bash
python scripts/body/train.py \
    --in_channels 6 \
    --n_classes 71 \
    --n_infers 1 \
    --heavy_decoder False \
    --lr 1e-4 \
    --bs 1
```
