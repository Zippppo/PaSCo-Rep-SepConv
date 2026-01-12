# PaSCo Body Scene Completion - Implementation Plan

## Overview

Migrate PaSCo from autonomous driving to human body scene completion.

**Core Principle:** Validate architecture feasibility first, then incrementally optimize.

---

## Phase 0: Data Analysis & Adaptation Layer

### 0.1 Dataset Statistics

Analyze 4,028 samples to determine:

| Metric | Purpose |
|--------|---------|
| `grid_occ_size` (D,H,W) distribution | Determine fixed grid size |
| `sensor_pc` point count distribution | Determine sampling strategy |
| Per-class voxel ratio | Design class weights |
| Sparsity ratio | Estimate memory requirements |

### 0.2 Grid Size Strategy

**Dimension Convention:** `[D, H, W]` = `[Front-Back, Left-Right, Height]`

**Statistics Analysis (from 4028 samples):**

| Dimension | Axis | Mean | P95 | P99 | Max | Chosen |
|-----------|------|------|-----|-----|-----|--------|
| D | Front-Back | 101 | 122 | 129 | 132 | 128 |
| H | Left-Right | 77 | 100 | 107 | 129 | 128 |
| W | Height | 114 | 237 | 251 | 256 | 256 |

**Decision: `scene_size = [128, 128, 256]`**

- Total voxels: 4.19M
- Effective sparse voxels (~18% organ ratio): ~750K
- Coverage: D ~97%, H >99%, W ~100%
- Samples exceeding D=128 (~3-5%): handle with center crop

### 0.3 Data Adaptation Layer

```
pasco/data/body/
├── body_dataset.py    # BodyDataset class
├── body_dm.py         # BodyDataModule (Lightning)
├── collate.py         # Handle dynamic sizes with padding
└── params.py          # 72 class definitions & frequencies
```

**Key adaptations:**
- Load `.npz` files with `sensor_pc`, `voxel_labels`, `grid_world_min/max`
- Pad/crop to fixed grid size
- Voxelize surface point cloud to sparse tensor

---

## Phase 1: Minimum Viable Product (MVP)

**Goal:** Verify end-to-end training with minimal modifications.

### 1.1 Input Feature Simplification

| Component | Original PaSCo | Body MVP |
|-----------|----------------|----------|
| Input dim | 283D (27 + 256 WaffleIron) | 38D (6 + 32 pos_enc) |
| Features | xyz, vote, intensity, embedding | xyz, xyz_rel, positional_encoding |

### 1.2 Architecture Simplification

- `n_infers = 1`: Disable MIMO, single subnet
- `num_queries = 72`: Queries = num_classes
- Keep all 72 classes (**do NOT merge air classes**)

**Why keep `inside_body_empty`:**
- Critical anatomical boundary marker (lung cavity, intestinal cavity)
- Model needs to distinguish outside vs inside empty space
- Minimal computational overhead

### 1.3 Loss Adjustment

- Remove instance-related losses
- Use frequency-weighted CE + Lovász
- Adjust class weights based on Phase 0 statistics

### 1.4 Success Criteria

- mIoU > 5% (better than random)
- Loss converges within 10 epochs
- No OOM errors

---

## Phase 2: Feature Enhancement

**Prerequisite:** Phase 1 MVP successfully converges.

### 2.1 Point Cloud Encoder Options

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| Reuse Encoder3DSepV2 | Architecture consistency | Need input layer modification | Recommended |
| PointNet++ | Mature, stable | Computational overhead | Backup |
| Simple MLP + local features | Lightweight | Limited expressiveness | Not recommended |

### 2.2 Training Strategy

- End-to-end training (no pre-training initially)
- If convergence issues, consider staged training

### 2.3 Success Criteria

- mIoU > 20%
- Reasonable per-class performance distribution

---

## Phase 3: Full Migration (Optional)

**Prerequisite:** Phase 2 mIoU > 30%.

### 3.1 Instance Label Construction

Based on anatomical priors for paired/multiple organs:

| Organ Type | Instances |
|------------|-----------|
| Ribs (left/right 1-12) | 24 instances |
| Kidneys (left/right) | 2 instances |
| Lungs (left/right) | 2 instances |
| Vertebrae | Multiple instances |

**Benefits:**
- Feature sharing across same organ type
- Sample balancing (e.g., 24× for ribs)
- Better generalization

### 3.2 MIMO Restoration

- `n_infers = 4`
- Enable uncertainty estimation
- Restore instance-level losses

---

## Configuration Changes

### New config: `body.yaml`

```yaml
# Scene geometry
n_classes: 72
scene_size: [128, 128, 256]  # TBD after Phase 0 statistics
voxel_size: 4  # mm

# Input features
in_channels: 38  # 6 (xyz + xyz_rel) + 32 (pos_enc)

# Architecture (Phase 1)
n_infers: 1
num_queries: 72

# Loss weights (TBD after class frequency analysis)
class_weights: [...]
```

---

## File Modification Summary

| File | Modification |
|------|--------------|
| `pasco/data/body/*` | New: Dataset, DataModule, collate, params |
| `pasco/models/net_panoptic_sparse.py` | Adjust `in_channels`, add pos_enc option |
| `pasco/models/unet3d_sparse_v2.py` | Modify CylinderFeat for new input dim |
| `configs/body.yaml` | New: Body-specific configuration |
| `scripts/train.py` | Add body dataset option |

---

## Action Items

### Immediate (Phase 0)

- [ ] Run dataset statistics script
- [ ] Determine fixed grid size from statistics
- [ ] Implement BodyDataset class
- [ ] Implement body collate function
- [ ] Create body.yaml config

### Next (Phase 1)

- [ ] Implement positional encoding for xyz
- [ ] Adjust CylinderFeat input dimension
- [ ] Set n_infers=1, disable MIMO
- [ ] Train MVP and verify convergence

### Later (Phase 2+)

- [ ] Add point cloud encoder if needed
- [ ] Hyperparameter tuning
- [ ] Instance label construction (Phase 3)
- [ ] MIMO restoration (Phase 3)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Grid size too large → OOM | Use smaller batch size, gradient accumulation |
| Poor convergence without rich features | Add simple encoder, consider pre-training |
| Class imbalance (72 classes) | Aggressive class weighting, focal loss |
| Dynamic grid sizes | Fixed size with padding (Phase 1), optimize later |

---

## References

- Original PaSCo: `pasco/data/semantic_kitti/`
- Body dataset spec: `Dataset/DATASET_README.md`
- Analysis notes: `docs/pasco-body.md`
