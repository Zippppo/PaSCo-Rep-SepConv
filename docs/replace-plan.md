# PaSCo MinkowskiEngine → TorchSparse Migration Plan (v3 Optimized)

## Overview

**Goal**: Replace MinkowskiEngine with TorchSparse for 4.6x training / 2.9x inference speedup

**Constraints**:
- Direct replacement, no backend switching
- No pretrained model conversion
- Priority: Speed first

**Key Metrics** (from codebase analysis):
- 16 files to modify (not 19 - merged redundant)
- 175 ME API calls total
- 80% direct replacement, 14% adaptation, 6% special handling

---

## Core Insight: Why Current Plan is Suboptimal

| Issue | Current Plan | Optimized |
|-------|--------------|-----------|
| Risk timing | High-risk `sparse_to_dense` in Phase 5 | Validate in Phase 1 |
| Parallelism | Sequential file processing | Parallel where dependency allows |
| Validation | End-to-end test only at Phase 6 | Vertical slice validation per phase |
| Dependency | Layer-based (all blocks → all encoders) | Dependency-chain based |

---

## Dependency Graph (Critical Path)

```
                    ┌─────────────────┐
                    │ torchsparse_utils│  ← Phase 1
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │ mink.py  │        │layers.py │        │dropout.py│  ← Phase 2 (parallel)
   └────┬─────┘        └────┬─────┘        └──────────┘
        │                   │
        ▼                   │
┌───────────────┐           │
│ encoder_v2.py │           │
└───────┬───────┘           │
        │                   │
        ▼                   ▼
┌───────────────────────────────────┐
│       unet3d_sparse_v2.py         │  ← Phase 3
└───────────────┬───────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌─────────────┐  ┌──────────────────────────┐
│decoder_v3.py│  │transformer_predictor_v2.py│  ← Phase 4 (highest risk)
└─────────────┘  └──────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│    net_panoptic_sparse*.py        │  ← Phase 5 (integration)
└───────────────────────────────────┘
```

---

## API Mapping (Verified)

### Direct Replacement (80%)
| ME | TorchSparse |
|----|-------------|
| `ME.MinkowskiConvolution` | `spnn.Conv3d` |
| `ME.MinkowskiBatchNorm` | `spnn.BatchNorm` |
| `ME.MinkowskiReLU/LeakyReLU` | `spnn.ReLU/LeakyReLU` |
| `ME.MinkowskiConvolutionTranspose` | `spnn.Conv3d(transposed=True)` |
| `ME.MinkowskiMaxPooling` | `spnn.MaxPool3d` |
| `ME.cat()` | `torchsparse.cat()` |

### Adaptation Required (14%)
| ME | TorchSparse | Pattern |
|----|-------------|---------|
| `ME.SparseTensor(F, coord_key, coord_mgr)` | `SparseTensor(feats=F, coords=x.coords, stride=x.stride)` | Explicit coord |
| `ME.MinkowskiLinear` | `nn.Linear` on `.feats` + wrap | Feature-only op |
| `ME.MinkowskiSigmoid` | `torch.sigmoid` on `.feats` + wrap | Feature-only op |
| `ME.MinkowskiGlobalPooling` | `torchsparse.nn.functional.global_avg_pool` | Returns dense (B,C) |
| `ME.MinkowskiPruning` | Boolean indexing | Custom helper |

### Special Handling (6%)
| ME | TorchSparse | Complexity |
|----|-------------|------------|
| `expand_coordinates=True` | `generative=True` | Coord generation |
| `.dense(shape, min_coord)` | Custom `sparse_to_dense()` | HIGH RISK |
| `ME.utils.batched_coordinates` | Custom `batched_coordinates()` | Batch assembly |
| `tensor_stride` attribute | `.stride` | Attribute rename |

---

## Phase 1: Foundation + Risk Validation

**Goal**: Create utilities AND validate high-risk patterns upfront

**Deliverables**: `pasco/backend/torchsparse_utils.py`

```python
# Core utilities (6 functions)
sparse_tensor_like(source, feats)      # Create with source's coords/stride
sparse_prune(x, mask)                  # Boolean mask pruning
broadcast_multiply(x, pooled)          # Pool + broadcast pattern
batched_coordinates(coords_list)       # Batch coord assembly
sparse_to_dense(x, shape, min_coord)   # Dense conversion - HIGH RISK
PointCloudField                        # TensorField replacement
```

**Risk Validation Tests** (must pass before Phase 2):
1. `sparse_to_dense` with edge cases (empty batch, coords at boundary)
2. `generative=True` outputs more coords than input
3. Gradient flow through all differentiable utilities

**Exit Criteria**:
- [ ] All 6 utilities implemented
- [ ] Unit tests pass with gradient checks
- [ ] `sparse_to_dense` handles OOB coords gracefully

---

## Phase 2: Foundation Blocks (Parallelizable)

**Goal**: Convert atomic building blocks

**Can run in parallel** (no inter-dependencies):

| File | Classes | ME Calls | Complexity |
|------|---------|----------|------------|
| `mink.py` | 11 | 68 | High - base for all |
| `layers.py` | 17 | 27 | Medium - independent utilities |
| `dropout.py` | 2 | 3 | Low - uses `sparse_tensor_like` |

**mink.py Conversion Order** (sequential within file):
1. `BasicConvolutionBlock` → Direct `spnn.Conv3d`
2. `BasicDeconvolutionBlock` → `spnn.Conv3d(transposed=True)`
3. `BasicGenerativeDeconvolutionBlock` → Add `generative=True`
4. `ResidualBlock` / `ResidualBlockOriginal`
5. `drop_path()` → Use `sparse_tensor_like`
6. `SELayer` → Manual broadcast pattern
7. `ASPP` → GlobalPool + concat + broadcast

**SELayer Pattern** (critical):
```python
# ME version (implicit broadcast via coordinate_manager)
y = self.pooling(x)  # Returns SparseTensor with same coords
y = self.fc(y)
return ME.SparseTensor(y.F * x.F, coord_key=x.coord_key, coord_mgr=x.coord_mgr)

# TorchSparse version (explicit broadcast)
y = F.global_avg_pool(x)           # Returns (B, C) dense tensor
y = self.fc(y)                     # (B, C) dense
batch_idx = x.coords[:, 0].long()  # Get batch index per point
return SparseTensor(feats=y[batch_idx] * x.feats, coords=x.coords, stride=x.stride)
```

**Exit Criteria**:
- [ ] Each block's forward pass succeeds independently
- [ ] Stride propagation verified (log stride at each layer)
- [ ] SELayer output coords == input coords

---

## Phase 3: Encoder-Decoder Pipeline

**Goal**: Wire up encoder-decoder with skip connections

**Files** (sequential - dependency chain):
1. `encoder_v2.py` - Depends on mink.py blocks
2. `unet3d_sparse_v2.py` - Assembles encoder + decoder
3. `decoder_v3.py` - Uses generative deconv + pruning

**Key Challenge**: Skip connection concatenation
```python
# ME.cat() requires identical coordinates
# Architecture ensures this via stride matching:
# - Encoder stores outputs at stride [1, 2, 4, 8]
# - Decoder upsamples back to matching stride
# - torchsparse.cat() has same requirement
```

**DecoderBlock Pattern** (decoder_v3.py):
```python
# Generative upsample creates new coordinates
upsampled = self.deconv(x)  # generative=True → more points
# Pruning removes invalid points
mask = self.pruning_predictor(upsampled)
pruned = sparse_prune(upsampled, mask > threshold)
```

**Vertical Slice Test**:
```python
# Minimal forward pass test
encoder = MinkEncoderDecoder(...)
x = create_test_sparse_tensor(batch_size=2)
outputs = encoder(x)
assert len(outputs) == 4  # [y1, y2, y3, y4]
assert outputs[0].stride == (8,8,8)
assert outputs[3].stride == (1,1,1)
```

**Exit Criteria**:
- [ ] Full encoder-decoder forward pass completes
- [ ] Output strides: [8, 4, 2, 1]
- [ ] Skip connections verified (coords match before cat)

---

## Phase 4: Transformer (Highest Risk)

**Goal**: Convert transformer predictor, especially `compute_attn_mask`

**Critical Function**: `compute_attn_mask()` (transformer_predictor_v2.py:220-289)

**ME API usage in this function**:
1. Line 230: `ME.utils.batched_coordinates(keep_mask_C)`
2. Line 231: `ME.SparseTensor(keep_mask_F, keep_mask_C)`
3. Line 254: `ME.utils.batched_coordinates([...])` (inside loop)
4. Line 258-261: `ME.SparseTensor(..., tensor_stride=...)`
5. Line 263-274: `.dense(shape, min_coordinate)` ← **HIGHEST RISK**

**sparse_to_dense Risk Analysis**:
```python
# ME version returns tuple, takes min_coordinate as IntTensor
keep_mask_dense = sparse.dense(
    shape=torch.Size([1, C, D, H, W]),
    min_coordinate=torch.IntTensor([*min_Cs[i]])
)[0]  # [0] to get tensor from tuple

# TorchSparse version must:
# 1. Handle coords potentially outside [0, shape) range
# 2. Clamp or skip out-of-bounds coords
# 3. Return single tensor (not tuple)
```

**Mitigation**:
```python
def sparse_to_dense(x, shape, min_coord):
    # Shift coords to start from 0
    shifted = x.coords[:, 1:] - min_coord
    # Clamp to valid range
    shifted = shifted.clamp(min=0, max=torch.tensor(shape[2:]) - 1)
    # Create dense tensor
    dense = torch.zeros(shape, device=x.feats.device, dtype=x.feats.dtype)
    batch_idx = x.coords[:, 0]
    dense[batch_idx, :, shifted[:, 0], shifted[:, 1], shifted[:, 2]] = x.feats
    return dense
```

**Other transformer_predictor_v2.py changes**:
- Line 364-368: `ME.SparseTensor(features=..., coordinates=..., tensor_stride=...)`
  → `SparseTensor(feats=..., coords=..., stride=...)`

**Exit Criteria**:
- [ ] `compute_attn_mask` output shape: `(B*nheads, num_queries, N)` boolean
- [ ] No index out of bounds errors
- [ ] Attention mask values match expected pattern

---

## Phase 5: Integration

**Goal**: Final assembly and end-to-end validation

**Files**:
- `net_panoptic_sparse.py` - Main model assembly
- `net_panoptic_sparse_kitti360.py` - KITTI360 variant
- `collate.py` - Data format for TorchSparse
- `criterion_sparse.py` - Loss on SparseTensor outputs

**Data Format Change**:
```python
# ME format
coords = ME.utils.batched_coordinates(coords_list)
sparse = ME.SparseTensor(feats, coords)

# TorchSparse format
coords = batched_coordinates(coords_list)  # Our utility
sparse = SparseTensor(feats=feats, coords=coords, stride=1)
```

**Integration Checklist**:
- [ ] Update all imports: `import MinkowskiEngine as ME` → `import torchsparse` + utils
- [ ] Verify collate produces correct TorchSparse input format
- [ ] Loss computation handles SparseTensor.feats correctly
- [ ] Model construction without errors
- [ ] Single training step completes
- [ ] Loss values finite (no NaN/Inf)

---

## File Modification Summary (16 files)

| Priority | File | ME Calls | Risk | Phase |
|----------|------|----------|------|-------|
| P1 | `pasco/backend/torchsparse_utils.py` | NEW | - | 1 |
| P1 | `pasco/maskpls/mink.py` | 68 | High | 2 |
| P1 | `pasco/models/layers.py` | 27 | Medium | 2 |
| P1 | `pasco/models/dropout.py` | 3 | Low | 2 |
| P2 | `pasco/models/encoder_v2.py` | 12 | Medium | 3 |
| P2 | `pasco/models/unet3d_sparse_v2.py` | 8 | Medium | 3 |
| P2 | `pasco/models/decoder_v3.py` | 21 | High | 3 |
| P3 | `pasco/models/transformer/transformer_predictor_v2.py` | 15 | **Critical** | 4 |
| P4 | `pasco/models/net_panoptic_sparse.py` | 5 | Low | 5 |
| P4 | `pasco/models/net_panoptic_sparse_kitti360.py` | 5 | Low | 5 |
| P4 | `pasco/data/semantic_kitti/collate.py` | 4 | Low | 5 |
| P4 | `pasco/loss/criterion_sparse.py` | 3 | Low | 5 |
| P4 | `pasco/models/helper.py` | 2 | Low | 5 |
| P4 | `pasco/models/misc.py` | 1 | Low | 5 |
| P4 | `pasco/models/augmenter.py` | 1 | Low | 5 |
| P4 | `pasco/loss/losses.py` | 0 | None | 5 |

---

## Risk Matrix (Prioritized)

| Risk | Location | Impact | Mitigation | Phase |
|------|----------|--------|------------|-------|
| `sparse_to_dense` OOB | transformer_predictor_v2.py:263-274 | **Critical** | Coord clamping, validate in Phase 1 | 1,4 |
| `generative=True` coord gen | mink.py (BasicGenerativeDeconvBlock) | High | Unit test: output.coords.shape[0] > input.coords.shape[0] | 2 |
| GlobalPool broadcast | ASPP, SELayer, TransformerInFeat | High | Manual batch_idx pattern | 2 |
| `tensor_stride` → `.stride` | transformer_predictor_v2.py:261,367 | Medium | Search-replace + verify | 4 |
| Stride propagation | All conv layers | Medium | Add stride assertions | 2-3 |
| SparseTensor addition | decoder_v3.py | Medium | Assert coord equality | 3 |

---

## Execution Timeline

```
Phase 1 (Foundation)     ████████░░░░░░░░░░░░░░░░░░░░░░
Phase 2 (Blocks)         ░░░░░░░░████████████░░░░░░░░░░  ← Parallel: mink.py | layers.py | dropout.py
Phase 3 (Enc-Dec)        ░░░░░░░░░░░░░░░░░░░░████████░░
Phase 4 (Transformer)    ░░░░░░░░░░░░░░░░░░░░░░░░░░████  ← Highest risk, needs Phase 1-3
Phase 5 (Integration)    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░██
```

---

## Verification Strategy

### Per-Phase Verification (Vertical Slices)
```python
# Phase 1: Utility tests
def test_sparse_to_dense_edge_cases():
    # Empty batch, boundary coords, negative coords

# Phase 2: Block tests
def test_selayer_preserves_coords():
    out = selayer(x)
    assert torch.equal(out.coords, x.coords)

# Phase 3: Pipeline test
def test_encoder_decoder_forward():
    outputs = model(x)
    assert len(outputs) == 4

# Phase 4: Attention mask test
def test_compute_attn_mask_shape():
    mask = predictor.compute_attn_mask(...)
    assert mask.shape == (B * nheads, num_queries, N)

# Phase 5: End-to-end
def test_training_step():
    loss = model.training_step(batch)
    assert torch.isfinite(loss)
```

### Rollback Points
Each phase creates a git tag. Rollback if:
- Tests fail after 4 hours debugging
- Performance regression >30%
- Memory increase >50%

---

## Acceptance Criteria

### Functional
- [ ] All unit tests pass
- [ ] SemanticKITTI single sample inference
- [ ] Single training epoch without errors
- [ ] Loss decreasing trend in first 100 steps

### Performance
- [ ] Training throughput ≥ 3x ME baseline
- [ ] Inference throughput ≥ 2x ME baseline
- [ ] Peak memory ≤ ME baseline

---

## Key Optimizations from v2

1. **Risk-first approach**: `sparse_to_dense` validated in Phase 1, not Phase 5
2. **Parallel execution**: Phase 2 files have no inter-dependencies
3. **Reduced phases**: 5 phases instead of 6 (merged Foundation + Risk Validation)
4. **Dependency-driven order**: Following actual import graph, not arbitrary layers
5. **Vertical slice testing**: Each phase has runnable validation, not just unit tests
6. **Accurate file count**: 16 files (removed duplicates from v2's 19)
7. **Explicit patterns**: Code patterns inline where needed for implementation clarity
