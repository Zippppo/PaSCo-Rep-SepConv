## Project Overview
- **PaSCo**: A scene completion method for autonomous driving with key innovations: (i) Mask-centric Transformer architecture (ii) Semantic prediction-based cropping for better small-class performance (iii) MIMO strategy using shared-weight sub-networks for voxel-level and instance-level uncertainty estimation.
- **New Task(Pasco-Body)**: Human body scene completion. Input: partial body surface point cloud. Output: complete human body voxel representation with semantic labels.

## Testing Strategy (TDD)
- **Core utilities**: Strict TDD (coordinate transforms, IoU computation, etc.)
- **Data loading/preprocessing**: Test data shape, type, and value ranges
- **Model code**: Smoke tests (verify forward pass, output shapes)
- **Exploratory code**: Tests optional, focus on reproducibility
