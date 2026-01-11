  核心架构

  主模型类：Net (pasco/models/net_panoptic_sparse.py:1)
  Net (PyTorch Lightning模块)
  ├── CylinderFeat: 点云特征处理
  ├── UNet3DV2: 编码器-解码器主干
  ├── TransformerPredictor: 基于查询的Transformer（全景分割）
  ├── Ensembler: 多推理模式集成
  ├── HungarianMatcher: 匈牙利匹配
  └── SetCriterion: 全景分割损失函数

  主干网络：UNet3DV2

  编码器 (Encoder3DSepV2)
  Input (稀疏3D点云)
    ↓ enc_in_feats (1×1卷积，通道: in→64)
    ↓ s1 (保持尺度1)
    ↓ s1s2 (下采样stride=2，3个残差块) → scale 2
    ↓ s2s4 (下采样stride=2，3个残差块) → scale 4
    ↓ s4s8 (下采样stride=2，3个残差块) → scale 8
    ↓ 输出多尺度特征 [s1, s2, s4, s8]

  瓶颈层
  - SPCDense3Dv2: 密集3D卷积，使用多核路径(3×3×3, 5×5×5, 7×7×7)

  解码器 (DecoderGenerativeSepConvV2)
  Bottleneck (scale 8)
    ↓ DecoderBlock[0]: 上采样→scale 4
    │   └── 7个残差块 + 语义完成头
    ↓ DecoderBlock[1]: 上采样→scale 2
    │   └── 7个残差块 + 语义完成头
    ↓ DecoderBlock[2]: 上采样→scale 1
        └── 7个残差块 + 语义完成头
    ↓ 输出多尺度语义logits

  Transformer模块

  TransformerPredictorV2 (pasco/models/transformer/transformer_predictor_v2.py:1)
  ├── 可学习查询 (num_queries=100 × n_infers)
  ├── 3个Transformer层 (对应scale 4,2,1)
  │   ├── Self-Attention (8个注意力头)
  │   ├── Cross-Attention (查询×稀疏特征)
  │   └── FFN层
  └── 输出: 查询logits + 体素logits

  数据流

  点云(27维)
  → CylinderFeat(投影到体素网格)
  → Encoder(多尺度特征提取)
  → Dense Bottleneck(密集处理)
  → Decoder(渐进上采样，3个尺度)
  → Transformer(实例/全景预测)
  → Ensemble(多推理聚合)
  → 最终输出(全景分割+不确定性)

  关键特性

  - 稀疏张量: 使用MinkowskiEngine高效处理3D稀疏数据
  - 多推理模式(MIMO): 支持1-4个并行推理，用于不确定性估计
  - 多尺度处理: 在尺度1,2,4上进行语义完成
  - 场景尺寸: 256×256×32体素网格

  主要实现文件在 pasco/models/ 目录下，核心是基于稀疏UNet的编码器-解码器架构，结合Transformer进行实例分割。