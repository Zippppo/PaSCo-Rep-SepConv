当前状态总结

  问题分析
  ┌────────────────────────┬─────────────────────────────────────────┬───────────────────────────────────────┐
  │          问题          │                当前状态                 │               目标状态                │
  ├────────────────────────┼─────────────────────────────────────────┼───────────────────────────────────────┤
  │ outside_body (class 0) │ 权重=0，但模型仍输出 37 类 (36+dustbin) │ 完全移除，模型输出 36 类 (35+dustbin) │
  ├────────────────────────┼─────────────────────────────────────────┼───────────────────────────────────────┤
  │ inside_body_empty      │ class 1, 权重=0.05                      │ class 1, 权重=0.1 (对应 KITTI empty)  │
  ├────────────────────────┼─────────────────────────────────────────┼───────────────────────────────────────┤
  │ organs (2-35)          │ 语义+实例分离                           │ 保持不变 ✓                            │
  ├────────────────────────┼─────────────────────────────────────────┼───────────────────────────────────────┤
  │ precomputed data       │ 基于 36 类                              │ 需要重新生成 (35 类)                  │
  └────────────────────────┴─────────────────────────────────────────┴───────────────────────────────────────┘
  类别映射变化

  旧 (36类)                    新 (35类)
  ─────────────────────────────────────────
  0: outside_body      →       移除 (ignore=255)
  1: inside_body_empty →       0: inside_body_empty
  2: skeleton          →       1: skeleton
  3: brain             →       2: brain
  ...
  35: rib              →       34: rib

  ---
  完整修改计划

  Phase 1: 数据层修改

  1.1 修改 pasco/data/body/label_mapping.py

  - 将 outside_body 映射到 255 (ignore label)
  - 所有其他类别 ID 减 1
  - 更新 remap_labels() 函数

  1.2 修改 pasco/data/body/params.py

  - n_classes: 36 → 35
  - class_names: 移除 "outside_body"
  - thing_ids: 所有 ID 减 1
  - single_instance_ids: 所有 ID 减 1
  - multi_instance_ids: 所有 ID 减 1
  - class_frequencies: 移除 class 0 的频率

  1.3 修改 pasco/data/body/body_dataset.py

  - _normalize_grid_size(): 填充值从 0 改为 255
  - _generate_multiscale_labels(): 更新空区域处理逻辑
  - _prepare_mask_label(): 更新 skip 逻辑 (原 class 0,1 → 新 255,0)

  1.4 重新运行预处理脚本

  python scripts/body/data/data_pre_process.py \
      --dataset_root Dataset/voxel_data \
      --output_root Dataset/voxel_data_precomputed_v2 \
      --split_file dataset_split.json

  ---
  Phase 2: 训练层修改

  2.1 修改 scripts/body/train.py

  # 旧
  body_n_classes = 36
  class_weight[0] = 0.0   # outside_body
  class_weight[1] = 0.05  # inside_body_empty

  # 新
  body_n_classes = 35
  class_weight[0] = 0.1   # inside_body_empty (对应 KITTI empty)
  class_weight[-1] = 0.1  # dustbin
  # 其他 organs 保持 1.0

  2.2 修改 compl_labelweights

  # 旧
  compl_labelweights[0] = 0.0  # outside_body
  compl_labelweights[1] = compl_labelweights[1] * 0.1

  # 新
  # class_frequencies 已经不包含 outside_body
  # inside_body_empty (新 class 0) 保持正常计算

  ---
  Phase 3: 模型层修改

  3.1 修改 pasco/models/decoder_v3.py

  # 旧: predict_completion_sem_logit()
  keep = sem_class != 0  # 移除 outside_body

  # 新: 
  keep = sem_class != 255  # 移除 ignore label (如果有的话)
  # 或者不需要修改，因为 class 0 现在是 inside_body_empty

  3.2 检查 pasco/models/helper.py

  - panoptic_inference(): 确认 labels.ne(0) 逻辑是否需要更新

  ---
  Phase 4: 评估层修改

  4.1 修改评估指标

  - 确保 IoU 计算忽略 255 (ignore label)
  - PQ 计算不包含移除的类别

  ---
  重要补充点

  补充 1: Instance 处理确认

  当前 All-Instance Strategy 已正确实现：
  - Single-instance organs (class 1-21 → 新 0-20): 每个语义类 = 1 个 instance
  - Multi-instance organs (class 22-34 → 新 21-33): 每个 (语义, instance_id) = 1 个 instance
  - 例：每根肋骨语义类相同 (rib)，但 instance_id 不同

  补充 2: 稀疏卷积行为

  - Decoder pruning 会移除预测为非器官的位置
  - 移除 class 0 后，pruning 条件需要更新为移除 inside_body_empty 或保留所有预测

  补充 3: 数据边界处理

  - _normalize_grid_size() 中 padding 填充值改为 255
  - 这确保超出原始数据范围的区域被忽略

  补充 4: Loss 计算

  - CrossEntropy 使用 ignore_index=255
  - 确保 255 不参与 loss 计算

  ---
  执行顺序

  1. label_mapping.py (核心映射逻辑)
         ↓
  2. params.py (参数更新)
         ↓
  3. body_dataset.py (数据加载适配)
         ↓
  4. 运行 data_pre_process.py (重新生成预处理数据)
         ↓
  5. train.py (训练参数更新)
         ↓
  6. decoder_v3.py / helper.py (模型逻辑检查)
         ↓
  7. 测试训练流程