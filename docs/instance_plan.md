     Body任务实例分割实现计划

     概述

     将72类body标签重新映射为36类语义标签 + 实例ID，添加panoptic分割支持。

     目标

     1. 利用Transformer queries: 14个thing类，最多50个实例/样本
     2. 缓解类别不平衡: 合并24根肋骨为单一"rib"类
     3. 减少分类难度: 72类→36类

     ---
     新类别体系

     Stuff类 (22个，无实例)
     ┌──────┬───────────────────┬──────┐
     │ 新ID │       类名        │ 原ID │
     ├──────┼───────────────────┼──────┤
     │ 0    │ outside_body      │ 0    │
     ├──────┼───────────────────┼──────┤
     │ 1    │ inside_body_empty │ 1    │
     ├──────┼───────────────────┼──────┤
     │ 2    │ liver             │ 2    │
     ├──────┼───────────────────┼──────┤
     │ 3    │ spleen            │ 3    │
     ├──────┼───────────────────┼──────┤
     │ 4    │ stomach           │ 6    │
     ├──────┼───────────────────┼──────┤
     │ 5    │ pancreas          │ 7    │
     ├──────┼───────────────────┼──────┤
     │ 6    │ gallbladder       │ 8    │
     ├──────┼───────────────────┼──────┤
     │ 7    │ urinary_bladder   │ 9    │
     ├──────┼───────────────────┼──────┤
     │ 8    │ prostate          │ 10   │
     ├──────┼───────────────────┼──────┤
     │ 9    │ heart             │ 11   │
     ├──────┼───────────────────┼──────┤
     │ 10   │ brain             │ 12   │
     ├──────┼───────────────────┼──────┤
     │ 11   │ thyroid_gland     │ 13   │
     ├──────┼───────────────────┼──────┤
     │ 12   │ spinal_cord       │ 14   │
     ├──────┼───────────────────┼──────┤
     │ 13   │ lung              │ 15   │
     ├──────┼───────────────────┼──────┤
     │ 14   │ esophagus         │ 16   │
     ├──────┼───────────────────┼──────┤
     │ 15   │ trachea           │ 17   │
     ├──────┼───────────────────┼──────┤
     │ 16   │ small_bowel       │ 18   │
     ├──────┼───────────────────┼──────┤
     │ 17   │ duodenum          │ 19   │
     ├──────┼───────────────────┼──────┤
     │ 18   │ colon             │ 20   │
     ├──────┼───────────────────┼──────┤
     │ 19   │ spine             │ 23   │
     ├──────┼───────────────────┼──────┤
     │ 20   │ skull             │ 48   │
     ├──────┼───────────────────┼──────┤
     │ 21   │ sternum           │ 49   │
     ├──────┼───────────────────┼──────┤
     │ 22   │ costal_cartilages │ 50   │
     └──────┴───────────────────┴──────┘
     Thing类 (14个，有实例)
     ┌──────┬─────────────────┬───────┬──────────────────────────────┐
     │ 新ID │      类名       │ 原ID  │            实例数            │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 23   │ kidney          │ 4,5   │ 2 (L=1, R=2)                 │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 24   │ adrenal_gland   │ 21,22 │ 2                            │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 25   │ rib             │ 24-47 │ 24 (L1-12→1-12, R1-12→13-24) │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 26   │ scapula         │ 51,52 │ 2                            │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 27   │ clavicula       │ 53,54 │ 2                            │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 28   │ humerus         │ 55,56 │ 2                            │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 29   │ hip             │ 57,58 │ 2                            │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 30   │ femur           │ 59,60 │ 2                            │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 31   │ gluteus_maximus │ 61,62 │ 2                            │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 32   │ gluteus_medius  │ 63,64 │ 2                            │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 33   │ gluteus_minimus │ 65,66 │ 2                            │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 34   │ autochthon      │ 67,68 │ 2                            │
     ├──────┼─────────────────┼───────┼──────────────────────────────┤
     │ 35   │ iliopsoas       │ 69,70 │ 2                            │
     └──────┴─────────────────┴───────┴──────────────────────────────┘
     注意: rectum (原ID 71) 无数据，不包含在映射中。

     实例ID格式

     - Panoptic格式: semantic_id * 1000 + instance_id
     - 例如: kidney左=23001, kidney右=23002, rib左1=25001, rib右12=25024

     ---
     实现步骤

     Step 1: 创建标签映射模块

     文件: pasco/data/body/label_mapping.py (新建)

     """72类→36类映射 + 实例ID生成"""
     import numpy as np

     N_CLASSES_NEW = 36
     THING_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

     CLASS_NAMES_NEW = [
         "outside_body", "inside_body_empty", "liver", "spleen", "stomach",
         "pancreas", "gallbladder", "urinary_bladder", "prostate", "heart",
         "brain", "thyroid_gland", "spinal_cord", "lung", "esophagus",
         "trachea", "small_bowel", "duodenum", "colon", "spine",
         "skull", "sternum", "costal_cartilages",
         # Thing classes
         "kidney", "adrenal_gland", "rib", "scapula", "clavicula",
         "humerus", "hip", "femur", "gluteus_maximus", "gluteus_medius",
         "gluteus_minimus", "autochthon", "iliopsoas",
     ]

     def create_label_mapping():
         """创建72→36的查找表"""
         semantic_lut = np.zeros(72, dtype=np.uint8)
         instance_lut = np.zeros(72, dtype=np.uint8)
         # ... 实现映射逻辑 ...
         return semantic_lut, instance_lut

     def remap_labels(voxel_labels_72):
         """转换标签"""
         semantic_lut, instance_lut = create_label_mapping()
         return semantic_lut[voxel_labels_72], instance_lut[voxel_labels_72]

     Step 2: 更新params.py

     文件: pasco/data/body/params.py (修改)

     - 导入新的映射配置
     - 设置 thing_ids = THING_IDS
     - 设置 n_classes = 36
     - 更新 class_names = CLASS_NAMES_NEW
     - 重新计算类别频率

     Step 3: 更新数据预处理脚本

     文件: scripts/body/data/data_pre_process.py (修改)

     - 使用 remap_labels() 转换标签
     - 在NPZ中保存 instance_label 字段
     - 使用 n_classes=36 生成多尺度标签

     Step 4: 更新Dataset

     文件: pasco/data/body/body_dataset.py (修改)

     主要修改:
     1. 添加 prepare_instance_target() 方法
     2. 修改 _prepare_mask_label() 支持thing/stuff分离
     3. 加载instance_label从预计算数据

     def _prepare_mask_label(self, semantic_label, instance_label):
         # 1. 为stuff类创建语义mask
         stuff_masks = self.prepare_target(semantic_label, ignore_labels=[0])
         stuff_filtered = [(l, m) for l, m in zip(...) if l not in self.thing_ids]

         # 2. 为thing类创建实例mask
         thing_masks = self.prepare_instance_target(semantic_label, instance_label)

         # 3. 合并返回
         return {"labels": [...], "masks": [...]}

     Step 5: 重新预计算数据

     # 删除旧的预计算数据
     rm -rf Dataset/voxel_data_precomputed/*

     # 重新运行预处理
     python scripts/body/data/data_pre_process.py \
         --dataset_root Dataset/voxel_data \
         --output_root Dataset/voxel_data_precomputed \
         --num_workers 8

     ---
     关键文件清单
     ┌───────────────────────────────────────┬──────┬──────────────────────────┐
     │                 文件                  │ 操作 │           说明           │
     ├───────────────────────────────────────┼──────┼──────────────────────────┤
     │ pasco/data/body/label_mapping.py      │ 新建 │ 标签映射逻辑             │
     ├───────────────────────────────────────┼──────┼──────────────────────────┤
     │ pasco/data/body/params.py             │ 修改 │ 配置thing_ids, n_classes │
     ├───────────────────────────────────────┼──────┼──────────────────────────┤
     │ pasco/data/body/body_dataset.py       │ 修改 │ 实例mask生成             │
     ├───────────────────────────────────────┼──────┼──────────────────────────┤
     │ scripts/body/data/data_pre_process.py │ 修改 │ 预计算添加instance_label │
     └───────────────────────────────────────┴──────┴──────────────────────────┘
     ---
     验证方法

     1. 单元测试: 验证label_mapping正确性
     # 测试所有72个类都有映射
     semantic_lut, instance_lut = create_label_mapping()
     assert semantic_lut.max() == 35
     assert len(np.unique(semantic_lut)) == 36
     2. 数据检查: 验证预计算数据
     data = np.load("sample.npz")
     assert "instance_label" in data
     assert data["semantic_label"].max() <= 35
     3. 训练测试: 运行短训练验证
     python scripts/body/train.py --check --use_precomputed
     # 应显示 n_classes: 36, thing_ids: [23, 24, ...]
     4. 指标验证: PQ指标不再全为0
       - Things PQ/SQ/RQ 应该有非零值
       - 实例分割开始工作

     ---
     预期效果
     ┌─────────────────┬─────────────────────┬─────────────────────┐
     │      指标       │ 当前 (72类, 无实例) │ 预期 (36类, 有实例) │
     ├─────────────────┼─────────────────────┼─────────────────────┤
     │ 语义类数        │ 72                  │ 36                  │
     ├─────────────────┼─────────────────────┼─────────────────────┤
     │ Thing类数       │ 0                   │ 14                  │
     ├─────────────────┼─────────────────────┼─────────────────────┤
     │ 最大实例数/样本 │ 0                   │ 50                  │
     ├─────────────────┼─────────────────────┼─────────────────────┤
     │ Query利用率     │ 0%                  │ ~50%                │
     ├─────────────────┼─────────────────────┼─────────────────────┤
     │ Things PQ       │ 0.0                 │ >0                  │
     ├─────────────────┼─────────────────────┼─────────────────────┤
     │ mIoU            │ ~7%                 │ 预期提升            │
     └─────────────────┴─────────────────────┴─────────────────────┘