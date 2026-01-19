"""
Label mapping for body instance segmentation.

Maps 72 original classes to 35 semantic classes + instance IDs.

NEW STRATEGY (All-Instance): Every organ is treated as an instance.
This fully utilizes Mask2Former's query mechanism where each query
predicts one organ's complete mask.

Instance ID format: simple sequential (1, 2, 3, ...)
- Single organs (liver, heart, etc.): instance_id = 1
- Paired organs (kidney L/R): instance_id = 1 (left), 2 (right)
- Ribs: left 1-12 get IDs 1-12, right 1-12 get IDs 13-24

Class mapping (35 classes):
- outside_body: mapped to 255 (IGNORE label, not in model output)
- Class 0 (inside_body_empty): Low weight background class
- Classes 1-34 (organs): All treated as instances
"""
import numpy as np
from typing import Tuple

# Number of classes
N_CLASSES_OLD = 72
N_CLASSES_NEW = 35  # Reduced from 36 (outside_body removed)

# Ignore label for outside_body
IGNORE_LABEL = 255

# Thing class IDs (have instance segmentation)
# All organ classes (1-34) are thing classes
# Class 0 (inside_body_empty) is background
THING_IDS = list(range(1, 35))  # [1, 2, 3, ..., 34]

# Classes that have multiple instances (paired organs and ribs)
# Old: [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
# New: all IDs reduced by 1
MULTI_INSTANCE_IDS = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

# Classes that have single instance (unpaired organs)
# Old: list(range(2, 23)) -> New: list(range(1, 22))
SINGLE_INSTANCE_IDS = list(range(1, 22))  # [1, 2, ..., 21]

# New class names (35 classes, outside_body removed)
CLASS_NAMES_NEW = [
    # Background class (0): low weight
    "inside_body_empty",    # 0: Low weight background (was class 1)
    # Single-instance organ classes (1-21): each is one instance
    "liver",                # 1: instance_id=1 (was class 2)
    "spleen",               # 2: instance_id=1 (was class 3)
    "stomach",              # 3: instance_id=1 (was class 4)
    "pancreas",             # 4: instance_id=1 (was class 5)
    "gallbladder",          # 5: instance_id=1 (was class 6)
    "urinary_bladder",      # 6: instance_id=1 (was class 7)
    "prostate",             # 7: instance_id=1 (was class 8)
    "heart",                # 8: instance_id=1 (was class 9)
    "brain",                # 9: instance_id=1 (was class 10)
    "thyroid_gland",        # 10: instance_id=1 (was class 11)
    "spinal_cord",          # 11: instance_id=1 (was class 12)
    "lung",                 # 12: instance_id=1 (was class 13)
    "esophagus",            # 13: instance_id=1 (was class 14)
    "trachea",              # 14: instance_id=1 (was class 15)
    "small_bowel",          # 15: instance_id=1 (was class 16)
    "duodenum",             # 16: instance_id=1 (was class 17)
    "colon",                # 17: instance_id=1 (was class 18)
    "spine",                # 18: instance_id=1 (was class 19)
    "skull",                # 19: instance_id=1 (was class 20)
    "sternum",              # 20: instance_id=1 (was class 21)
    "costal_cartilages",    # 21: instance_id=1 (was class 22)
    # Multi-instance organ classes (22-34): have 2+ instances
    "kidney",               # 22: 2 instances (was class 23)
    "adrenal_gland",        # 23: 2 instances (was class 24)
    "rib",                  # 24: 24 instances (was class 25)
    "scapula",              # 25: 2 instances (was class 26)
    "clavicula",            # 26: 2 instances (was class 27)
    "humerus",              # 27: 2 instances (was class 28)
    "hip",                  # 28: 2 instances (was class 29)
    "femur",                # 29: 2 instances (was class 30)
    "gluteus_maximus",      # 30: 2 instances (was class 31)
    "gluteus_medius",       # 31: 2 instances (was class 32)
    "gluteus_minimus",      # 32: 2 instances (was class 33)
    "autochthon",           # 33: 2 instances (was class 34)
    "iliopsoas",            # 34: 2 instances (was class 35)
]

# Mapping from old 72-class IDs to new semantic + instance IDs
# Format: old_id: (new_semantic_id, instance_id)
# Note: outside_body (old class 0) maps to 255 (IGNORE_LABEL)
# All other semantic IDs are reduced by 1 compared to old mapping
_LABEL_MAPPING = {
    # outside_body -> IGNORE (255), not part of model output
    0: (IGNORE_LABEL, 0),    # outside_body - IGNORED (mapped to 255)

    # inside_body_empty -> new class 0 (was class 1)
    1: (0, 0),    # inside_body_empty - low weight background

    # Single-instance organs (instance_id = 1)
    # All semantic IDs reduced by 1
    2: (1, 1),    # liver (was 2 -> now 1)
    3: (2, 1),    # spleen (was 3 -> now 2)
    6: (3, 1),    # stomach (was 4 -> now 3)
    7: (4, 1),    # pancreas (was 5 -> now 4)
    8: (5, 1),    # gallbladder (was 6 -> now 5)
    9: (6, 1),    # urinary_bladder (was 7 -> now 6)
    10: (7, 1),   # prostate (was 8 -> now 7)
    11: (8, 1),   # heart (was 9 -> now 8)
    12: (9, 1),   # brain (was 10 -> now 9)
    13: (10, 1),  # thyroid_gland (was 11 -> now 10)
    14: (11, 1),  # spinal_cord (was 12 -> now 11)
    15: (12, 1),  # lung (was 13 -> now 12)
    16: (13, 1),  # esophagus (was 14 -> now 13)
    17: (14, 1),  # trachea (was 15 -> now 14)
    18: (15, 1),  # small_bowel (was 16 -> now 15)
    19: (16, 1),  # duodenum (was 17 -> now 16)
    20: (17, 1),  # colon (was 18 -> now 17)
    23: (18, 1),  # spine (was 19 -> now 18)
    48: (19, 1),  # skull (was 20 -> now 19)
    49: (20, 1),  # sternum (was 21 -> now 20)
    50: (21, 1),  # costal_cartilages (was 22 -> now 21)

    # Multi-instance organs - paired organs (L=1, R=2)
    # All semantic IDs reduced by 1
    4: (22, 1),   # kidney_left -> kidney (was 23 -> now 22), instance 1
    5: (22, 2),   # kidney_right -> kidney, instance 2
    21: (23, 1),  # adrenal_gland_left -> adrenal_gland (was 24 -> now 23), instance 1
    22: (23, 2),  # adrenal_gland_right -> adrenal_gland, instance 2
    51: (25, 1),  # scapula_left -> scapula (was 26 -> now 25), instance 1
    52: (25, 2),  # scapula_right -> scapula, instance 2
    53: (26, 1),  # clavicula_left -> clavicula (was 27 -> now 26), instance 1
    54: (26, 2),  # clavicula_right -> clavicula, instance 2
    55: (27, 1),  # humerus_left -> humerus (was 28 -> now 27), instance 1
    56: (27, 2),  # humerus_right -> humerus, instance 2
    57: (28, 1),  # hip_left -> hip (was 29 -> now 28), instance 1
    58: (28, 2),  # hip_right -> hip, instance 2
    59: (29, 1),  # femur_left -> femur (was 30 -> now 29), instance 1
    60: (29, 2),  # femur_right -> femur, instance 2
    61: (30, 1),  # gluteus_maximus_left -> gluteus_maximus (was 31 -> now 30), instance 1
    62: (30, 2),  # gluteus_maximus_right -> gluteus_maximus, instance 2
    63: (31, 1),  # gluteus_medius_left -> gluteus_medius (was 32 -> now 31), instance 1
    64: (31, 2),  # gluteus_medius_right -> gluteus_medius, instance 2
    65: (32, 1),  # gluteus_minimus_left -> gluteus_minimus (was 33 -> now 32), instance 1
    66: (32, 2),  # gluteus_minimus_right -> gluteus_minimus, instance 2
    67: (33, 1),  # autochthon_left -> autochthon (was 34 -> now 33), instance 1
    68: (33, 2),  # autochthon_right -> autochthon, instance 2
    69: (34, 1),  # iliopsoas_left -> iliopsoas (was 35 -> now 34), instance 1
    70: (34, 2),  # iliopsoas_right -> iliopsoas, instance 2

    # Ribs: 24 instances (L1-12 -> 1-12, R1-12 -> 13-24)
    # rib semantic ID was 25 -> now 24
    24: (24, 1),   # rib_left_1 -> rib, instance 1
    25: (24, 2),   # rib_left_2 -> rib, instance 2
    26: (24, 3),   # rib_left_3 -> rib, instance 3
    27: (24, 4),   # rib_left_4 -> rib, instance 4
    28: (24, 5),   # rib_left_5 -> rib, instance 5
    29: (24, 6),   # rib_left_6 -> rib, instance 6
    30: (24, 7),   # rib_left_7 -> rib, instance 7
    31: (24, 8),   # rib_left_8 -> rib, instance 8
    32: (24, 9),   # rib_left_9 -> rib, instance 9
    33: (24, 10),  # rib_left_10 -> rib, instance 10
    34: (24, 11),  # rib_left_11 -> rib, instance 11
    35: (24, 12),  # rib_left_12 -> rib, instance 12
    36: (24, 13),  # rib_right_1 -> rib, instance 13
    37: (24, 14),  # rib_right_2 -> rib, instance 14
    38: (24, 15),  # rib_right_3 -> rib, instance 15
    39: (24, 16),  # rib_right_4 -> rib, instance 16
    40: (24, 17),  # rib_right_5 -> rib, instance 17
    41: (24, 18),  # rib_right_6 -> rib, instance 18
    42: (24, 19),  # rib_right_7 -> rib, instance 19
    43: (24, 20),  # rib_right_8 -> rib, instance 20
    44: (24, 21),  # rib_right_9 -> rib, instance 21
    45: (24, 22),  # rib_right_10 -> rib, instance 22
    46: (24, 23),  # rib_right_11 -> rib, instance 23
    47: (24, 24),  # rib_right_12 -> rib, instance 24

    # Rectum (71) has no data in current dataset, map to IGNORE
    71: (IGNORE_LABEL, 0),
}


def create_label_mapping() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lookup tables for label remapping.

    Returns:
        semantic_lut: np.ndarray[72] - maps old_id -> new_semantic_id (255 for ignored)
        instance_lut: np.ndarray[72] - maps old_id -> instance_id (0 for stuff)
    """
    # Use uint8 to accommodate 255 (IGNORE_LABEL)
    semantic_lut = np.zeros(N_CLASSES_OLD, dtype=np.uint8)
    instance_lut = np.zeros(N_CLASSES_OLD, dtype=np.uint8)

    for old_id, (new_sem_id, inst_id) in _LABEL_MAPPING.items():
        semantic_lut[old_id] = new_sem_id
        instance_lut[old_id] = inst_id

    return semantic_lut, instance_lut


# Pre-compute lookup tables for efficiency
_SEMANTIC_LUT, _INSTANCE_LUT = create_label_mapping()


def remap_labels(voxel_labels_72: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remap 72-class labels to 35-class semantic + instance labels.

    Args:
        voxel_labels_72: np.ndarray with values 0-71

    Returns:
        semantic_labels: np.ndarray with values 0-34 or 255 (IGNORE_LABEL)
        instance_labels: np.ndarray with instance IDs (0 for stuff classes)

    Note:
        - outside_body (old class 0) is mapped to 255 (IGNORE_LABEL)
        - inside_body_empty is now class 0
        - All organ classes are shifted by -1
    """
    # Clip to valid range to avoid index errors
    voxel_labels_72 = np.clip(voxel_labels_72, 0, N_CLASSES_OLD - 1)

    semantic_labels = _SEMANTIC_LUT[voxel_labels_72]
    instance_labels = _INSTANCE_LUT[voxel_labels_72]

    return semantic_labels, instance_labels


def compute_new_class_frequencies(old_counts: np.ndarray) -> np.ndarray:
    """
    Compute class frequencies for the new 35-class scheme.

    Args:
        old_counts: np.ndarray[72] - voxel counts for original 72 classes

    Returns:
        new_counts: np.ndarray[35] - aggregated voxel counts for new 35 classes
                    (outside_body counts are excluded)
    """
    new_counts = np.zeros(N_CLASSES_NEW, dtype=np.float64)

    for old_id in range(N_CLASSES_OLD):
        new_sem_id = _SEMANTIC_LUT[old_id]
        # Skip IGNORE_LABEL (255) - outside_body is not counted
        if new_sem_id != IGNORE_LABEL:
            new_counts[new_sem_id] += old_counts[old_id]

    return new_counts


def get_instance_count_for_class(semantic_id: int) -> int:
    """
    Get the number of instances for a given semantic class.

    Args:
        semantic_id: New semantic class ID (0-34)

    Returns:
        Number of instances (0 for background, 1+ for organ classes)
    """
    # Background classes have no instances
    if semantic_id not in THING_IDS:
        return 0

    # Single-instance organs
    if semantic_id in SINGLE_INSTANCE_IDS:
        return 1

    # Multi-instance organs: count unique instance IDs
    instance_ids = set()
    for old_id, (new_sem_id, inst_id) in _LABEL_MAPPING.items():
        if new_sem_id == semantic_id and inst_id > 0:
            instance_ids.add(inst_id)

    return len(instance_ids)


def get_total_instances_per_sample() -> int:
    """
    Calculate the total number of instances in a typical sample.

    Returns:
        Total instance count (used for setting num_queries)
    """
    total = 0
    for sem_id in THING_IDS:
        total += get_instance_count_for_class(sem_id)
    return total


def validate_mapping():
    """Validate the label mapping is correct."""
    semantic_lut, instance_lut = create_label_mapping()

    # Check all old IDs are mapped
    assert len(_LABEL_MAPPING) == N_CLASSES_OLD, \
        f"Expected {N_CLASSES_OLD} mappings, got {len(_LABEL_MAPPING)}"

    # Check semantic IDs are in valid range (0-34 or 255 for IGNORE)
    valid_sem_ids = semantic_lut[semantic_lut != IGNORE_LABEL]
    assert valid_sem_ids.max() <= N_CLASSES_NEW - 1, \
        f"Max semantic ID {valid_sem_ids.max()} exceeds {N_CLASSES_NEW - 1}"

    # Check all 35 new classes are used (except possibly some)
    unique_sem_ids = np.unique(valid_sem_ids)
    print(f"Unique semantic IDs used (excluding IGNORE): {len(unique_sem_ids)}")
    print(f"IGNORE_LABEL (255) count: {np.sum(semantic_lut == IGNORE_LABEL)}")

    # Check thing classes (all organs 1-34) have instances
    print("\n=== All-Instance Strategy (35 classes) ===")
    print(f"Total thing classes: {len(THING_IDS)}")
    print(f"Single-instance classes: {len(SINGLE_INSTANCE_IDS)}")
    print(f"Multi-instance classes: {len(MULTI_INSTANCE_IDS)}")

    total_instances = 0
    for thing_id in THING_IDS:
        inst_count = get_instance_count_for_class(thing_id)
        assert inst_count > 0, f"Thing class {thing_id} has no instances"
        total_instances += inst_count
        print(f"  Class {thing_id:2d} ({CLASS_NAMES_NEW[thing_id]:20s}): {inst_count} instance(s)")

    print(f"\nTotal instances per sample: {total_instances}")
    print(f"Recommended num_queries: {int(total_instances * 5)} (5x redundancy)")

    # Check background/ignore classes have no instances
    for old_id, (new_sem_id, inst_id) in _LABEL_MAPPING.items():
        if new_sem_id not in THING_IDS and new_sem_id != IGNORE_LABEL:
            # Only class 0 (inside_body_empty) is non-thing, non-ignore
            assert inst_id == 0, \
                f"Background class {new_sem_id} has non-zero instance ID {inst_id}"
        if new_sem_id == IGNORE_LABEL:
            assert inst_id == 0, \
                f"IGNORE class (old_id={old_id}) has non-zero instance ID {inst_id}"

    print("\nLabel mapping validation passed!")
    return True


if __name__ == "__main__":
    validate_mapping()
