"""
Label mapping for body instance segmentation.

Full 71-class scheme: Maps 72 original classes to 71 semantic classes.
Class 0 (outside_body) is removed and mapped to 255 (ignore).
All 71 remaining classes are treated as "thing" classes with instance_id=1.

This is a direct 1:1 mapping with no class merging:
- Original class 0 (outside_body) → 255 (ignore)
- Original class 1-71 → new class 0-70 (each with instance_id=1)

All classes are instance-aware (thing classes), enabling full panoptic segmentation.
"""
import numpy as np
from typing import Tuple

# Number of classes
N_CLASSES_OLD = 72
N_CLASSES_NEW = 71  # 71 classes (outside_body removed, no merging)

# Thing class IDs (have instance segmentation)
# All 71 classes are thing classes (each has instance_id=1)
THING_IDS = list(range(71))  # [0, 1, 2, ..., 70]

# New class names (71 classes, outside_body removed, direct mapping from datainfo.json)
CLASS_NAMES_NEW = [
    "inside_body_empty",      # 0 (was 1)
    "liver",                  # 1 (was 2)
    "spleen",                 # 2 (was 3)
    "kidney_left",            # 3 (was 4)
    "kidney_right",           # 4 (was 5)
    "stomach",                # 5 (was 6)
    "pancreas",               # 6 (was 7)
    "gallbladder",            # 7 (was 8)
    "urinary_bladder",        # 8 (was 9)
    "prostate",               # 9 (was 10)
    "heart",                  # 10 (was 11)
    "brain",                  # 11 (was 12)
    "thyroid_gland",          # 12 (was 13)
    "spinal_cord",            # 13 (was 14)
    "lung",                   # 14 (was 15)
    "esophagus",              # 15 (was 16)
    "trachea",                # 16 (was 17)
    "small_bowel",            # 17 (was 18)
    "duodenum",               # 18 (was 19)
    "colon",                  # 19 (was 20)
    "adrenal_gland_left",     # 20 (was 21)
    "adrenal_gland_right",    # 21 (was 22)
    "spine",                  # 22 (was 23)
    "rib_left_1",             # 23 (was 24)
    "rib_left_2",             # 24 (was 25)
    "rib_left_3",             # 25 (was 26)
    "rib_left_4",             # 26 (was 27)
    "rib_left_5",             # 27 (was 28)
    "rib_left_6",             # 28 (was 29)
    "rib_left_7",             # 29 (was 30)
    "rib_left_8",             # 30 (was 31)
    "rib_left_9",             # 31 (was 32)
    "rib_left_10",            # 32 (was 33)
    "rib_left_11",            # 33 (was 34)
    "rib_left_12",            # 34 (was 35)
    "rib_right_1",            # 35 (was 36)
    "rib_right_2",            # 36 (was 37)
    "rib_right_3",            # 37 (was 38)
    "rib_right_4",            # 38 (was 39)
    "rib_right_5",            # 39 (was 40)
    "rib_right_6",            # 40 (was 41)
    "rib_right_7",            # 41 (was 42)
    "rib_right_8",            # 42 (was 43)
    "rib_right_9",            # 43 (was 44)
    "rib_right_10",           # 44 (was 45)
    "rib_right_11",           # 45 (was 46)
    "rib_right_12",           # 46 (was 47)
    "skull",                  # 47 (was 48)
    "sternum",                # 48 (was 49)
    "costal_cartilages",      # 49 (was 50)
    "scapula_left",           # 50 (was 51)
    "scapula_right",          # 51 (was 52)
    "clavicula_left",         # 52 (was 53)
    "clavicula_right",        # 53 (was 54)
    "humerus_left",           # 54 (was 55)
    "humerus_right",          # 55 (was 56)
    "hip_left",               # 56 (was 57)
    "hip_right",              # 57 (was 58)
    "femur_left",             # 58 (was 59)
    "femur_right",            # 59 (was 60)
    "gluteus_maximus_left",   # 60 (was 61)
    "gluteus_maximus_right",  # 61 (was 62)
    "gluteus_medius_left",    # 62 (was 63)
    "gluteus_medius_right",   # 63 (was 64)
    "gluteus_minimus_left",   # 64 (was 65)
    "gluteus_minimus_right",  # 65 (was 66)
    "autochthon_left",        # 66 (was 67)
    "autochthon_right",       # 67 (was 68)
    "iliopsoas_left",         # 68 (was 69)
    "iliopsoas_right",        # 69 (was 70)
    "rectum",                 # 70 (was 71)
]

# Mapping from old 72-class IDs to new semantic + instance IDs
# Format: old_id: (new_semantic_id, instance_id)
# All classes have instance_id=1 (full instance segmentation)
# NOTE: Simple 1:1 mapping with offset -1 (except class 0 -> 255)
_LABEL_MAPPING = {
    0: (255, 0),   # outside_body -> 255 (ignore)
    1: (0, 1),     # inside_body_empty -> 0, instance 1
    2: (1, 1),     # liver -> 1, instance 1
    3: (2, 1),     # spleen -> 2, instance 1
    4: (3, 1),     # kidney_left -> 3, instance 1
    5: (4, 1),     # kidney_right -> 4, instance 1
    6: (5, 1),     # stomach -> 5, instance 1
    7: (6, 1),     # pancreas -> 6, instance 1
    8: (7, 1),     # gallbladder -> 7, instance 1
    9: (8, 1),     # urinary_bladder -> 8, instance 1
    10: (9, 1),    # prostate -> 9, instance 1
    11: (10, 1),   # heart -> 10, instance 1
    12: (11, 1),   # brain -> 11, instance 1
    13: (12, 1),   # thyroid_gland -> 12, instance 1
    14: (13, 1),   # spinal_cord -> 13, instance 1
    15: (14, 1),   # lung -> 14, instance 1
    16: (15, 1),   # esophagus -> 15, instance 1
    17: (16, 1),   # trachea -> 16, instance 1
    18: (17, 1),   # small_bowel -> 17, instance 1
    19: (18, 1),   # duodenum -> 18, instance 1
    20: (19, 1),   # colon -> 19, instance 1
    21: (20, 1),   # adrenal_gland_left -> 20, instance 1
    22: (21, 1),   # adrenal_gland_right -> 21, instance 1
    23: (22, 1),   # spine -> 22, instance 1
    24: (23, 1),   # rib_left_1 -> 23, instance 1
    25: (24, 1),   # rib_left_2 -> 24, instance 1
    26: (25, 1),   # rib_left_3 -> 25, instance 1
    27: (26, 1),   # rib_left_4 -> 26, instance 1
    28: (27, 1),   # rib_left_5 -> 27, instance 1
    29: (28, 1),   # rib_left_6 -> 28, instance 1
    30: (29, 1),   # rib_left_7 -> 29, instance 1
    31: (30, 1),   # rib_left_8 -> 30, instance 1
    32: (31, 1),   # rib_left_9 -> 31, instance 1
    33: (32, 1),   # rib_left_10 -> 32, instance 1
    34: (33, 1),   # rib_left_11 -> 33, instance 1
    35: (34, 1),   # rib_left_12 -> 34, instance 1
    36: (35, 1),   # rib_right_1 -> 35, instance 1
    37: (36, 1),   # rib_right_2 -> 36, instance 1
    38: (37, 1),   # rib_right_3 -> 37, instance 1
    39: (38, 1),   # rib_right_4 -> 38, instance 1
    40: (39, 1),   # rib_right_5 -> 39, instance 1
    41: (40, 1),   # rib_right_6 -> 40, instance 1
    42: (41, 1),   # rib_right_7 -> 41, instance 1
    43: (42, 1),   # rib_right_8 -> 42, instance 1
    44: (43, 1),   # rib_right_9 -> 43, instance 1
    45: (44, 1),   # rib_right_10 -> 44, instance 1
    46: (45, 1),   # rib_right_11 -> 45, instance 1
    47: (46, 1),   # rib_right_12 -> 46, instance 1
    48: (47, 1),   # skull -> 47, instance 1
    49: (48, 1),   # sternum -> 48, instance 1
    50: (49, 1),   # costal_cartilages -> 49, instance 1
    51: (50, 1),   # scapula_left -> 50, instance 1
    52: (51, 1),   # scapula_right -> 51, instance 1
    53: (52, 1),   # clavicula_left -> 52, instance 1
    54: (53, 1),   # clavicula_right -> 53, instance 1
    55: (54, 1),   # humerus_left -> 54, instance 1
    56: (55, 1),   # humerus_right -> 55, instance 1
    57: (56, 1),   # hip_left -> 56, instance 1
    58: (57, 1),   # hip_right -> 57, instance 1
    59: (58, 1),   # femur_left -> 58, instance 1
    60: (59, 1),   # femur_right -> 59, instance 1
    61: (60, 1),   # gluteus_maximus_left -> 60, instance 1
    62: (61, 1),   # gluteus_maximus_right -> 61, instance 1
    63: (62, 1),   # gluteus_medius_left -> 62, instance 1
    64: (63, 1),   # gluteus_medius_right -> 63, instance 1
    65: (64, 1),   # gluteus_minimus_left -> 64, instance 1
    66: (65, 1),   # gluteus_minimus_right -> 65, instance 1
    67: (66, 1),   # autochthon_left -> 66, instance 1
    68: (67, 1),   # autochthon_right -> 67, instance 1
    69: (68, 1),   # iliopsoas_left -> 68, instance 1
    70: (69, 1),   # iliopsoas_right -> 69, instance 1
    71: (70, 1),   # rectum -> 70, instance 1
}


def create_label_mapping() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lookup tables for label remapping.

    Returns:
        semantic_lut: np.ndarray[72] - maps old_id -> new_semantic_id (255 for ignored)
        instance_lut: np.ndarray[72] - maps old_id -> instance_id (1 for all valid classes)
    """
    # Use 255 as default for unmapped classes (will be ignored)
    semantic_lut = np.full(N_CLASSES_OLD, 255, dtype=np.uint8)
    instance_lut = np.zeros(N_CLASSES_OLD, dtype=np.uint8)

    for old_id, (new_sem_id, inst_id) in _LABEL_MAPPING.items():
        semantic_lut[old_id] = new_sem_id
        instance_lut[old_id] = inst_id

    return semantic_lut, instance_lut


# Pre-compute lookup tables for efficiency
_SEMANTIC_LUT, _INSTANCE_LUT = create_label_mapping()


def remap_labels(voxel_labels_72: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remap 72-class labels to 71-class semantic + instance labels.

    Args:
        voxel_labels_72: np.ndarray with values 0-71

    Returns:
        semantic_labels: np.ndarray with values 0-70 or 255 (ignore)
        instance_labels: np.ndarray with instance IDs (1 for all valid classes)
    """
    # Clip to valid range to avoid index errors
    voxel_labels_72 = np.clip(voxel_labels_72, 0, N_CLASSES_OLD - 1)

    semantic_labels = _SEMANTIC_LUT[voxel_labels_72]
    instance_labels = _INSTANCE_LUT[voxel_labels_72]

    return semantic_labels, instance_labels


def compute_new_class_frequencies(old_counts: np.ndarray) -> np.ndarray:
    """
    Compute class frequencies for the new 71-class scheme.

    Simple 1:1 mapping with offset -1:
    - old_counts[0] (outside_body) is excluded
    - old_counts[1:72] -> new_counts[0:71]

    Args:
        old_counts: np.ndarray[72] - voxel counts for original 72 classes

    Returns:
        new_counts: np.ndarray[71] - voxel counts for new 71 classes
                   (excludes outside_body/ignore class)
    """
    # Simple mapping: new class i = old class i+1
    new_counts = old_counts[1:N_CLASSES_OLD].copy().astype(np.float64)
    return new_counts


def get_instance_count_for_class(semantic_id: int) -> int:
    """
    Get the number of instances for a given semantic class.

    In the 71-class full instance scheme, every class has exactly 1 instance.

    Args:
        semantic_id: New semantic class ID (0-70)

    Returns:
        1 for all valid thing classes, 0 for invalid IDs
    """
    if semantic_id in THING_IDS:
        return 1
    return 0


def validate_mapping():
    """Validate the label mapping is correct."""
    semantic_lut, instance_lut = create_label_mapping()

    # Check all old IDs are mapped
    assert len(_LABEL_MAPPING) == N_CLASSES_OLD, \
        f"Expected {N_CLASSES_OLD} mappings, got {len(_LABEL_MAPPING)}"

    # Check semantic IDs are in valid range (0-70 or 255)
    valid_sem_ids = set(range(N_CLASSES_NEW)) | {255}
    for sem_id in semantic_lut:
        assert sem_id in valid_sem_ids, f"Invalid semantic ID {sem_id}"

    # Check valid class range
    unique_sem_ids = np.unique(semantic_lut)
    print(f"Unique semantic IDs used: {unique_sem_ids.tolist()}")
    print(f"Number of unique semantic classes (excluding 255): {len([s for s in unique_sem_ids if s != 255])}")

    # Check all thing classes have instances
    print(f"\nAll {len(THING_IDS)} classes are thing classes (instance_id=1):")
    for thing_id in THING_IDS[:5]:  # Show first 5
        inst_count = get_instance_count_for_class(thing_id)
        print(f"  Thing class {thing_id} ({CLASS_NAMES_NEW[thing_id]}): {inst_count} instance")
    print(f"  ... (and {len(THING_IDS) - 5} more)")

    # Verify class 0 (original outside_body) maps to 255
    assert semantic_lut[0] == 255, f"Original class 0 should map to 255, got {semantic_lut[0]}"

    # Verify class 1 (original inside_body_empty) maps to 0
    assert semantic_lut[1] == 0, f"Original class 1 should map to 0, got {semantic_lut[1]}"

    # Verify simple 1:1 mapping
    for old_id in range(1, N_CLASSES_OLD):
        expected_new_id = old_id - 1
        assert semantic_lut[old_id] == expected_new_id, \
            f"Old class {old_id} should map to {expected_new_id}, got {semantic_lut[old_id]}"

    print(f"\nLabel mapping validation passed!")
    print(f"  - {N_CLASSES_NEW} output classes (0-70)")
    print(f"  - Original class 0 (outside_body) -> 255 (ignore)")
    print(f"  - Original class 1-71 -> 0-70 (all with instance_id=1)")
    print(f"  - All {len(THING_IDS)} classes are thing classes")
    return True


if __name__ == "__main__":
    validate_mapping()
