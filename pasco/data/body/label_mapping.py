"""
Label mapping for body instance segmentation.

Maps 72 original classes to 36 semantic classes + instance IDs.
This enables panoptic segmentation by:
- Merging left/right paired organs into single semantic class with 2 instances
- Merging 24 individual ribs into single "rib" class with 24 instances

Instance ID format: simple sequential (1, 2, 3, ...)
- Left organs/structures get instance ID 1
- Right organs/structures get instance ID 2
- Ribs: left 1-12 get IDs 1-12, right 1-12 get IDs 13-24
"""
import numpy as np
from typing import Tuple

# Number of classes
N_CLASSES_OLD = 72
N_CLASSES_NEW = 36

# Thing class IDs (have instance segmentation)
# These are the new semantic IDs for classes that will have instances
THING_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

# New class names (36 classes)
CLASS_NAMES_NEW = [
    # Stuff classes (0-22): no instances
    "outside_body",         # 0
    "inside_body_empty",    # 1
    "liver",                # 2
    "spleen",               # 3
    "stomach",              # 4
    "pancreas",             # 5
    "gallbladder",          # 6
    "urinary_bladder",      # 7
    "prostate",             # 8
    "heart",                # 9
    "brain",                # 10
    "thyroid_gland",        # 11
    "spinal_cord",          # 12
    "lung",                 # 13
    "esophagus",            # 14
    "trachea",              # 15
    "small_bowel",          # 16
    "duodenum",             # 17
    "colon",                # 18
    "spine",                # 19
    "skull",                # 20
    "sternum",              # 21
    "costal_cartilages",    # 22
    # Thing classes (23-35): have instances
    "kidney",               # 23: 2 instances (L=1, R=2)
    "adrenal_gland",        # 24: 2 instances
    "rib",                  # 25: 24 instances (L1-12=1-12, R1-12=13-24)
    "scapula",              # 26: 2 instances
    "clavicula",            # 27: 2 instances
    "humerus",              # 28: 2 instances
    "hip",                  # 29: 2 instances
    "femur",                # 30: 2 instances
    "gluteus_maximus",      # 31: 2 instances
    "gluteus_medius",       # 32: 2 instances
    "gluteus_minimus",      # 33: 2 instances
    "autochthon",           # 34: 2 instances
    "iliopsoas",            # 35: 2 instances
]

# Mapping from old 72-class IDs to new semantic + instance IDs
# Format: old_id: (new_semantic_id, instance_id)
# instance_id = 0 means stuff class (no instance)
_LABEL_MAPPING = {
    # Stuff classes (direct mapping, instance_id = 0)
    0: (0, 0),    # outside_body
    1: (1, 0),    # inside_body_empty
    2: (2, 0),    # liver
    3: (3, 0),    # spleen
    6: (4, 0),    # stomach
    7: (5, 0),    # pancreas
    8: (6, 0),    # gallbladder
    9: (7, 0),    # urinary_bladder
    10: (8, 0),   # prostate
    11: (9, 0),   # heart
    12: (10, 0),  # brain
    13: (11, 0),  # thyroid_gland
    14: (12, 0),  # spinal_cord
    15: (13, 0),  # lung
    16: (14, 0),  # esophagus
    17: (15, 0),  # trachea
    18: (16, 0),  # small_bowel
    19: (17, 0),  # duodenum
    20: (18, 0),  # colon
    23: (19, 0),  # spine
    48: (20, 0),  # skull
    49: (21, 0),  # sternum
    50: (22, 0),  # costal_cartilages

    # Thing classes - paired organs (L=1, R=2)
    4: (23, 1),   # kidney_left -> kidney, instance 1
    5: (23, 2),   # kidney_right -> kidney, instance 2
    21: (24, 1),  # adrenal_gland_left -> adrenal_gland, instance 1
    22: (24, 2),  # adrenal_gland_right -> adrenal_gland, instance 2
    51: (26, 1),  # scapula_left -> scapula, instance 1
    52: (26, 2),  # scapula_right -> scapula, instance 2
    53: (27, 1),  # clavicula_left -> clavicula, instance 1
    54: (27, 2),  # clavicula_right -> clavicula, instance 2
    55: (28, 1),  # humerus_left -> humerus, instance 1
    56: (28, 2),  # humerus_right -> humerus, instance 2
    57: (29, 1),  # hip_left -> hip, instance 1
    58: (29, 2),  # hip_right -> hip, instance 2
    59: (30, 1),  # femur_left -> femur, instance 1
    60: (30, 2),  # femur_right -> femur, instance 2
    61: (31, 1),  # gluteus_maximus_left -> gluteus_maximus, instance 1
    62: (31, 2),  # gluteus_maximus_right -> gluteus_maximus, instance 2
    63: (32, 1),  # gluteus_medius_left -> gluteus_medius, instance 1
    64: (32, 2),  # gluteus_medius_right -> gluteus_medius, instance 2
    65: (33, 1),  # gluteus_minimus_left -> gluteus_minimus, instance 1
    66: (33, 2),  # gluteus_minimus_right -> gluteus_minimus, instance 2
    67: (34, 1),  # autochthon_left -> autochthon, instance 1
    68: (34, 2),  # autochthon_right -> autochthon, instance 2
    69: (35, 1),  # iliopsoas_left -> iliopsoas, instance 1
    70: (35, 2),  # iliopsoas_right -> iliopsoas, instance 2

    # Ribs: 24 instances (L1-12 -> 1-12, R1-12 -> 13-24)
    24: (25, 1),   # rib_left_1 -> rib, instance 1
    25: (25, 2),   # rib_left_2 -> rib, instance 2
    26: (25, 3),   # rib_left_3 -> rib, instance 3
    27: (25, 4),   # rib_left_4 -> rib, instance 4
    28: (25, 5),   # rib_left_5 -> rib, instance 5
    29: (25, 6),   # rib_left_6 -> rib, instance 6
    30: (25, 7),   # rib_left_7 -> rib, instance 7
    31: (25, 8),   # rib_left_8 -> rib, instance 8
    32: (25, 9),   # rib_left_9 -> rib, instance 9
    33: (25, 10),  # rib_left_10 -> rib, instance 10
    34: (25, 11),  # rib_left_11 -> rib, instance 11
    35: (25, 12),  # rib_left_12 -> rib, instance 12
    36: (25, 13),  # rib_right_1 -> rib, instance 13
    37: (25, 14),  # rib_right_2 -> rib, instance 14
    38: (25, 15),  # rib_right_3 -> rib, instance 15
    39: (25, 16),  # rib_right_4 -> rib, instance 16
    40: (25, 17),  # rib_right_5 -> rib, instance 17
    41: (25, 18),  # rib_right_6 -> rib, instance 18
    42: (25, 19),  # rib_right_7 -> rib, instance 19
    43: (25, 20),  # rib_right_8 -> rib, instance 20
    44: (25, 21),  # rib_right_9 -> rib, instance 21
    45: (25, 22),  # rib_right_10 -> rib, instance 22
    46: (25, 23),  # rib_right_11 -> rib, instance 23
    47: (25, 24),  # rib_right_12 -> rib, instance 24

    # Rectum (71) has no data in current dataset, map to 0 (outside_body)
    71: (0, 0),
}


def create_label_mapping() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lookup tables for label remapping.

    Returns:
        semantic_lut: np.ndarray[72] - maps old_id -> new_semantic_id
        instance_lut: np.ndarray[72] - maps old_id -> instance_id (0 for stuff)
    """
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
    Remap 72-class labels to 36-class semantic + instance labels.

    Args:
        voxel_labels_72: np.ndarray with values 0-71

    Returns:
        semantic_labels: np.ndarray with values 0-35
        instance_labels: np.ndarray with instance IDs (0 for stuff classes)
    """
    # Clip to valid range to avoid index errors
    voxel_labels_72 = np.clip(voxel_labels_72, 0, N_CLASSES_OLD - 1)

    semantic_labels = _SEMANTIC_LUT[voxel_labels_72]
    instance_labels = _INSTANCE_LUT[voxel_labels_72]

    return semantic_labels, instance_labels


def compute_new_class_frequencies(old_counts: np.ndarray) -> np.ndarray:
    """
    Compute class frequencies for the new 36-class scheme.

    Args:
        old_counts: np.ndarray[72] - voxel counts for original 72 classes

    Returns:
        new_counts: np.ndarray[36] - aggregated voxel counts for new 36 classes
    """
    new_counts = np.zeros(N_CLASSES_NEW, dtype=np.float64)

    for old_id in range(N_CLASSES_OLD):
        new_sem_id = _SEMANTIC_LUT[old_id]
        new_counts[new_sem_id] += old_counts[old_id]

    return new_counts


def get_instance_count_for_class(semantic_id: int) -> int:
    """
    Get the number of instances for a given semantic class.

    Args:
        semantic_id: New semantic class ID (0-35)

    Returns:
        Number of instances (0 for stuff, >0 for thing classes)
    """
    if semantic_id not in THING_IDS:
        return 0

    # Count unique instance IDs for this semantic class
    instance_ids = set()
    for old_id, (new_sem_id, inst_id) in _LABEL_MAPPING.items():
        if new_sem_id == semantic_id and inst_id > 0:
            instance_ids.add(inst_id)

    return len(instance_ids)


def validate_mapping():
    """Validate the label mapping is correct."""
    semantic_lut, instance_lut = create_label_mapping()

    # Check all old IDs are mapped
    assert len(_LABEL_MAPPING) == N_CLASSES_OLD, \
        f"Expected {N_CLASSES_OLD} mappings, got {len(_LABEL_MAPPING)}"

    # Check semantic IDs are in valid range
    assert semantic_lut.max() <= N_CLASSES_NEW - 1, \
        f"Max semantic ID {semantic_lut.max()} exceeds {N_CLASSES_NEW - 1}"

    # Check all 36 new classes are used (except possibly some)
    unique_sem_ids = np.unique(semantic_lut)
    print(f"Unique semantic IDs used: {len(unique_sem_ids)}")

    # Check thing classes have instances
    for thing_id in THING_IDS:
        inst_count = get_instance_count_for_class(thing_id)
        assert inst_count > 0, f"Thing class {thing_id} has no instances"
        print(f"Thing class {thing_id} ({CLASS_NAMES_NEW[thing_id]}): {inst_count} instances")

    # Check stuff classes have no instances
    for old_id, (new_sem_id, inst_id) in _LABEL_MAPPING.items():
        if new_sem_id not in THING_IDS:
            assert inst_id == 0, \
                f"Stuff class {new_sem_id} has non-zero instance ID {inst_id}"

    print("Label mapping validation passed!")
    return True


if __name__ == "__main__":
    validate_mapping()
