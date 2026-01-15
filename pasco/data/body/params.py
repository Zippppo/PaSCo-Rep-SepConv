"""
Body task parameters for panoptic segmentation.

This module defines:
- 36 semantic classes (merged from original 72)
- 14 thing classes with instance segmentation
- Class frequencies for loss weighting
"""
import numpy as np

from .label_mapping import (
    N_CLASSES_NEW,
    THING_IDS,
    CLASS_NAMES_NEW,
    compute_new_class_frequencies,
)

# Thing class IDs (have instance segmentation)
# kidney, adrenal_gland, rib, scapula, clavicula, humerus, hip, femur,
# gluteus_maximus, gluteus_medius, gluteus_minimus, autochthon, iliopsoas
thing_ids = THING_IDS

# Number of classes
n_classes = N_CLASSES_NEW  # 36

# Class names
class_names = CLASS_NAMES_NEW

# Original 72-class counts (from dataset_statistics.json)
# Used to compute aggregated frequencies for new 36-class scheme
_counts_72_classes = np.array([
    1835148880,  # 0: outside_body
    1305194053,  # 1: inside_body_empty
    103424463,   # 2: liver
    15898014,    # 3: spleen
    10398763,    # 4: kidney_left
    10230115,    # 5: kidney_right
    21630858,    # 6: stomach
    4658872,     # 7: pancreas
    1498865,     # 8: gallbladder
    6206109,     # 9: urinary_bladder
    581862,      # 10: prostate
    22878556,    # 11: heart
    7404653,     # 12: brain
    215713,      # 13: thyroid_gland
    3974045,     # 14: spinal_cord
    118037994,   # 15: lung
    1145998,     # 16: esophagus
    499939,      # 17: trachea
    52689766,    # 18: small_bowel
    3548894,     # 19: duodenum
    53168008,    # 20: colon
    281643,      # 21: adrenal_gland_left
    235155,      # 22: adrenal_gland_right
    40370405,    # 23: spine
    137679,      # 24: rib_left_1
    171008,      # 25: rib_left_2
    217854,      # 26: rib_left_3
    293088,      # 27: rib_left_4
    421241,      # 28: rib_left_5
    656967,      # 29: rib_left_6
    887295,      # 30: rib_left_7
    931431,      # 31: rib_left_8
    1022843,     # 32: rib_left_9
    920110,      # 33: rib_left_10
    613550,      # 34: rib_left_11
    253999,      # 35: rib_left_12
    141291,      # 36: rib_right_1
    173831,      # 37: rib_right_2
    219897,      # 38: rib_right_3
    305529,      # 39: rib_right_4
    437490,      # 40: rib_right_5
    666370,      # 41: rib_right_6
    906619,      # 42: rib_right_7
    971404,      # 43: rib_right_8
    1032106,     # 44: rib_right_9
    916029,      # 45: rib_right_10
    598933,      # 46: rib_right_11
    237644,      # 47: rib_right_12
    4698411,     # 48: skull
    1267918,     # 49: sternum
    6773519,     # 50: costal_cartilages
    1688279,     # 51: scapula_left
    1663694,     # 52: scapula_right
    399816,      # 53: clavicula_left
    415169,      # 54: clavicula_right
    1443002,     # 55: humerus_left
    1479224,     # 56: humerus_right
    14086197,    # 57: hip_left
    14028382,    # 58: hip_right
    6587179,     # 59: femur_left
    6593151,     # 60: femur_right
    19427736,    # 61: gluteus_maximus_left
    20124925,    # 62: gluteus_maximus_right
    9583218,     # 63: gluteus_medius_left
    9346327,     # 64: gluteus_medius_right
    2135654,     # 65: gluteus_minimus_left
    2329493,     # 66: gluteus_minimus_right
    25831270,    # 67: autochthon_left
    25352202,    # 68: autochthon_right
    12660719,    # 69: iliopsoas_left
    12221147,    # 70: iliopsoas_right
    0,           # 71: rectum (no samples in current dataset)
], dtype=np.float64)

# Compute aggregated class frequencies for 36-class scheme
_counts_1_1 = compute_new_class_frequencies(_counts_72_classes)

# Avoid zero frequency (add small epsilon)
_counts_1_1[_counts_1_1 == 0] = 1.0

# Class frequencies at different scales
class_frequencies = {
    "1_1": _counts_1_1,
    "1_2": _counts_1_1 / 8.0,    # 2x downsampling: volume / 2^3
    "1_4": _counts_1_1 / 64.0,   # 4x downsampling: volume / 4^3
}


def print_class_info():
    """Print class information for debugging."""
    print(f"Number of classes: {n_classes}")
    print(f"Thing class IDs: {thing_ids}")
    print(f"\nClass frequencies (36 classes):")
    for i, (name, count) in enumerate(zip(class_names, _counts_1_1)):
        thing_marker = " [THING]" if i in thing_ids else ""
        print(f"  {i:2d}: {name:25s} {count:15.0f}{thing_marker}")


if __name__ == "__main__":
    print_class_info()
