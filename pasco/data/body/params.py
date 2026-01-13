import numpy as np

# Body task: no instance segmentation, only semantic
# All 72 classes are treated as "stuff" (background/semantic only)
thing_ids = []

# Class frequencies at different scales (voxel counts)
# Scale 1_1: original resolution
# Scale 1_2: 2x downsampling (volume / 8)
# Scale 1_4: 4x downsampling (volume / 64)
# Data derived from docs/dataset_statistics.json
_counts_1_1 = np.array([
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

# Avoid zero frequency for rectum class (add small epsilon)
_counts_1_1[_counts_1_1 == 0] = 1.0

class_frequencies = {
    "1_1": _counts_1_1,
    "1_2": _counts_1_1 / 8.0,    # 2x downsampling: volume / 2^3
    "1_4": _counts_1_1 / 64.0,   # 4x downsampling: volume / 4^3
}

# 72 class names from Dataset/datainfo.json
class_names = [
    "outside_body",         # 0
    "inside_body_empty",    # 1
    "liver",                # 2
    "spleen",               # 3
    "kidney_left",          # 4
    "kidney_right",         # 5
    "stomach",              # 6
    "pancreas",             # 7
    "gallbladder",          # 8
    "urinary_bladder",      # 9
    "prostate",             # 10
    "heart",                # 11
    "brain",                # 12
    "thyroid_gland",        # 13
    "spinal_cord",          # 14
    "lung",                 # 15
    "esophagus",            # 16
    "trachea",              # 17
    "small_bowel",          # 18
    "duodenum",             # 19
    "colon",                # 20
    "adrenal_gland_left",   # 21
    "adrenal_gland_right",  # 22
    "spine",                # 23
    "rib_left_1",           # 24
    "rib_left_2",           # 25
    "rib_left_3",           # 26
    "rib_left_4",           # 27
    "rib_left_5",           # 28
    "rib_left_6",           # 29
    "rib_left_7",           # 30
    "rib_left_8",           # 31
    "rib_left_9",           # 32
    "rib_left_10",          # 33
    "rib_left_11",          # 34
    "rib_left_12",          # 35
    "rib_right_1",          # 36
    "rib_right_2",          # 37
    "rib_right_3",          # 38
    "rib_right_4",          # 39
    "rib_right_5",          # 40
    "rib_right_6",          # 41
    "rib_right_7",          # 42
    "rib_right_8",          # 43
    "rib_right_9",          # 44
    "rib_right_10",         # 45
    "rib_right_11",         # 46
    "rib_right_12",         # 47
    "skull",                # 48
    "sternum",              # 49
    "costal_cartilages",    # 50
    "scapula_left",         # 51
    "scapula_right",        # 52
    "clavicula_left",       # 53
    "clavicula_right",      # 54
    "humerus_left",         # 55
    "humerus_right",        # 56
    "hip_left",             # 57
    "hip_right",            # 58
    "femur_left",           # 59
    "femur_right",          # 60
    "gluteus_maximus_left", # 61
    "gluteus_maximus_right",# 62
    "gluteus_medius_left",  # 63
    "gluteus_medius_right", # 64
    "gluteus_minimus_left", # 65
    "gluteus_minimus_right",# 66
    "autochthon_left",      # 67
    "autochthon_right",     # 68
    "iliopsoas_left",       # 69
    "iliopsoas_right",      # 70
    "rectum",               # 71
]

# Number of classes
n_classes = 72
