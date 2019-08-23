# Parameters for VOI finding


# Old phantom, T1 spheres
# T1 spheres broken up into 4 layers:
#
#              Anterior
#       ------------------------
#      / layer 4 with 3 spheres \
#     |  layer 3 with 3 spheres  |
#     |  layer 2 with 9 spheres  |
#     |  layer 1 with 9 spheres  |
#     ----------------------------
#              Posterior
#
# Layers 1 and 2 have 9 spheres numbered as so:
#
#            Superior
#
#               9
#           2        8
#
# Left   3      1      7    Right
#
#           4        6
#               5
#
#            Inferior
#
#
# Layer 3 has 3 spheres numbered as so:
#
#      Superior
#
#          2
# Left       3   Right
#        1
#
#      Inferior
#
# Layer 4 has 3 spheres numbered as so:
#
#      Superior
#
#            2
# Left   1       Right
#          3
#
#      Inferior
#
# Note: the above sphere labeling is based off UCSF conventions.

# TODO: update centers and add expected values and labels from manual
BREAST_131_SPHERES_T1 = {
    "config": {
        "definition_name": "BREAST_131_SPHERES_T1",
        "thermometry": False,
        # increasing kernel size to 5,5 better finds circles (with less false circles) but centers aren't as precise
        "gaussian_blur_kernel_size": (3,3),
        "clahe_limit": 1.0,
        "clahe_grid_size": (3,3),
        "dp": 1.3,
        # increase canny_threshold to detect more edges, leading to more circles being found
        "canny_threshold": 50,
        "min_distance_between_circle_centers": 1.5, #cm
        "min_radius": 0.5,
        "max_radius": 1.2,
         # a lower acc_threshold allows lower quality circles to be found
        "acc_threshold": 13,
        "radius_max_percentage": 0.5,
        "height_max_percentage": 0.8,
        "graph_match_threshold_factor": 1.5,
        "max_error": 50.0,
        "expected_total_number_vois": 24,
    },
    "vois": {
        # see numbering above for orientation, 1-9 layer 1, 10-18 layer 2, 19-21 layer 3, 22-24 layer 4
        # these centers are from the UCSF data
        1:  { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-9.7, 0.8, 0.4] },
        2:  { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-7.5, -1.4, 0.4] },
        3:  { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-6.6, 0.7, 0.4] },
        4:  { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-7.6, 2.8, 0.4] },
        5:  { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-9.6, 3.7, 0.4] },
        6:  { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-11.8, 2.9, 0.4] },
        7:  { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-12.7, 0.7, 0.4] },
        8:  { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-11.9, -1.4, 0.4] },
        9:  { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-9.7, -2.3, 0.4] },
        10: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-9.6, 0.7, -2.9] },
        11: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-7.6, -1.5, -2.9] },
        12: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-6.6, 0.7, -2.9] },
        13: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-7.4, 2.9, -2.9] },
        14: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-9.6, 3.8, -2.9] },
        15: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-11.8, 2.9, -2.9] },
        16: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-12.8, 0.7, -2.9] },
        17: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-11.9, -1.4, -2.9] },
        18: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-9.7, -2.3, -2.9] },
        19: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-8.8, -0.2, -5.8] },
        20: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-9.3, 2.0, -5.8] },
        21: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-10.9, 0.5, -5.8] },
        22: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-8.4, 1.0, -8.9] },
        23: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-10.6, 1.6, -8.9] },
        24: { "content_type": "Unknown", "expected_value": None, "shape": "sphere", "radius_cm": 1.0, "center_cm": [-10.0, -0.5, -8.9] },
    }
}

# Old phantom, ADC vials

#       Current numbering, update once we have the order used by UCSF (will that always be constant?)
#
#               7      8
#           6              9

#       5        1    2      10
#
#       16       3    4       11
#
#          15               12
#                 14   13
#

# TODO: update centers and add expected values and labels from manual
BREAST_131_SPHERES_ADC = {
    "config": {
        "definition_name": "BREAST_131_SPHERES_ADC",
        "thermometry": False,
        "gaussian_blur_kernel_size": (3,3),
        "clahe_limit": 1.0,
        "clahe_grid_size": (3,3),
        "dp": 1,
        "canny_threshold": 50,
        "min_distance_between_circle_centers": 1.5,
        "min_radius": 0.3,
        "max_radius": 0.8,
        "acc_threshold": 6,
        "radius_max_percentage": 0.4,
        "height_max_percentage": 0.8,
        "max_error": 30.0,
        "graph_match_threshold_factor": 1.5,
        "expected_total_number_vois": 16,
    },
    "vois": {
        # the centers were found manually and are rough
        1: { "content_type": "Unknown", "expected_value": None, "shape": "cylinder", "radius_cm": 0.75, "height_cm": 11.0, "center_cm": [-8.1, -1.8, -2.7] },
        2: { "content_type": "Unknown", "expected_value": None, "shape": "cylinder", "radius_cm": 0.75, "height_cm": 11.0, "center_cm": [-10.3, -1.8, -2.7] },
        3: { "content_type": "Unknown", "expected_value": None, "shape": "cylinder", "radius_cm": 0.75, "height_cm": 11.0, "center_cm": [-10.6, 0.6, -2.7] },
        4: { "content_type": "Unknown", "expected_value": None, "shape": "cylinder", "radius_cm": 0.75, "height_cm": 11.0, "center_cm": [-7.8, 0.8, -2.7] },
        5: { "content_type": "Unknown", "expected_value": None, "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-5.0, -1.8, -1.2] },
        6: { "content_type": "Unknown", "expected_value": None, "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-6.3, -3.6, -1.2] },
        7: { "content_type": "Unknown", "expected_value": None, "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-7.8, -4.8, -1.2] },
        8: { "content_type": "Unknown", "expected_value": None, "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-10.6, -4.8, -1.2] },
        9: { "content_type": "Unknown", "expected_value": None, "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-12.4, -3.8, -1.2] },
        10: {"content_type": "Unknown", "expected_value": None,  "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-13.3, -2.2, -1.2] },
        11: {"content_type": "Unknown", "expected_value": None,  "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-13.6, 0.8, -1.2] },
        12: {"content_type": "Unknown", "expected_value": None,  "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-12.2, 2.6, -1.2] },
        13: {"content_type": "Unknown", "expected_value": None,  "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-10.6, 3.6, -1.2] },
        14: {"content_type": "Unknown", "expected_value": None,  "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-8.0, 3.8, -1.2] },
        15: {"content_type": "Unknown", "expected_value": None,  "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-6.3, 2.8, -1.2] },
        16: {"content_type": "Unknown", "expected_value": None,  "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-5.0, 1.2, -1.2] },
    }
}

# New phantom, T1 vials
#   4 layers
#   Layer 1,2 9 vials
#   Layer 3,4 3 vials
#
#                S
#
#                4
#           5         3
#
# L      6      1       2     R
#
#          7         9
#               8
#
#                 I


#           S
#
#
#         2      1
#   L                  R
#
#             3
#
#           I


BREAST_131_CYLS_T1 = {
    "config": {
        "definition_name": "BREAST_131_CYLS_T1",
        "thermometry": True,
        # increasing kernel size to 5,5 better finds circles (with less false circles) but centers aren't as precise
        "gaussian_blur_kernel_size": (3,3),
        "clahe_limit": 1.0,
        "clahe_grid_size": (3,3),
        "dp": 1.3,
        # increase canny_threshold to detect more edges, leading to more circles being found
        "canny_threshold": 50,
        "min_distance_between_circle_centers": 1.5,
        "min_radius": 0.5,
        "max_radius": 1.2,
         # a lower acc_threshold allows lower quality circles to be found
        "acc_threshold": 13,
        "radius_max_percentage": 0.5,
        "height_max_percentage": 0.8,
        "graph_match_threshold_factor": 1.5,
        "max_error": 20.0,
        "expected_total_number_vois": 24,
    },
    "vois": {
        # see numbering above for orientation, 1-9 layer 1, 10-18 layer 2, 19-21 layer 3, 22-24 layer 4
        # z coordinates and center x,y are eyeballed in Slicer on case E7217 7, then vial centers are calculated with vial_centers.py
        1: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [9.6, 0.3, 1.0], "content_type": "fat", "expected_value": 335.9 },
        2: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [6.6, 0.3, 1.0], "content_type": "FBG", "expected_value": 1587.8 },
        3: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [7.5, 2.5, 1.0], "content_type": "fat", "expected_value": 335.9 },
        4: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [9.6, 3.3, 1.0], "content_type": "fat", "expected_value": 335.9 },
        5: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [11.7, 2.5, 1.0], "content_type": "FBG", "expected_value": 1587.8 },
        6: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [12.6, 0.3, 1.0], "content_type": "fat", "expected_value": 335.9 },
        7: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [11.7, -1.8, 1.0], "content_type": "FBG", "expected_value": 1587.8 },
        8: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [9.6, -2.7, 1.0], "content_type": "fat", "expected_value": 335.9 },
        9: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [7.5, -1.8, 1.0], "content_type": "fat", "expected_value": 335.9 },
        10: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [9.6, 0.3, -1.8], "content_type": "PVP25", "expected_value": None },
        11: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [6.6, 0.3, -1.8], "content_type": "FBG", "expected_value": 1587.8 },
        12: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [7.5, 2.5, -1.8], "content_type": "FBG", "expected_value": 1587.8 },
        13: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [9.6, 3.3, -1.8], "content_type": "PVP25", "expected_value": None },
        14: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [11.7, 2.5, -1.8], "content_type": "FBG", "expected_value": 1587.8 },
        15: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [12.6, 0.3, -1.8], "content_type": "FBG", "expected_value": 1587.8 },
        16: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [11.7, -1.8, -1.8], "content_type": "PVP25", "expected_value": None },
        17: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [9.6, -2.7, -1.8], "content_type": "FBG", "expected_value": 1587.8 },
        18: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [7.5, -1.8, -1.8], "content_type": "PVP25", "expected_value": None },
        19: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [8.5, 1.0, -5.8], "content_type": "fat", "expected_value": 335.9 },
        20: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [10.7, 1.0, -5.8], "content_type": "fat", "expected_value": 335.9 },
        21: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [9.6, -1.0, -5.8], "content_type": "fat", "expected_value": 335.9 },
        22: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [8.5, 1.0, -8.2], "content_type": "FBG", "expected_value": 1587.8 },
        23: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [10.7, 1.0, -8.2], "content_type": "FBG", "expected_value": 1587.8 },
        24: { "shape": "cylinder", "radius_cm": 1.0, "height_cm": 2.0, "center_cm": [9.6, -1.0, -8.2], "content_type": "fat", "expected_value": 335.9 },
    }
}

# New phantom, ADC vials
#   inner ring 4 vials (15mm x 110mm)
#   outer ring 12 vials (15mm x 80mm)
#                   S
#
#               8      7
#           9              6

#       10        2    1      5
# L                                 R
#       11       3    4       16
#
#          12               15
#                 13   14
#
#                   I

BREAST_131_CYLS_ADC = {
    "config": {
        "definition_name": "BREAST_131_CYLS_ADC",
        "thermometry": False,
        "gaussian_blur_kernel_size": (3,3),
        "clahe_limit": 1.0,
        "clahe_grid_size": (3,3),
        "dp": 1,
        "canny_threshold": 50,
        "min_distance_between_circle_centers": 1.5,
        "min_radius": 0.3,
        "max_radius": 0.8,
        "acc_threshold": 6,
        "radius_max_percentage": 0.4,
        "height_max_percentage": 0.8,
        "max_error": 30.0,
        "graph_match_threshold_factor": 1.5,
        "expected_total_number_vois": 16,
    },
    "vois": {
        # z coordinates and center x,y are eyeballed in Slicer on case E7217 10, then vial centers are calculated with vial_centers.py
        1: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 11.0, "center_cm": [-10.7, 2.1, -3.5], "content_type": "PVP40", "expected_value": 676 },
        2: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 11.0, "center_cm": [-7.6, 2.1, -3.5], "content_type": "PVP10", "expected_value": 1657 },
        3: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 11.0, "center_cm": [-7.6, -1.0, -3.5], "content_type": "PVP25", "expected_value": 1084 },
        4: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 11.0, "center_cm": [-10.7, -1.0, -3.5], "content_type": "PVP10", "expected_value": 1657 },
        5: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-13.4, 1.7, -2.0], "content_type": "PVP25", "expected_value": 1084 },
        6: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-12.3, 3.7, -2.0], "content_type": "fat", "expected_value": None },
        7: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-10.3, 4.8, -2.0], "content_type": "PVP10", "expected_value": 1657 },
        8: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-8.0, 4.8, -2.0], "content_type": "water", "expected_value": 2025 },
        9: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-6.0, 3.7, -2.0], "content_type": "fat", "expected_value": None },
        10: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-4.9, 1.7, -2.0], "content_type": "PVP40", "expected_value": 676 },
        11: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-4.9, -0.6, -2.0], "content_type": "PVP10", "expected_value": 1657 },
        12: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-6.0, -2.6, -2.0], "content_type":"fat", "expected_value": None },
        13: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-8.0, -3.7, -2.0], "content_type": "PVP40", "expected_value": 676 },
        14: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-10.3, -3.7, -2.0], "content_type": "fat", "expected_value": None },
        15: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-12.3, -2.6, -2.0], "content_type": "PVP18", "expected_value": 1339 },
        16: { "shape": "cylinder", "radius_cm": 0.75, "height_cm": 8.0, "center_cm": [-13.4, -0.6, -2.0], "content_type": "PVP14", "expected_value": 1503 },
    }
}

NEW_BREAST_PHANTOM_NAME = 'Model 131 Breast Phantom'
OLD_BREAST_PHANTOM_NAME = 'Model 131 Breast Phantom (Spheres)'

PHANTOM_CATALOG = {
    OLD_BREAST_PHANTOM_NAME: {
        "ADC": BREAST_131_SPHERES_ADC,
        "T1": BREAST_131_SPHERES_T1
    },
    NEW_BREAST_PHANTOM_NAME: {
        "ADC": BREAST_131_CYLS_ADC,
        "T1": BREAST_131_CYLS_T1
    }
}
