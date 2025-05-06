import copy
import numpy as np

PLANE_GEN_PARAMS_DEFAULT = {
    "anchor_intensity_threshold": 0.8,
    "align_intensity_threshold": 0.4,
    "z_threshold": 5,
    "max_tilt_deg": 30.0,
    "projection_dist_thresh":  0.5,
    "normalize_intensity" : True,
    "plane_boundaries" : [-np.inf, np.inf, -np.inf, np.inf],
    "margin" : 2, # distance between boundary point and pixel in img to be considered an edge roi
    "match_anchors" : True,
    "fixed_basis" : True,
    "regenerate_planes" : False, # always re-generate planes even if this z-stack has a set
    "max_alignments" : 500,
    "z_guess": -1, # guess at the z-level where the plane match is located in stack-> -1 means no guess
    "z_range": 1 # +- tolerance to search for in z in both planes
}

# Default parameters for matching two planes
MATCH_PLANE_PARAM_DEFAULTS = {
    "bin_match_params" : {
        "resolution" : 360,
        "angle_tolerance" : 2,
        "radians" : True,
        "min_matches" : 2,
        "outlier_thresh" : 2,
        "mse_threshold" : 1e-3
    },
    "angle_match_params" : {
        "blur_sigma" : 2,
        "resolution" : 360
    },
    "angle_mse_threshold" : 1.0,
    "magnitude_mse_threshold" : 1.0,
    "circular_fft" : True,
}

# Default parameters for matching two lists of planes
PLANE_LIST_PARAM_DEFAULTS = {
        "min_score" : 0.7,
        "traits": {
            "angle" : {
                "weight": 0.6,
                "max_value" : 5.0
            },
            "magnitude" : {
                "weight": 0.4,
                "max_value" : 10.0
            }
        }
    }

# Default parameters for matching 2 zstacks by a single plane in 2D
DEFAULT_2D_MATCH_PARAMS = {
    "plane_gen_params" : PLANE_GEN_PARAMS_DEFAULT,
    "stack_a_boundary" : None,
    "stack_b_boundary" : None,
    "plane_list_params" : PLANE_LIST_PARAM_DEFAULTS,
    "match_plane_params" : MATCH_PLANE_PARAM_DEFAULTS,
    "seg_params": {
        "method" : "split",
        "eps": 3.0,
        "min_samples" : 5
    },
    "min_uoi" : 0.5, # min uoi for a match between planes + transformations
    "plot_uoi" : False, # plot the matched rois and their intersection
    "plot_match" : False, # plot the matched rois only
    "z_guess_a": -1, # guess at the z-level where the plane match is located in stack a -> -1 means no guess
    "z_guess_b": -1 # guess at the z-level where the plane match is located in stack b -> -1 means no guess
}

# Used for updating a parameter dict to overwrite a subset of defaults
def deep_update(base, updates):
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)  # Recurse
        else:
            base[key] = value

# Establish a pattern for updating a default dictionary
def create_param_dict(default, updates = None):
    params = copy.deepcopy(default)
    if updates:
        deep_update(params, updates)

    return params
