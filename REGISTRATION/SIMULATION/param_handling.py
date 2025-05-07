import copy
import numpy as np

PLANE_GEN_PARAMS_DEFAULT = {
    "read_filename" : None, # file to read plane parameters from
    "save_filename" : None, # file to save plane parameters to 
    "anchor_intensity_threshold": 0.8, # threshold to compare transformed intensity to for a point to be considered an anchor pt (see transform_intensity)
    "align_intensity_threshold": 0.4, # as above, but for alignment points
    "z_threshold": 5, # max distance in z between two points in a plane
    "max_tilt_deg": 30.0, # max magnitude of tilt in the vector between two points in a plane
    "projection_dist_thresh":  0.5, # euclidean distance for a point considered close enough to a plane for it to be projected
    "transform_intensity" : "quantile", # "quantile", "raw", "minmax" - transforms avg intensity per roi and compares to anchor/align intensity threshold
    "plane_boundaries" : [-np.inf, np.inf, -np.inf, np.inf],
    "margin" : 2, # distance between boundary point and pixel in img to be considered an edge roi
    "match_anchors" : True, # require two planes to have a common anchor to be equivalent (turning this off will reduce the number of planes and accuracy)
    "fixed_basis" : True, # define the projected 2d x-axis as [1,0,0]
    "regenerate_planes" : False, # always re-generate planes even if this z-stack has a set
    "max_alignments" : 500, # maximum number of alignment points allowed per plane
    "z_guess": -1, # guess at the z-level where the plane match is located in stack-> -1 means no guess
    "z_range": 0 # +- tolerance to search for in z in both planes
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
    "circular_fft" : True,
    "traits": {
        "angle" : {
            "weight": 0.6,
            "max_value" : 5.0,
            "metric" : "mse",
            "terminate_after": np.inf,
        },
        "magnitude" : {
            "weight": 0.4,
            "max_value" : 10.0,
            "metric" : "mse",
            "terminate_after": np.inf,
        }
    }
}

# Default parameters for matching two lists of planes
PLANE_LIST_PARAM_DEFAULTS = {
        "min_score" : 0.7,
        "max_matches" : 2, # max matches to scale score between
        "min_score_modifier" : 0.8, # if matches for a plane = min_matches, score is modified by min score
        "max_score_modifier" : 1.0, # interpolated to max_score for >= max_matches
        "z_guess_a": -1, # guess at the z-level where the plane match is located in plane list a -> -1 means no guess -> used for optimization in time here for a pre-generated plane list
        "z_guess_b": -1, # guess at the z-level where the plane match is located in plane list k b -> -1 means no guess
        "z_range" : 0, # # +- tolerance to search for in z in both planes
    }

# Default parameters for matching 2 zstacks by a single plane in 2D
DEFAULT_2D_MATCH_PARAMS = {
    "plane_gen_params" : PLANE_GEN_PARAMS_DEFAULT,
    "stack_a_boundary" : None,
    "stack_b_boundary" : None,
    "planes_a_read_file" : None,
    "planes_b_read_file" : None,
    "planes_a_write_file" : None,
    "planes_b_write_file" : None,
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
