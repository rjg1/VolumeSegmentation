from zstack import *
from registration_utils import extract_zstack_plane, match_zstacks_2d, filter_region_by_shape, project_angled_plane_points, project_flat_plane_points
import random
import time

# Debug
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from skimage.measure import EllipseModel
from collections import defaultdict
import numpy as np

STACK_IN_FILE = "drg_complete_3i_algo_VOLUMES.csv"
NEW_PLANE_IN_FILE = "roi_coords_single_plane.csv"
A_PLANE_OUT_FILE = f"{STACK_IN_FILE}".split('.csv')[0] + "_planes.csv"
B_PLANE_OUT_FILE = f"{NEW_PLANE_IN_FILE}".split('.csv')[0] + "_planes.csv"


# TODO - find the plane in the z-stack, work out an approximate scale ratio
# Compare exact planes (flat match)
# See what score it gets
# Diagnose convex hull errors
# Diagnose degenerate plane creation/loading

plane_gen_params = {
    "read_filename" : A_PLANE_OUT_FILE, # file to read plane parameters from
    "save_filename" : A_PLANE_OUT_FILE, # file to save plane parameters to 
    "anchor_intensity_threshold": 0.5, # threshold to compare transformed intensity to for a point to be considered an anchor pt (see transform_intensity)
    "align_intensity_threshold": 0.4, # as above, but for alignment points
    "z_threshold": 5, # max distance in z between two points in a plane
    "max_tilt_deg": 30.0, # max magnitude of tilt in the vector between two points in a plane
    "projection_dist_thresh":  0.5, # euclidean distance for a point considered close enough to a plane for it to be projected
    "transform_intensity" : "quantile", # "quantile", "raw", "minmax" - transforms avg intensity per roi and compares to anchor/align intensity threshold
    "plane_boundaries" : [0, 512, 0, 512],
    "margin" : 2, # distance between boundary point and pixel in img to be considered an edge roi
    "match_anchors" : True, # require two planes to have a common anchor to be equivalent (turning this off will reduce the number of planes and accuracy)
    "fixed_basis" : True, # define the projected 2d x-axis as [1,0,0]
    "regenerate_planes" : False, # always re-generate planes even if this z-stack has a set
    "max_alignments" : 500, # maximum number of alignment points allowed per plane
    "z_guess": -1, # guess at the z-level where the plane match is located in stack-> -1 means no guess
    "z_range": 0, # +- tolerance to generate for in z
    "n_threads" : 10, # Number of threads to spawn when generating planes
    "anchor_dist_thresh": 20, # acceptable euclidean distance between an anchor and an alignment point
    "reconstruct_angled_rois" : False, # projects points to angled planes and reconstructs ROIs for a more accurate centroid placement
    "filter_params": { # used for filtering rois on reconstructed angular planes, prior to centroid calculation
        "disable_filtering": True,
        "min_area": 40,
        "max_area": 1000,
        "max_eccentricity": 0.69,
        "preserve_anchor": True
    },
    "seg_params": { # used for segmenting rois when reconstructing angled planes
        "method" : "volume",
        "eps": 3.0,
        "min_samples" : 5
    }
}


match_plane_params = {
        "bin_match_params" : {
            "min_matches" : 2,
            "fixed_scale" : None, # TODO find a workable fixed scale
            "outlier_thresh" : 2, #10, # 2
            "mse_threshold": 1, # 0.3
            "angle_tolerance" : 5 #5, # 2
        },
        "traits": {
            "angle" : {
                "weight": 0.5,
                "max_value" : 0.1
            },
            "magnitude" : {
                "weight": 0.1,
                "max_value" : 0.8
            },
            "avg_radius": {
                "weight": 0.05,
                "max_value" : 1
            },
            "circularity": {
                "weight": 0.05,
                "max_value" : 0.1
            },
            "area": {
                "weight": 0.1,
                "max_value" : 5
            }
        }
    }

plane_list_params = {
    "min_score" : 0.5,
    "max_matches" : 2, # max number of alignment point matches - score scaled up with more matches
    "min_score_modifier" : 1.0, # if num align matches for a plane = min_matches, score is modified by min score
    "max_score_modifier" : 1.0, # interpolated to max_score for >= max_matches
    "z_guess_a": 88, #114, # guess at the z-level where the plane match is located in plane list a -> -1 means no guess -> used for optimization in time here for a pre-generated plane list
    "z_guess_b": -1, # guess at the z-level where the plane match is located in plane list k b -> -1 means no guess
    "z_range" : 20, # # +- tolerance to search for in z in both planes
}

match_params = {
    "plane_gen_params" : plane_gen_params,
    "match_plane_params" : match_plane_params,
    "plane_list_params" : plane_list_params,
    "planes_a_read_file" : A_PLANE_OUT_FILE,
    "planes_b_read_file" : B_PLANE_OUT_FILE,
    "planes_a_write_file" : A_PLANE_OUT_FILE,
    "planes_b_write_file" : B_PLANE_OUT_FILE,
    "plot_uoi" : True,
    "plot_match" : False,
    "use_gpu" : True,
    "min_uoi": 0.9,
    "seg_params": {
        "method" : "volume",
        "eps": 1.5,
        "min_samples" : 5
    },
    "filter_params": {
        "disable_filtering": True,
        "min_area": 40,
        "max_area": 1000,
        "max_eccentricity": 0.69,
        "preserve_anchor": True
    },
    "rematch_angled_planes": False, # re-run 2d matching on new centroids of angled planes after projection (before uoi calculation)
    "match_planes_only": False # Do not perform UoI comparison and only return the coarse match results
}

# Test plane generation for z-stack data
def main():
    # Seed random
    random.seed(10)
    # Start the clock
    start = time.perf_counter()
    z_stack = ZStack(data=STACK_IN_FILE) # Load z-stack of existing data
    z_stack.generate_random_intensities(0,1000) # Generate random average intensities for it per ROI
    generation_start = time.perf_counter()
    z_stack.generate_planes_gpu(plane_gen_params)
    generation_end = time.perf_counter()
    # Choose a random plane to make a new z-stack
    # End debug

    # DEBUG STEP
    # Generate planes within this plane
    new_stack = ZStack(data=NEW_PLANE_IN_FILE)
    plane_gen_params['read_filename'] = B_PLANE_OUT_FILE
    plane_gen_params['save_filename'] = B_PLANE_OUT_FILE
    plane_gen_params['align_intensity_threshold'] = 0       # Any point can be an alignment point
    plane_gen_params['anchor_intensity_threshold'] = 0      # Any point can be an anchor point
    plane_gen_params['regenerate_planes'] = True            # Always regenerate b-planes
    planes_b = new_stack.generate_planes_gpu(plane_gen_params)

    # DEBUG
    # Restore params for other plane gen step
    plane_gen_params['align_intensity_threshold'] = 0.4
    plane_gen_params['anchor_intensity_threshold'] = 0.5
    plane_gen_params['read_filename'] = A_PLANE_OUT_FILE
    plane_gen_params['save_filename'] = A_PLANE_OUT_FILE
    plane_gen_params['anchor_dist_thresh'] = 200 # more liberal with anchors pts that can be made
    plane_gen_params['regenerate_planes'] = False
    # matches = compare_planes_by_geometry_2d(selected_plane, planes_b, new_stack)

    # if matches:
    #     print(f"Found matching reconstructed plane(s): {matches}")
    # else:
    #     print("No exact reconstructed plane match found in B planes.")
    # return
    # END DEBUG
    # Attempt to match planes
    match_start = time.perf_counter()
    result = match_zstacks_2d(zstack_a=z_stack, zstack_b=new_stack, match_params=match_params)
    # Stop the clock
    end = time.perf_counter()

    print(result)
    print(f"Total time taken: {end - start:.4f} seconds")
    print(f"Plane generation time taken: {generation_end - generation_start:.4f} seconds")
    print(f"Match time taken: {end - match_start:.4f} seconds")




if __name__ == "__main__":
    main()



