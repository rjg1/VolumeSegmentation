from zstack import *
import time

# IN_FILE = "real_data_filtered_algo_VOLUMES_g.csv"
IN_FILE = "drg_complete_2_algo_VOLUMES.csv"
OUT_FILE = f"{IN_FILE}".split('.csv')[0] + "_planes.csv"

plane_gen_params = {
    "read_filename" : OUT_FILE, # file to read plane parameters from
    "save_filename" : OUT_FILE, # file to save plane parameters to 
    "anchor_intensity_threshold": 0.8, # threshold to compare transformed intensity to for a point to be considered an anchor pt (see transform_intensity)
    "align_intensity_threshold": 0.4, # as above, but for alignment points
    "z_threshold": 5, # max distance in z between two points in a plane
    "max_tilt_deg": 30.0, # max magnitude of tilt in the vector between two points in a plane
    "projection_dist_thresh":  0.5, # euclidean distance for a point considered close enough to a plane for it to be projected
    "transform_intensity" : "quantile", # "quantile", "raw", "minmax" - transforms avg intensity per roi and compares to anchor/align intensity threshold
    "plane_boundaries" : [0, 1024, 0, 1024],
    "margin" : 2, # distance between boundary point and pixel in img to be considered an edge roi
    "match_anchors" : True, # require two planes to have a common anchor to be equivalent (turning this off will reduce the number of planes and accuracy)
    "fixed_basis" : True, # define the projected 2d x-axis as [1,0,0]
    "regenerate_planes" : True, # always re-generate planes even if this z-stack has a set
    "max_alignments" : 500, # maximum number of alignment points allowed per plane
    "z_guess": -1, # guess at the z-level where the plane match is located in stack-> -1 means no guess
    "z_range": 0, # +- tolerance to search for in z in both planes
    "n_threads" : 10, # Number of threads to spawn when generating planes
    "anchor_dist_thresh": 20 # acceptable euclidean distance between an anchor and an alignment point
}


# Test plane generation for z-stack data
def main():
    # Start the clock
    start = time.perf_counter()
    z_stack_subset = ZStack(data=IN_FILE) # Load z-stack of existing data
    z_stack_subset.generate_random_intensities(0,1000) # Generate random average intensities for it per ROI
    generation_start = time.perf_counter()
    z_stack_subset.generate_planes_gpu(plane_gen_params)
    # z_stack_subset.generate_planes(plane_gen_params)
    # Stop the clock
    end = time.perf_counter()

    print(f"Total time taken: {end - start:.4f} seconds")
    print(f"Generation time taken: {end - generation_start:.4f} seconds")

if __name__ == "__main__":
    main()