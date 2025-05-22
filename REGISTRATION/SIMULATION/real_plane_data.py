from zstack import *
from registration_utils import extract_zstack_plane, match_zstacks_2d
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

# STACK_IN_FILE = "real_data_filtered_algo_VOLUMES_g.csv"
STACK_IN_FILE = "drg_complete_2_algo_VOLUMES.csv"
PLANE_OUT_FILE = f"{STACK_IN_FILE}".split('.csv')[0] + "_planes.csv"
NEW_PLANE_FILE = f"{STACK_IN_FILE}".split('.csv')[0] + "new_planes.csv"
USE_FLAT_PLANE = False
AREA_THRESHOLD = 20  
MAX_ATTEMPTS = 50   


plane_gen_params = {
    "read_filename" : PLANE_OUT_FILE, # file to read plane parameters from
    "save_filename" : None, # file to save plane parameters to 
    "anchor_intensity_threshold": 0.5, # threshold to compare transformed intensity to for a point to be considered an anchor pt (see transform_intensity)
    "align_intensity_threshold": 0.4, # as above, but for alignment points
    "z_threshold": 5, # max distance in z between two points in a plane
    "max_tilt_deg": 30.0, # max magnitude of tilt in the vector between two points in a plane
    "projection_dist_thresh":  0.5, # euclidean distance for a point considered close enough to a plane for it to be projected
    "transform_intensity" : "quantile", # "quantile", "raw", "minmax" - transforms avg intensity per roi and compares to anchor/align intensity threshold
    "plane_boundaries" : [0, 1024, 0, 1024],
    "margin" : 2, # distance between boundary point and pixel in img to be considered an edge roi
    "match_anchors" : True, # require two planes to have a common anchor to be equivalent (turning this off will reduce the number of planes and accuracy)
    "fixed_basis" : True, # define the projected 2d x-axis as [1,0,0]
    "regenerate_planes" : False, # always re-generate planes even if this z-stack has a set
    "max_alignments" : 500, # maximum number of alignment points allowed per plane
    "z_guess": -1, # guess at the z-level where the plane match is located in stack-> -1 means no guess
    "z_range": 0, # +- tolerance to search for in z in both planes
    "n_threads" : 10, # Number of threads to spawn when generating planes
    "anchor_dist_thresh": 10 # acceptable euclidean distance between an anchor and an alignment point
}


match_plane_params = {
        "bin_match_params" : {
            "min_matches" : 2,
            "fixed_scale" : 1 # Using same scale at the moment
        },
        "traits": {
            "angle" : {
                "weight": 0.6,
                "max_value" : 0.1
            },
            "magnitude" : {
                "weight": 0.4,
                "max_value" : 0.1
            }
        }
    }

plane_list_params = {
    "min_score" : 0.97,
    "max_matches" : 4, # max matches to scale score between
    "min_score_modifier" : 0.8, # if matches for a plane = min_matches, score is modified by min score
    "max_score_modifier" : 1.0, # interpolated to max_score for >= max_matches
    "z_guess_a": -1, #114, # guess at the z-level where the plane match is located in plane list a -> -1 means no guess -> used for optimization in time here for a pre-generated plane list
    "z_guess_b": -1, # guess at the z-level where the plane match is located in plane list k b -> -1 means no guess
    "z_range" : 1, # # +- tolerance to search for in z in both planes
}

match_params = {
    "plane_gen_params" : plane_gen_params,
    "match_plane_params" : match_plane_params,
    "plane_list_params" : plane_list_params,
    "planes_a_read_file" : PLANE_OUT_FILE,
    "planes_b_read_file" : None,
    "planes_a_write_file" : None,
    "planes_b_write_file" : None,
    "plot_uoi" : True,
    "plot_match" : False,
    "use_gpu" : True,
    "min_uoi": -1,
    "seg_params": {
        "method" : "split",
        "eps": 1.5,
        "min_samples" : 5
    }
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
    # z_stack_subset.generate_planes(plane_gen_params) # CPU Version debug
    # Choose a random plane to make a new z-stack

    print("Starting plane selection")

    random.seed(11) # Re-seed random...
    attempt = 0
    plane_pool = (
        [p for p in z_stack.planes if np.allclose(p.normal, [0, 0, 1], atol=1e-6)]
        if USE_FLAT_PLANE
        else [p for p in z_stack.planes if not np.allclose(p.normal, [0, 0, 1], atol=1e-6)]
    )
    new_stack = None

    if not plane_pool:
        raise ValueError("No suitable planes found in the z-stack.")
    
    while attempt < MAX_ATTEMPTS:
        selected_plane = random.choice(plane_pool)
        temp_stack = extract_zstack_plane(z_stack, selected_plane, threshold=plane_gen_params['projection_dist_thresh'], method="volume")
        mean_area = temp_stack._average_roi_area()

        if mean_area > AREA_THRESHOLD:
            new_stack = temp_stack
            print(f"Selected plane ID {selected_plane.anchor_point.id} with mean ROI area {mean_area:.2f}")
            break

        attempt += 1

    if new_stack is None:
        raise RuntimeError(f"Could not find a suitable plane with average ROI area > {AREA_THRESHOLD} after {MAX_ATTEMPTS} attempts.")
    print("Finished plane selection")

    # Test z-stack filtering
    # filtered_stack = filter_zstack_by_shape(
    #     zstack=new_stack,
    #     min_area=10,
    #     max_area=1000,
    #     max_eccentricity=0.95
    # )

    # filtered_stack = filter_zstack_by_shape(
    #     zstack=new_stack,
    #     min_area=35,
    #     max_area=1000,
    #     max_eccentricity=0.95
    # )

    filtered_stack = filter_zstack_by_shape(
        zstack=new_stack,
        min_area=35,
        max_area=1000,
        max_eccentricity=0.7
    )

    plot_zstack_rois(filtered_stack)
    # plot_zstack_points(new_stack)
    return

    # Extract data points close to this plane for the new z-stack
    new_stack = extract_zstack_plane(z_stack, selected_plane, threshold=plane_gen_params['projection_dist_thresh'])
    # Generate planes within this plane
    plane_gen_params['read_filename'] = None
    plane_gen_params['save_filename'] = None
    # Attempt to match planes
    match_start = time.perf_counter()
    uoi_match_data = match_zstacks_2d(zstack_a=z_stack, zstack_b=new_stack, match_params=match_params)
    # Stop the clock
    end = time.perf_counter()

    print(uoi_match_data)
    print(f"Total time taken: {end - start:.4f} seconds")
    print(f"Plane generation time taken: {generation_end - generation_start:.4f} seconds")
    print(f"Match time taken: {end - match_start:.4f} seconds")

def plot_zstack_rois(zstack, title="ZStack ROI Areas"):
    zstack._build_xy_rois()  # Ensure ROI data is built

    fig, ax = plt.subplots(figsize=(10, 10))
    patches = []
    areas = []

    for roi in zstack.xy_rois.values():
        boundary = roi.points
        if not boundary:
            continue
        coords_2d = [(x, y) for x, y, _ in boundary]
        coords_2d = get_ordered_boundary(coords_2d)
        patch = MplPolygon(coords_2d, closed=True)
        patches.append(patch)
        areas.append(roi.get_area())

    if not patches:
        print("No valid ROI polygons to plot.")
        return

    collection = PatchCollection(patches, cmap='viridis', edgecolor='black', linewidths=0.5)
    collection.set_array(np.array(areas))  # Color by area
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_title(title)
    cbar = plt.colorbar(collection, ax=ax)
    cbar.set_label("ROI Area")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_zstack_points(zstack, title="ZStack ROI Points (Colored by ROI_ID)"):
    colors = []
    xs = []
    ys = []

    for z in zstack.z_planes:
        for roi_id, roi_data in zstack.z_planes[z].items():
            coords = roi_data["coords"]
            for x, y in coords:
                xs.append(x)
                ys.append(y)
                colors.append(roi_id)

    if not xs:
        print("No points to plot.")
        return

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(xs, ys, c=colors, cmap='viridis', s=5)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(scatter, label="ROI_ID")
    plt.tight_layout()
    plt.show()


def get_ordered_boundary(coords_2d):
    """
    Given a list of (x, y) points, return them ordered along the convex hull.
    """
    if len(coords_2d) < 3:
        return coords_2d
    try:
        hull = ConvexHull(coords_2d)
        return [coords_2d[i] for i in hull.vertices]
    except:
        return coords_2d

def filter_zstack_by_shape(
    zstack,
    min_area=None,
    max_area=None,
    max_eccentricity=None
):
    """
    Filters the ZStack ROIs by shape and area.

    Args:
        zstack: Original ZStack
        min_area: Minimum ROI area
        max_area: Maximum ROI area
        max_eccentricity: Maximum allowed eccentricity (0=circle, 1=line)

    Returns:
        A new ZStack with filtered ROIs
    """
    filtered_z_planes = defaultdict(lambda: defaultdict(dict))

    for (z, roi_id), region in zstack.xy_rois.items():
        coords = region.points
        if len(coords) < 5:
            continue

        x, y = zip(*[(pt[0], pt[1]) for pt in coords])
        x = np.array(x)
        y = np.array(y)

        area = region.get_area()

        # Area filtering
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        # Shape filtering
        model = EllipseModel()
        if not model.estimate(np.column_stack([x, y])):
            continue

        _, _, a_axis, b_axis, _ = model.params
        a, b = sorted([a_axis, b_axis])  # Ensure a ≤ b
        if b == 0:
            continue
        eccentricity = np.sqrt(1 - (a ** 2) / (b ** 2))
        if max_eccentricity is not None and eccentricity > max_eccentricity:
            continue

        # Keep the ROI
        original = zstack.z_planes[z][roi_id]
        filtered_z_planes[z][roi_id]["coords"] = original["coords"]
        if "intensity" in original:
            filtered_z_planes[z][roi_id]["intensity"] = original["intensity"]
        if "volume" in original:
            filtered_z_planes[z][roi_id]["volume"] = int(original["volume"])

    return ZStack(filtered_z_planes)

if __name__ == "__main__":
    main()



