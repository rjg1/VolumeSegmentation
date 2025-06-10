from zstack import *
from registration_utils import extract_zstack_plane, match_zstacks_2d, filter_region_by_shape
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

STACK_IN_FILE = "real_data_filtered_algo_VOLUMES_g.csv"
# STACK_IN_FILE = "drg_complete_2s_algo_VOLUMES.csv"
PLANE_OUT_FILE = f"{STACK_IN_FILE}".split('.csv')[0] + "_planes.csv"
USE_FLAT_PLANE = True
PLOT_PLANE_COMPARISON = True
AREA_THRESHOLD = 20  
MAX_ATTEMPTS = 50   


plane_gen_params = {
    "read_filename" : PLANE_OUT_FILE, # file to read plane parameters from
    "save_filename" : PLANE_OUT_FILE, # file to save plane parameters to 
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
    "regenerate_planes" : True, # always re-generate planes even if this z-stack has a set
    "max_alignments" : 500, # maximum number of alignment points allowed per plane
    "z_guess": -1, # guess at the z-level where the plane match is located in stack-> -1 means no guess
    "z_range": 0, # +- tolerance to generate for in z
    "n_threads" : 10, # Number of threads to spawn when generating planes
    "anchor_dist_thresh": 100 # acceptable euclidean distance between an anchor and an alignment point
}


match_plane_params = {
        "bin_match_params" : {
            "min_matches" : 2,
            "fixed_scale" : 1 # Using same scale at the moment
        },
        "traits": {
            "angle" : {
                "weight": 0.4,
                "max_value" : 0.1
            },
            "magnitude" : {
                "weight": 0.3,
                "max_value" : 0.1
            },
            "avg_radius": {
                "weight": 0.1,
                "max_value" : 0.1
            },
            "circularity": {
                "weight": 0.1,
                "max_value" : 0.1
            },
            "area": {
                "weight": 0.1,
                "max_value" : 0.1
            }
        }
    }

plane_list_params = {
    "min_score" : 0.9,
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
    "planes_a_write_file" : PLANE_OUT_FILE,
    "planes_b_write_file" : None,
    "plot_uoi" : True,
    "plot_match" : False,
    "use_gpu" : True,
    "min_uoi": -1,
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

    random.seed(15) # Re-seed random...
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

    # filtered_stack = filter_zstack_by_shape(
    #     zstack=new_stack,
    #     min_area=35,
    #     max_area=1000,
    #     max_eccentricity=0.7
    # )

    # filtered_stack = filter_region_by_shape(
    #     new_stack,
    #     selected_plane,
    #     min_area=40,
    #     max_area=1000,
    #     max_eccentricity=0.69,
    #     preserve_anchor_regions=False
    # )

    # plot_zstack_rois(new_stack)
    # plot_zstack_rois(filtered_stack)
    # plot_zstack_points(new_stack)


    # # Extract data points close to this plane for the new z-stack
    # Generate planes within this plane
    plane_gen_params['read_filename'] = None
    plane_gen_params['save_filename'] = None

    # # DEBUG STEP ugh
    # # Generate more planes
    # plane_gen_params['align_intensity_threshold'] = 0
    # plane_gen_params['anchor_intensity_threshold'] = 0
    # planes_b = new_stack.generate_planes_gpu(plane_gen_params)
    # plane_gen_params['align_intensity_threshold'] = 0.4
    # plane_gen_params['anchor_intensity_threshold'] = 0.5
    # matches = compare_planes_by_geometry_2d(selected_plane, planes_b)

    # if matches:
    #     print(f"Found matching reconstructed plane(s): {matches}")
    # else:
    #     print("No exact reconstructed plane match found in B planes.")
    # return
    # # END DEBUG
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



def compare_planes_by_geometry(reference_plane, candidate_planes, tol=1e-6):
    """
    Compare a reference plane to a list of candidate planes by:
    - Anchor ID
    - Alignment IDs
    - Corresponding point positions

    Args:
        reference_plane: Plane object to compare against
        candidate_planes: list of Plane objects
        tol: numerical tolerance for position matching

    Returns:
        List of tuples: (candidate_idx, match_reason)
    """
    results = []

    ref_anchor_id = reference_plane.anchor_point.id
    ref_anchor_pos = reference_plane.anchor_point.position
    ref_alignments = {
        pt.id: pt.position for i, pt in reference_plane.plane_points.items() if i != 0
    }

    for i, candidate in enumerate(candidate_planes):
        print("")
        cand_anchor_id = candidate.anchor_point.id
        cand_anchor_pos = candidate.anchor_point.position

        if cand_anchor_id != ref_anchor_id:
            continue  # anchor ID must match

        if not np.allclose(ref_anchor_pos, cand_anchor_pos, atol=tol):
            continue  # anchor position must match

        cand_alignments = {
            pt.id: pt.position for j, pt in candidate.plane_points.items() if j != 0
        }

        if set(cand_alignments.keys()) != set(ref_alignments.keys()):
            continue  # alignment IDs must match exactly

        # Check all alignment point positions
        positions_match = all(
            np.allclose(ref_alignments[aid], cand_alignments[aid], atol=tol)
            for aid in ref_alignments
        )

        if not positions_match:
            continue

        # Passed all checks
        results.append((i, f"Match with candidate plane index {i}"))

    return results


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
        print(f"RID={roi_id}, eccentricity={eccentricity:.4f}")
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

# more debugging

def compare_planes_by_geometry_2d(reference_plane, candidate_planes, tol=1e-4):
    """
    Compare a reference plane to a list of candidate planes by:
    - Anchor ID
    - Alignment Point IDs
    - 2D projected positions (into local plane basis)

    Args:
        reference_plane: Plane object
        candidate_planes: list of Plane objects
        tol: position matching tolerance (in 2D)

    Returns:
        List of tuples: (candidate_idx, match_reason)
    """
    results = []

    # Project ref plane points to 2D
    ref_proj = reference_plane.get_local_2d_coordinates()
    ref_anchor_id = reference_plane.anchor_point.id
    ref_alignments = {
        ppt.id: ref_proj[i]
        for i, ppt in reference_plane.plane_points.items()
        if i != 0
    }

    for i, candidate in enumerate(candidate_planes):
        if candidate.anchor_point.id != ref_anchor_id:
            continue
        else:
            print(f"Found matching anchors in b plane idx: {i}")
        if PLOT_PLANE_COMPARISON:
            plot_plane_comparison_2d(reference_plane, candidate)
        cand_proj = candidate.get_local_2d_coordinates()

        if not np.allclose(ref_proj[0], cand_proj[0], atol=tol):
            continue  # anchor positions don't match

        cand_alignments = {
            ppt.id: cand_proj[j]
            for j, ppt in candidate.plane_points.items()
            if j != 0
        }

        if set(cand_alignments.keys()) != set(ref_alignments.keys()):
            continue

        positions_match = all(
            np.allclose(ref_alignments[aid], cand_alignments[aid], atol=tol)
            for aid in ref_alignments
        )

        if not positions_match:
            continue

        results.append((i, f"2D projected match with candidate plane index {i}"))

    return results


def plot_plane_comparison_2d(ref_plane, cand_plane, tol=1e-4):
    """
    Plot 2D projections of reference and candidate planes to visually compare geometry.
    Anchor is shown in red, alignments in blue (ref) and green (cand).
    Matching point IDs are labeled.
    """
    ref_proj = ref_plane.get_local_2d_coordinates()
    cand_proj = cand_plane.get_local_2d_coordinates()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("2D Plane Comparison")

    # Plot reference plane (anchor = red, alignments = blue)
    for i, (x, y) in ref_proj.items():
        if i == 0:
            ax.scatter(x, y, color='red', label='Ref Anchor', s=60)
            ax.text(x + 0.2, y + 0.2, f"A{ref_plane.plane_points[i].id}", color='red')
        else:
            ax.scatter(x, y, color='blue')
            ax.text(x + 0.2, y + 0.2, f"P{ref_plane.plane_points[i].id}", color='blue')

    # Plot candidate plane (anchor = orange, alignments = green)
    for j, (x, y) in cand_proj.items():
        if j == 0:
            ax.scatter(x, y, color='orange', label='Cand Anchor', s=60)
            ax.text(x + 0.2, y + 0.2, f"A{cand_plane.plane_points[j].id}", color='orange')
        else:
            ax.scatter(x, y, color='green')
            ax.text(x + 0.2, y + 0.2, f"P{cand_plane.plane_points[j].id}", color='green')

    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    plt.show()

# end debug


if __name__ == "__main__":
    main()



