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

STACK_IN_FILE = "real_data_filtered_algo_VOLUMES_g.csv"
# STACK_IN_FILE = "drg_complete_2s_algo_VOLUMES.csv"
PLANE_OUT_FILE = f"{STACK_IN_FILE}".split('.csv')[0] + "_planes.csv"
USE_FLAT_PLANE = False
FILTER_PLANES = True # Only assess a specific A plane
PLOT_PLANE_COMPARISON = True
AREA_THRESHOLD = 50
MIN_ROI_NUMBER = 8
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
    "anchor_dist_thresh": 100, # acceptable euclidean distance between an anchor and an alignment point
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
            "fixed_scale" : 1, # Using same scale at the moment
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
    "rematch_angled_planes": True, # re-run 2d matching on new centroids of angled planes after projection (before uoi calculation)
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
    # z_stack_subset.generate_planes(plane_gen_params) # CPU Version debug
    # Choose a random plane to make a new z-stack

    print(f"Starting plane selection: {len(z_stack.planes)} planes identified")

    random.seed(17) # Re-seed random...
    attempt = 0
    plane_ids = []
    for idx, plane in enumerate(z_stack.planes):
        if USE_FLAT_PLANE and np.allclose(plane.normal, [0,0,1], atol=1e-6):
            plane_ids.append(idx)
        elif not USE_FLAT_PLANE and not np.allclose(plane.normal, [0,0,1], atol=1e-6):
            plane_ids.append(idx)
    new_stack = None

    if not plane_ids:
        raise ValueError("No suitable planes found in the z-stack.")
    
    while attempt < MAX_ATTEMPTS:
        
        selected_idx = 1248
        # selected_idx = 2086
        # selected_idx = 620

        # # BEGIN DEBUG SELECTION
        # tested_idxs = [1248, 2086, 1542, 1033, 620, 2194, 2303, 1433, 3082, 1612, 1257, 3317]
        # selected_idx = tested_idxs[0]
        # while selected_idx in tested_idxs:
        #     selected_idx = random.choice(plane_ids)
        # # END DEBUG SELECTION


        selected_plane = z_stack.planes[selected_idx]

        temp_stack = extract_zstack_plane(z_stack, selected_plane, threshold=plane_gen_params['projection_dist_thresh'], method="volume")
        mean_area = temp_stack._average_roi_area()
        if len(list(temp_stack.z_planes.keys())) > 0:
            num_rois = len(list(temp_stack.z_planes[list(temp_stack.z_planes.keys())[0]].keys()))
        else:
            num_rois = 0

        if mean_area > AREA_THRESHOLD and num_rois >= MIN_ROI_NUMBER:
            new_stack = temp_stack
            print(f"Selected plane ID {selected_idx} with mean ROI area {mean_area:.2f}")
            break

        attempt += 1


    if new_stack is None:
        raise RuntimeError(f"Could not find a suitable plane with average ROI area > {AREA_THRESHOLD} after {MAX_ATTEMPTS} attempts.")
    print("Finished plane selection")

    # # # Debug - view 2d projection of new stack
    # plot_zstack_rois(new_stack)
    
    # # # Debug - view 2d projection of original points
    # plot_projected_regions_with_plane_points(z_stack, selected_plane)

    # plot_zstack_rois(filtered_stack)
    # plot_zstack_points(new_stack)

    # End debug

    # DEBUG STEP
    # Generate planes within this plane
    plane_gen_params['read_filename'] = None
    plane_gen_params['save_filename'] = None                # Never load/save B-planes
    plane_gen_params['align_intensity_threshold'] = 0       # Any point can be an alignment point
    plane_gen_params['anchor_intensity_threshold'] = 0      # Any point can be an anchor point
    plane_gen_params['regenerate_planes'] = True            # Always regenerate b-planes
    planes_b = new_stack.generate_planes_gpu(plane_gen_params)

    # DEBUG - extract all non-tested planes
    # if len(planes_b) > 0:
    #     planes_b = [planes_b[0]] # Extract first plane of planes_b
    #     new_stack.planes = planes_b
    # else:
    #     print(f"No b planes generated... check that out maybe")
    #     return
    if FILTER_PLANES:
        planes_a = [z_stack.planes[selected_idx]]
        z_stack.planes = planes_a
    # DEBUG
    # Restore params for other plane gen step
    plane_gen_params['align_intensity_threshold'] = 0.4
    plane_gen_params['anchor_intensity_threshold'] = 0.5
    plane_gen_params['read_filename'] = PLANE_OUT_FILE
    plane_gen_params['save_filename'] = PLANE_OUT_FILE
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

    

def export_trait_mse_to_csv(results_dict, filtered_list, output_path="trait_mse_export.csv"):
    """
    Export trait MSE values from match results to a CSV file.

    Args:
        results_dict (dict): Dict of plane matches with nested ["result"]["trait_values"].
        filtered_list (list): List of booleans (True = correct, False = incorrect).
        output_path (str): CSV file path to write.
    """
    if len(results_dict) != len(filtered_list):
        raise ValueError("Mismatch between result count and correctness list length.")

    rows = []
    for (_, data), is_correct in zip(results_dict.items(), filtered_list):
        match_result = data.get("result", {})
        trait_values = match_result.get("trait_values", {})
        for trait, error in trait_values.items():
            rows.append({
                "trait": trait,
                "error": error,
                "outcome": "Correct" if is_correct else "Incorrect"
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Exported trait MSEs to: {output_path}")

def plot_trait_mse_by_match(results_dict, filtered_list, exclude_traits=("angle", "magnitude")):
    """
    Plot trait MSEs across plane matches using results dictionary and correctness list.

    Args:
        results_dict (dict): dictionary where each value contains a "result" key with "trait_values".
        filtered_list (list): list of 1.0 or 0.0 indicating correct (1.0) or incorrect (0.0) match.
        exclude_traits (tuple): traits to exclude from the plot (default: angle, magnitude).
    """
    if not results_dict or not filtered_list:
        print("[WARN] Empty results or correctness list.")
        return

    # Extract ordered list of trait dictionaries from the results
    trait_mse_data = [v["result"]["trait_values"] for v in results_dict.values()]
    all_traits = [k for k in trait_mse_data[0].keys() if k not in exclude_traits]

    for trait in all_traits:
        correct_vals = [entry[trait] for entry, correct in zip(trait_mse_data, filtered_list) if correct == 1.0]
        incorrect_vals = [entry[trait] for entry, correct in zip(trait_mse_data, filtered_list) if correct == 0.0]

        plt.figure()
        plt.boxplot([correct_vals, incorrect_vals], labels=["Correct", "Incorrect"])
        plt.title(f"MSE Distribution for Trait: '{trait}'")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



def evaluate_volume_overlap(matched_planes, zstack_a, zstack_b):
    """
    For each matched plane, compute the percentage of matched (z, roi_id) pairs
    whose volume ids match in zstack_a and zstack_b.
    """
    ratio_list = []
    for score, match_data in matched_planes.items():
        og_matches = match_data["result"].get("og_matches", [])
        if not og_matches:
            match_data["result"]["volume_match_ratio"] = 0.0
            continue

        match_count = 0
        total = 0

        for (z_a, id_a), (z_b, id_b) in og_matches:
            vol_a = zstack_a.z_planes.get(z_a, {}).get(id_a, {}).get("volume")
            vol_b = zstack_b.z_planes.get(z_b, {}).get(id_b, {}).get("volume")
            if vol_a is not None and vol_b is not None:
                total += 1
                if vol_a == vol_b:
                    match_count += 1

        ratio_list.append((match_count / total) if total > 0 else 0.0)
    
    return ratio_list


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
def compare_planes_by_geometry_2d(reference_plane, candidate_planes, z_stack, tol=1e-4, min_matches=2):
    """
    Compare a reference plane to a list of candidate planes by:
    - Anchor position (3D)
    - Alignment point 2D projections (geometry-based, not ID-based)

    Args:
        reference_plane: Plane object
        candidate_planes: list of Plane objects
        tol: position matching tolerance (in 2D)
        min_matches: minimum number of 2D-aligned points that must match to consider a plane valid

    Returns:
        List of tuples: (candidate_idx, match_reason)
    """
    results = []

    # Project reference plane points to 2D
    ref_proj = reference_plane.get_local_2d_coordinates()
    ref_anchor_pos = reference_plane.anchor_point.position
    ref_alignment_pts = np.array([ref_proj[i] for i in range(1, len(ref_proj))])

    for i, candidate in enumerate(candidate_planes):
        cand_proj = candidate.get_local_2d_coordinates()
        cand_anchor_pos = candidate.anchor_point.position

        # Check anchor positions match
        if not np.allclose(ref_anchor_pos, cand_anchor_pos, atol=tol):
            pass
            #continue

        if PLOT_PLANE_COMPARISON:
            plot_projected_regions_with_plane_points(z_stack, candidate, datatype="flat")
            plot_plane_comparison_2d(reference_plane, candidate)


        cand_alignment_pts = np.array([cand_proj[j] for j in range(1, len(cand_proj))])

        # Compare geometric similarity (how many 2D points match)
        matched = 0
        for ref_pt in ref_alignment_pts:
            if any(np.allclose(ref_pt, cand_pt, atol=tol) for cand_pt in cand_alignment_pts):
                matched += 1

        if matched >= min_matches:
            results.append((i, f"{matched} matching 2D alignment points at candidate index {i}"))

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

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Polygon as MplPolygon
def plot_projected_regions_with_plane_points(z_stack, plane, threshold=0.5, method="volume", eps=5.0, min_samples=5, datatype="angled"):
    """
    Projects the ZStack onto a given plane, clusters regions, and visualizes:
    - Colored 2D projected regions (from DBSCAN or volume)
    - Anchor/alignment points (overlaid as markers)

    Args:
        z_stack: the ZStack object
        plane: the Plane object (angled)
        threshold: max distance to consider point part of the plane
        method: 'split' or 'merge'
    """
    from matplotlib.colors import to_rgba
    if datatype == "angled":
        output_regions, pid_mapping, _ = project_angled_plane_points(
            z_stack,
            angled_plane=plane,
            threshold=threshold,
            method=method,
            eps=eps,
            min_samples=min_samples
        )
    else:
        output_regions = project_flat_plane_points(z_stack, plane.anchor_point)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Colormap for visual variety
    cmap = get_cmap('tab20')
    region_colors = {}

    for idx, (pid, pts3d) in enumerate(output_regions.items()):
        # Project each point to 2D
        projected_pts = [plane._project_point_2d(np.array(pt)) for pt in pts3d]
        projected_pts = np.array(projected_pts)

        color = cmap(idx % 20)
        region_colors[pid] = color

        ax.scatter(projected_pts[:, 0], projected_pts[:, 1], s=8, color=color, label=f"Region {pid}")

        # Draw convex hull if possible
        if len(projected_pts) >= 3:
            try:
                hull = ConvexHull(projected_pts)
                polygon = projected_pts[hull.vertices]
                patch = MplPolygon(polygon, closed=True, fill=False, edgecolor=to_rgba(color, 0.8), linewidth=1.5)
                ax.add_patch(patch)
            except:
                pass  # Fallback if ConvexHull fails

    # Overlay anchor and alignment points
    projected_plane_pts = {
        (pt.id, pt.position[2]): plane._project_point_2d(pt.position)
        for pt in plane.plane_points.values()
    }

    for (pid, z), pt2d in projected_plane_pts.items():
        ax.scatter(pt2d[0], pt2d[1], s=80, marker='x', color='black')
        ax.text(pt2d[0] + 0.5, pt2d[1] + 0.5, f"PID {pid}", fontsize=8)

    ax.set_title("Projected Regions and Plane Points")
    ax.set_xlabel("u-axis")
    ax.set_ylabel("v-axis")
    ax.set_aspect('equal')
    # ax.legend(loc='upper right', fontsize='x-small', markerscale=1.5)
    plt.tight_layout()
    plt.show()

# end debug


if __name__ == "__main__":
    main()



