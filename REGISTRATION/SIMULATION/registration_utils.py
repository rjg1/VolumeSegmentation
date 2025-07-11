# Scripts used for registration
from region import Region, BoundaryRegion
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
from param_handling import DEFAULT_2D_MATCH_PARAMS, PLANE_GEN_PARAMS_DEFAULT, create_param_dict
from plane import Plane
from zstack import ZStack
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
from collections import defaultdict
import copy
from sklearn.cluster import DBSCAN, KMeans
from shapely.errors import TopologicalError
from scipy.spatial import ConvexHull
import random
from skimage.measure import EllipseModel
from tqdm import tqdm

def safe_polygon(region):
    pts = region.get_boundary_points()
    if len(pts) < 4:
        return None
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if poly.is_valid else None
    except (ValueError, TopologicalError):
        return None

# Takes a dict of {roi id: region} for each set of regions, and an existing match list
def compute_avg_uoi(regions_a, regions_b, matches, min_uoi = 0, plot=False):
    """
    Computes average UoI between region groups.
    Handles many-to-many, unmatched ROIs with bipartite matching.
    """
    grouped_a = {}
    grouped_b = {}

    for i, j in matches:
        grouped_a.setdefault(i, []).append(regions_a[i])
        grouped_b.setdefault(j, []).append(regions_b[j])

    matched_ids_a = set(i for i, _ in matches)
    matched_ids_b = set(j for _, j in matches)

    # Identify unmatched IDs
    unmatched_ids_a = [i for i in regions_a if i not in matched_ids_a]
    unmatched_ids_b = [j for j in regions_b if j not in matched_ids_b]

    # New matches for unmatched regions - match based off maximizing total UoI over matches
    new_matches = []
    if unmatched_ids_a and unmatched_ids_b:
        cost_matrix = np.zeros((len(unmatched_ids_a), len(unmatched_ids_b)))

        for i_idx, a in enumerate(unmatched_ids_a):
            for j_idx, b in enumerate(unmatched_ids_b):
                poly_a = safe_polygon(regions_a[a])
                poly_b = safe_polygon(regions_b[b])

                if poly_a is None or poly_b is None:
                    uoi = 0.0
                else:
                    inter = poly_a.intersection(poly_b)
                    union = poly_a.union(poly_b)
                    uoi = inter.area / union.area if union.area > 0 else 0.0

                cost_matrix[i_idx, j_idx] = -uoi  # NEGATE because we maximize UoI

        # Solve the assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i_idx, j_idx in zip(row_ind, col_ind):
            i = unmatched_ids_a[i_idx]
            j = unmatched_ids_b[j_idx]

            grouped_a.setdefault(i, []).append(regions_a[i])
            grouped_b.setdefault(j, []).append(regions_b[j])

            new_matches.append((i, j))

    uoi_scores = []

    # Loop over all matches (original and new)
    all_matches = matches + new_matches

    for i, j in all_matches:
        if i not in grouped_a or j not in grouped_b:
            continue

        # Use safe_polygon to clean and validate
        polygons_a = [safe_polygon(region) for region in grouped_a[i]]
        polygons_b = [safe_polygon(region) for region in grouped_b[j]]

        # Filter out None entries
        polygons_a = [p for p in polygons_a if p is not None]
        polygons_b = [p for p in polygons_b if p is not None]

        if not polygons_a or not polygons_b:
            # print(f"Skipping match ({i}, {j}) due to invalid polygons.")
            continue

        poly_a = unary_union(polygons_a)
        if not poly_a.is_valid:
            poly_a = poly_a.buffer(0)

        poly_b = unary_union(polygons_b)
        if not poly_b.is_valid:
            poly_b = poly_b.buffer(0)

        if not poly_a.is_valid or not poly_b.is_valid:
            # print(f"Skipping invalid merged match ({i}, {j})")
            continue

        # Proceed with valid UoI computation
        inter = poly_a.intersection(poly_b)
        union = poly_a.union(poly_b)
        uoi = inter.area / union.area if union.area > 0 else 0.0
        uoi_scores.append(uoi)


    # Calc the avg uoi across all pair matches
    avg_uoi = sum(uoi_scores) / len(uoi_scores) if uoi_scores else 0.0

    if plot:
        plot_uoi_matches(grouped_a, grouped_b, all_matches, min_uoi, avg_uoi)

    return avg_uoi

def plot_uoi_matches(grouped_a, grouped_b, matches, min_uoi, uoi):
    """
    Plot matched ROI regions with their union-over-intersection overlays.

    Args:
        grouped_a (dict): {id: [regions]} for dataset A
        grouped_b (dict): {id: [regions]} for dataset B
        matches (list of tuples): [(i, j), ...] match pairs
        min_uoi (float): minimum UoI threshold to trigger plotting
        uoi (float): current UoI score
    """
    if uoi <= min_uoi:
        return  # Don't plot if below threshold

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Matched ROIs with UoI Shading")
    ax.set_aspect('equal')

    for i, j in matches:
        if i not in grouped_a or j not in grouped_b:
            continue

        polygons_a = [safe_polygon(region) for region in grouped_a[i]]
        polygons_b = [safe_polygon(region) for region in grouped_b[j]]
        polygons_a = [p for p in polygons_a if p is not None]
        polygons_b = [p for p in polygons_b if p is not None]
        if not polygons_a or not polygons_b:
            continue

        poly_a = unary_union(polygons_a)
        poly_b = unary_union(polygons_b)
        inter = poly_a.intersection(poly_b)

        try:
            centroid_a = np.array(poly_a.centroid.xy).flatten()
            centroid_b = np.array(poly_b.centroid.xy).flatten()
            ax.scatter(*centroid_a, color="blue", marker="o", s=25)
            ax.scatter(*centroid_b, color="green", marker="x", s=25)
            ax.text(centroid_a[0]+0.1, centroid_a[1]+0.1, f"A{i}", color="blue", fontsize=8)
            ax.text(centroid_b[0]+0.1, centroid_b[1]+0.1, f"B{j}", color="green", fontsize=8)

            for poly, color, label in [(poly_a, "blue", f"A{i}"), (poly_b, "green", f"B{j}"), (inter, "purple", "Intersection")]:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.geom_type == "Polygon":
                    x, y = poly.exterior.xy
                    if len(x) >= 3:
                        ax.fill(x, y, color=color, alpha=0.15 if color != "purple" else 0.3, zorder=1)
                        ax.plot(x, y, color[0] + "--", linewidth=1, zorder=2)
                elif poly.geom_type == "MultiPolygon":
                    for subpoly in poly.geoms:
                        if not subpoly.is_valid:
                            subpoly = subpoly.buffer(0)
                        x, y = subpoly.exterior.xy
                        if len(x) >= 3:
                            ax.fill(x, y, color=color, alpha=0.15 if color != "purple" else 0.3, zorder=1)
                            ax.plot(x, y, color[0] + "--", linewidth=1, zorder=2)

        except Exception as e:
            print(f"[Plot Error for pair ({i}, {j})]: {e}")

    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Complete pipeline for matching the planes of 2 Z-stack objects
def match_zstacks_2d(zstack_a : ZStack, zstack_b : ZStack, 
                  match_params = None):
    # Copy defaults
    params = create_param_dict(DEFAULT_2D_MATCH_PARAMS, match_params)

    print(params)

    plane_gen_params = create_param_dict(PLANE_GEN_PARAMS_DEFAULT, params["plane_gen_params"])
    plane_gen_params["z_range"] = params["plane_list_params"]["z_range"] # Inherit z-range from plane list params

    if params["stack_a_boundary"]: # Update custom image boundaries for plane A if they are uneven
            plane_gen_params["plane_boundaries"] = params["stack_a_boundary"]
    # Insert plane-a specific parameters
    plane_gen_params["z_guess"] = params["plane_list_params"]["z_guess_a"]
    plane_gen_params["read_filename"] = params["planes_a_read_file"]
    plane_gen_params["save_filename"] = params["planes_a_write_file"]
    print(f"Generating A planes")
    if params["use_gpu"]:
        planes_a = zstack_a.generate_planes_gpu(plane_gen_params)
    else:
        planes_a = zstack_a.generate_planes(plane_gen_params)

    if params["stack_b_boundary"]: # Update custom image boundaries for plane B if they are uneven
        plane_gen_params["plane_boundaries"] = params["stack_b_boundary"]
    # Insert plane-b specific parameters
    plane_gen_params["z_guess"] = params["plane_list_params"]["z_guess_b"]
    plane_gen_params["read_filename"] = params["planes_b_read_file"]
    plane_gen_params["save_filename"] = params["planes_b_write_file"]
    print(f"Generating B planes")
    if params["use_gpu"]:
        planes_b = zstack_b.generate_planes_gpu(plane_gen_params)
    else:
        planes_b = zstack_b.generate_planes(plane_gen_params)

    # Perform the grid-search matching between generated planes
    print(f"[DEBUG] Generated/loaded {len(planes_a)} planes in A")
    print(f"[DEBUG] Generated/loaded {len(planes_b)} planes in B")
    print(f"Beginning Matching")
    matched_planes = Plane.match_plane_lists(planes_a, planes_b, plane_list_params=params["plane_list_params"], match_plane_params=params["match_plane_params"])

    if len(matched_planes) > 0:
        print(f"Best plane match score: {list(matched_planes.keys())[0]}")

    if params["match_planes_only"]:
        return matched_planes

    # Determine required 2D transformations per plane match
    unique_transformations = []

    for plane_match in tqdm(matched_planes.values(), desc="Calculating 2D transforms"):
        plane_a = plane_match["plane_a"]
        plane_b = plane_match["plane_b"]
        match_data = plane_match["result"]

        _, _, transform_info = plane_a.get_aligned_2d_projection(
            plane_b,
            offset_deg=match_data["offset"],
            scale_factor=match_data["scale_factor"]
        )

        rotation_2d = transform_info["rotation_deg"]
        scale_2d = transform_info["scale"]
        translation_2d = transform_info["translation"]

        unique_transformations.append((rotation_2d, scale_2d, translation_2d[0], translation_2d[1], match_data, plane_a, plane_b))

    # Calculate UoIs of all unique 2D transformations
    UoIs = []
    uoi_match_data = [] # list of data relating to plane matches by UoI

    # Project and transform points on both planes for each transformation identified
    for rotation_2d, scale_2d, tx, ty, match_data, plane_a, plane_b in tqdm(unique_transformations, desc=f"Evaluating transforms"):
        # Determine whether either plane is flat
        flat_plane_a = np.allclose(np.abs(plane_a.normal), [0, 0, 1], atol=1e-6)
        flat_plane_b = np.allclose(np.abs(plane_b.normal), [0, 0, 1], atol=1e-6)
        
        translation_2d = (tx, ty)

        # Get dicts of {idx : [x,y,z]}
        pid_mapping_a = {}
        pid_mapping_b = {}

        if flat_plane_a:
            rois_a_3d = project_flat_plane_points(zstack_a, plane_a.anchor_point)
        else:
            rois_a_3d, pid_mapping_a, _ = project_angled_plane_points(zstack_a, plane_a, threshold=params["plane_gen_params"]["projection_dist_thresh"], **params["seg_params"])
        if flat_plane_b:
            rois_b_3d = project_flat_plane_points(zstack_b, plane_b.anchor_point)
        else:
            rois_b_3d, pid_mapping_b, _ = project_angled_plane_points(zstack_b, plane_b, threshold=params["plane_gen_params"]["projection_dist_thresh"], **params["seg_params"])
        
        # Filtering for angled planes
        filter_params = params["filter_params"]
        if not filter_params["disable_filtering"]:
            print(f"Filtering B Regions")
            rois_b_3d = filter_region_by_shape(
                rois_b_3d,
                plane_b,
                min_area=filter_params["min_area"],
                max_area=filter_params["max_area"],
                max_eccentricity=filter_params["max_eccentricity"],
                preserve_anchor_regions=filter_params["preserve_anchor"]
            )
            print(f"Filtering A Regions")
            rois_a_3d = filter_region_by_shape(
                    rois_a_3d,
                    plane_a,
                    min_area=filter_params["min_area"],
                    max_area=filter_params["max_area"],
                    max_eccentricity=filter_params["max_eccentricity"],
                    preserve_anchor_regions=filter_params["preserve_anchor"]
            )

        matches = match_data["og_matches"]
        updated_matches = []
        for a, b in matches:
            new_a = pid_mapping_a.get(a, a)  # if mapping exists, remap
            new_b = pid_mapping_b.get(b, b)
            updated_matches.append((new_a, new_b))

        rois_a_2d_proj = {}
        rois_b_2d_proj = {}

        # Prepare for angled plane refinement step
        plane_a_refined = plane_a
        plane_b_refined = plane_b

        # DEBUG attempt to correct angled 2d mismatch
        if not (flat_plane_a and flat_plane_b) and params["rematch_angled_planes"]:
            # print("Recalculating 2D Plane match for angled plane")
            # Angled plane (realistically only plane a) was used - recalculate more precise 2d transformations
            def compute_centroids(rois_dict):
                return {pid: (np.min(pts, axis=0) + np.max(pts, axis=0)) / 2.0 for pid, pts in rois_dict.items()}
            
            def make_plane(centroid_dict, zstack, original_plane, label="A"):
                if len(centroid_dict) < 3: # Must have at least 3 points to make a new plane
                    return original_plane
                anchor_id = original_plane.anchor_point.id
                if anchor_id not in centroid_dict: # anchor not projected 
                    anchor_id = next(iter(centroid_dict)) # pick any roi in centroid_dict as the new anchor

                anchor_z, _ = anchor_id

                anchor_pos = centroid_dict[anchor_id]
                anchor_point = zstack.construct_planepoint(anchor_id, anchor_pos, idx_z = anchor_z)

                align_pts = []
                for pid, centroid_3d in centroid_dict.items():
                    if pid == anchor_id:
                        continue
                    try:
                        z, _ = pid # TODO traits go missing here if segmented by volume label - alignment pts dont help this step
                        pp = zstack.construct_planepoint(pid, centroid_3d, idx_z = z)
                        align_pts.append(pp)
                    except Exception as e:
                        print(f"[{label}] Warning for PID {pid}: {e}")

                return Plane(anchor_point, align_pts, max_alignments=params["plane_gen_params"]["max_alignments"], fixed_basis=params["plane_gen_params"]["fixed_basis"])

            # Make planes
            centroids_a = compute_centroids(rois_a_3d)
            centroids_b = compute_centroids(rois_b_3d)
            plane_a_refined = make_plane(centroids_a, zstack_a, plane_a, label="A") if not flat_plane_a else plane_a
            plane_b_refined = make_plane(centroids_b, zstack_b, plane_b, label="B") if not flat_plane_b else plane_b

            # Match planes
            match_data = plane_a_refined.match_planes(plane_b_refined, 
                                    match_plane_params=params["match_plane_params"])
            # print(f"New match data:\n{match_data}")
            if match_data["match"]:
                # Get aligned 2d projection
                _, _, transform_info = plane_a_refined.get_aligned_2d_projection(
                    plane_b_refined,
                    offset_deg=match_data["offset"],
                    scale_factor=match_data["scale_factor"]
                )

                rotation_2d = transform_info["rotation_deg"]
                scale_2d = transform_info["scale"]
                translation_2d = transform_info["translation"]
                # print(f"[Refinement] Updated alignment")
            else:
                print("[Refinement] Plane match failed, falling back to previous transform")
        # END DEBUG

        # Iterate through A rois and project to 2D plane
        for idx, coords_3d in rois_a_3d.items():
            rois_a_2d_proj[idx] = np.array([plane_a_refined._project_point_2d(pt) for pt in coords_3d])

        # Iterate through B rois and project to 2D plane and transform
        for idx, coords_3d in rois_b_3d.items():
            rois_b_2d_proj[idx] = np.array(
                plane_a_refined.project_and_transform_points(coords_3d, plane_b_refined, rotation_deg=rotation_2d, scale=scale_2d, translation=translation_2d)
            )

        # TEST DEBUG
        # print(f"Number of rois on angled plane A: {len(rois_a_2d)}, num rois on flat plane a: {len(rois_a_2d_proj)}")
        # plt.figure(figsize=(8, 8))
        # for idx, points in rois_a_2d_proj.items():
        #     if len(points) == 0:
        #         continue
        #     x_vals, y_vals = zip(*points)
        #     plt.scatter(x_vals, y_vals, label=f"ROI {idx}", s=5)
        # plt.gca().set_aspect('equal')
        # plt.title("test")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.legend(loc="upper right", fontsize='x-small', ncol=2)
        # plt.grid(True)
        # plt.tight_layout()

        # plot_projected_regions_with_plane_points(zstack_a, plane_a)
        # END TEST DEBUG

        # # TEST DEBUG
        # expected_points = []
        # projected_points = []
        # for z, roi_dict in zstack_b.z_planes.items():
        #     for roi_id, roi_data in roi_dict.items():
        #         key = roi_id
        #         coords_3d = [(x, y, z) for x, y in roi_data["coords"]]

        #         # Check if ROI exists in rois_b_2d
        #         if key not in rois_b_2d_proj:
        #             print(f"[MISSING] ROI {key} not found in rois_b_2d")
        #             continue
        #         else:
        #             print(f"[FOUND] ROI {key} found in rois_b_2d")


        #         # Re-project coords_3d and compare
        #         coords_2d_expected = np.array(plane_a.project_and_transform_points(
        #             coords_3d, plane_b, rotation_deg=rotation_2d, scale=scale_2d, translation=translation_2d
        #         ))

        #         coords_2d_projected = np.array(rois_b_2d_proj[key])[:, :2]
                
        #         expected_points.append(coords_2d_expected)           # shape (N, 2)
        #         projected_points.append(coords_2d_projected)         # shape (N, 2)


        # plt.figure(figsize=(8, 8))
        # plt.title("All ROI Points – Before vs After Transformation")
        # plt.axis("equal")
        # plt.grid(True)

        # # Stack all into single arrays
        # expected_points_np = np.vstack(expected_points)
        # projected_points_np = np.vstack(projected_points)

        # # Plot recomputed (expected) transformed projections
        # plt.scatter(expected_points_np[:, 0], expected_points_np[:, 1], color='blue', alpha=0.6, s=10, label="Expected (Recomputed)")

        # # Plot stored projections (from rois_b_2d)
        # plt.scatter(projected_points_np[:, 0], projected_points_np[:, 1], color='red', alpha=0.6, s=10, label="Stored (rois_b_2d)")

        # plt.legend()
        # plt.tight_layout()
        # plt.show()        
        # # END TEST DEBUG


        # Calculate UoI for all unique transformations
        regions_a = {pid : BoundaryRegion([(x, y, 0) for x,y in rois_a_2d_proj[pid]]) for pid in rois_a_2d_proj}
        regions_b = {pid : BoundaryRegion([(x, y, 0) for x,y in rois_b_2d_proj[pid]]) for pid in rois_b_2d_proj}

        # TEST DEBUG
        # plot_regions_and_alignment_points(regions_a, plane_a, title="Projected Regions A")
        # plot_regions_and_alignment_points(regions_b, plane_b, title="Projected Regions B")
        # END TEST DEBUG

        # Filter matches to only those where both PIDs survived projection
        filtered_matches = [
            (i, j) for (i, j) in updated_matches
            if i in regions_a and j in regions_b
        ]

        UoI = compute_avg_uoi(regions_a, regions_b, filtered_matches, params["min_uoi"], plot=params["plot_uoi"])
        if UoI >= params["min_uoi"]:
            UoIs.append(UoI)
            uoi_match_data.append({"plane_a" : plane_a,
                                   "plane_b": plane_b,
                                   "rotation_2d": rotation_2d,
                                   "scale_2d": scale_2d,
                                   "translation_2d" : (tx, ty),
                                   "UoI": UoI})


    # Calculate best UoI
    if len(UoIs) > 0:
        best_UoI = max(UoIs)
        best_idx = UoIs.index(best_UoI)
        best_transformation = list(unique_transformations)[best_idx]
        best_transformation = best_transformation[0:4] # extract only the operations
        print(f"Best UoI: {best_UoI}, Best Transformation: {best_transformation}")
    else:
        print(f"No plane matches found with UoI above min: {params['min_uoi']}")

    # Sort by descending UoI
    sorted_data = sorted(uoi_match_data, key=lambda x: x["UoI"], reverse=True)

    # Plot matches if required
    if params["plot_match"]:
        plot_uoi_match_data(sorted_data, zstack_a, zstack_b)
    
    return sorted_data, matched_planes

# Test
from matplotlib.patches import Polygon as MplPolygon
def plot_projected_rois(rois_a, rois_b, title="Projected Regions A vs B"):
    fig, ax = plt.subplots(figsize=(8, 10))
    colors = {}

    # Plot A regions
    for rid, pts in rois_a.items():
        if len(pts) < 3:
            continue
        # Extract XY only
        pts_2d = [(pt[0], pt[1]) for pt in pts if len(pt) >= 2]
        color = colors.get(rid, plt.cm.tab20(random.randint(0, 19)))
        colors[rid] = color
        poly = MplPolygon(pts_2d, closed=True, edgecolor=color, fill=False, linewidth=1.5, label=f"A {rid}")
        ax.add_patch(poly)

    # Plot B regions
    for rid, pts in rois_b.items():
        if len(pts) < 3:
            continue
        pts_2d = [(pt[0], pt[1]) for pt in pts if len(pt) >= 2]
        color = colors.get(rid, plt.cm.tab20b(random.randint(0, 19)))
        colors[rid] = color
        poly = MplPolygon(pts_2d, closed=True, edgecolor=color, fill=False, linestyle='--', linewidth=1.2, label=f"B {rid}")
        ax.add_patch(poly)

    ax.set_aspect('equal')
    ax.set_title(title)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    plt.show()

def plot_uoi_match_data(sorted_data, zstack_a, zstack_b, threshold=0.5, seg_params=None):
    """
    Plots the projected 2D ROIs of plane A and transformed plane B,
    correctly handling both flat and angled planes.
    """
    if seg_params is None:
        seg_params = {"method": "volume", "eps": 5.0, "min_samples": 5}

    for entry in sorted_data:
        plane_a = entry["plane_a"]
        plane_b = entry["plane_b"]
        rotation_2d = entry["rotation_2d"]
        scale_2d = entry["scale_2d"]
        tx, ty = entry["translation_2d"]
        uoi = entry["UoI"]

        # Check plane types
        flat_plane_a = np.allclose(np.abs(plane_a.normal), [0, 0, 1], atol=1e-6)
        flat_plane_b = np.allclose(np.abs(plane_b.normal), [0, 0, 1], atol=1e-6)

        # Project ROIs to 2D
        if flat_plane_a:
            rois_a_2d = project_flat_plane_points(zstack_a, plane_a.anchor_point)
        else:
            rois_a_2d, _, _ = project_angled_plane_points(zstack_a, plane_a, threshold=threshold, **seg_params)

        if flat_plane_b:
            rois_b_2d = project_flat_plane_points(zstack_b, plane_b.anchor_point)
        else:
            rois_b_2d, _, _ = project_angled_plane_points(zstack_b, plane_b, threshold=threshold, **seg_params)

        # Project each 3D ROI onto plane A’s 2D space
        rois_a_2d_proj = {
            pid: np.array([plane_a._project_point_2d(np.array(pt)) for pt in coords])
            for pid, coords in rois_a_2d.items()
        }

        # Project + transform plane B ROIs into plane A's 2D space
        rois_b_2d_proj = {
            pid: np.array(plane_a.project_and_transform_points(coords, plane_b, rotation_deg=rotation_2d,
                                                               scale=scale_2d, translation=(tx, ty)))
            for pid, coords in rois_b_2d.items()
        }

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"Projected ROIs A and B (UoI={uoi:.2f})")
        ax.set_aspect('equal')

        for pid, points in rois_a_2d_proj.items():
            if len(points) > 0:
                ax.plot(points[:, 0], points[:, 1], 'o', markersize=2, color='blue')
                centroid_x = np.min(points[:, 0]) + (np.max(points[:, 0]) - np.min(points[:, 0])) / 2
                centroid_y = np.min(points[:, 1]) + (np.max(points[:, 1]) - np.min(points[:, 1])) / 2
                ax.text(centroid_x, centroid_y, f"A{pid}", fontsize=8, color='blue')

        for pid, points in rois_b_2d_proj.items():
            if len(points) > 0:
                ax.plot(points[:, 0], points[:, 1], 'x', markersize=2, color='green')
                centroid_x = np.min(points[:, 0]) + (np.max(points[:, 0]) - np.min(points[:, 0])) / 2
                centroid_y = np.min(points[:, 1]) + (np.max(points[:, 1]) - np.min(points[:, 1])) / 2
                ax.text(centroid_x, centroid_y, f"B{pid}", fontsize=8, color='green')

        ax.grid(True)
        plt.tight_layout()
        plt.show()


def project_flat_plane_points(z_stack, anchor_point):
    """
    Projects points for a flat plane (normal close to (0,0,1)).
    
    Args:
        z_stack: The ZStack object containing all ROIs.
        anchor_point: The PlanePoint anchor (must have z info).

    Returns:
        Dictionary { pid : [(x,y,z)] } mapping ROI ids to points enclosing boundary regions.
    """
    z_level = int(round(anchor_point.position[2]))  # Round to nearest int z-plane
    if z_level not in z_stack.z_planes:
        raise ValueError(f"Z level {z_level} not found in z-stack.")

    rois_on_z = z_stack.z_planes[z_level]

    boundary_regions = {}
    for roi_id, roi_data in rois_on_z.items():
        coords = roi_data["coords"]
        points_3d = [(x, y, z_level) for (x, y) in coords]
        boundary_regions[roi_id] = points_3d

    return boundary_regions

def project_angled_plane_points(
    z_stack, angled_plane: Plane,
    threshold=0.5, method="volume", eps=5.0, min_samples=5, 
):
    """
    Projects points onto an angled plane, clusters them into ROIs, 
    and assigns anchor/alignment points either via splitting (default) or merging.

    Args:
        method: "split" (default) or "merge"

    Returns:
        output_regions: {pid: [(x, y, z)]}
        pid_mapping: {old_pid: new_pid}  # (empty dict if method == "split")
    """
    # Project ROI points
    projected_pts_2d = []
    original_pts_3d = []
    volume_ids = []

    for z in z_stack.z_planes:
        for roi_id, roi_data in z_stack.z_planes[z].items():
            coords = roi_data["coords"]
            vol_id = int(roi_data.get("volume", -1))
            for (x, y) in coords:
                pt3d = np.array([x, y, z])
                if angled_plane._distance_to_plane(pt3d) < threshold:
                    proj2d = angled_plane._project_point_2d(pt3d)
                    projected_pts_2d.append(proj2d)
                    original_pts_3d.append((x, y, z))
                    volume_ids.append(vol_id)

    if not projected_pts_2d:
        return {}, {}, {}

    projected_pts_2d = np.array(projected_pts_2d)

    if method == "volume" and z_stack.has_volume:
        # Group by volume ID
        clusters = defaultdict(list)
        for i, vol_id in enumerate(volume_ids):
            clusters[vol_id].append((projected_pts_2d[i], original_pts_3d[i]))

        # Get all pts (anchor/alignment) on the plane and project them into 2D
        plane_anchor_id = angled_plane.anchor_point.id
        projected_anchors = {
            pt.id: angled_plane._project_point_2d(pt.position)
            for pt in angled_plane.plane_points.values()
        }
        output_regions = {}
        output_volumes = {}
        pid_mapping = {}
        # Iterate through all ROI clusters (clustered by volume id)
        for vol_id, points in clusters.items():
            cluster_pts_2d = np.array([pt2d for pt2d, _ in points])
            cluster_polygon = None
            # Make each ROI into a shapely polygon
            if len(cluster_pts_2d) >= 3:
                # Remove duplicate points first
                cluster_pts_2d_unique = np.unique(cluster_pts_2d, axis=0)
                if cluster_pts_2d_unique.shape[0] >= 3 and np.linalg.matrix_rank(cluster_pts_2d_unique - cluster_pts_2d_unique[0]) >= 2:
                    try:
                        hull = ConvexHull(cluster_pts_2d_unique)
                        cluster_polygon = Polygon(cluster_pts_2d_unique[hull.vertices])
                    except Exception as e:
                        print(f"[Warning] ConvexHull failed: {e}")
                        cluster_polygon = None

            # Determine which regions belong to anchors/alignment points - retains original plane point mapping
            anchors_inside = []
            if cluster_polygon:
                for pid, anchor_pt in projected_anchors.items():
                    point = Point(anchor_pt)
                    if cluster_polygon.contains(point):
                        anchors_inside.append(pid)

            if len(anchors_inside) == 1: # A single anchor/alignment point is inside this ROI - the output region gets the ROI id of that point
                pid = anchors_inside[0]
                output_regions[pid] = [pt3d for _, pt3d in points]
                output_volumes[pid] = vol_id
            elif len(anchors_inside) > 1:
                # Pick first as master, map rest to it
                master_pid = anchors_inside[0] if plane_anchor_id not in anchors_inside else plane_anchor_id # Prioritize keeping anchor pt id
                output_regions[master_pid] = [pt3d for _, pt3d in points]
                output_volumes[master_pid] = vol_id

                # Map all other pids to master
                for pid in anchors_inside:
                    if pid != master_pid:
                        pid_mapping[pid] = master_pid
            else: # No anchor/alignment points inside
                region_id = (-1, vol_id)
                output_regions[region_id] = [pt3d for pt2d, pt3d in points]
                output_volumes[region_id] = vol_id

        return output_regions, pid_mapping, output_volumes
    elif method == "volume": # Default to split method if volume labels not in z-stack
        method = "split"

    # Cluster projected points
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(projected_pts_2d)
    labels = clustering.labels_

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # Skip noise
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((projected_pts_2d[idx], original_pts_3d[idx]))

    # Project anchor/alignment points
    projected_anchors = {}
    for _, plane_point in angled_plane.plane_points.items():
        proj2d = angled_plane._project_point_2d(plane_point.position)
        projected_anchors[plane_point.id] = proj2d

    # Build final regions
    output_regions = {}
    assigned_pids = set()
    pid_mapping = {}
    next_pid = max(projected_anchors.keys()) + 1 # used to identify new regions

    for label, points in clusters.items():
        cluster_pts_2d = np.array([pt2d for pt2d, pt3d in points])

        if len(cluster_pts_2d) >= 3:
            if cluster_pts_2d.shape[0] < 3 or np.linalg.matrix_rank(cluster_pts_2d - cluster_pts_2d[0]) < 2:
                # Not enough distinct points or not full rank -> skip
                continue  
            hull = ConvexHull(cluster_pts_2d)

            cluster_polygon = Polygon(cluster_pts_2d[hull.vertices])
        else:
            cluster_polygon = None

        # Find anchors inside cluster
        anchors_inside = []
        if cluster_polygon is not None:
            for pid, anchor_pt in projected_anchors.items():
                point = Point(anchor_pt)
                if cluster_polygon.contains(point):
                    anchors_inside.append(pid)

        if len(anchors_inside) == 0: # new cluster of previously unprojected pts
            output_regions[next_pid] = [pt3d for pt2d, pt3d in points]
            next_pid += 1
            continue

        if len(anchors_inside) == 1:
            # Exactly one alignment point inside
            pid = anchors_inside[0]
            output_regions[pid] = [pt3d for pt2d, pt3d in points]
            assigned_pids.add(pid)
        else:
            # Multiple anchors inside -> split or merge
            if method == "split":
                # SPLIT
                anchor_pts = np.array([projected_anchors[pid] for pid in anchors_inside])
                kmeans = KMeans(
                    n_clusters=len(anchor_pts),
                    init=anchor_pts,
                    n_init=1,
                    max_iter=1,
                    random_state=42
                )
                cluster_pts_only = np.array([pt2d for pt2d, pt3d in points])
                labels_split = kmeans.fit_predict(cluster_pts_only)

                split_clusters = {pid: [] for pid in anchors_inside}
                for idx, split_label in enumerate(labels_split):
                    assigned_anchor = anchors_inside[split_label]
                    split_clusters[assigned_anchor].append(points[idx][1])  # Keep original 3D point

                for pid, pts3d in split_clusters.items():
                    output_regions[pid] = pts3d
                    assigned_pids.add(pid)

            elif method == "merge":
                # MERGE
                # Pick first PID as the master
                master_pid = anchors_inside[0]
                merged_pts = [pt3d for pt2d, pt3d in points]

                output_regions[master_pid] = merged_pts
                assigned_pids.add(master_pid)

                # Record that all other PIDs map to this master PID
                for pid in anchors_inside:
                    pid_mapping[pid] = master_pid

    return output_regions, pid_mapping, {}

# TEST DEBUG
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull
import numpy as np

def plot_regions_and_alignment_points(
    regions, 
    plane, 
    title="Regions + Alignment Points",
    transform=None,
    reference_plane=None
):
    """
    Plot boundary polygons of projected regions alongside projected anchor/alignment points.
    
    Args:
        regions: dict {pid : BoundaryRegion}
        plane: Plane object used for projection
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title)
    ax.set_aspect('equal')

    # Plot region boundaries
    for pid, region in regions.items():
        points = region.get_boundary_points()
        if len(points) < 3:
            continue  # Need at least 3 points to make a polygon

        pts2d = np.array([(x, y) for x, y, z in points])

        try:
            hull = ConvexHull(pts2d)
            hull_pts = pts2d[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])

            ax.plot(hull_pts[:, 0], hull_pts[:, 1], '-', label=f"Region {pid}")
            centroid_x = np.min(pts2d[:, 0]) + (np.max(pts2d[:, 0]) - np.min(pts2d[:, 0])) / 2
            centroid_y = np.min(pts2d[:, 1]) + (np.max(pts2d[:, 1]) - np.min(pts2d[:, 1])) / 2
            ax.scatter(centroid_x, centroid_y, color='black', s=10)
        except Exception as e:
            print(f"[plot_regions_and_alignment_points] Warning: Could not form polygon for region {pid}: {e}")
            continue

    # Now project alignment points and plot
    for _, plane_point in plane.plane_points.items():
        pos = plane_point.position

        if transform is not None and reference_plane is not None:
            # Apply the same transformation used on rois_b_2d
            transformed_2d = reference_plane.project_and_transform_points(
                [pos], plane, rotation_deg=transform[0], scale=transform[1], translation=transform[2]
            )[0]
            # Convert to 3D by appending z = 0.0
            pos = np.array([transformed_2d[0], transformed_2d[1], 0.0])

        proj2d = reference_plane._project_point_2d(pos) if reference_plane else plane._project_point_2d(pos)

        og_id = plane_point.id
        ax.scatter(proj2d[0], proj2d[1], color='red', s=30, marker='x')
        ax.text(proj2d[0] + 0.5, proj2d[1] + 0.5, f"{og_id}", color='red', fontsize=7)



    ax.grid(True)
    ax.legend(fontsize=6, loc='best')
    plt.tight_layout()
    plt.show()


def extract_zstack_plane(z_stack, plane, threshold=0.5, eps=5.0, min_samples=5, method="volume"):
    """
    Extracts all points near a plane from the ZStack and returns a new ZStack.
    If the plane is angled (not flat in Z), the function:
      - Projects points onto the plane
      - Clusters projected points into ROIs
      - Assigns them fixed z=0
    If the plane is flat, it behaves as before.

    Args:
        z_stack: The source ZStack
        plane: A Plane object
        threshold: Distance threshold to consider points "on" the plane
        eps: DBSCAN epsilon for projected clustering (angled mode)
        min_samples: DBSCAN min_samples (angled mode)
        method: "split" or "merge" for assigning anchor points in ambiguous clusters

    Returns:
        ZStack with a single plane of projected ROIs at z=0
    """
    print("[INFO] Extracting plane from z-stack")
    new_z_planes = defaultdict(lambda: defaultdict(dict))
    is_flat = np.allclose(plane.normal, [0, 0, 1])
    
    if is_flat: # Simply copy points from the original z-dict
        z = plane.anchor_point.position[2]
        # print(f"Anchor point z: {z}")
        for roi_id, roi_data in z_stack.z_planes[z].items():
            coords = roi_data["coords"]
            for (x, y) in coords:
                new_z_planes[z][roi_id].setdefault("coords", []).append((x, y))
            if "intensity" in roi_data:
                new_z_planes[z][roi_id]["intensity"] = roi_data["intensity"]
            if "volume" in roi_data:
                new_z_planes[z][roi_id]["volume"] = int(roi_data["volume"])
            if "traits" in roi_data:
                new_z_planes[z][roi_id]["traits"] = roi_data["traits"]
        new_stack = ZStack(new_z_planes)

        return new_stack

    # Angled plane: project + cluster
    output_regions, _, output_volumes = project_angled_plane_points(
        z_stack, plane, threshold=threshold, method=method, eps=eps, min_samples=min_samples
    ) 

    z_fixed = 0 # Set an arbitrary z for projected plane
    remap_rois = []
    for (z, roi_id), points in output_regions.items():
        if not points:
            continue
        coords_2d = [plane._project_point_2d(np.array([x, y, z])) for (x, y, z) in points]
        
        # Skip volume segmented rois for now
        if (z == -1) or roi_id in new_z_planes[z_fixed]:
            remap_rois.append((z, roi_id))
            continue

        new_z_planes[z_fixed][roi_id]["coords"] = coords_2d
        new_z_planes[z_fixed][roi_id]["volume"] = output_volumes[(z, roi_id)]

        # Just assign a random intensity -- TODO could be improved later
        new_z_planes[z_fixed][roi_id]["intensity"] =  random.uniform(0.8, 1)
    
    # Bandaid fix for volume segmented labels/duplicate roi ids
    existing_ids = list(new_z_planes[z_fixed].keys())
    new_id = max(existing_ids) + 1 if existing_ids else 0
    for z, roi_id in remap_rois:
        points = output_regions[(z, roi_id)]
        if not points:
            continue
        coords_2d = [plane._project_point_2d(np.array([x, y, z])) for (x, y, z) in points]
        new_z_planes[z_fixed][new_id]["coords"] = coords_2d
        new_z_planes[z_fixed][new_id]["intensity"] =  random.uniform(0.8, 1)
        new_z_planes[z_fixed][new_id]["volume"] = output_volumes[(z, roi_id)]
        new_id += 1

    # Apply fixed 2d offset to all points (fixes margin roi exclusion errors in plane generation)
    all_coords = [
        (x, y)
        for roi_data in new_z_planes[z_fixed].values()
        for (x, y) in roi_data["coords"]
    ]

    if all_coords: 
        min_x = min(x for x, y in all_coords)
        min_y = min(y for x, y in all_coords)
        x_offset = -min_x + 10  # shift into positive range + margin
        y_offset = -min_y + 10

        # Apply the offset to all ROIs
        for roi_data in new_z_planes[z_fixed].values():
            roi_data["coords"] = [
                (x + x_offset, y + y_offset) for (x, y) in roi_data["coords"]
            ]

    new_stack = ZStack(new_z_planes)

    print("[INFO] Successfully extracted plane from z-stack")
    return new_stack

def filter_region_by_shape(
    data,
    plane,
    min_area=None,
    max_area=None,
    max_eccentricity=None,
    preserve_anchor_regions=True
):
    """
    Filters either a ZStack or a dictionary of regions by shape, area, and overlap.

    Args:
        data: ZStack or output_regions dict from project_angled_plane_points
        plane: Plane object used to check anchor/alignment point IDs and projection
        min_area: Minimum ROI area
        max_area: Maximum ROI area
        max_eccentricity: Maximum allowed eccentricity (0=circle, 1=line)
        preserve_anchor_regions: If True, keeps any region with an anchor or alignment point

    Returns:
        Same type as input: filtered ZStack or filtered dict
    """

    def get_zstack_coords(z, rid):
        region = data.z_planes[z][rid]
        if isinstance(region, dict) and "coords" in region:
            return region["coords"]
        raise ValueError(f"[ERROR] Missing 'coords' for z={z}, rid={rid}")
    
    def get_ordered_boundary(coords_2d):
        if len(coords_2d) < 3:
            return coords_2d
        try:
            hull = ConvexHull(coords_2d)
            return [coords_2d[i] for i in hull.vertices]
        except:
            return coords_2d

    is_zstack = isinstance(data, ZStack) # Detect if input is a ZStack or region dict
    if is_zstack:
        data._build_xy_rois()
        region_items = list(data.xy_rois.keys())
        get_coords = lambda z, rid: get_zstack_coords(z, rid)
        get_z = lambda z, rid: z
    else:
        region_items = list(data.keys())
        get_coords = lambda _, rid: data[rid]
        get_z = lambda _, rid: np.median([pt[2] for pt in data[rid]])

    region_data = []

    for key in region_items:
        z, rid = key if is_zstack else (None, key)
        coords = get_coords(z, rid)
        if len(coords) < 5:
            continue

        # Ensure coords are 2D
        if len(coords[0]) == 3:
            coords_2d = [(x, y) for x, y, _ in coords]
        else:
            coords_2d = coords

        coords_2d = get_ordered_boundary(coords_2d)
        if len(coords_2d) < 5:
            # print(f"Skipped low point roi: {rid}")
            continue

        x, y = zip(*coords_2d)
        x = np.array(x)
        y = np.array(y)

        poly = Polygon(coords_2d)
        if not poly.is_valid or poly.area == 0:
            # print(f"Skipped invalid poly roi: {rid}")
            continue

        area = poly.area
        is_anchor = any(
            (pt.id == rid and np.isclose(get_z(z, rid), pt.position[2]))
            for pt in plane.plane_points.values()
        )
        # if is_anchor:
        #     # print(f"Found anchor/alignment point ID: {rid}")

        if min_area is not None and area < min_area and not (preserve_anchor_regions and is_anchor):
            # print(f"Skipped small area roi: {rid}")
            continue
        if max_area is not None and area > max_area and not (preserve_anchor_regions and is_anchor):
            # print(f"Skipped high area roi: {rid}")
            continue

        # print(f"ROI ID:{key} passed area check with area = {area}")

        # Shape filtering
        model = EllipseModel()
        if not model.estimate(np.column_stack([x, y])) and not is_anchor:
            # print(f"Failed to estimate ellipsemodel for roi: {rid}")
            continue

        _, _, a_axis, b_axis, _ = model.params
        a, b = sorted([a_axis, b_axis])  # Ensure a ≤ b
        if b == 0 and not is_anchor:
            # print(f"Failed to calculate eccentricity for roi: {rid}")
            continue
        eccentricity = np.sqrt(1 - (a ** 2) / (b ** 2))

        if max_eccentricity is not None and eccentricity > max_eccentricity and not (preserve_anchor_regions and is_anchor):
            continue

        region_data.append({
            "rid": rid,
            "z": z,
            "coords": coords,
            "polygon": poly,
            "area": area,
            "is_anchor": is_anchor
        })

    # Overlap filtering
    kept_ids = set()
    kept_regions = []
    region_data.sort(key=lambda r: -r["area"])  # Sort by area descending to help prioritization

    for i, reg_i in enumerate(region_data):
        poly_i = reg_i["polygon"]
        rid_i = reg_i["rid"]
        is_anchor = reg_i["is_anchor"]
        if rid_i in kept_ids:
            continue
        overlaps_existing = any(poly_i.intersects(reg_j["polygon"]) for reg_j in kept_regions)
        if overlaps_existing and not is_anchor:
            # print(f"Skipping overlapped roi: {rid_i}")
            continue  # Skip reg_i if it overlaps anything already accepted
        keep = True
        for j in range(i + 1, len(region_data)):
            reg_j = region_data[j]
            if reg_j["rid"] in kept_ids:
                continue

            if reg_i["polygon"].intersects(reg_j["polygon"]):
                # Decide which to keep -> Filter by anchor pt inside > area > distance to plane
                if reg_i["is_anchor"] and not reg_j["is_anchor"]:
                    kept_ids.add(reg_i["rid"])
                    kept_regions.append(reg_i)
                    continue
                elif reg_j["is_anchor"] and not reg_i["is_anchor"]:
                    keep = False
                    break
                elif reg_j["is_anchor"] and reg_i["is_anchor"] and preserve_anchor_regions: # Keep both and allow overlap if both anchor regions
                    keep = True
                    break

                # Keep the roi with the highest area
                keep = True if region_data[i]["area"] >= region_data[j]["area"] else False
        if keep:
            kept_ids.add(reg_i["rid"])
            kept_regions.append(reg_i)

    # Build output
    if is_zstack:
        new_zplanes = defaultdict(lambda: defaultdict(dict))
        for reg in region_data:
            if reg["rid"] not in kept_ids:
                continue
            original = data.z_planes[reg["z"]][reg["rid"]]
            new_zplanes[reg["z"]][reg["rid"]]["coords"] = original["coords"]
            if "intensity" in original:
                new_zplanes[reg["z"]][reg["rid"]]["intensity"] = original["intensity"]
            if "volume" in original:
                new_zplanes[reg["z"]][reg["rid"]]["volume"] = int(original["volume"])
        return ZStack(new_zplanes)
    else:
        return {rid: data[rid] for rid in kept_ids}
    

    # TEST DEBUG
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

    # END TEST DEBUG