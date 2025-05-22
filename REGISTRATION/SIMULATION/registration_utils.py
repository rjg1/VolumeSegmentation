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
from planepoint import PlanePoint
from scipy.spatial.distance import cdist
from collections import defaultdict
import copy
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import ConvexHull
import random


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
                poly_a = Polygon(regions_a[a].get_boundary_points())
                poly_b = Polygon(regions_b[b].get_boundary_points())

                if not poly_a.is_valid:
                    poly_a = poly_a.buffer(0)
                if not poly_b.is_valid:
                    poly_b = poly_b.buffer(0)

                if not poly_a.is_valid or not poly_b.is_valid:
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

    # Init plot
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Matched ROIs with UoI Shading")
        ax.set_aspect('equal')

    uoi_scores = []

    # Loop over all matches (original and new)
    all_matches = matches + new_matches

    for i, j in all_matches:
        if i not in grouped_a or j not in grouped_b:
            continue

        # Merge polygons
        poly_a = unary_union([
            Polygon(region.get_boundary_points()) for region in grouped_a[i]
        ])
        if not poly_a.is_valid:
            poly_a = poly_a.buffer(0)

        poly_b = unary_union([
            Polygon(region.get_boundary_points()) for region in grouped_b[j]
        ])
        if not poly_b.is_valid:
            poly_b = poly_b.buffer(0)

        if not poly_a.is_valid or not poly_b.is_valid:
            print(f"Skipping invalid merged match ({i}, {j})")
            continue

        # Compute intersection & union
        inter = poly_a.intersection(poly_b)
        union = poly_a.union(poly_b)
        uoi = inter.area / union.area if union.area > 0 else 0
        uoi_scores.append(uoi)

        if plot:
            centroid_a = np.array(poly_a.centroid.xy).flatten()
            centroid_b = np.array(poly_b.centroid.xy).flatten()

            ax.scatter(*centroid_a, color="blue", marker="o", s=25)
            ax.scatter(*centroid_b, color="green", marker="x", s=25)

            ax.text(centroid_a[0]+0.1, centroid_a[1]+0.1, f"A{i}", color="blue", fontsize=8)
            ax.text(centroid_b[0]+0.1, centroid_b[1]+0.1, f"B{j}", color="green", fontsize=8)

            # Plot Region A
            if not poly_a.is_empty:
                try:
                    if poly_a.geom_type == "Polygon":
                        x, y = poly_a.exterior.xy
                        ax.fill(x, y, color="blue", alpha=0.15, label="Region A" if i == matches[0][0] else "")
                    elif poly_a.geom_type == "MultiPolygon":
                        for subpoly in poly_a.geoms:
                            x, y = subpoly.exterior.xy
                            ax.fill(x, y, color="blue", alpha=0.15)
                except:
                    pass

            # Plot Region B
            if not poly_b.is_empty:
                try:
                    if poly_b.geom_type == "Polygon":
                        x, y = poly_b.exterior.xy
                        ax.fill(x, y, color="green", alpha=0.15, label="Region B" if i == matches[0][0] else "")
                    elif poly_b.geom_type == "MultiPolygon":
                        for subpoly in poly_b.geoms:
                            x, y = subpoly.exterior.xy
                            ax.fill(x, y, color="green", alpha=0.15)
                except:
                    pass

            # Plot Intersection
            if not inter.is_empty:
                try:
                    if inter.geom_type == "Polygon":
                        x, y = inter.exterior.xy
                        ax.fill(x, y, color="purple", alpha=0.3, label="Intersection" if i == matches[0][0] else "")
                    elif inter.geom_type == "MultiPolygon":
                        for subpoly in inter.geoms:
                            x, y = subpoly.exterior.xy
                            ax.fill(x, y, color="purple", alpha=0.3)
                except:
                    pass

    uoi = sum(uoi_scores) / len(uoi_scores) if uoi_scores else 0.0

    if plot and uoi > min_uoi:
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return uoi

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
    if params["use_gpu"]:
        planes_b = zstack_b.generate_planes_gpu(plane_gen_params)
    else:
        planes_b = zstack_b.generate_planes(plane_gen_params)

    # Perform the grid-search matching between generated planes
    print(f"[DEBUG] Generated {len(planes_a)} planes in A")
    print(f"[DEBUG] Generated {len(planes_b)} planes in B")
    print(f"Beginning Matching")
    matched_planes = Plane.match_plane_lists(planes_a, planes_b, plane_list_params=params["plane_list_params"], match_plane_params=params["match_plane_params"])

    if len(matched_planes) > 0:
        print(f"Best plane match score: {list(matched_planes.keys())[0]}")

    # Stores just (rounded) values for uniqueness checking
    seen_transformations = set()

    # Stores full records including planes
    unique_transformations = []

    # DEBUG
    seen_transformations_count = {}

    for match in matched_planes.values():
        plane_a = match["plane_a"]
        plane_b = match["plane_b"]
        match_data = match["result"]

        proj_a, proj_b_aligned, transform_info = plane_a.get_aligned_2d_projection(
            plane_b,
            offset_deg=match_data["offset"],
            scale_factor=match_data["scale_factor"]
        )

        rotation_2d = transform_info["rotation_deg"]
        scale_2d = transform_info["scale"]
        translation_2d = transform_info["translation"]

        rounded = (
            round(rotation_2d, 2),
            round(scale_2d, 2),
            round(translation_2d[0], 2),
            round(translation_2d[1], 2),
        )

        if rounded not in seen_transformations:
            seen_transformations.add(rounded)
            unique_transformations.append((rotation_2d, scale_2d, translation_2d[0], translation_2d[1], match_data, plane_a, plane_b))
            seen_transformations_count[rounded] = 1
        else:
            seen_transformations_count[rounded] += 1

    print(f"Unique transformations found (rot, scale, t_x, t_y): {[(rot, scale, tx, ty) for rot, scale, tx, ty, _, _, _ in unique_transformations]}")

    print(seen_transformations_count)

    # Calculate UoIs of all unique 2D transformations
    UoIs = []
    uoi_match_data = [] # list of data relating to plane matches by UoI

    # Project and transform points on both planes for each transformation identified
    for rotation_2d, scale_2d, tx, ty, match_data, plane_a, plane_b in unique_transformations:
        # Determine whether either plane is flat
        flat_plane_a = np.allclose(np.abs(plane_a.normal), [0, 0, 1], atol=0.05)
        flat_plane_b = np.allclose(np.abs(plane_b.normal), [0, 0, 1], atol=0.05)
        translation_2d = (tx, ty)

        # Get dicts of {idx : [x,y,z]}
        pid_mapping_a = {}
        pid_mapping_b = {}

        if flat_plane_a:
            rois_a_2d = project_flat_plane_points(zstack_a, plane_a.anchor_point)
        else:
            rois_a_2d, pid_mapping_a = project_angled_plane_points(zstack_a, plane_a, threshold=params["plane_gen_params"]["projection_dist_thresh"], **params["seg_params"])

        if flat_plane_b:
            rois_b_2d = project_flat_plane_points(zstack_b, plane_b.anchor_point)
        else:
            rois_b_2d, pid_mapping_b = project_angled_plane_points(zstack_b, plane_b, threshold=params["plane_gen_params"]["projection_dist_thresh"], **params["seg_params"])

        matches = match_data["og_matches"]
        updated_matches = []
        for a, b in matches:
            new_a = pid_mapping_a.get(a, a)  # if mapping exists, remap
            new_b = pid_mapping_b.get(b, b)
            updated_matches.append((new_a, new_b))

        rois_a_2d_proj = {}
        rois_b_2d_proj = {}
        # Iterate through A rois and project to 2D plane
        for idx, coords_3d in rois_a_2d.items():
            rois_a_2d_proj[idx] = np.array([plane_a._project_point_2d(pt) for pt in coords_3d])

        # Iterate through B rois and project to 2D plane and transform
        for idx, coords_3d in rois_b_2d.items():
            rois_b_2d_proj[idx] = np.array(
                plane_a.project_and_transform_points(coords_3d, plane_b, rotation_deg=rotation_2d, scale=scale_2d, translation=translation_2d)
            )

        # Calculate UoI for all unique transformations
        regions_a = {pid : BoundaryRegion([(x, y, 0) for x,y in rois_a_2d_proj[pid]]) for pid in rois_a_2d_proj}
        regions_b = {pid : BoundaryRegion([(x, y, 0) for x,y in rois_b_2d_proj[pid]]) for pid in rois_b_2d_proj}

        # Debug code
        # plot_regions_and_alignment_points(regions_a, plane_a, title="Projected Regions A")
        # plot_regions_and_alignment_points(regions_b, plane_a, title="Projected Regions B")

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
    
    return sorted_data

def plot_uoi_match_data(sorted_data, zstack_a, zstack_b):
    for entry in sorted_data:
        plane_a = entry["plane_a"]
        plane_b = entry["plane_b"]
        rotation_2d = entry["rotation_2d"]
        scale_2d = entry["scale_2d"]
        tx, ty = entry["translation_2d"]
        uoi = entry["UoI"]

        # Recompute projections
        rois_a_2d_proj = {pid: np.array([plane_a._project_point_2d(pt) for pt in coords])
                        for pid, coords in project_flat_plane_points(zstack_a, plane_a.anchor_point).items()}
        rois_b_2d_proj = {pid: np.array(
            plane_a.project_and_transform_points(coords, plane_b, rotation_deg=rotation_2d, scale=scale_2d, translation=(tx, ty)))
                        for pid, coords in project_flat_plane_points(zstack_b, plane_b.anchor_point).items()}

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"Projected ROIs A and B (UoI={uoi:.2f})")
        ax.set_aspect('equal')

        for pid, points in rois_a_2d_proj.items():
            if len(points) > 0:
                ax.plot(points[:, 0], points[:, 1], 'o', markersize=2, color='blue')
                centroid = points.mean(axis=0)
                ax.text(centroid[0], centroid[1], f"A{pid}", fontsize=8, color='blue')

        for pid, points in rois_b_2d_proj.items():
            if len(points) > 0:
                ax.plot(points[:, 0], points[:, 1], 'x', markersize=2, color='green')
                centroid = points.mean(axis=0)
                ax.text(centroid[0], centroid[1], f"B{pid}", fontsize=8, color='green')

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
    threshold=0.5, method="split", eps=5.0, min_samples=5, 
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
            vol_id = roi_data.get("volume", None)
            for (x, y) in coords:
                pt3d = np.array([x, y, z])
                if angled_plane._distance_to_plane(pt3d) < threshold:
                    proj2d = angled_plane._project_point_2d(pt3d)
                    projected_pts_2d.append(proj2d)
                    original_pts_3d.append((x, y, z))
                    volume_ids.append(vol_id)

    if not projected_pts_2d:
        return {}

    projected_pts_2d = np.array(projected_pts_2d)

    if method == "volume":
        # Group by volume ID
        clusters = defaultdict(list)
        for i, vol_id in enumerate(volume_ids):
            clusters[vol_id].append((projected_pts_2d[i], original_pts_3d[i]))
        pid_mapping = {}  # No remapping done in volume mode
        output_regions = {
            vol_id: [pt3d for _, pt3d in cluster]
            for vol_id, cluster in clusters.items()
            if vol_id is not None
        }
        return output_regions, pid_mapping

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

    return output_regions, pid_mapping

# TEST DEBUG
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull
import numpy as np

def plot_regions_and_alignment_points(
    regions, 
    plane, 
    title="Regions + Alignment Points"
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
            centroid = pts2d.mean(axis=0)
            ax.scatter(centroid[0], centroid[1], color='black', s=10)

        except Exception as e:
            print(f"Warning: Could not form polygon for region {pid}: {e}")
            continue

    # Now project alignment points and plot
    for _, plane_point in plane.plane_points.items():
        proj2d = plane._project_point_2d(plane_point.position)
        og_id = plane_point.id  # original ID (real ROI id)

        ax.scatter(proj2d[0], proj2d[1], color='red', s=30, marker='x')
        ax.text(proj2d[0] + 0.5, proj2d[1] + 0.5, f"{og_id}", color='red', fontsize=7)

    ax.grid(True)
    ax.legend(fontsize=6, loc='best')
    plt.tight_layout()
    plt.show()

# # Extract all points of a z-stack that coincide with a plane into a new z-stack
# def extract_zstack_plane(z_stack, plane, threshold = 0.5):

#     new_z_planes = defaultdict(lambda: defaultdict(dict))  # {z: {roi_id: {'coords': [...], 'intensity': ...}}}
#     for z in z_stack.z_planes:
#         for roi_id, roi_data in z_stack.z_planes[z].items():
#             coords = roi_data["coords"]
#             point_projected = False
#             for (x, y) in coords:
#                 pt3d = np.array([x, y, z])
#                 if plane._distance_to_plane(pt3d) < threshold:
#                     if not new_z_planes[z][roi_id].get('coords', None):
#                         new_z_planes[z][roi_id]['coords'] = []
#                     # Add this point to this ROI in the new z-stack
#                     new_z_planes[z][roi_id]['coords'].append((x,y))
#                     point_projected = True
#             # Data from this ROI was used
#             if point_projected:
#                 # Add intensity if it is available
#                 if z_stack.z_planes[z][roi_id].get('intensity', None):
#                     new_z_planes[z][roi_id]['intensity'] = z_stack.z_planes[z][roi_id]['intensity']
#                 # Add volume if it is available
#                 if z_stack.z_planes[z][roi_id].get('volume', None):
#                     new_z_planes[z][roi_id]['volume'] = z_stack.z_planes[z][roi_id]['volume']
#     # Create a z-stack using this dict
#     new_z_stack = ZStack(new_z_planes)
#     # # Add its plane - Not sure if this is necessary
#     new_z_stack.planes.append(plane)
#     return new_z_stack

def extract_zstack_plane(z_stack, plane, threshold=0.5, eps=5.0, min_samples=5, method="split"):
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
        print(f"Anchor point z: {z}")
        for roi_id, roi_data in z_stack.z_planes[z].items():
            coords = roi_data["coords"]
            for (x, y) in coords:
                new_z_planes[z][roi_id].setdefault("coords", []).append((x, y))
            if "intensity" in roi_data:
                new_z_planes[z][roi_id]["intensity"] = roi_data["intensity"]
            if "volume" in roi_data:
                new_z_planes[z][roi_id]["volume"] = int(roi_data["volume"])
        new_stack = ZStack(new_z_planes)

        return new_stack

    # Angled plane: project + cluster
    output_regions, _ = project_angled_plane_points(
        z_stack, plane, threshold=threshold, method=method, eps=eps, min_samples=min_samples
    ) # Project to 2d

    z_fixed = 0 # Set an arbitrary z for projected plane
    for roi_id, points in output_regions.items():
        if not points:
            continue
        coords_2d = [(x, y) for (x, y, z) in points]
        new_z_planes[z_fixed][roi_id]["coords"] = coords_2d

        # Just assign a random intensity -- TODO could be improved later
        new_z_planes[z_fixed][roi_id]["intensity"] =  random.uniform(0.8, 1)

    new_stack = ZStack(new_z_planes)
    new_stack.planes.append(plane)
    print("[INFO] Successfully extracted plane from z-stack")
    return new_stack
