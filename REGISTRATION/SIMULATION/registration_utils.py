# Scripts used for registration
from region import Region, BoundaryRegion
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
from zstack import ZStack, PLANE_GEN_PARAMS_DEFAULT
from plane import Plane, PLANE_LIST_PARAM_DEFAULTS, MATCH_PLANE_PARAM_DEFAULTS
from shapely.geometry import Polygon
from planepoint import PlanePoint
from scipy.spatial.distance import cdist
import copy
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

# Default parameters to match 2 z-stacks in 2D via their best matching planes
DEFAULT_2D_MATCH_PARAMS = {
    "plane_gen_params" : PLANE_GEN_PARAMS_DEFAULT,
    "stack_a_boundary" : None,
    "stack_b_boundary" : None,
    "plane_list_params" : PLANE_LIST_PARAM_DEFAULTS,
    "match_plane_params" : MATCH_PLANE_PARAM_DEFAULTS,
    "plot_uoi" : False,
    "plot_match" : False
}

# Takes a dict of {roi id: region} for each set of regions, and an existing match list
# matches remaining regoins based on minimising centroid distance using a bipartite match
# calculates the UoI of all matched regions.
# returns the average UoI
def compute_avg_uoi(regions_a, regions_b, matches, plot=False):
    matched_ids_a = set(i for i, _ in matches)
    matched_ids_b = set(j for _, j in matches)

    # Get unmatched ids
    unmatched_ids_a = [i for i in regions_a if i not in matched_ids_a]
    unmatched_ids_b = [j for j in regions_b if j not in matched_ids_b]

    # Match unmatched regions using centroid proximity
    if unmatched_ids_a and unmatched_ids_b:
        centroids_a = np.array([regions_a[i].get_centroid()[:2] for i in unmatched_ids_a])
        centroids_b = np.array([regions_b[j].get_centroid()[:2] for j in unmatched_ids_b])

        cost_matrix = cdist(centroids_a, centroids_b, metric='euclidean')
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        new_matches = [(unmatched_ids_a[i], unmatched_ids_b[j]) for i, j in zip(row_ind, col_ind)]
        matches += new_matches

    # Init plot if needed
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Matched ROIs with UoI Shading")
        ax.set_aspect("equal")

    uoi_scores = []

    for i, j in matches:
        region_a = regions_a[i]
        region_b = regions_b[j]

        pts_a = np.array(region_a.get_boundary_points())
        pts_b = np.array(region_b.get_boundary_points())

        # Build polygons
        poly_a = Polygon(pts_a)
        poly_b = Polygon(pts_b)

        if not poly_a.is_valid or not poly_b.is_valid:
            continue

        # Compute intersection & union
        inter = poly_a.intersection(poly_b)
        union = poly_a.union(poly_b)
        uoi = inter.area / union.area if union.area > 0 else 0
        uoi_scores.append(uoi)

        if plot:
            # Plot centroids
            cent_a = region_a.get_centroid()
            cent_b = region_b.get_centroid()
            ax.scatter(*cent_a[:2], color="blue", marker="o", s=25)
            ax.scatter(*cent_b[:2], color="green", marker="x", s=25)

            # Label centroids
            ax.text(cent_a[0] + 0.1, cent_a[1] + 0.1, f"A{i}", color="blue", fontsize=8)
            ax.text(cent_b[0] + 0.1, cent_b[1] + 0.1, f"B{j}", color="green", fontsize=8)

            # Shade intersection
            if not inter.is_empty:
                try:
                    if inter.geom_type == "Polygon":
                        x, y = inter.exterior.xy
                        ax.fill(x, y, color="purple", alpha=0.3, label="Intersection" if uoi_scores.count(uoi) == 1 else "")
                    elif inter.geom_type == "MultiPolygon":
                        for subpoly in inter.geoms:
                            x, y = subpoly.exterior.xy
                            ax.fill(x, y, color="purple", alpha=0.3)
                except:
                    pass

            # Shade non-overlapping parts of A
            a_only = poly_a.difference(poly_b)
            if not a_only.is_empty:
                try:
                    if a_only.geom_type == "Polygon":
                        x, y = a_only.exterior.xy
                        ax.fill(x, y, color="red", alpha=0.2, label="A only" if i == matches[0][0] else "")
                    elif a_only.geom_type == "MultiPolygon":
                        for subpoly in a_only.geoms:
                            x, y = subpoly.exterior.xy
                            ax.fill(x, y, color="red", alpha=0.2)
                except:
                    pass

            # Shade non-overlapping parts of B
            b_only = poly_b.difference(poly_a)
            if not b_only.is_empty:
                try:
                    if b_only.geom_type == "Polygon":
                        x, y = b_only.exterior.xy
                        ax.fill(x, y, color="red", alpha=0.2, label="B only" if i == matches[0][0] else "")
                    elif b_only.geom_type == "MultiPolygon":
                        for subpoly in b_only.geoms:
                            x, y = subpoly.exterior.xy
                            ax.fill(x, y, color="red", alpha=0.2)
                except:
                    pass

    if plot:
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return sum(uoi_scores) / len(uoi_scores) if uoi_scores else 0.0

# Complete pipeline for matching the planes of 2 Z-stack objects
def match_zstacks_2d(zstack_a : ZStack, zstack_b : ZStack, 
                  match_params = None):
    # Define function for updating dictionary to match defaults
    def deep_update(base, updates):
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_update(base[key], value)  # Recurse
            else:
                base[key] = value
    # Copy defaults
    params = copy.deepcopy(DEFAULT_2D_MATCH_PARAMS)
    if match_params: # If user specified non-default parameters
        deep_update(params, match_params) # Update them

    # Extract or generate planes from each z-stack
    if len(zstack_a.planes) == 0:
        print(f"Generating A planes")
        plane_gen_params = copy.deepcopy(params["plane_gen_params"])
        if params["stack_a_boundary"]: # Update custom image boundaries if they are uneven
            plane_gen_params["plane_boundaries"] = params["stack_a_boundary"]

        planes_a = zstack_a.generate_planes(plane_gen_params)
    else:
        planes_a = zstack_a.planes

    if len(zstack_b.planes) == 0:
        print(f"Generating B planes")
        plane_gen_params = copy.deepcopy(params["plane_gen_params"])
        if params["stack_b_boundary"]: # Update custom image boundaries if they are uneven
            plane_gen_params["plane_boundaries"] = params["stack_b_boundary"]
        planes_b = zstack_b.generate_planes(plane_gen_params)
    else:
        planes_b = zstack_b.planes

    # Perform the grid-search matching between generated planes
    matched_planes = Plane.match_plane_lists(planes_a, planes_b, plane_list_params=params["plane_list_params"], match_plane_params=params["match_plane_params"])

    # Get all unique 2D transformations from best matches
    unique_transformations = set()

    for match in list(matched_planes.values()):
        plane_a = match["plane_a"]
        plane_b = match["plane_b"]
        match_data = match["result"]

        proj_a, proj_b_aligned, transform_info = plane_a.get_aligned_2d_projection(
            plane_b,
            offset_deg=match_data["offset"],
            scale_factor= match_data["scale_factor"]
        )

        rotation_2d = transform_info["rotation_deg"]
        scale_2d = transform_info["scale"]
        translation_2d = transform_info["translation"]
        rounded = (
            round(rotation_2d, 2),
            round(scale_2d, 2),
            round(translation_2d[0], 2),
            round(translation_2d[1], 2),
            plane_a,
            plane_b
        )


    print(f"Unique transformations found (rot, scale, t_x, t_y, plane_a, plane_b): {unique_transformations}")

    # Calculate UoIs of all unique 2D transformations
    UoIs = []

    # Project and transform points on both planes for each transformation identified
    for rotation_2d, scale_2d, tx, ty, plane_a, plane_b in unique_transformations:
        # Determine whether either plane is flat
        flat_plane_a = np.allclose(np.abs(plane_a.normal), [0, 0, 1], atol=0.05)
        flat_plane_b = np.allclose(np.abs(plane_b.normal), [0, 0, 1], atol=0.05)
        translation_2d = (tx, ty)

        rois_a_2d_proj = {}
        rois_b_2d_proj = {}
        # Get dicts of {idx : [x,y,z]}
        if flat_plane_a:
            rois_a_2d = project_flat_plane_points(zstack_a, plane_a.anchor_point)
        else:
            rois_a_2d = project_angled_plane_points(zstack_a, plane_a)

        if flat_plane_b:
            rois_b_2d = project_flat_plane_points(zstack_b, plane_b.anchor_point)
        else:
            rois_b_2d = project_angled_plane_points(zstack_b, plane_b)

        # Iterate through A rois and project to 2D plane
        for idx, coords_3d in rois_a_2d.items():
            rois_a_2d_proj[idx] = np.array([plane_a._project_point_2d(pt) for pt in coords_3d])

        # Iterate through B rois and project to 2D plane and transform
        for idx, coords_3d in rois_b_2d.items():
            rois_b_2d_proj[idx] = np.array(
                plane_a.project_and_transform_points(coords_3d, plane_b, rotation_deg=rotation_2d, scale=scale_2d, translation=translation_2d)
            )

        # TODO Plot match here if designated in params

        # Calculate UoI for all unique transformations
        regions_a = {pid : BoundaryRegion([(x, y, z) for x,y,z in rois_a_2d[pid]]) for pid, pts in rois_a_2d.items()}
        regions_b = {pid : BoundaryRegion([(x, y, z) for x,y,z in rois_b_2d[pid]]) for pid, pts in rois_b_2d.items()}
        UoI = compute_avg_uoi(regions_a, regions_b, match_data["og_matches"], plot=params["plot_uoi"])
        UoIs.append(UoI)


    # Calculate best UoI
    best_UoI = max(UoIs)
    best_idx = UoIs.index(best_UoI)
    best_transformation = list(unique_transformations)[best_idx]
    print(f"Best UoI: {best_UoI}, Best Transformation: {best_transformation}")


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

def project_angled_plane_points(z_stack, angled_plane, fixed_basis=True, eps=5.0, min_samples=5):
    """
    Projects points onto an angled plane and segments clusters into ROIs.

    Args:
        z_stack: The ZStack object containing ROIs.
        angled_plane: The Plane object.
        fixed_basis: Whether to use a fixed basis for projections.
        eps: DBSCAN eps value for clustering.
        min_samples: Minimum points per cluster.

    Returns:
        Dictionary {pid : [(x, y, z)]}, where pid is a cluster label.
    """

    all_projected_pts = []
    original_coords = []
    idx_map = []  # map from projected point index to (x, y, z) original

    # Project all points
    for z in z_stack.z_planes:
        for roi_id, roi_data in z_stack.z_planes[z].items():
            coords = roi_data["coords"]
            for (x, y) in coords:
                pt3d = np.array([x, y, z])
                proj = angled_plane.project_point(pt3d, fixed_basis=fixed_basis)
                if proj is not None:
                    all_projected_pts.append(proj)
                    original_coords.append((x, y, z))  # save original 3D point
                    idx_map.append(len(idx_map))  # just sequential indexing

    if not all_projected_pts:
        return {}

    all_projected_pts = np.array(all_projected_pts)

    # Cluster projected points
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(all_projected_pts)
    labels = clustering.labels_

    # Group points by cluster
    regions_raw = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # skip noise
        if label not in regions_raw:
            regions_raw[label] = []
        regions_raw[label].append(original_coords[idx])  # store 3D points

    return regions_raw