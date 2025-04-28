# Scripts used for registration
from region import Region, BoundaryRegion
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
from zstack import ZStack, PLANE_GEN_PARAMS_DEFAULT
from plane import Plane, PLANE_LIST_PARAM_DEFAULTS, MATCH_PLANE_PARAM_DEFAULTS, PLANE_ANCHOR_ID
from shapely.geometry import Polygon, Point
from planepoint import PlanePoint
from scipy.spatial.distance import cdist
import copy
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import ConvexHull

# Default parameters to match 2 z-stacks in 2D via their best matching planes
DEFAULT_2D_MATCH_PARAMS = {
    "plane_gen_params" : PLANE_GEN_PARAMS_DEFAULT,
    "stack_a_boundary" : None,
    "stack_b_boundary" : None,
    "plane_list_params" : PLANE_LIST_PARAM_DEFAULTS,
    "match_plane_params" : MATCH_PLANE_PARAM_DEFAULTS,
    "seg_params": {
        "eps": 3.0,
        "min_samples" : 5
    },
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

            # # Shade non-overlapping parts of A
            # a_only = poly_a.difference(poly_b)
            # if not a_only.is_empty:
            #     try:
            #         if a_only.geom_type == "Polygon":
            #             x, y = a_only.exterior.xy
            #             ax.fill(x, y, color="red", alpha=0.2, label="A only" if i == matches[0][0] else "")
            #         elif a_only.geom_type == "MultiPolygon":
            #             for subpoly in a_only.geoms:
            #                 x, y = subpoly.exterior.xy
            #                 ax.fill(x, y, color="red", alpha=0.2)
            #     except:
            #         pass

            # # Shade non-overlapping parts of B
            # b_only = poly_b.difference(poly_a)
            # if not b_only.is_empty:
            #     try:
            #         if b_only.geom_type == "Polygon":
            #             x, y = b_only.exterior.xy
            #             ax.fill(x, y, color="red", alpha=0.2, label="B only" if i == matches[0][0] else "")
            #         elif b_only.geom_type == "MultiPolygon":
            #             for subpoly in b_only.geoms:
            #                 x, y = subpoly.exterior.xy
            #                 ax.fill(x, y, color="red", alpha=0.2)
            #     except:
            #         pass

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

    print(matched_planes)

    # Stores just (rounded) values for uniqueness checking
    seen_transformations = set()

    # Stores full records including planes
    unique_transformations = []

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

    print(f"Unique transformations found (rot, scale, t_x, t_y): {[(rot, scale, tx, ty) for rot, scale, tx, ty, _, _, _ in unique_transformations]}")

    # Calculate UoIs of all unique 2D transformations
    UoIs = []

    # Project and transform points on both planes for each transformation identified
    for rotation_2d, scale_2d, tx, ty, match_data, plane_a, plane_b in unique_transformations:
        # Determine whether either plane is flat
        # TODO turn off Falses so script can run as normal
        flat_plane_a = False #np.allclose(np.abs(plane_a.normal), [0, 0, 1], atol=0.05)
        flat_plane_b = False #np.allclose(np.abs(plane_b.normal), [0, 0, 1], atol=0.05)
        translation_2d = (tx, ty)

        # Get dicts of {idx : [x,y,z]}
        if flat_plane_a:
            rois_a_2d = project_flat_plane_points(zstack_a, plane_a.anchor_point)
        else:
            rois_a_2d = project_angled_plane_points(zstack_a, plane_a, threshold=params["plane_gen_params"]["projection_dist_thresh"], **params["seg_params"])

        if flat_plane_b:
            rois_b_2d = project_flat_plane_points(zstack_b, plane_b.anchor_point)
        else:
            rois_b_2d = project_angled_plane_points(zstack_b, plane_b, threshold=params["plane_gen_params"]["projection_dist_thresh"], **params["seg_params"])

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

        # Plot match here if designated in params
        if params["plot_match"]:
            # Setup plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_title("Projected and Transformed ROIs (Plane A and Plane B)")
            ax.set_aspect('equal')

            # Plot ROIs from Plane A (always blue)
            for pid, points in rois_a_2d_proj.items():
                if points.shape[0] == 0:
                    continue
                ax.plot(points[:, 0], points[:, 1], 'o', markersize=2, color='blue')  # ← explicitly blue
                centroid = points.mean(axis=0)
                ax.text(centroid[0], centroid[1], f"A{pid}", fontsize=8, color='blue')

            # Plot ROIs from Plane B (always green)
            for pid, points in rois_b_2d_proj.items():
                if points.shape[0] == 0:
                    continue
                ax.plot(points[:, 0], points[:, 1], 'x', markersize=2, color='green')  # ← explicitly green
                centroid = points.mean(axis=0)
                ax.text(centroid[0], centroid[1], f"B{pid}", fontsize=8, color='green')

            ax.grid(True)
            plt.tight_layout()
            plt.show()



        # Calculate UoI for all unique transformations
        regions_a = {pid : BoundaryRegion([(x, y, 0) for x,y in rois_a_2d_proj[pid]]) for pid in rois_a_2d_proj}
        regions_b = {pid : BoundaryRegion([(x, y, 0) for x,y in rois_b_2d_proj[pid]]) for pid in rois_b_2d_proj}

        # plot_regions_2d_polygons(regions_a, title="Projected Regions A")
        # plot_regions_2d_polygons(regions_b, title="Projected Regions B")

        # And you have original matches from match_data["og_matches"]
        matches = match_data["og_matches"]

        # Filter matches to only those where both PIDs survived projection
        filtered_matches = [
            (i, j) for (i, j) in matches
            if i in regions_a and j in regions_b
        ]

        print(regions_a.keys())
        print(regions_b.keys())

        print(matches)

        UoI = compute_avg_uoi(regions_a, regions_b, filtered_matches, plot=params["plot_uoi"])
        UoIs.append(UoI)


    # Calculate best UoI
    if len(UoIs) > 0:
        best_UoI = max(UoIs)
        best_idx = UoIs.index(best_UoI)
        best_transformation = list(unique_transformations)[best_idx]
        best_transformation = best_transformation[0:4] # extract only the operations
        print(f"Best UoI: {best_UoI}, Best Transformation: {best_transformation}")
    else:
        print("No matches found - no UoI calculated")


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
    threshold=0.5, eps=5.0, min_samples=5
):
    """
    Projects points onto an angled plane, clusters them into ROIs, 
    and assigns each alignment point to exactly one unique cluster region.

    Args:
        z_stack: The ZStack object.
        angled_plane: The Plane object.
        threshold: Distance threshold to consider points near the plane.
        eps: DBSCAN epsilon.
        min_samples: DBSCAN minimum samples.

    Returns:
        Dictionary {pid : [(x, y, z)]} where pid is a unique ROI ID.
        Only alignment/anchor points that are assigned are kept.
    """

    # 1. Project ROI points
    projected_pts_2d = []
    original_pts_3d = []

    for z in z_stack.z_planes:
        for roi_id, roi_data in z_stack.z_planes[z].items():
            coords = roi_data["coords"]
            for (x, y) in coords:
                pt3d = np.array([x, y, z])
                if angled_plane._distance_to_plane(pt3d) < threshold:
                    proj2d = angled_plane._project_point_2d(pt3d)
                    projected_pts_2d.append(proj2d)
                    original_pts_3d.append((x, y, z))

    if not projected_pts_2d:
        return {}

    projected_pts_2d = np.array(projected_pts_2d)

    # 2. Cluster projected points
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(projected_pts_2d)
    labels = clustering.labels_

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # Skip noise
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((projected_pts_2d[idx], original_pts_3d[idx]))

    # 3. Project anchor/alignment points
    projected_anchors = {}
    for _, plane_point in angled_plane.plane_points.items():
        proj2d = angled_plane._project_point_2d(plane_point.position)
        projected_anchors[plane_point.id] = proj2d

    # 4. Build final regions
    output_regions = {}
    assigned_pids = set()

    for label, points in clusters.items():
        cluster_pts_2d = np.array([pt2d for pt2d, pt3d in points])

        if len(cluster_pts_2d) >= 3:
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

        if len(anchors_inside) == 0:
            continue  # No alignment points inside → discard cluster

        if len(anchors_inside) == 1:
            # Exactly one alignment point inside
            pid = anchors_inside[0]
            output_regions[pid] = [pt3d for pt2d, pt3d in points]
            assigned_pids.add(pid)
        else:
            # Multiple anchors inside -> split
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

    return output_regions

# TEST DEBUG
def plot_regions_2d_polygons(regions, title="Regions 2D Projection"):
    """
    Plot the boundary polygons of projected regions (2D).
    Expects {pid : BoundaryRegion}.
    """
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    from scipy.spatial import ConvexHull
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title)
    ax.set_aspect('equal')

    for pid, region in regions.items():
        points = region.get_boundary_points()  # <-- Extract boundary points from BoundaryRegion
        if len(points) < 3:
            continue  # Need at least 3 points to form a polygon

        pts2d = np.array([(x, y) for x, y, z in points])

        try:
            hull = ConvexHull(pts2d)
            hull_pts = pts2d[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])

            ax.plot(hull_pts[:, 0], hull_pts[:, 1], '-', label=f"Region {pid}")

            centroid = pts2d.mean(axis=0)
            ax.scatter(centroid[0], centroid[1], color='black', s=10)
            ax.text(centroid[0], centroid[1], f"{pid}", fontsize=8, color='black')

        except Exception as e:
            print(f"Warning: Could not form polygon for region {pid}: {e}")
            continue

    ax.grid(True)
    ax.legend(fontsize=6, loc='best')
    plt.tight_layout()
    plt.show()
