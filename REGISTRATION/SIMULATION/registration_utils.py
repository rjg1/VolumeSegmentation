# Scripts used for registration
from region import Region, BoundaryRegion
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
from zstack import ZStack
from plane import Plane

# Takes a dict of {roi id: region} for each set of regions, and an existing match list
# matches remaining regoins based on minimising centroid distance using a bipartite match
# calculates the UoI of all matched regions.
# returns the average UoI
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

def compute_avg_uoi(regions_a, regions_b, matches, plot=False):
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

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
                  plane_gen_params = None, 
                  match_list_params = None, 
                  match_plane_params = None, 
                  plot_UoI = False,
                  plot_match = False):
    
    # Extract or generate planes from each z-stack
    if len(zstack_a.planes) == 0:
        planes_a = zstack_a.generate_planes(*plane_gen_params)
    else:
        planes_a = zstack_a.planes

    if len(zstack_b.planes) == 0:
        planes_b = zstack_b.generate_planes(*plane_gen_params)
    else:
        planes_b = zstack_b.planes

    # Perform the grid-search matching between generated planes
    matched_planes = Plane.match_plane_lists(planes_a, planes_b, *match_list_params)

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
            round(translation_2d[1], 2)
        )

        unique_transformations.add(rounded)

    # Project all points near plane A/B to a new plane in the respective Z-stacks. These planes may have unlimited* points (hence the remake)
    plane_a_points = Plane(plane_a.anchor_point, plane_a.alignment_points, max_alignments=np.inf, fixed_basis=plane_gen_params["fixed_basis"])
    plane_b_points = Plane(plane_b.anchor_point, plane_b.alignment_points, max_alignments=np.inf, fixed_basis=plane_gen_params["fixed_basis"])
    
    for z in plane_a.z_planes:
        for roi_id, roi_dict in planes_a.z_planes[z].items():
            coords = roi_dict["coords"]
            coords_3d = [(x,y,z) for x,y in coords]
            # Project all nearby points onto the new plane
            plane_a_points.project_points(coords_3d, threshold=plane_gen_params["projection_dist_thresh"])

    for z in plane_b.z_planes:
        for roi_id, roi_dict in planes_b.z_planes[z].items():
            coords = roi_dict["coords"]
            coords_3d = [(x,y,z) for x,y in coords]
            # Project all nearby points onto the new plane
            plane_b_points.project_points(coords_3d, threshold=plane_gen_params["projection_dist_thresh"])

    # Calculate UoIs of all unique 2D transformations
    UoIs = []

    # Project and transform points on both planes for each transformation identified
    for rotation_2d, scale_2d, _, _ in unique_transformations:
        proj_a, proj_b_transformed, _ = plane_a_points.get_aligned_2d_projection(plane_b, rotation_2d, scale_2d)

        # Calculate UoI for all unique transformations
        regions_a = {pid : BoundaryRegion([(x, y, 0) for x, y in proj_a[pid]]) for pid in proj_a}
        regions_b = {pid : BoundaryRegion([(x, y, 0) for x, y in proj_b_transformed[pid]]) for pid in proj_b_transformed}
        UoI = compute_avg_uoi(regions_a, regions_b, match_data["matches"], plot=plot_UoI)
        UoIs.append(UoI)

    # Calculate best UoI
    best_UoI = max(UoIs)
    best_idx = UoIs.index(best_UoI)
    best_transformation = list(unique_transformations)[best_idx]
    print(f"Best UoI: {best_UoI}, Best Transformation: {best_transformation}")

