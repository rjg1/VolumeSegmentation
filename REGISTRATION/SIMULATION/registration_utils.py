# Scripts used for registration
from region import Region, BoundaryRegion
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np

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

            # Shade intersection if exists
            if not inter.is_empty:
                try:
                    x, y = inter.exterior.xy
                    ax.fill(x, y, color="purple", alpha=0.3, label="Intersection" if uoi_scores.count(uoi) == 1 else "")
                except:
                    pass  # Handle non-polygonal intersections (e.g., line)

    if plot:
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return sum(uoi_scores) / len(uoi_scores) if uoi_scores else 0.0
