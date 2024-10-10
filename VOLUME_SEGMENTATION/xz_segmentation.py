# IMPORTS
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull, Delaunay, QhullError
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from shapely import LineString
import itertools

# GLOBAL VARIABLES
# Additional gap between clusters - make bigger to increase likelihood of grouping in XZ ROIs
CLUSTER_GAP_BUFFER = 0.1
DEFAULT_MIN_SAMPLES = 4
MIN_PTS_PERCENTILE = 10
EPS_REDUCTION = 0.8

def cluster_xz_rois(xz_roi_points):
    clustered_hulls = {}
    # Iterate through xyz points and separate into xz groups
    # Create dict of {y : [ConvexHull of ROI pts]}
    xz_points = {}
    z_diff_min = np.inf
    y_diff_min = np.inf 
    prev_point = None
    for x, y, z in xz_roi_points:
        if y not in xz_points:
            xz_points[y] = []
        xz_points[y].append((x,z))
        # Determine min distance between y and z points
        if prev_point:
           px, py, pz = prev_point
           z_diff = abs(pz - z)
           y_diff = abs(py - y)
           if z_diff < z_diff_min and z_diff > 0:
               z_diff_min = z_diff
           if y_diff < y_diff_min and y_diff > 0:
               y_diff_min = y_diff
        prev_point = (x,y,z) 
    
    # Determine max clustering distance
    # Candidate estimate is max(gap in z planes, gap in y planes)
    dist = max(y_diff_min, z_diff_min) + CLUSTER_GAP_BUFFER

    for y, xz_list in xz_points.items():
        # Get all x,z values for fixed y and make numpy array
        xz_values = np.array(xz_list)
        # Use DBSCAN clustering
        db = DBSCAN(eps=dist, min_samples=5).fit(xz_values)
        # Extract the clusters
        labels = db.labels_
        for label in set(labels):
            if label == -1:  # Skip outliers
                continue

            # Extract points belonging to the current cluster
            cluster_points = xz_values[labels == label]
            
            # Ensure points not co-linear
            if points_collinear(cluster_points):
                continue # cant create convex hull around co-linear points
            
            # Try to make a ConvexHull for this cluster
            try:
                hull = ConvexHull(cluster_points)
                if y not in clustered_hulls:
                    clustered_hulls[y] = []
                clustered_hulls[y].append(hull)
            except QhullError as e:
                print(f"QhullError: {e}")
                print(cluster_points)
                # Plot the problematic points
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c='r', label='Problem Points')
                plt.title(f"Problematic Cluster at y={y}")
                plt.xlabel("X")
                plt.ylabel("Z")
                plt.legend()
                plt.show()
    return clustered_hulls

"""
Autonomously tunes minpts and eps of dbscan based on XZ planes
"""
def cluster_xz_rois_tuned(xz_roi_points, parameters = {}, eps=None, min_samples=None):
    """
    Cluster points on each XZ plane independently using tuned eps and min_samples for each plane.
    
    Args:
        xz_roi_points (list): List of (x, y, z) points.
    
    Returns:
        clustered_hulls (dict): A dictionary of y-planes with lists of ConvexHull objects for each cluster.
    """
    clustered_hulls = {}
    xz_points = {}

    # Organize points into y-groups
    for x, y, z in xz_roi_points:
        if y not in xz_points:
            xz_points[y] = []
        xz_points[y].append((x, z))

    # Cluster points on each y-plane using individually tuned parameters
    for y, xz_list in xz_points.items():
        print(f"Clustering y={y}")
        if not xz_list:  # Skip empty y-planes
            continue

        xz_values = np.array(xz_list)

        if len(xz_list) < 2:
            # print(f"Only {len(xz_list)} point(s) on y-plane {y}, returning the point.")
            if not parameters.get('ignore_colinear_xz'):
                if y not in clustered_hulls:
                    clustered_hulls[y] = []
                clustered_hulls[y].append([(x,y,z) for x,z in xz_list]) 
            continue

        # min_samples = estimate_min_samples(xz_values)
        min_samples = 2
        # print(f"y= {y} min_samples = {min_samples} len_xz = {len(xz_values)}")
        eps = determine_eps(xz_values, min_samples=min_samples)

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(xz_values)
        labels = db.labels_

        # Process each cluster
        for label in set(labels):
            if label == -1:  # Skip outliers
                continue
            cluster_points = xz_values[labels == label]
            point_list = []
            if points_collinear(cluster_points):
                # Ignore colinear xz if designated
                if parameters.get("ignore_colinear_xz"):
                    continue
                # Add colinear points
                if y not in clustered_hulls:
                    clustered_hulls[y] = []

                # Determine if points are collinear along x or z
                unique_x = len(set(cluster_points[:, 0])) == 1  # Check if all x values are the same
                unique_z = len(set(cluster_points[:, 1])) == 1  # Check if all z values are the same
                
                if unique_x or unique_z:
                    if unique_x:
                        # Points are collinear along x, so sort by z
                        start_z = cluster_points[cluster_points[:, 1].argmin(), 1].item()
                        end_z = cluster_points[cluster_points[:, 1].argmax(), 1].item()    
                        start_tuple = (cluster_points[0][0].item(), y, start_z)  
                        end_tuple = (cluster_points[0][0].item(), y, end_z)      
                    else:
                        # Points are collinear along z, so sort by x
                        start_x = cluster_points[cluster_points[:, 0].argmin(), 0].item()
                        end_x = cluster_points[cluster_points[:, 0].argmax(), 0].item()    
                        start_tuple = (start_x, y, cluster_points[0][1].item()) 
                        end_tuple = (end_x, y, cluster_points[0][1].item())      
                    point_list = [start_tuple, end_tuple]
                    
                    # print(f"Added colinear line segment! : {point_list}")
            else:
                try:
                    hull = ConvexHull(cluster_points)
                    if y not in clustered_hulls:
                        clustered_hulls[y] = []
                    # Extract boundary points of convex hull
                    vertices = hull.points[hull.vertices]
                    point_list = [((x, y, z) for x, z in vertices)]
                except QhullError as e:
                    print(f"QhullError: {e}")
            
            # Interpolate boundary points and store
            if parameters.get('cache_interpolated_xz', None) and len(point_list) > 1:
                xz_points_2d = [(x,z) for x,y,z in point_list]
                cx, cz = find_centroid(xz_points_2d)
                sorted_points = sort_points_by_angle(xz_points_2d, (cx, cz))
                sorted_points.append(sorted_points[0]) # wrap polygon back around on itself
                line = LineString(sorted_points)
                # Generate n zx points around the boundary of the ROI polygon
                distances = np.linspace(0, line.length, parameters['roi_projection_n_points'])
                points = [line.interpolate(distance) for distance in distances]
                # Update 2d xz points to use new point projection
                xz_points_2d = [(point.x, point.y) for point in points]
                # Translate back into 3d
                point_list = [(x,y,z) for x,z in xz_points_2d]
            # Add all valid point lists to the dictionary at this y-plane
            if len(point_list) > 0:
                clustered_hulls[y].append(point_list)

    return clustered_hulls


# Function to sort points by angle from centroid
def sort_points_by_angle(points, center):
    cx, cz = center
    def angle(point):
        return np.arctan2(point[1] - cz, point[0] - cx)
    return sorted(points, key=angle)

# Function to calculate centroid of the points
def find_centroid(points):
    x, z = zip(*points)
    cx = sum(x) / len(points)
    cz = sum(z) / len(points)
    return cx, cz

def points_collinear(points):
    # points is a list of tuples [(x1, z1), (x2, z2), ...]
    if len(points) < 3:
        return True  # Less than 3 points are trivially collinear
    
    for i in range(2, len(points)):
        x1, z1 = points[i - 2]
        x2, z2 = points[i - 1]
        x3, z3 = points[i]
        # Calculate determinant
        det = np.linalg.det([[x1, z1, 1], [x2, z2, 1], [x3, z3, 1]])
        if not np.isclose(det, 0):  # Check if determinant is close to zero
            return False
    return True

def visualize_random_clusters(clustered_hulls, num_planes=5):
    """
    Visualize all clusters on each selected y-plane in one figure per y-plane,
    showing all points within each cluster along with convex hull boundaries or points/line segments.
    
    Args:
        clustered_hulls (dict): A dictionary containing y-plane keys with lists of ConvexHull objects or raw points/line segments.
        num_planes (int): Number of random y-planes to visualize.
    """
    # Randomly select a set of y-planes to visualize
    selected_y_planes = random.sample(list(clustered_hulls.keys()), min(num_planes, len(clustered_hulls)))

    # Define a set of distinct colors for variety in cluster visualization
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

    # Plot each selected y-plane
    for y in selected_y_planes:
        clusters = clustered_hulls[y]
        if not clusters:
            continue

        plt.figure(figsize=(10, 8))
        # Iterate over each cluster in the current y-plane
        for idx, cluster in enumerate(clusters):
            color = next(colors)

            # Check if the cluster is a ConvexHull object
            if isinstance(cluster, ConvexHull):
                hull_points = cluster.points  # All points used for creating the hull
                hull_vertices = hull_points[cluster.vertices]  # Boundary points of the hull

                # Plot all the points in the cluster
                plt.scatter(hull_points[:, 0], hull_points[:, 1], color=color, s=10, alpha=0.6, label=f'Cluster {idx + 1}')

                # Plot the convex hull boundaries
                for simplex in cluster.simplices:
                    plt.plot(hull_points[simplex, 0], hull_points[simplex, 1], color=color, linestyle='-')

            # Handle line segments (2 points)
            elif isinstance(cluster, np.ndarray) and len(cluster) == 2:
                plt.plot(cluster[:, 0], cluster[:, 1], color=color, linestyle='-', label=f'Line Segment {idx + 1}')
                plt.scatter(cluster[:, 0], cluster[:, 1], color=color, s=20)

            # Handle single points
            elif isinstance(cluster, np.ndarray) and len(cluster) == 1:
                plt.scatter(cluster[:, 0], cluster[:, 1], color=color, s=50, label=f'Single Point {idx + 1}')
            else:
                raise ValueError(f"Unexpected cluster type for y-plane {y}: {cluster}")

        plt.title(f'Clusters on XZ Plane at y = {y:.2f}')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.grid(True)
        plt.legend()
        plt.show()


def determine_eps(xz_values, min_samples=DEFAULT_MIN_SAMPLES):
    """
    Determine the optimal eps value using the Elbow method without plotting.
    The optimal eps is identified as the distance at the point with the maximum delta between sorted distances.
    """
    if len(xz_values) <= min_samples:
        print(f"Warning: Only {len(xz_values)} samples available. Using default epsilon.")
        return 1
    # Fit NearestNeighbors model to determine k-distance graph
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(xz_values)
    distances, _ = neighbors.kneighbors(xz_values)

    # Sort the k-th nearest neighbor distances in ascending order
    sorted_distances = np.sort(distances[:, -1])  # Get the min_samples-th nearest neighbor distances

    # Calculate the difference between consecutive sorted distances to find the largest delta
    deltas = np.diff(sorted_distances)
    max_delta_index = np.argmax(deltas)  # Index of the maximum delta
    optimal_k_index = max(0, max_delta_index - 1)
    optimal_eps = sorted_distances[optimal_k_index]  # eps is the distance at the point before the elbow

    return max(1,optimal_eps * EPS_REDUCTION)


def estimate_min_samples(points, percentile=10):
    """
    Estimate an appropriate min_samples value as a fraction of the total number of points.
    
    Args:
        points (ndarray): 2D array of points on the XZ-plane.
        percentile (float): Fraction of total points to use as min_samples.
        
    Returns:
        int: Estimated min_samples value.
    """
    # Use the specified percentile as a fraction of the total number of points
    num_points = len(points)
    estimated_min_samples = max(4, int((percentile / 100) * num_points))
    
    return estimated_min_samples

def tune_parameters(xz_roi_points, sample_fraction=0.2, min_samples=4):
    """
    Tune eps and min_samples based on a random sample of y-planes with sufficient points.
    
    Args:
        xz_roi_points (list): List of (x, y, z) points.
        sample_fraction (float): Fraction of y-planes to sample for tuning.
        min_samples (int): Fixed value of min_samples used for determining eps.

    Returns:
        median_eps (float): The median of tuned eps values across sampled y-planes.
        min_samples (int): The fixed value of min_samples used.
    """
    xz_points = {}
    # Separate points into groups based on y-values
    for x, y, z in xz_roi_points:
        if y not in xz_points:
            xz_points[y] = []
        xz_points[y].append((x, z))

    # Filter y-planes to only include those with points
    non_empty_y_planes = {y: points for y, points in xz_points.items() if points}

    # Calculate the mean and standard deviation of points per y-plane
    points_counts = [len(points) for points in non_empty_y_planes.values()]
    mean_points = np.mean(points_counts)
    std_points = np.std(points_counts)

    # Define a threshold as 2 standard deviations below the mean
    threshold = mean_points - 2 * std_points

    # Filter y-planes that have at least the threshold number of points
    sufficient_y_planes = [y for y, points in non_empty_y_planes.items() if len(points) >= threshold]

    # If no y-planes meet the threshold, fall back to using all non-empty planes
    if not sufficient_y_planes:
        sufficient_y_planes = list(non_empty_y_planes.keys())

    # Randomly sample y-planes with a sufficient number of points
    sampled_y_planes = random.sample(sufficient_y_planes, max(1, int(len(sufficient_y_planes) * sample_fraction)))

    # Tune eps on sampled y-planes
    eps_values = []
    for y in sampled_y_planes:
        xz_values = np.array(xz_points[y])
        eps = determine_eps(xz_values, min_samples=min_samples)
        eps_values.append(eps)

    # Use the median eps from the sampled y-planes
    median_eps = np.median(eps_values)

    return median_eps