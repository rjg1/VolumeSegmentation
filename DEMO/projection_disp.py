import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from shapely import Polygon, Point
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

NUM_XZ_SLICES = 60
INTERPOLATION_RESOLUTION = 10
Y_SAMPLE_RATIO = 1 # Number of y points to sample per pixel
X_SAMPLE_RATIO = 1 # Number of x points to sample per pixel
CLUSTER_GAP_BUFFER = 0.1
DEFAULT_MIN_SAMPLES = 4
MIN_PTS_PERCENTILE = 10
MIN_EPS = 0.2
EPS_REDUCTION = 0.8

def show_projection_start(volume_id = 1, roi_interval = 5):
    file_path = '../VALIDATION/validation_runs/drg_subset_1_r/real_data_filtered_v0_VOLUMES.csv'
    df = pd.read_csv(file_path)
    # First, load the CSV and generate the necessary polygons
    volume_data = df[df['VOLUME_ID'] == volume_id]
    max_z = volume_data['z'].max()

    # Prepare the points and generate polygons for the max z-plane
    points = list(zip(volume_data['x'], volume_data['y'], volume_data['z'], volume_data['ROI_ID']))
    z_polys = generate_xz_single_y(points, max_z)

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-20)


    volume_data_adj = volume_data[volume_data['z'] != max_z]
    volume_data_adj = volume_data_adj[volume_data_adj['z'] % roi_interval == 0]
    # Scatter all points in the volume in a light color
    sc = ax.scatter(volume_data_adj['x'], volume_data_adj['y'], volume_data_adj['z'],
                    c=volume_data_adj['z'], cmap=plt.cm.Oranges.reversed(), alpha=0.5)

    # Group by z-plane so we can plot smooth lines per z-plane
    grouped_data = volume_data_adj.groupby('z')

    # # Loop through each z-plane group and plot smooth lines
    # for z_val, group in grouped_data:
    #     # Extract x, y values as tuples
    #     xy_points = list(zip(group['x'].values, group['y'].values))
        
    #     # Find the centroid for the current z-plane
    #     centroid = find_centroid(xy_points)
        
    #     # Sort the points by angle relative to the centroid
    #     sorted_points = sort_points_by_angle(xy_points, centroid)
        
    #     # Separate the sorted x and y values
    #     x_vals = [p[0] for p in sorted_points]
    #     y_vals = [p[1] for p in sorted_points]
        
    #     # Plot a smooth line across these sorted points on the current z-plane
    #     ax.plot(x_vals, y_vals, [z_val] * len(x_vals), color='orange', alpha=0.5, linewidth=2)


    # Buffer size for floating-point tolerance in the point-polygon containment check
    buffer_size = 0.001


    x_min, x_max = volume_data['x'].min(), volume_data['x'].max()
    y_min, y_max = volume_data['y'].min(), volume_data['y'].max()
    tolerance = 0.01 
    num_lines = 50 
    y_lines = np.linspace(y_min, y_max, num_lines)

    num_x_points = 50 # Adjust the number of points along the x-axis
    x_values = np.linspace(x_min, x_max, num_x_points)
   # Highlight points that intersect with ROI polygons
    for y_line in y_lines:
        for x in x_values:
            point = Point(x, y_line)  # Create the point at (x, y_line)
            
            # Check each polygon on the z-plane
            for roi_id, roi_dict in z_polys[max_z].items():
                poly = roi_dict['poly']
                
                # If the polygon exists and the point is within the polygon
                if poly is not None and poly.contains(point):
                    # Plot this point in dark blue
                    ax.scatter(x, y_line, max_z, color='darkblue', s=2)
                else:
                    ax.scatter(x, y_line, max_z, color='skyblue', s=2)

    num_lines = int((y_max - y_min) / 5)  # Adjust as needed
    y_values = range(int(y_min), int(y_max), max(1, int((y_max - y_min) / num_lines)))

    # Make z plane roi red
    for roi_id, roi_dict in z_polys[max_z].items():
        poly = roi_dict['poly']
    
        # Check if the polygon exists
        if poly is not None:
            for idx, row in volume_data.iterrows():
                x, y, z = row['x'], row['y'], row['z']
                if z == max_z and poly.buffer(buffer_size).contains(Point(x, y)):
                    # Highlight these points
                    ax.scatter(x, y, z, color='green', s=3)  # Darker blue for points inside the polygon


    # Customize and show plot
    ax.set_title(f'XZ Projection')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def main():
    # show_projection_start(volume_id=1, roi_interval=3)
    # plot_projection(volume_id =1, roi_interval = 2)
    # plot_intersection()
    plot_roi_coloring2(volume_id=1, roi_interval=2, green_ids=[1,2,3,4,5,6,7,8,9,10,11,12,13], red_ids=[])
    # plot_roi_restrictions(volume_id=1, roi_interval=2, green_ids=[4,5,6,7,8,9,10,11,12,13], red_ids=[1,2,3], n_rois=5, gap_interval = 3)

def plot_projection(volume_data = None, volume_id = 1, roi_interval = 5):
    # Load the CSV containing the volume data
    file_path = '../VALIDATION/validation_runs/drg_subset_1_r/real_data_filtered_v0_VOLUMES.csv'
    df = pd.read_csv(file_path)

    # Load the volume data for the specific volume_id
    volume_data = df[df['VOLUME_ID'] == volume_id]
    max_z = volume_data['z'].max()

    # Load the CSV containing the projection points
    projection_file_path = '../VOLUME_SEGMENTATION/xz_cache/real_data_filtered_v0_ROIS_XZ.csv'
    projection_data = pd.read_csv(projection_file_path)

    # Get the min and max values from the current volume data
    x_min, x_max = volume_data['x'].min(), volume_data['x'].max()
    y_min, y_max = volume_data['y'].min(), volume_data['y'].max()
    z_min, z_max = volume_data['z'].min(), volume_data['z'].max()

    # Filter projection points to be within the x, y, z range of the current volume
    filtered_projection_data = projection_data[
        (projection_data['x'] >= x_min) & (projection_data['x'] <= x_max) &
        (projection_data['y'] >= y_min) & (projection_data['y'] <= y_max) &
        (projection_data['z'] >= z_min) & (projection_data['z'] <= z_max)
    ]

    # Apply a tolerance to select y-planes that are close to multiples of roi_interval
    tolerance = 0.99
    # Get the unique sorted y-values
    y_values = np.unique(filtered_projection_data['y'])
    # Determine the gaps between consecutive y-values
    y_gaps = np.diff(y_values)
    # Assume the most common y-gap as the constant interval (mode)
    y_gap = np.median(y_gaps)  # Use median instead of mode for robustness
    filtered_projection_data = filtered_projection_data[
        np.abs(filtered_projection_data['y'] % (roi_interval * y_gap)) <= tolerance
    ]

    # Set up 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-20)

    # Group points by y-plane
    grouped_data = filtered_projection_data.groupby('y')

    removed_pts = []
    # Loop through each y-plane and plot the convex hull
    for y_plane, group in grouped_data:
        # Get the x, z coordinates for the current y-plane
        points = group[['x', 'z']].values
        
        # # Only create a convex hull if there are enough points (at least 3)
        # if len(points) >= 3:
        #     hull = ConvexHull(points)
           
        #     for simplex in hull.simplices:
        #         for vertex in simplex:
        #             removed_pts.append((points[vertex, 0], y_plane, points[vertex, 1]))
        #     # Plot the boundar           y of the convex hull in red using the original points
        #     for simplex in hull.simplices:
        #         ax.plot(points[simplex, 0], [y_plane] * len(simplex), points[simplex, 1], color='red', linewidth=2)
        
        # Filter out points that are in the convex hull (removed points)
        filtered_group = group[~group.apply(lambda row: (row['x'], row['y'], row['z']) in removed_pts, axis=1)]

        # Scatter all points, but the convex hull points will be outlined with red
        ax.scatter(filtered_group['x'], filtered_group['y'], filtered_group['z'], c='darkblue', alpha=0.7, s=0.5)
        
        eps = determine_eps(points, min_samples=4)
        db = DBSCAN(eps=eps, min_samples=4).fit(points)
        labels = db.labels_

        # Loop through each cluster and compute a convex hull
        unique_labels = set(labels)
        for cluster_label in unique_labels:
            if cluster_label == -1:
                # Noise points (DBSCAN labels noise as -1)
                continue
            
            # Get the points corresponding to the current cluster
            cluster_points = points[labels == cluster_label]
            
            # Only create a convex hull if there are enough points (at least 3)
            if len(cluster_points) >= 3:
                if points_collinear(cluster_points):
                    sorted_points = cluster_points[np.argsort(cluster_points[:, 0])]  # Sort by x-coordinate
                    ax.plot(sorted_points[:, 0], [y_plane] * len(sorted_points), sorted_points[:, 1], 
                            color='red', linewidth=2)
                else:
                    hull = ConvexHull(cluster_points)
                    
                    # Store the boundary points from the convex hull
                    for simplex in hull.simplices:
                        for vertex in simplex:
                            removed_pts.append((cluster_points[vertex, 0], y_plane, cluster_points[vertex, 1]))
                    
                    # Plot the boundary of the convex hull in red for this cluster
                    for simplex in hull.simplices:
                        ax.plot(cluster_points[simplex, 0], [y_plane] * len(simplex), cluster_points[simplex, 1], 
                                color='red', linewidth=2)

            else:
                # Plot the individual points if there are fewer than 3 in the cluster
                ax.scatter(cluster_points[:, 0], [y_plane] * len(cluster_points), cluster_points[:, 1], color='red', s=10)
                

    # Set labels and show plot
    ax.set_title('XZ Segmentation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def determine_eps(xz_values, min_samples=DEFAULT_MIN_SAMPLES):
    """
    Determine the optimal eps value using the Elbow method without plotting.
    The optimal eps is identified as the distance at the point with the maximum delta between sorted distances.
    """
    if len(xz_values) <= min_samples:
        print(f"Warning: Only {len(xz_values)} samples available. Using default epsilon.")
        return MIN_EPS
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

    return max(MIN_EPS, optimal_eps * EPS_REDUCTION)

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

def plot_intersection(volume_data = None, volume_id = 1, roi_interval =5):
    # Load the CSV containing the volume data
    file_path = '../VALIDATION/validation_runs/drg_subset_1_r/real_data_filtered_v0_VOLUMES.csv'
    df = pd.read_csv(file_path)

    # Load the volume data for the specific volume_id
    volume_data = df[df['VOLUME_ID'] == volume_id]
    max_z = volume_data['z'].max()

    # Load the CSV containing the projection points
    projection_file_path = '../VOLUME_SEGMENTATION/xz_cache/real_data_filtered_v0_ROIS_XZ.csv'
    projection_data = pd.read_csv(projection_file_path)

    # Get the min and max values from the current volume data
    x_min, x_max = volume_data['x'].min(), volume_data['x'].max()
    y_min, y_max = volume_data['y'].min(), volume_data['y'].max()
    z_min, z_max = volume_data['z'].min(), volume_data['z'].max()
    max_y = volume_data['y'].max()
    min_y = volume_data['y'].min()
    # Find the unique sorted y-values
    y_values = sorted(volume_data['y'].unique())

    # Find the midpoint in the range of y-values
    mid_y = y_values[len(y_values) // 2]


    # Filter projection points to be within the x, y, z range of the current volume
    filtered_projection_data = projection_data[
        (projection_data['x'] >= x_min) & (projection_data['x'] <= x_max) &
        (projection_data['y'] >= y_min) & (projection_data['y'] <= y_max) &
        (projection_data['z'] >= z_min) & (projection_data['z'] <= z_max)
    ]

    # Apply a tolerance to select y-planes that are close to multiples of roi_interval
    tolerance = 0.99
    filtered_projection_data = filtered_projection_data[
        np.abs(filtered_projection_data['y'] % roi_interval) <= tolerance
    ]

    # Set up 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the top ROI at max_y (Top y-plane)
    top_roi_data = volume_data[volume_data['y'] == mid_y]
    
    # Get the x, z coordinates for the top ROI
    if len(top_roi_data) >= 3:
        points = top_roi_data[['x', 'z']].values

        
        # Plot the boundary of the convex hull in red
        # hull = ConvexHull(points)
        # for simplex in hull.simplices:
        #     ax.plot(points[simplex, 0], [mid_y] * len(simplex), points[simplex, 1], color='red', linewidth=2)
        
        #TEST interpolation
        from shapely import LineString
        cx, cz = find_centroid(points)
        # Sort xz boundary points in predicted order around a polygon
        sorted_points = sort_points_by_angle(points, (cx, cz))
        sorted_points.append(sorted_points[0]) # wrap polygon back around on itself
        line = LineString(sorted_points)
        # Generate n zx points around the boundary of the ROI polygon
        distances = np.linspace(0, line.length, 300)
        points = [line.interpolate(distance) for distance in distances]
        # Update 2d xz points to use new point projection
        x_points = [point.x for point in points]
        y_points = [587 for _ in points]
        z_points = [point.y for point in points]
        ax.scatter(x_points, y_points, z_points, c='darkblue', alpha=0.7, s=0.5)
        #ENDTEST
        print(max(z_points))
        print(max_z)
        xyz_pts = [(point.x, 587, point.y) for point in points]
        xyz_pts = [(x,y,z) for x,y,z in xyz_pts if abs(z - max_z) < 0.4]
        print(xyz_pts)
        x_vals, y_vals, z_vals = zip(*xyz_pts)

        ax.plot(x_vals, y_vals, z_vals, color='red')

        # Plot the remaining points of the top ROI
        # ax.scatter(top_roi_data['x'], top_roi_data['y'], top_roi_data['z'], c='darkblue', alpha=0.7, s=0.5)
    
    # Plot the middle xz projection points (y around mid_y)
    middle_projection_data = filtered_projection_data[
        (filtered_projection_data['y'] >= (mid_y - 0.5)) & (filtered_projection_data['y'] <= (mid_y + 0.5))
    ]
    
    max_z_plane_data = volume_data[volume_data['z'] == max_z]
    ax.scatter(max_z_plane_data['x'], max_z_plane_data['y'], max_z_plane_data['z'], 
               c='green', alpha=0.7, s=2)

    # Scatter the middle projection points
    ax.scatter(middle_projection_data['x'], middle_projection_data['y'], middle_projection_data['z'], 
            c='orange', alpha=0.7, s=0.5)
    
    # Set labels and show plot
    ax.set_title('Top ROI and Middle XZ Projection Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
            

def plot_roi_coloring(volume_data=None, volume_id=1, roi_interval=5, green_ids=None, red_ids=None):
    # Load the CSV containing the volume data
    file_path = '../VALIDATION/validation_runs/drg_subset_1_r/real_data_filtered_v0_VOLUMES.csv'
    df = pd.read_csv(file_path)

    # Load the volume data for the specific volume_id
    volume_data = df[df['VOLUME_ID'] == volume_id]
    max_z = volume_data['z'].max()

    # Load the CSV containing the projection points
    projection_file_path = '../VOLUME_SEGMENTATION/xz_cache/real_data_filtered_v0_ROIS_XZ.csv'
    projection_data = pd.read_csv(projection_file_path)

    # Get the min and max values from the current volume data
    x_min, x_max = volume_data['x'].min(), volume_data['x'].max()
    y_min, y_max = volume_data['y'].min(), volume_data['y'].max()
    z_min, z_max = volume_data['z'].min(), volume_data['z'].max()

    # Find the unique sorted y-values
    y_values = sorted(volume_data['y'].unique())

    # # Filter projection points to be within the x, y, z range of the current volume
    filtered_projection_data = projection_data[
        (projection_data['x'] >= x_min) & (projection_data['x'] <= x_max) &
        (projection_data['y'] >= y_min) & (projection_data['y'] <= y_max) &
        (projection_data['z'] >= z_min) & (projection_data['z'] <= z_max)
    ]

    # Apply a tolerance to select y-planes that are close to multiples of roi_interval
    tolerance = 0.99
    # Get the unique sorted y-values
    y_values = np.unique(filtered_projection_data['y'])
    # Determine the gaps between consecutive y-values
    y_gaps = np.diff(y_values)
    # Assume the most common y-gap as the constant interval (mode)
    y_gap = np.median(y_gaps)  # Use median instead of mode for robustness
    filtered_projection_data = filtered_projection_data[
        np.abs(filtered_projection_data['y'] % (roi_interval * y_gap)) <= tolerance
    ]

    # Set up 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-20)

    # Group points by y-plane (for the ROI)
    grouped_data = filtered_projection_data.groupby('y')
    
    # Assign an ID to each ROI and color them according to user specification
    roi_id = 0
    for y_plane, group in grouped_data:
        # Get the x, z coordinates for the current y-plane
        points = group[['x', 'z']].values
        
        # Only create a convex hull if there are enough points (at least 3)
        if len(points) >= 3:
            hull = ConvexHull(points)
            roi_id += 1  # Increment ROI ID for each new region
            
            # Check if this ROI ID should be colored green or red
            color = 'darkblue'  # Default color
            if green_ids and roi_id in green_ids:
                color = 'green'
            elif red_ids and roi_id in red_ids:
                color = 'red'

            # Plot the convex hull
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], [y_plane] * len(simplex), points[simplex, 1], color=color, linewidth=2)
        else:
             # If fewer than 3 points, plot them directly
            roi_id += 1  # Increment ROI ID for each new region
            
            # Check if this ROI ID should be colored green or red
            color = 'darkblue'
            if green_ids and roi_id in green_ids:
                color = 'green'
            elif red_ids and roi_id in red_ids:
                color = 'red'
            
            # Plot the individual points
            ax.scatter(points[:, 0], [y_plane] * len(points), points[:, 1], color=color, s=10)

    
    max_z_plane_data = volume_data[volume_data['z'] == max_z]
    # ax.scatter(max_z_plane_data['x'], max_z_plane_data['y'], max_z_plane_data['z'], 
    #            c='green', alpha=0.7, s=2)
    # Extract x, y values from the max_z_plane_data
    xy_points = list(zip(max_z_plane_data['x'].values, max_z_plane_data['y'].values))

    # Find the centroid of the points in max_z_plane
    centroid = find_centroid(xy_points)

    # Sort the points by angle relative to the centroid
    sorted_points = sort_points_by_angle(xy_points, centroid)

    # Separate the sorted x and y values
    x_vals = [p[0] for p in sorted_points]
    y_vals = [p[1] for p in sorted_points]

    # Plot a continuous line for the max z-plane using sorted points
    ax.plot(x_vals, y_vals, [max_z] * len(x_vals), color='green', alpha=0.7, linewidth=2)

    # Set labels and show plot
    ax.set_title('Volumetric Segmentation via XY and XZ ROI Intersection')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_roi_coloring2(volume_data=None, volume_id=1, roi_interval=5, green_ids=None, red_ids=None):
    # Load the CSV containing the volume data
    file_path = '../VALIDATION/validation_runs/drg_subset_1_r/real_data_filtered_v0_VOLUMES.csv'
    df = pd.read_csv(file_path)

    # Load the volume data for the specific volume_id
    volume_data = df[df['VOLUME_ID'] == volume_id]
    max_z = volume_data['z'].max()

    # Load the CSV containing the projection points
    projection_file_path = '../VOLUME_SEGMENTATION/xz_cache/real_data_filtered_v0_ROIS_XZ.csv'
    projection_data = pd.read_csv(projection_file_path)

    # Get the min and max values from the current volume data
    x_min, x_max = volume_data['x'].min(), volume_data['x'].max()
    y_min, y_max = volume_data['y'].min(), volume_data['y'].max()
    z_min, z_max = volume_data['z'].min(), volume_data['z'].max()

    # Find the unique sorted y-values
    y_values = sorted(volume_data['y'].unique())

    # # Filter projection points to be within the x, y, z range of the current volume
    filtered_projection_data = projection_data[
        (projection_data['x'] >= x_min) & (projection_data['x'] <= x_max) &
        (projection_data['y'] >= y_min) & (projection_data['y'] <= y_max) &
        (projection_data['z'] >= z_min) & (projection_data['z'] <= z_max)
    ]

    # Apply a tolerance to select y-planes that are close to multiples of roi_interval
    tolerance = 0.99
    # Get the unique sorted y-values
    y_values = np.unique(filtered_projection_data['y'])
    # Determine the gaps between consecutive y-values
    y_gaps = np.diff(y_values)
    # Assume the most common y-gap as the constant interval (mode)
    y_gap = np.median(y_gaps)  # Use median instead of mode for robustness
    filtered_projection_data = filtered_projection_data[
        np.abs(filtered_projection_data['y'] % (roi_interval * y_gap)) <= tolerance
    ]

    # Set up 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-20)

    # Group points by y-plane (for the ROI)
    grouped_data = filtered_projection_data.groupby('y')
    
    # Assign an ID to each ROI and color them according to user specification
    roi_id = 0
    for y_plane, group in grouped_data:
        # Get the x, z coordinates for the current y-plane
        points = group[['x', 'z']].values
        
        # Only create a convex hull if there are enough points (at least 3)
        if len(points) >= 3:
            hull = ConvexHull(points)
            roi_id += 1  # Increment ROI ID for each new region
            
            # Check if this ROI ID should be colored green or red
            color = 'darkblue'  # Default color
            if green_ids and roi_id in green_ids:
                color = 'green'
            elif red_ids and roi_id in red_ids:
                color = 'red'

            # Plot the convex hull
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], [y_plane] * len(simplex), points[simplex, 1], color=color, linewidth=2)
        else:
             # If fewer than 3 points, plot them directly
            roi_id += 1  # Increment ROI ID for each new region
            
            # Check if this ROI ID should be colored green or red
            color = 'darkblue'
            if green_ids and roi_id in green_ids:
                color = 'green'
            elif red_ids and roi_id in red_ids:
                color = 'red'
            
            # Plot the individual points
            ax.scatter(points[:, 0], [y_plane] * len(points), points[:, 1], color=color, s=10)

    volume_data = volume_data[volume_data['z'] % roi_interval == 0]
    z_unique = volume_data['z'].unique()
    # ax.scatter(max_z_plane_data['x'], max_z_plane_data['y'], max_z_plane_data['z'], 
    #            c='green', alpha=0.7, s=2)
    # Extract x, y values from the max_z_plane_data
    for z in z_unique:
        z_plane_data = volume_data[volume_data['z'] == z]
        xy_points = list(zip(z_plane_data['x'].values, z_plane_data['y'].values))

        # Find the centroid of the points in max_z_plane
        centroid = find_centroid(xy_points)

        # Sort the points by angle relative to the centroid
        sorted_points = sort_points_by_angle(xy_points, centroid)

        # Separate the sorted x and y values
        x_vals = [p[0] for p in sorted_points]
        y_vals = [p[1] for p in sorted_points]

        # Plot a continuous line for the max z-plane using sorted points
        ax.plot(x_vals, y_vals, [z] * len(x_vals), color='green', alpha=0.7, linewidth=2)

    # Set labels and show plot
    ax.set_title('Volumetric Segmentation via XY and XZ ROI Intersection')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_roi_restrictions(volume_data=None, n_rois = 3, volume_id=1, roi_interval=5, green_ids=None, red_ids=None, gap_interval = 3):
    # Load the CSV containing the volume data
    file_path = '../VALIDATION/validation_runs/drg_subset_1_r/real_data_filtered_v0_VOLUMES.csv'
    df = pd.read_csv(file_path)

    # Load the volume data for the specific volume_id
    volume_data = df[df['VOLUME_ID'] == volume_id]
    max_z = volume_data['z'].max()

    # Load the CSV containing the projection points
    projection_file_path = '../VOLUME_SEGMENTATION/xz_cache/real_data_filtered_v0_ROIS_XZ.csv'
    projection_data = pd.read_csv(projection_file_path)

    # Get the min and max values from the current volume data
    x_min, x_max = volume_data['x'].min(), volume_data['x'].max()
    y_min, y_max = volume_data['y'].min(), volume_data['y'].max()
    z_min, z_max = volume_data['z'].min(), volume_data['z'].max()

    # Find the unique sorted y-values
    y_values = sorted(volume_data['y'].unique())

    # # Filter projection points to be within the x, y, z range of the current volume
    filtered_projection_data = projection_data[
        (projection_data['x'] >= x_min) & (projection_data['x'] <= x_max) &
        (projection_data['y'] >= y_min) & (projection_data['y'] <= y_max) &
        (projection_data['z'] >= z_min) & (projection_data['z'] <= z_max)
    ]

    # Apply a tolerance to select y-planes that are close to multiples of roi_interval
    tolerance = 0.99
    # Get the unique sorted y-values
    y_values = np.unique(filtered_projection_data['y'])
    # Determine the gaps between consecutive y-values
    y_gaps = np.diff(y_values)
    # Assume the most common y-gap as the constant interval (mode)
    y_gap = np.median(y_gaps)  # Use median instead of mode for robustness
    filtered_projection_data = filtered_projection_data[
        np.abs(filtered_projection_data['y'] % (roi_interval * y_gap)) <= tolerance
    ]

    # Set up 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-20)

    # Group points by y-plane (for the ROI)
    grouped_data = filtered_projection_data.groupby('y')
    
    # Assign an ID to each ROI and color them according to user specification
    roi_id = 0
    for y_plane, group in grouped_data:
        # Get the x, z coordinates for the current y-plane
        points = group[['x', 'z']].values
        
        # Only create a convex hull if there are enough points (at least 3)
        if len(points) >= 3:
            hull = ConvexHull(points)
            roi_id += 1  # Increment ROI ID for each new region
            
            # Check if this ROI ID should be colored green or red
            color = 'darkblue'  # Default color
            if green_ids and roi_id in green_ids:
                color = 'green'
            elif red_ids and roi_id in red_ids:
                color = 'red'

            # Plot the convex hull
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], [y_plane] * len(simplex), points[simplex, 1], color=color, linewidth=2)
        else:
             # If fewer than 3 points, plot them directly
            roi_id += 1  # Increment ROI ID for each new region
            
            # Check if this ROI ID should be colored green or red
            color = 'darkblue'
            if green_ids and roi_id in green_ids:
                color = 'green'
            elif red_ids and roi_id in red_ids:
                color = 'red'
            
            # Plot the individual points
            ax.scatter(points[:, 0], [y_plane] * len(points), points[:, 1], color=color, s=10)

    for i in range(n_rois):
        z =  max_z - (i * gap_interval)
        z_plane_data = volume_data[volume_data['z'] == z]
        # ax.scatter(max_z_plane_data['x'], max_z_plane_data['y'], max_z_plane_data['z'], 
        #            c='green', alpha=0.7, s=2)
        # Extract x, y values from the max_z_plane_data
        xy_points = list(zip(z_plane_data['x'].values, z_plane_data['y'].values))

        # Find the centroid of the points in max_z_plane
        centroid = find_centroid(xy_points)

        # Sort the points by angle relative to the centroid
        sorted_points = sort_points_by_angle(xy_points, centroid)

        # Separate the sorted x and y values
        x_vals = [p[0] for p in sorted_points]
        y_vals = [p[1] for p in sorted_points]

        # Plot a continuous line for the max z-plane using sorted points
        ax.plot(x_vals, y_vals, [z] * len(x_vals), color='green', alpha=0.7, linewidth=2)

    # Plot data from the XY ROI at max-z with an x and y shift of +10, in red
    max_z_plane_data = volume_data[volume_data['z'] == max_z]
    last_roi_z = max_z - ((n_rois-1) * gap_interval)
    last_roi_plane_data = volume_data[volume_data['z'] == last_roi_z]
    shifted_x = max_z_plane_data['x'] + 20  # Shift x 
    shifted_y = max_z_plane_data['y'] + 10  # Shift y
    shifted_z = last_roi_z - gap_interval

    last_roi_points = list(zip(last_roi_plane_data['x'], last_roi_plane_data['y']))
    lr_cx, lr_cy = find_centroid(last_roi_points)
    shifted_points = list(zip(shifted_x, shifted_y))
    sh_cx, sh_cy = find_centroid(shifted_points)
    last_roi_centroid = (lr_cx, lr_cy, last_roi_z)
    shifted_points_centroid = (sh_cx, sh_cy, shifted_z)

    # Unpack the coordinates of the two centroids
    centroid_x = [last_roi_centroid[0], shifted_points_centroid[0]]  # X coordinates of both centroids
    centroid_y = [last_roi_centroid[1], shifted_points_centroid[1]]  # Y coordinates of both centroids
    centroid_z = [last_roi_centroid[2], shifted_points_centroid[2]]  # Z coordinates of both centroids

    # Plot a dotted line between the two centroids
    ax.plot(centroid_x, centroid_y, centroid_z, color='orange', linestyle='dotted', linewidth=2, label="Centroid Line")


    # Scatter plot the shifted points in red
    xy_points = list(zip(shifted_x.values, shifted_y.values))

    # Find the centroid of the points in max_z_plane
    centroid = find_centroid(xy_points)

    # Sort the points by angle relative to the centroid
    sorted_points = sort_points_by_angle(xy_points, centroid)

    # Separate the sorted x and y values
    x_vals = [p[0] for p in sorted_points]
    y_vals = [p[1] for p in sorted_points]

    # Plot a continuous line for the max z-plane using sorted points
    ax.plot(x_vals, y_vals, [last_roi_z - gap_interval] * len(x_vals), color='orange', alpha=0.7, linewidth=2.5)

    # ax.scatter(shifted_x, shifted_y, shifted_z, color='orange', alpha=0.7, s=2, label="Shifted ROI at max-z")


    # Set labels and show plot
    ax.set_title('Volumetric Segmentation via XY and XZ ROI Intersection')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_roi_coloring_2(volume_data=None, volume_id=1, roi_interval=5, green_ids=None, red_ids=None):
    # Load the CSV containing the volume data
    file_path = '../VALIDATION/validation_runs/drg_subset_1_r/real_data_filtered_v0_VOLUMES.csv'
    df = pd.read_csv(file_path)

    # Load the volume data for the specific volume_id
    volume_data = df[df['VOLUME_ID'] == volume_id]
    max_z = volume_data['z'].max()

    # Load the CSV containing the projection points
    projection_file_path = '../VOLUME_SEGMENTATION/xz_cache/real_data_filtered_v0_ROIS_XZ.csv'
    projection_data = pd.read_csv(projection_file_path)

    # Get the min and max values from the current volume data
    x_min, x_max = volume_data['x'].min(), volume_data['x'].max()
    y_min, y_max = volume_data['y'].min(), volume_data['y'].max()
    z_min, z_max = volume_data['z'].min(), volume_data['z'].max()

    # Find the unique sorted y-values
    y_values = sorted(volume_data['y'].unique())

    # Filter projection points to be within the x, y, z range of the current volume
    filtered_projection_data = projection_data[
        (projection_data['x'] >= x_min) & (projection_data['x'] <= x_max) &
        (projection_data['y'] >= y_min) & (projection_data['y'] <= y_max) &
        (projection_data['z'] >= z_min) & (projection_data['z'] <= z_max)
    ]

    # Apply a tolerance to select y-planes that are close to multiples of roi_interval
    tolerance = 0.99
    y_values = np.unique(filtered_projection_data['y'])
    y_gaps = np.diff(y_values)
    y_gap = np.median(y_gaps)  # Use median instead of mode for robustness
    filtered_projection_data = filtered_projection_data[
        np.abs(filtered_projection_data['y'] % (roi_interval * y_gap)) <= tolerance
    ]

    # Set up 3D plot (Main plot with ROIs)
    fig, (ax_main, ax_right) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(15, 7))

    # Plot the filtered projection points directly in main subplot
    ax_main.scatter(filtered_projection_data['x'], filtered_projection_data['y'], filtered_projection_data['z'], 
                    c='blue', alpha=0.5, s=5)

    # Group points by y-plane (for the ROI)
    grouped_data = filtered_projection_data.groupby('y')
    
    # Assign an ID to each ROI and color them according to user specification
    roi_id = 0
    for y_plane, group in grouped_data:
        points = group[['x', 'z']].values
        
        if len(points) >= 3:
            hull = ConvexHull(points)
            roi_id += 1  # Increment ROI ID for each new region
            
            # Check if this ROI ID should be colored green or red
            color = 'darkblue'
            if green_ids and roi_id in green_ids:
                color = 'green'
            elif red_ids and roi_id in red_ids:
                color = 'red'

            # Plot the convex hull in the main plot
            for simplex in hull.simplices:
                ax_main.plot(points[simplex, 0], [y_plane] * len(simplex), points[simplex, 1], color=color, linewidth=2)
                ax_right.plot(points[simplex, 0], [y_plane] * len(simplex), points[simplex, 1], color='green', linewidth=2)
        else:
            roi_id += 1
            
            color = 'darkblue'
            if green_ids and roi_id in green_ids:
                color = 'green'
            elif red_ids and roi_id in red_ids:
                color = 'red'
            
            # Plot the individual points in the main plot
            ax_main.scatter(points[:, 0], [y_plane] * len(points), points[:, 1], color=color, s=10)

    # Sort and plot the max_z_plane_data as a continuous line in the main plot
    max_z_plane_data = volume_data[volume_data['z'] == max_z]
    xy_points = list(zip(max_z_plane_data['x'].values, max_z_plane_data['y'].values))
    centroid = find_centroid(xy_points)
    sorted_points = sort_points_by_angle(xy_points, centroid)
    x_vals = [p[0] for p in sorted_points]
    y_vals = [p[1] for p in sorted_points]
    ax_main.plot(x_vals, y_vals, [max_z] * len(x_vals), color='green', alpha=0.7, linewidth=2)

    # Plot both XZ and XY ROIs in green on the right subplot
    for y_plane, group in grouped_data:
        points_xz = group[['x', 'z']].values
        points_xy = group[['x', 'y']].values
        
        # Plot only the boundary points of XZ ROIs
        if len(points_xz) >= 3:
            hull_xz = ConvexHull(points_xz)
            for simplex in hull_xz.simplices:
                pass
                # 

    ax_right.scatter(volume_data['x'], volume_data['y'], volume_data['z'], color='green', alpha=0.5)

    # Set labels and show plots
    ax_main.set_title('Main Plot: ROIs with Color-Coded IDs')
    ax_main.set_xlabel('X')
    ax_main.set_ylabel('Y')
    ax_main.set_zlabel('Z')

    ax_right.set_title('Right Subplot: XZ & XY ROIs (Green)')
    ax_right.set_xlabel('X')
    ax_right.set_ylabel('Y')

    plt.show()



def generate_xz_single_y(points, max_z, min_y = None, max_y = None, include_xy_id = False):
    xz_roi_points = []
    # Filter points only for the highest z-plane
    points_on_z = [(x, y, z, id) for x, y, z, id in points if z == max_z]
    
    # Determine y plane intervals (same as your original code)
    if min_y is None:
        min_y = min([y for _, y, _, _ in points_on_z])
    if max_y is None:
        max_y = max([y for _, y, _, _ in points_on_z])
    num_y_slices = int(np.ceil((max_y - min_y) * Y_SAMPLE_RATIO))
    y_points = np.linspace(min_y, max_y, num_y_slices)
    
    # Build XY polygons (for current z-plane)
    z_planes = {}
    for x, y, z, id in points_on_z:
        if not z_planes.get(z, None):
            z_planes[z] = {}
        if not z_planes[z].get(id, None):
            z_planes[z][id] = []
        z_planes[z][id].append((x, y))

    z_polys = {}
    for z, id_dict in z_planes.items():
        z_polys[z] = {}
        for id, xy_list in id_dict.items():
            z_polys[z][id] = {}  # Default sentinel value
            
            # Calculate centroid and sorted points
            cx, cy = find_centroid(xy_list)
            sorted_points = sort_points_by_angle(xy_list, (cx, cy))
            sorted_points.append(sorted_points[0])  # Close polygon
            
            # Create a polygon if there are enough points
            poly = None
            if len(sorted_points) >= 4:
                poly = Polygon(sorted_points)

            # Save polygon metadata
            z_polys[z][id]['poly'] = poly
            z_polys[z][id]['min_x'] = min(x for x, _ in xy_list)
            z_polys[z][id]['max_x'] = max(x for x, _ in xy_list)
            z_polys[z][id]['min_y'] = min(y for _, y in xy_list)
            z_polys[z][id]['max_y'] = max(y for _, y in xy_list)
    
    return z_polys

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

if __name__ == "__main__":
    main()