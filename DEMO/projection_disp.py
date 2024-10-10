import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from shapely import Polygon, Point
from scipy.spatial import ConvexHull

NUM_XZ_SLICES = 60
INTERPOLATION_RESOLUTION = 10
Y_SAMPLE_RATIO = 1 # Number of y points to sample per pixel
X_SAMPLE_RATIO = 1 # Number of x points to sample per pixel

def show_projection_start():
    file_path = '../VALIDATION/validation_runs/drg_subset_1_r/real_data_filtered_v0_VOLUMES.csv'
    df = pd.read_csv(file_path)
    volume_id = 1
    # First, load the CSV and generate the necessary polygons
    volume_data = df[df['VOLUME_ID'] == volume_id]
    max_z = volume_data['z'].max()

    # Prepare the points and generate polygons for the max z-plane
    points = list(zip(volume_data['x'], volume_data['y'], volume_data['z'], volume_data['ROI_ID']))
    z_polys = generate_xz_single_y(points, max_z)

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    volume_data_adj = volume_data[volume_data['z'] != max_z]
    volume_data_adj = volume_data_adj[volume_data_adj['z'] % 5 == 0]
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
    num_lines = 20  # Define how many lines you want to generate
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
                    ax.scatter(x, y, z, color='red', s=2)  # Darker blue for points inside the polygon


    # Customize and show plot
    ax.set_title(f'3D Volume and Projected Lines (z = {max_z}) for Volume {volume_id}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def main():
    # show_projection_start()
    # plot_projection()
    plot_intersection()

def plot_projection(volume_data = None):
    # Load the CSV containing the volume data
    file_path = '../VALIDATION/validation_runs/drg_subset_1_r/real_data_filtered_v0_VOLUMES.csv'
    df = pd.read_csv(file_path)
    volume_id = 1

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

    # Apply a tolerance to select y-planes that are close to multiples of 5
    tolerance = 0.99
    filtered_projection_data = filtered_projection_data[
        np.abs(filtered_projection_data['y'] % 5) <= tolerance
    ]

    # Set up 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Group points by y-plane
    grouped_data = filtered_projection_data.groupby('y')

    removed_pts = []
    # Loop through each y-plane and plot the convex hull
    for y_plane, group in grouped_data:
        # Get the x, z coordinates for the current y-plane
        points = group[['x', 'z']].values
        
        # Only create a convex hull if there are enough points (at least 3)
        if len(points) >= 3:
            hull = ConvexHull(points)
           
            for simplex in hull.simplices:
                for vertex in simplex:
                    removed_pts.append((points[vertex, 0], y_plane, points[vertex, 1]))
            # Plot the boundar           y of the convex hull in red using the original points
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], [y_plane] * len(simplex), points[simplex, 1], color='red', linewidth=2)
        
        # Filter out points that are in the convex hull (removed points)
        filtered_group = group[~group.apply(lambda row: (row['x'], row['y'], row['z']) in removed_pts, axis=1)]

        # Scatter all points, but the convex hull points will be outlined with red
        ax.scatter(filtered_group['x'], filtered_group['y'], filtered_group['z'], c='darkblue', alpha=0.7, s=0.5)

            

    # Set labels and show plot
    ax.set_title('Filtered Projection Points with Convex Hull Outlines per y-plane')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_intersection(volume_data = None):
    # Load the CSV containing the volume data
    file_path = '../VALIDATION/validation_runs/drg_subset_1_r/real_data_filtered_v0_VOLUMES.csv'
    df = pd.read_csv(file_path)
    volume_id = 1

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

    # Apply a tolerance to select y-planes that are close to multiples of 5
    tolerance = 0.99
    filtered_projection_data = filtered_projection_data[
        np.abs(filtered_projection_data['y'] % 5) <= tolerance
    ]

       # Set up 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the top ROI at max_y (Top y-plane)
    top_roi_data = volume_data[volume_data['y'] == mid_y]
    
    # Get the x, z coordinates for the top ROI
    if len(top_roi_data) >= 3:
        points = top_roi_data[['x', 'z']].values
        hull = ConvexHull(points)
        
        # Plot the boundary of the convex hull in red
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], [mid_y] * len(simplex), points[simplex, 1], color='red', linewidth=2)
        
        # Plot the remaining points of the top ROI
        ax.scatter(top_roi_data['x'], top_roi_data['y'], top_roi_data['z'], c='darkblue', alpha=0.7, s=0.5)
    
    # Plot the middle xz projection points (y around mid_y)
    middle_projection_data = filtered_projection_data[
        (filtered_projection_data['y'] >= (mid_y - 0.5)) & (filtered_projection_data['y'] <= (mid_y + 0.5))
    ]
    
    max_z_plane_data = volume_data[volume_data['z'] == max_z]
    ax.scatter(max_z_plane_data['x'], max_z_plane_data['y'], max_z_plane_data['z'], 
               c='red', alpha=0.7, s=0.5)

    # Scatter the middle projection points
    ax.scatter(middle_projection_data['x'], middle_projection_data['y'], middle_projection_data['z'], 
            c='orange', alpha=0.7, s=0.5)
    
    # Set labels and show plot
    ax.set_title('Top ROI and Middle XZ Projection Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
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