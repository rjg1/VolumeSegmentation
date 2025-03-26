import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os
from scipy.spatial import cKDTree

PLOT_TITLE = "Sample plane generation"
CSV_FILENAME = 'real_data_filtered_algo_VOLUMES_g.csv'  # Input volume csv
MAX_POINTS = 10000 # Maximum points to sample
AX_LIMIT_MAX = 1024

def main():
    random.seed(13)
    
    # Use the parsed arguments or fallback to the defaults if no arguments are provided
    # Load and sample the data
    sampled_df = load_and_sample_data(CSV_FILENAME, MAX_POINTS)

    # Assign colors based on volume IDs
    colors = assign_colors_by_volume_id(sampled_df)

    # Define a plane at an arbitrary z-level and angle
    center = sampled_df[['x', 'y', 'z']].mean().to_numpy()
    normal, d = define_plane(center, [('X', 2)])

    # Get points near this plane
    df_near, pts_near = get_points_near_plane(sampled_df, normal, d, threshold=2)
    projected_pts = project_points_onto_plane(pts_near, normal, center)
    print(projected_pts)

    # Visualize the points with assigned colors
    visualize_points(sampled_df, colors, PLOT_TITLE, projected_pts, normal, center)


def get_rot_matrix(rotation):
    axis, degrees = rotation
    radians = np.deg2rad(degrees)

    axis = axis.lower()

    match axis:
        case 'x':
            return  np.array([
                            [1, 0, 0],
                            [0, np.cos(radians), -np.sin(radians)],
                            [0, np.sin(radians), np.cos(radians)]
                        ])
        case 'y':
            return  np.array([
                            [np.cos(radians), 0, np.sin(radians)],
                            [0, 1, 0],
                            [-np.sin(radians), 0, np.cos(radians)]
                        ])
        case 'z':
            return  np.array(
                            [[np.cos(radians), -np.sin(radians), 0],
                            [np.sin(radians), np.cos(radians), 0],
                            [0, 0, 1]
                        ])
        case _:
            return None

def define_plane(center, rotations):
    """
    Define a plane tilted by theta1 degrees in the y–z plane,
    passing through a given center point.
    """
    # Perpendicular to flat x-y plane
    normal = np.array([0,0,1]) # Perpendicular to flat x-y plane
    
    # Iterate through all rotations sequentially and apply them
    for rotation in rotations:
        rotation_matrix = get_rot_matrix(rotation)
        if rotation is not None:
            normal = np.matmul(normal, rotation_matrix)

    # Plane equation: n . (x, y, z) = d
    d = np.dot(normal, center)
    return normal, d

def project_points_onto_plane(points, normal, center):
    """
    Orthogonally project 3D points onto the defined plane.
    """
    normal = normal / np.linalg.norm(normal)
    relative = points - center
    dot = np.dot(relative, normal)
    projections = points - np.outer(dot, normal)
    return projections

def get_points_near_plane(df, normal, d, threshold=0.5):
    """
    Return all points in df that are within `threshold` distance of the plane.
    """
    points = df[['x', 'y', 'z']].to_numpy()
    distances = np.abs(points @ normal - d) # dot product of points and normal. Denominator of absolute value of normal disappears as it is 1
    mask = distances < threshold
    return df[mask], points[mask]

def load_and_sample_data(filename, sample_point_max):
    """Load the CSV data and sample a subset of points for visualization."""
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    df = df[df['VOLUME_ID'] != -1] # Filter noise points
    # Test filter good parts of DRG
    # df = df[df['z'] <= 80]
    # df = df[df['z'] >= 30]

    # Display the first few rows to check data
    # print("Loaded Data Preview:")
    # print(df.head())
    subset_ratio = min(1, sample_point_max/len(df))
    # Check if volume IDs are present
    if 'VOLUME_ID' not in df.columns:
        raise ValueError("CSV does not contain VOLUME_ID column. Ensure export was done with volumes enabled.")

    # Sample a subset of the data for visualization
    sampled_df = df.sample(frac=subset_ratio, random_state=42)  # Randomly sample a subset for plotting
    print(f"Sampled {len(sampled_df)} points out of {len(df)} total points.")

    return sampled_df

def assign_colors_by_volume_id(df):
    """Assign colors to points based on their VOLUME_ID."""
    # Extract unique volume IDs and assign a color for each ID
    unique_volume_ids = df['VOLUME_ID'].unique()
    num_volumes = len(unique_volume_ids)
    print(f"{num_volumes} volumes found")

    # Generate random colors for each volume ID
    color_map = {vol_id: [random.random(), random.random(), random.random()] for vol_id in unique_volume_ids}

    # Map colors to each row in the DataFrame
    colors = df['VOLUME_ID'].map(color_map)

    # Convert to a numpy array for Matplotlib
    colors_array = np.array(colors.tolist())

    return colors_array

def visualize_points(df, colors, title, projected_pts, normal=None, center=None):
    """Visualize 3D data + 2D projected slice (in x/y view) using consistent colors and axes."""
    fig = plt.figure(figsize=(16, 8))

    # === LEFT SUBPLOT: 3D original data ===
    ax1 = fig.add_subplot(121, projection='3d')
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()

    ax1.scatter(x, y, z, c=colors, s=1, alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f"{title} (3D view)")

    # Axis limits for consistency
    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(0, 80)

    # === Plot rectangle on the plane in 3D ===
    if normal is not None and center is not None:
        normal = normal / np.linalg.norm(normal)
        nx, ny, nz = normal
        d = np.dot(normal, center)

        # Define x/y corners using data bounds
        x0, x1 = x.min(), x.max()
        y0, y1 = y.min(), y.max()

        rect_xy = np.array([
            [x1, y1],
            [x1, y0],
            [x0, y0],
            [x0, y1],
            [x1, y1]
        ])

        # Compute z values using plane equation
        rect_z = [(d - nx * x_val - ny * y_val) / nz for x_val, y_val in rect_xy]

        # Stack into 3D points
        rect_3d = np.column_stack((rect_xy, rect_z))
        ax1.plot(rect_3d[:, 0], rect_3d[:, 1], rect_3d[:, 2], color='red', linewidth=2)

    # === RIGHT SUBPLOT: 2D view of projected points ===
    ax2 = fig.add_subplot(122)
    ax2.set_title(f"{title} (Projected X/Y)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    if projected_pts is not None and len(projected_pts) > 0:
        # Get x/y of projected points
        x_proj = projected_pts[:, 0]
        y_proj = projected_pts[:, 1]

        # Match colors to corresponding points
        # We assume projected_pts[i] corresponds to df_near.iloc[i]
        # So we need to subset colors to just those points

        # Find mask of close points (from projected_pts) against original df
        tree = cKDTree(df[['x', 'y', 'z']].to_numpy())
        _, idxs = tree.query(projected_pts, k=1)
        projected_colors = colors[idxs]

        ax2.scatter(x_proj, y_proj, s=1, c=projected_colors)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_aspect('equal')

    # set_axes_equal_3d(ax1)

    plt.tight_layout()
    plt.show()

def set_axes_equal_3d(ax):
    """Make 3D axes have equal scale (otherwise aspect is distorted)."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

if __name__ == "__main__":
    main()