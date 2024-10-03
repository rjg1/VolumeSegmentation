# IMPORTS
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, Delaunay, QhullError
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from region import ClusterRegion, BoundaryRegion
from volume import Volume
import open3d as o3d
import random
import os
import pandas as pd
from scipy.spatial import distance
from xz_projection import generate_xz_single_y, plot_xz_planes
from xz_segmentation import cluster_xz_rois_tuned
from volume_seg_ffp import segment_volumes_drg

# GLOBAL VARIABLES
# Generate bonus points per ROI 
BONUS_ROI_POINTS = 300
# Hyperparameter alpha for generating an AlphaShape
ALPHASHAPE_ALPHA = 0.3

def main():
    # Dummy values - TBD
    volumes = {}
    clustered_hulls = {}
    xz_roi_points = []
    #  Test point generation
    for volume_id, volume in volumes.items():
        roi_list = [volume.get_xz_rois(), volume.get_xy_rois()]
        for roi_dict in roi_list:
            for roi_id, roi in roi_dict.items():
                roi.add_points(roi.generate_region_points(BONUS_ROI_POINTS))

    # Plot XZ ROI areas
    colour_ids = []
    unique_y = list(set([y for _,y, z in xz_roi_points]))
    for x, y, z in xz_roi_points:
        colour_ids.append(unique_y.index(y))

    # # Plot XY ROI boundaries
    plot_xyz_points(xz_roi_points, colour_ids)
    # # Show the plot
    plt.show()


    # Plot Clustered XZ ROIs in 3D
    plot_clustered_hulls(clustered_hulls)

    # Plot volume output
    plot_segmented_volumes(volumes)

    # Generate surface
    generate_alpha_surfaces(volumes)
    gen_mesh_2(volumes)

def plot_xyz_points(points, colour_ids):
    # Unzip the list of points into three lists: x, y, z
    x_vals, y_vals, z_vals = zip(*points)
    # Create a colormap
    cmap = cm.get_cmap('bone')

    if colour_ids:
        # Normalize the color IDs to range between 0 and 1 for the colormap
        norm = plt.Normalize(min(colour_ids), max(colour_ids))

        # Map color IDs to colors using the colormap
        colours = cmap(norm(colour_ids))
    else:
        colours = 'b'

    # Create a new figure for plotting
    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points as a scatter plot
    ax.scatter(x_vals, y_vals, z_vals, c=colours, marker='o')

    # Label the axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Set the title
    ax.set_title('3D Scatter Plot of Points')


def plot_segmented_volumes(volumes):
    # Create a new figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate a random color for each volume
    colours = {}
    for volume_id in volumes:
        colours[volume_id] = [random.random() for _ in range(3)]  # RGB color

    # Iterate through each volume
    for volume_id, volume in volumes.items():
        print(f"Showing volume: {volume_id}")
        base_color = colours[volume_id]
        xz_color =  colours[volume_id]
        xy_color =  colours[volume_id]
        # Plot the XZ ROIs
        for xz_id, xz_roi in volume.get_xz_rois().items():
            boundary_points = xz_roi.get_boundary_points()
            if boundary_points:
                x_points, y_points, z_points = zip(*boundary_points)
                ax.scatter(x_points, y_points, z_points, color=xz_color, label=f'Volume {volume_id} - XZ {xz_id}')

        # Plot the XY ROIs
        for xy_id, xy_roi in volume.get_xy_rois().items():
            boundary_points = xy_roi.get_boundary_points()
            if boundary_points:
                x_points, y_points, z_points = zip(*boundary_points)
                ax.scatter(x_points, y_points, z_points, color=xy_color, label=f'Volume {volume_id} - XY {xy_id}')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Plot of Volume ROIs')
    plt.show()

def gen_mesh_2(volumes):
    # List to store all meshes
    all_meshes = []

    for volume_id, volume in volumes.items():
        points = []
        roi_list = [volume.get_xz_rois(), volume.get_xy_rois()]
        
        # Collect all points from the ROIs
        for roi_dict in roi_list:
            for roi_id, roi in roi_dict.items():
                points += roi.points
        
        # Convert the list of tuples to a numpy array
        points = np.array(points)

        if len(points) == 0:
            continue  # Skip if there are no points to process

        # Create an Open3D PointCloud object from the points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate the alpha shape (concave hull)
        alpha = 0.3  # Adjust this value depending on the density and scale of your points
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        # Estimate normals for shading
        mesh.compute_vertex_normals()

        # Generate a color based on the volume_id
        color = generate_color_from_id(volume_id)
        mesh.paint_uniform_color(color)

        # Add the mesh to the list of all meshes
        all_meshes.append(mesh)

    # Visualize all meshes together with smooth shading
    o3d.visualization.draw_geometries(all_meshes, mesh_show_back_face=True)

# Function to generate a color based on the volume_id
def generate_color_from_id(volume_id):
    np.random.seed(volume_id)  # Seed the random generator with volume_id
    return np.random.rand(3)  # Generate a random color


def generate_alpha_surfaces(volumes):
    all_meshes = []    
    colours = [
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [1, 1, 0],  # Yellow
    [1, 0, 1],  # Magenta
    [0, 1, 1],  # Cyan
    [0.5, 0.5, 0.5]  # Grey
]


    for volume_id, volume in volumes.items():
        points = []
        roi_list = [volume.get_xz_rois(), volume.get_xy_rois()]
        
        # Collect all points from the ROIs
        for roi_dict in roi_list:
            for roi_id, roi in roi_dict.items():
                points += roi.points
        
        # Convert the list of tuples to a numpy array
        points = np.array(points)

        if len(points) == 0:
            continue  # Skip if there are no points to process

        # Create an Open3D PointCloud object from the points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate the alpha shape (concave hull)
        alpha = ALPHASHAPE_ALPHA  # Adjust this value depending on the density and scale of your points
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

       
        colour_index = volume_id % len(colours)  # Cycle through predefined colors
        mesh.paint_uniform_color(colours[colour_index])

        # Add the mesh to the list of all meshes
        all_meshes.append(mesh)

    # Visualize all meshes together
    o3d.visualization.draw_geometries(all_meshes)


def plot_clustered_hulls(clustered_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define a colormap for shading
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(0, len(clustered_points.keys()))
    id = 0
    for y, hulls in clustered_points.items():
        for hull in hulls:
            id += 1
            colour = cmap(norm(id))
            # Extract the vertices from the original points based on simplices
            for simplex in hull.simplices:
                # Get the vertices of the simplex
                vertices = hull.points[simplex]

                # Create the 3D polygon by adding the fixed y-value
                poly3d = Poly3DCollection([[(v[0], y, v[1]) for v in vertices]], alpha=0.3, color=colour)
                ax.add_collection3d(poly3d)

                # Plot the edges of the convex hull
                ax.plot(vertices[:, 0], [y]*len(vertices), vertices[:, 1], 'k-', lw=0.5, color=colour)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Convex Hulls in 3D with Shading')

    print(id)
    plt.show()


# Create a number of points within a convex hull
def sample_hull_points(hull, num_points):
    points = []
     # Extract the vertices of the convex hull
    vertices = hull.points[hull.vertices]
    
    # Create a Delaunay triangulation from the convex hull vertices
    delaunay = Delaunay(vertices)

    for _ in range(num_points):
        # Randomly choose a simplex (triangle) from the Delaunay triangulation
        simplex = random.choice(delaunay.simplices)
        # Get the vertices of the chosen simplex
        tri_vertices = vertices[simplex]
        # Generate a random point within the triangle using barycentric coordinates
        r1, r2 = sorted([random.random(), random.random()])
        point = (1 - r1) * tri_vertices[0] + (r1 - r2) * tri_vertices[1] + r2 * tri_vertices[2]
        points.append(tuple(point))

    return points

def adjust_color_shade(color, factor):
    """
    Adjust the shade of a given RGB color.
    A factor > 1.0 will make the color lighter,
    and a factor < 1.0 will make it darker.
    """
    return tuple(min(1, max(0, c * factor)) for c in color)

if __name__ == "__main__":
    main()