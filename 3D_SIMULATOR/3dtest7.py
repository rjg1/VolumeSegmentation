import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from picker import Picker
from shapely.geometry import Polygon, Point, LineString
import math
import pickle
import os
import argparse
import pandas as pd
from actions import ActionSet, AddSliceAction, MergeAction, ProjectionAction
from collections import defaultdict

NUM_XY_SLICES = 60
NUM_XZ_SLICES = 60
# NUM_XY_SLICES = 10
# NUM_XZ_SLICES = 15
RAND_SEED = 42
cmap = plt.cm.get_cmap('tab10')

def create_prism(center, size):
    """Creates a rectangular prism."""
    x, y, z = center
    dx, dy, dz = size
    return pv.Cube(center=center, x_length=dx, y_length=dy, z_length=dz)


def create_sphere(center, radius):
    """Creates a sphere."""
    return pv.Sphere(center=center, radius=radius)


def calculate_bounds(objects):
    """Calculates the bounding box for all objects."""
    bounds = np.array([obj.bounds for obj in objects])
    min_bounds = bounds[:, ::2].min(axis=0)
    max_bounds = bounds[:, 1::2].max(axis=0)
    center = (min_bounds + max_bounds) / 2
    size = max_bounds - min_bounds
    return center, size


def generate_spheres(num_spheres, radius_range, coordinate_range):
    spheres = []
    for _ in range(num_spheres):
        # Generate random coordinates within the coordinate range
        x = np.random.uniform(*coordinate_range[0])
        y = np.random.uniform(*coordinate_range[1])
        z = np.random.uniform(*coordinate_range[2])
        center = np.array([x, y, z])
        radius = np.random.uniform(*radius_range)

        
        intersection_occurred = False
        sphere = create_sphere(center=center, radius=radius)

        spheres.append(sphere)

    return spheres

#TODO Generate xz stack from xy stack
#TODO Recreate volumes from meshes

def main():
    # Create PyVista plotter and subplot grid
    plotter = pv.Plotter(shape=(2, 2))
    np.random.seed(RAND_SEED) # Seed random for sphere gen
    # Define objects
    sphere = create_sphere(center=(1, 1, 1), radius=0.8)
    prism = create_prism(center=(-1, -1, 1), size=(1.5, 1.5, 2))
    # num_spheres = 5
    num_spheres = 12
    radius_range = (0.1, 0.9)  # e.g with 13 bonus spheres
    # radius_range = (0.2, 1)  # e.g with 5 bonus spheres
    coordinate_range = ((-2, 2), (-3, 3), (0, 2)) 
    objects = [sphere, prism] 
    objects += generate_spheres(num_spheres, radius_range, coordinate_range)

    # Calculate bounds for the enclosing prism
    center, size = calculate_bounds(objects)
    bounding_box = create_prism(center, size)

    # Add bounding box to the plotter's first subplot
    plotter.subplot(0, 0)
    plotter.add_mesh(bounding_box, color='grey', opacity=0.25)

    z_planes = np.linspace(center[2] - size[2] / 2, center[2] + size[2] / 2, num=NUM_XY_SLICES)
    y_planes = np.linspace(center[1] - size[1] / 2, center[1] + size[1] / 2, num=NUM_XZ_SLICES)

    xy_slice_meshes = {}

    for z in z_planes:
        plane = pv.Plane(center=(center[0], center[1], z), direction=(0, 0, 1), i_size=size[0], j_size=size[1])
        plotter.subplot(0, 0)
        plotter.add_mesh(plane, color='green', style='wireframe', opacity=0.2)
        xy_slice_meshes[z] = {}
        for obj_num, obj in enumerate(objects):
            slice = obj.slice(origin=(0, 0, z), normal=(0, 0, 1))
            if slice.n_points > 0:
                plotter.subplot(0, 0)
                plotter.add_mesh(slice, color=cmap(obj_num))
                xy_slice_meshes[z][obj_num] = slice


    plotter.subplot(1, 0)
    plotter.add_mesh(bounding_box, color='grey', opacity=0.25)
    xz_slice_meshes = {}
    for y in y_planes:
        plane = pv.Plane(center=(center[0], y, center[2]), direction=(0, 1, 0), i_size=size[2], j_size=size[0])
        plotter.subplot(1, 0)
        plotter.add_mesh(plane, color='green', style='wireframe', opacity=0.2)
        xz_slice_meshes[y] = {}
        for obj_num, obj in enumerate(objects):
            slice = obj.slice(origin=(0, y, 0), normal=(0, 1, 0))
            if slice.n_points > 0:
                plotter.subplot(1, 0)
                plotter.add_mesh(slice, color=cmap(obj_num))
                xz_slice_meshes[y][obj_num] = slice



    picker = Picker(plotter, objects, z_planes, y_planes, xy_slice_meshes, xz_slice_meshes, (center, size))
    plotter.track_click_position(picker, side="right")

    # # Export data
    # out_rows = []
    # volume_mapping = find_overlapping_volumes(objects)
    # # iterate over all ROIs in all XY planes
    # for z in z_planes: # iterate over all z-planes
    #     current_roi_id = 0 # unique ROI index resets per z plane
    #     if (xy_slice_meshes.get(z, None)):  # ensure slices exist at this z-plane
    #         for obj_id, xy_roi in xy_slice_meshes[z].items(): # For each slice, of an object treat it as a unique ROI
    #             xy_roi_boundaries = [tuple(point) for point in xy_roi.points]
    #             vol_id = volume_mapping[obj_id]
    #             for x, y, z in xy_roi_boundaries:
    #                 out_rows.append([x, y, z, current_roi_id, vol_id])
    #             current_roi_id += 1
    # # Create dfs for each type of output file
    # df_volumes = pd.DataFrame(out_rows, columns=['x', 'y', 'z', 'ROI_ID', 'VOLUME_ID'])
    # df_rois = df_volumes.drop(columns=['VOLUME_ID'])

    # df_rois.to_csv('xyz_complex_points_ROIS.csv', index=False)
    # df_volumes.to_csv('xyz_complex_points_VOLUMES.csv', index=False)


    # Show the plot
    plotter.show()
    return

    # PART TWO - VOLUME SEGMENTATION
    plotter.clear()
    plotter = pv.Plotter(shape=(1, 2))
    # Show true segmentation on left side
    for obj in objects:
        plotter.subplot(0, 0)
        plotter.add_mesh(obj, color='grey', opacity=0.25)
    plotter.subplot(0, 0)
    plotter.add_mesh(bounding_box, color='grey', opacity=0.25)
    plotter.show()
    return
    segmented_objects = {}
    # Attempt to load a prior segmentation
    seg_file_path = f'segmentation_s{RAND_SEED}_o{len(objects)}.pickle'
    if os.path.exists(seg_file_path):
        with open(seg_file_path, 'rb') as f:
            segmented_objects = pickle.load(f)
        print("Loaded saved segmentation")
    else:
        print("No saved segmentation found")
    # Demonstrate the algorithm - only using 2D segmented planes
    if len(segmented_objects) == 0: # if no cached segmentation - perform one
        print("Beginning 3D segmentation")
        obj_id = 0
        removed_objects = []
        actions = ActionSet()
        projections = ActionSet() # track projections
        # xy_slice_meshes == dict of(z_value, dict of(obj_num, pv_slice))
        for z, xy_slices in xy_slice_meshes.items():
            for xy_obj_num, xy_slice in xy_slices.items(): # Go through the slice of every object on this xy plane
                # get boundaries of xy slice ROI
                xy_slice_boundaries = [tuple(point) for point in xy_slice.points]
                # Iterate through all slices in xz_slice_meshes (any slice that may intersect with it)
                for y, xz_slices in xz_slice_meshes.items():
                    for xz_obj_num, xz_slice in xz_slices.items():
                        # get boundaries of xz slice
                        xz_slice_boundaries = [tuple(point) for point in xz_slice.points]
                        # Check for intersection between xy_slice and xz_slice
                        projection = ProjectionAction(xy_slice_boundaries, xz_slice_boundaries, xy_slice, xz_slice, 
                                                      check_region_intersection(z, xy_slice_boundaries, xz_slice_boundaries, xz_slice))
                        projections.add_action(projection)
                        if projection.outcome:
                            # These slices intersect - decide what object to allocate these slices to
                            xz_slice_obj = None
                            xy_slice_obj = None
                            for segmented_obj_id, segmented_slices in segmented_objects.items():
                                    if segmented_obj_id in removed_objects: # object removed in a merge action
                                        continue
                                    if xz_slice in segmented_slices: # the xz slice belongs to an object
                                        xz_slice_obj = segmented_obj_id
                                    if xy_slice in segmented_slices: # the xy slice belongs to an object
                                        xy_slice_obj = segmented_obj_id
                                    if xz_slice_obj and xy_slice_obj:
                                        break # No need to keep looking
                            # Case 1 : both slices have an object - merge objects
                            if xz_slice_obj and xy_slice_obj:
                                if xy_slice_obj == xz_slice_obj:
                                    continue # Ignore if they are already on the same object
                                keep_obj = min(xz_slice_obj, xy_slice_obj)
                                discard_obj = max(xz_slice_obj, xy_slice_obj)
                                segmented_objects[keep_obj] += segmented_objects[discard_obj]
                                # Remove old objects
                                removed_objects.append(discard_obj)
                                # Track for debugging
                                actions.add_action(MergeAction(keep_obj, discard_obj))
                            # Case 2 : XY slice has an object - add XZ slice
                            elif xy_slice_obj and not xz_slice_obj:
                                segmented_objects[xy_slice_obj].append(xz_slice)
                                actions.add_action(AddSliceAction(xy_slice_obj, xz_slice))
                            # Case 3 : XZ slice has an object - add XY slice
                            elif xz_slice_obj and not xy_slice_obj:
                                segmented_objects[xz_slice_obj].append(xy_slice)
                                actions.add_action(AddSliceAction(xz_slice_obj, xy_slice))
                            # Case 4 : Neither slice has an object - allocate a new object
                            else:
                                obj_id += 1 # new object
                                new_id = obj_id
                                # Add both slices
                                segmented_objects[new_id] = [xz_slice]
                                segmented_objects[new_id].append(xy_slice)
                                actions.add_action(AddSliceAction(new_id, xy_slice))
                                actions.add_action(AddSliceAction(new_id, xz_slice))
        # DEBUG serialisation
        with open(f'DEBUG_SEG_OBJ_s{RAND_SEED}_o{len(objects)}.pickle', 'wb') as f:
            pickle.dump(segmented_objects, f)
        with open(f'DEBUG_ACTIONS_s{RAND_SEED}_o{len(objects)}.pickle', 'wb') as f:
            pickle.dump(actions, f)
        with open(f'DEBUG_PROJECTIONS_s{RAND_SEED}_o{len(objects)}.pickle', 'wb') as f:
            pickle.dump(projections, f)
        
        # Remove merged objects
        for obj in removed_objects:
            segmented_objects.pop(obj)

        # Serialise dict - speedy
        with open(seg_file_path, 'wb') as f:
            pickle.dump(segmented_objects, f)

    print(f"segobs: {segmented_objects.keys()}")
    for key, slice_list in segmented_objects.items():
        print(f"Key: {key}, slice list size: {len(slice_list)}")

    colour = 0
    for obj_id, slice_list in segmented_objects.items():
        for slice_obj in slice_list:
            plotter.subplot(0, 1)
            plotter.add_mesh(slice_obj, color=cmap(colour), opacity=0.25)
        colour += 1
    plotter.subplot(0, 1)
    plotter.add_mesh(bounding_box, color='grey', opacity=0.25)

    plotter.show()


import pyvista as pv
from collections import defaultdict

def find_overlapping_volumes(objects):
    """
    Determine overlapping objects and assign unique volume IDs.

    Args:
    - objects (list of pyvista.PolyData): List of PyVista objects to check for overlap.

    Returns:
    - volume_dict (dict): Dictionary where keys are object indices and values are volume IDs.
    """
    # Number of objects
    n = len(objects)

    # Graph to store overlaps (adjacency list)
    overlap_graph = defaultdict(list)

    # Check for overlaps between each pair of objects
    for i in range(n):
        obj1 = objects[i].triangulate()  # Triangulate the first object
        for j in range(i + 1, n):
            obj2 = objects[j].triangulate()  # Triangulate the second object
            if obj1.bounds != obj2.bounds:  # Quick bounds check to avoid unnecessary intersections
                try:
                    intersection = obj1.boolean_intersection(obj2)
                    # Check if the intersection has points or cells to determine if it is non-empty
                    if intersection.n_points > 0 and intersection.n_cells > 0:
                        # If there's an overlap, add both indices to the graph
                        overlap_graph[i].append(j)
                        overlap_graph[j].append(i)
                except Exception as e:
                    # Handle known warnings and errors gracefully
                    print(f"Warning: Unable to compute intersection between objects {i} and {j}: {e}")

    # Function to perform DFS and assign volume IDs
    def assign_volume_id(node, volume_id):
        stack = [node]
        visited.add(node)
        volume_dict[node] = volume_id
        while stack:
            current = stack.pop()
            for neighbor in overlap_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    volume_dict[neighbor] = volume_id
                    stack.append(neighbor)

    # Assign unique volume IDs
    volume_dict = {}
    visited = set()
    current_volume_id = 1

    for node in range(n):
        if node not in visited:
            assign_volume_id(node, current_volume_id)
            current_volume_id += 1

    return volume_dict



def check_region_intersection(z, xy_list, zx_list, xz_slice):
    if len(xy_list) < 4:  
        return False
    
    # Generate a polygon for the XY roi being compared
    poly1 = Polygon(xy_list)

    # Get a list of xz points for an object in the XZ slice
    point_data = xz_slice.points
    points_2d = [(x,z) for x,y,z in point_data]
    # Calculate centroid of the points
    cx, cz = centroid(points_2d)
    # Sort points in predicted order
    sorted_points = sort_points_by_angle(points_2d, (cx, cz))
    sorted_points.append(sorted_points[0]) # wrap polygon back around on itself
    line = LineString(sorted_points)
    n = 300 # 300 points along this line
    # Generate n zx points around the boundary of the ROI polygon
    distances = np.linspace(0, line.length, n)
    points = [line.interpolate(distance) for distance in distances]
    zx_points = [(point.x, point.y) for point in points]
    # Determine the fixed y at which the ZX plane is situated
    _ ,fixed_y, _ = zx_list[0] 


    # Generate a list of x points on the XZ roi boundary at fixed y and z sufficiently 
    # close to the z location of the XY plane. Stored as an (x,y) value
    close_points = [Point(x, fixed_y) for x, cz in zx_points if abs(z - cz) < 0.1]

    # Check if any points of the XZ roi can be found at fixed y,z value
    if not close_points:
        return False  # No points are close enough in the z dimension

    # A single point - check if the XY roi also contains this point
    if len(close_points) == 1:
        # Only one point, check intersection and containment
        return poly1.intersects(close_points[0]) or poly1.contains(close_points[0])

    # More than one point, create a LineString, and determine if it intersects the xy ROI
    line = LineString([(p.x, p.y) for p in close_points])
    return poly1.intersects(line) or poly1.contains(line)


# Function to sort points by angle from centroid
def sort_points_by_angle(points, center):
    cx, cz = center
    def angle(point):
        return np.arctan2(point[1] - cz, point[0] - cx)
    return sorted(points, key=angle)


# Function to calculate centroid of the points
def centroid(points):
    x, z = zip(*points)
    cx = sum(x) / len(points)
    cz = sum(z) / len(points)
    return cx, cz

if __name__ == "__main__":
    main()




