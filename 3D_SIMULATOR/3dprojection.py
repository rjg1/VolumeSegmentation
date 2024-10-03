import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from picker_2 import Picker
from shapely.geometry import Polygon, Point, LineString
import math
import pickle
import os
from actions import ActionSet, AddSliceAction, MergeAction, ProjectionAction

NUM_XY_SLICES = 100
NUM_XZ_SLICES = 100
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
    
    num_spheres = 5 #13
    # radius_range = (0.1, 0.9)  # e.g with 13 bonus spheres
    radius_range = (0.2, 1)  # e.g with 5 bonus spheres
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
    xz_generated = {}
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

        # For this XY plane, divide it into `y_planes` number of y slices
        for y in y_planes:
            # project all x points onto a XZ plane with fixed y and z
            if not xy_slice_meshes.get(z, None) or len(xy_slice_meshes[z]) == 0:
                # no XY ROIs on this XY plane
                continue
            # Project points for each object on this XY plane onto an XZ plane
            for obj_num, slice in xy_slice_meshes[z].items():
                # project x points of each object at this fixed y,z value onto an XZ plane
                if not xz_generated.get(y, None):
                    xz_generated[y] = []
                
                # Get xy points of this ROI
                point_data = slice.points
                points_2d = [(x,y) for x,y,z in point_data]
                # Calculate centroid of the points
                cx, cy = centroid(points_2d)
                # Sort points in predicted order
                sorted_points = sort_points_by_angle(points_2d, (cx, cy))
                sorted_points.append(sorted_points[0]) # wrap polygon back around on itself
                # ensure enough pts to generate polygon
                if (len(sorted_points) < 4):
                    continue
                # Create a polygon
                poly = Polygon(sorted_points)
                # print(intersection.coords)
                n = 20 # num points along this line to generate
                # n points across the x range
                test_points = np.linspace(min([x for x, _ in sorted_points]), max([x for x, _ in sorted_points]), n)
                point_additions = [(x,z) for x in test_points if poly.contains(Point(x,y))]
                xz_generated[y] += point_additions


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

    picker = Picker(plotter, objects, z_planes, y_planes, xy_slice_meshes, xz_slice_meshes, xz_generated, (center, size))
    plotter.track_click_position(picker, side="right")

    # Show the plot
    plotter.show()
    return
    # PART TWO
    plotter.clear()
    plotter = pv.Plotter(shape=(1, 2))
    # Show true segmentation on left side
    for obj in objects:
        plotter.subplot(0, 0)
        plotter.add_mesh(obj, color='grey', opacity=0.25)
    plotter.subplot(0, 0)
    plotter.add_mesh(bounding_box, color='grey', opacity=0.25)
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
        for z, xy_slices in xy_slice_meshes.items():
            for xy_obj_num, xy_slice in xy_slices.items(): # Go through the slice of every object on this xy plane
                # get boundaries of  xy slice
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
                                    if segmented_obj_id in removed_objects:
                                        continue
                                    if xz_slice in segmented_slices:
                                        xz_slice_obj = segmented_obj_id
                                    if xy_slice in segmented_slices:
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



def check_region_intersection(z, xy_list, zx_list, xz_slice):
    if len(xy_list) < 4:  
        return False
    poly1 = Polygon(xy_list)

    point_data = xz_slice.points
    points_2d = [(x,z) for x,y,z in point_data]
    # Calculate centroid of the points
    cx, cz = centroid(points_2d)
    # Sort points in predicted order
    sorted_points = sort_points_by_angle(points_2d, (cx, cz))
    sorted_points.append(sorted_points[0]) # wrap polygon back around on itself
    line = LineString(sorted_points)
    n = 300 # 300 points along this line
    distances = np.linspace(0, line.length, n)
    points = [line.interpolate(distance) for distance in distances]
    zx_points = [(point.x, point.y) for point in points]
    _ ,fixed_y, _ = zx_list[0] 


    # Filter points in zx_list that are within an arbitrary z threshold
    close_points = [Point(x, fixed_y) for x, cz in zx_points if abs(z - cz) < 0.1]

    if not close_points:
        return False  # No points are close enough in the z dimension

    if len(close_points) == 1:
        # Only one point, check intersection and containment
        return poly1.intersects(close_points[0]) or poly1.contains(close_points[0])

    # More than one point, create a LineString
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




