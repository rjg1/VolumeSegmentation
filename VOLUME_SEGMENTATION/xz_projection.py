import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
import pandas as pd

NUM_XZ_SLICES = 60
INTERPOLATION_RESOLUTION = 10
DEFAULT_X_POINTS = 10

'''
Original Script
'''
def generate_xz_rois(points, parameters = {}):
    # Make output list
    xz_roi_points = []
    # Determine y plane intervals
    min_y = min([y for _,y,_,_ in points])
    max_y = max([y for _,y,_,_ in points])
    if parameters.get('num_xz_slices', None) and not parameters.get('determine_xz_slices', None):
        num_y_slices = parameters.get('num_xz_slices')
    else: # Auto determine as the number of unique y-values
        # At least NUM_XZ_SLICES and at most 1024
        num_y_slices = min(1024,max(NUM_XZ_SLICES,len(set([y for _,y,_,_ in points]))))
    y_intervals = np.linspace(min_y, max_y, num_y_slices) # NUM_XZ_SLICES between min and max y value
    # Determine z-plane intervals
    z_intervals = set([z for _, _, z,_ in points])
    # Make dict of {z: {id: [(x,y)]}} for each z-plane 
    z_planes = {}
    for x, y, z, id in points:
        if not z_planes.get(z, None):
            z_planes[z] = {}
        if not z_planes[z].get(id, None):
            z_planes[z][id] = []
        z_planes[z][id].append((x, y))

    # Test Caching {z: {id: polygon}} for each z-plane 
    z_polys = {}
    for z, id_dict in z_planes.items():
        z_polys[z] = {}
        for id, xy_list in id_dict.items():
            z_polys[z][id] = {} # default sentinel value
            
            # Calculate centroid of the points
            cx, cy = find_centroid(xy_list)
            
            # Sort points in predicted order
            sorted_points = sort_points_by_angle(xy_list, (cx, cy))
            sorted_points.append(sorted_points[0]) # wrap polygon back around on itself
            
            # Ensure ROI has enough pts to generate polygon
            if (len(sorted_points) < 4):
                poly = None
            else:
                # Create a polygon using the sorted points of this ROI
                poly = Polygon(sorted_points)

            # Save metadata of this polygon
            min_x = min(x for x,_ in xy_list)
            max_x = max(x for x,_ in xy_list)
            min_y = min(y for _,y in xy_list)
            max_y = max(y for _,y in xy_list)
            z_polys[z][id]['poly'] = poly
            z_polys[z][id]['min_x'] = min_x
            z_polys[z][id]['max_x'] = max_x

    # Fix y and z, and project x points of an ROI onto a XZ plane
    for count, y in enumerate(y_intervals):
        print(f"[{count + 1}/{num_y_slices}] Generating XZ Plane at y = {y} ")
        for z in z_intervals:
            # Ensure there are points for this z interval
            if not z_planes.get(z, None) or len(z_planes[z]) == 0:
                continue
            # Iterate through the xy point lists for each XY ROI
            if z_polys.get(z, None):
                for id, roi_dict in z_polys.get(z, {}).items():
                    poly = roi_dict ['poly']
                    min_x = roi_dict ['min_x']
                    max_x = roi_dict ['max_x']
                    
                    if poly is not None:
                        # n points across the x range
                        test_points = np.linspace(min_x, max_x, parameters.get('num_x_points'), DEFAULT_X_POINTS)
                        point_additions = [(x,y,z) for x in test_points if poly.contains(Point(x,y))]
                        xz_roi_points += point_additions

    return xz_roi_points

def generate_xz_single_y(points, min_y = None, max_y = None, include_xy_id = False, parameters = {}):
    # Make output list
    xz_roi_points = []
    # Determine y plane intervals
    if min_y is None:
        min_y = min([y for _,y,_,_ in points])
    if max_y is None:
        max_y = max([y for _,y,_,_ in points])
    if parameters.get('num_xz_slices', None) and not parameters.get('determine_xz_slices', None):
        num_y_slices = parameters.get('num_xz_slices')
    else: # Auto determine as the number of unique y-values
        # At least NUM_XZ_SLICES and at most 2048
        num_y_slices = min(2048,max(NUM_XZ_SLICES,len(set([y for _,y,_,_ in points]))))
    y_points = np.linspace(min_y, max_y, num_y_slices) 
    # Determine z-plane intervals
    z_intervals = set([z for _, _, z,_ in points])
    # Make dict of {z: {id: [(x,y)]}} for each z-plane 
    z_planes = {}
    for x, y, z, id in points:
        if not z_planes.get(z, None):
            z_planes[z] = {}
        if not z_planes[z].get(id, None):
            z_planes[z][id] = []
        z_planes[z][id].append((x, y))

    # Test Caching {z: {id: polygon}} for each z-plane 
    z_polys = {}
    for z, id_dict in z_planes.items():
        z_polys[z] = {}
        for id, xy_list in id_dict.items():
            z_polys[z][id] = {} # default sentinel value
            
            # Calculate centroid of the points
            cx, cy = find_centroid(xy_list)
            
            # Sort points in predicted order
            sorted_points = sort_points_by_angle(xy_list, (cx, cy))
            sorted_points.append(sorted_points[0]) # wrap polygon back around on itself
            
            # Ensure ROI has enough pts to generate polygon
            if (len(sorted_points) < 4):
                poly = None
            else:
                # Create a polygon using the sorted points of this ROI
                poly = Polygon(sorted_points)

            # Save metadata of this polygon
            min_x = min(x for x,_ in xy_list)
            max_x = max(x for x,_ in xy_list)
            min_y = min(y for _,y in xy_list)
            max_y = max(y for _,y in xy_list)
            z_polys[z][id]['poly'] = poly
            z_polys[z][id]['min_x'] = min_x
            z_polys[z][id]['max_x'] = max_x
            z_polys[z][id]['min_y'] = min_y
            z_polys[z][id]['max_y'] = max_y
        
    # Determine the x points to interpolate across for fixed y and z
    min_x = min([x for x,_,_,_ in points])
    max_x = max([x for x,_,_,_ in points])
    if parameters.get('num_x_points', None):
        num_x_points = parameters.get('num_x_points')
    else:
        num_x_points = DEFAULT_X_POINTS
    x_points = np.linspace(min_x, max_x, num_x_points)

    # Fix y and z, and project x points of an ROI onto a XZ plane
    for count, y in enumerate(y_points):
        print(f"[{count + 1}/{num_y_slices}] Generating XZ Plane at y = {y} ")
        for z in z_intervals:
            # Ensure there are points for this z interval
            if not z_planes.get(z, None) or len(z_planes[z]) == 0:
                continue
            # Project a set of x points
            for x in x_points:
                # Iterate through the xy point lists for each XY ROI
                for id, roi_dict in z_polys.get(z, {}).items(): # For each ROI on this z-plane
                    # print(f"Evaluating poly {count} in z-plane: {z}")
                    poly = roi_dict ['poly']
                    min_x = roi_dict ['min_x']
                    max_x = roi_dict ['max_x']
                    min_y = roi_dict ['min_y']
                    max_y = roi_dict ['max_y']
                    # print(f"Poly {count} minx: {min_x}, maxx: {max_x}, miny: {min_y}, maxy: {max_x}")
                    # Ensure the current test y point is within the ROI on this z-plane
                    if poly is not None and x >= min_x and x <= max_x and y >= min_y and y <= max_y and poly.contains(Point(x,y)):
                        if include_xy_id:
                            xz_roi_points.append((x,y,z,id))
                        else:
                            xz_roi_points.append((x,y,z))
                    else:
                        pass
                        # print(f"Poly cannot spacially align with (x,y)==({x},{y})")
    return xz_roi_points


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

def plot_xz_planes(points, num_planes=5):
    """
    Plot a specified number of random XZ planes at fixed Y intervals.
    
    Args:
        points (list of tuples): List of (x, y, z) points.
        num_planes (int): Number of random XZ planes to plot.
    """
    # Extract all Y values from the points
    y_values = [y for _, y, _ in points]

    # Get the minimum and maximum Y values to determine the Y range
    min_y = min(y_values)
    max_y = max(y_values)

    # Generate random y-planes within the range of min_y and max_y
    y_planes = np.linspace(min_y, max_y, num_planes)

    # Plot each XZ plane at fixed Y intervals
    for i, y_plane in enumerate(y_planes):
        # Filter points that are near the current y_plane
        plane_points = [(x, z) for x, y, z in points if abs(y - y_plane) < 0.1]

        # If there are no points near the current y_plane, skip the plot
        if not plane_points:
            continue

        # Extract X and Z values from the filtered points
        x_coords, z_coords = zip(*plane_points)

        # Plotting the XZ plane
        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, z_coords, c='blue', marker='o', s=5)
        plt.title(f'XZ Plane at Y = {y_plane:.2f}')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.grid(True)
        plt.show()