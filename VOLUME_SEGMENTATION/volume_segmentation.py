"""
Takes an input of x,y,z points representing boundaries of each ROI.

Generates XZ planes of points from x,y,z points

"""
# IMPORTS
import csv
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial import ConvexHull
from region import BoundaryRegion
from volume import Volume
import os
import pandas as pd
import argparse
from xz_projection import generate_xz_single_y, generate_xz_rois
from xz_segmentation import cluster_xz_rois_tuned
from volume_seg_ffp import segment_volumes_drg

# GLOBAL VARIABLES
# Name of input file. Structured csv with |x|y|z|roi ID|. Note ID is per z layer
OUT_FILE = './results/real_data_filtered_algo_VOLUMES.csv'
XZ_IO_PATH = "./xz_cache/real_data_filtered_v0_ROIS_XZ.csv"
IN_FILE = './run_data/real_data_filtered_v0_ROIS.csv'
# Whether to use fit-for-purpose DRG segmentation or generalised segmentation
RESTRICTED_MODE = True
# Default FFP Parameters
ROI_PROJECTION_N_POINTS = 300           # Number of points to project around an ROI boundary when deciding intersection (Higher -> more computation)
CENTROID_DISTANCE_MAX = 10              # Maximum allowable distance between centroids of two adjacent XY ROIs
CENTROID_DISTANCE_PERC = 68             # Maximum allowable distance between centroids of two adjacent XY ROIs as a percentage of the radius of a centroid
MATCH_Z_THRESHOLD = 4                   # Maximum distance between two adjacent XY rois in Z to be part of same volume
AREA_DELTA_PERC_THRESHOLD = 150         # Maximum change in percent of ROI area between two XY rois in a volume (from small to large -> change of 150%)
AR_CHANGE_PERC = 30                     # Maximum allowed change in average area per ROI over a number of ROI additions
AR_CHANGE_NUM_SAMPLES = 2               # Number of xy roi samples that are considered when comparing change in average area per roi for a volume
AR_CHANGE_ACTIVATION_THRESH = 10        # Number of XY ROIs required in a volume before area restriction kicks in
RESTRICT_AREA = False                   # Restriction in the difference in area between two adjacent XY ROIs in a volume
RESTRICT_CENTROID_DISTANCE = True       # Restriction on the distance between centroid of two adjacent XY ROIs in a volume
USE_PERCENT_CENTROID_DISTANCE = False   # Centroid distance threshold is a percent of the radius of an XY ROI instead of a flat value
RESTRICT_ADJACENCY = True               # Restriction on how close an XY ROI must be in Z to its nearest neighbour in the volume
RESTRICT_MULTIPLE_XY_PER_VOL = True     # Restriction on whether or not two XY ROIs are allowed to be in the same z-level per volume
RESTRICT_AREA_CHANGE = False            # Restriction on how fast a volume's average area per ROI can change
CACHE_INTERPOLATED_XZ = True            # Whether to interpolate and store boundary points before volume segmentation (higher memory + speed)
IGNORE_COLINEAR_XZ = True               # Whether to discard colinear XZ ROIs as noise (keeping -> higher memory usage, higher linking)
NUM_XZ_SLICES = 60                      # Default number of XZ slices (y-planes) to generate
DETERMINE_XZ_SLICES = False             # Whether or not to automatically determine the number of xz slices
DEFAULT_X_POINTS = 10                   # Default number of x points to project across each ROI in a z-plane when making a XZ plane

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a volume segmentation on a dataset")
    
    # Add arguments for file paths
    parser.add_argument('--in_file', default=IN_FILE, help='Input filepath to csv used to run segmentation on (default: %(default)s)')
    parser.add_argument('--out_file', default=OUT_FILE, help='Output filepath where the results csv will be placed (default: %(default)s)')
    parser.add_argument('--xz_io_path', default=XZ_IO_PATH, help='Directory to look in for/generate an XZ plane cache (default: %(default)s)')
    parser.add_argument('--restricted_mode', default=RESTRICTED_MODE, help='Use DRG segmentation technique (default: %(default)s)')
    # Arguments for FFP parameters
    parser.add_argument('--use_flat_centroid', default=RESTRICT_CENTROID_DISTANCE or USE_PERCENT_CENTROID_DISTANCE, help='Restrict relative centroid distance of adjacent ROIs in z in a volume by a flat amount (default: %(default)s)')
    parser.add_argument('--use_perc_centroid', default=USE_PERCENT_CENTROID_DISTANCE, help='Restrict relative centroid distance of adjacent ROIs in z per volume by a percentage of the avg radius of the roi (default: %(default)s)')
    parser.add_argument('--use_z_threshold', default=RESTRICT_ADJACENCY, help='Only allow ROIs to be added to the same volume if their relative z value is within this threshold to its adjacent ROI in z(default: %(default)s)')
    parser.add_argument('--use_area_restriction', default=RESTRICT_AREA, help='Prevent adjacent ROIs in z from being added to the same volume if the percent area change is too large (default: %(default)s)')
    parser.add_argument('--use_xy_restriction', default=RESTRICT_MULTIPLE_XY_PER_VOL, help='Only allow one xy roi per z per volume (default: %(default)s)')
    parser.add_argument('--use_area_deriv_restriction', default=RESTRICT_AREA_CHANGE, help='Enable volume per roi change restriction (default: %(default)s)')
    parser.add_argument('--num_projection_points', default=ROI_PROJECTION_N_POINTS, help='Number of points to project when determining intersection (default: %(default)s)')
    parser.add_argument('--flat_centroid_dist', default=CENTROID_DISTANCE_MAX, help='Flat threshold for centroid check(default: %(default)s)')
    parser.add_argument('--perc_centroid_dist', default=CENTROID_DISTANCE_PERC, help='Percent threshold for centroid check(default: %(default)s)')
    parser.add_argument('--match_z_threshold', default=MATCH_Z_THRESHOLD, help='Match z threshold (default: %(default)s)')
    parser.add_argument('--area_delta_perc', default=AREA_DELTA_PERC_THRESHOLD, help='Area percent threshold for area restriction (default: %(default)s)')
    parser.add_argument('--ar_change_perc', default=AR_CHANGE_PERC, help='Allowed percentage difference in avg vol per roi in two volumes (default: %(default)s)')
    parser.add_argument('--ar_change_num_samples', default=AR_CHANGE_NUM_SAMPLES, help='Number of XY rois to compare to the overall volume in volume change (default: %(default)s)')
    parser.add_argument('--ar_change_activation_thresh', default=AR_CHANGE_ACTIVATION_THRESH, help='Number of XY rois to add to a volume before volume change kicks in (default: %(default)s)')
    parser.add_argument('--ignore_colinear_xz', default=IGNORE_COLINEAR_XZ, help='Whether or not to ignore XZ ROIs made entirely of colinear points (default: %(default)s)')
    parser.add_argument('--cache_interpolated_xz', default=IGNORE_COLINEAR_XZ, help='Whether or not to cache interpolated boundaries of XZ ROIs (default: %(default)s)')
    parser.add_argument('--num_xz_slices', default=NUM_XZ_SLICES, help='Number of XZ Slices (y-planes) to project (default: %(default)s)')
    parser.add_argument('--determine_xz_slices', default=DETERMINE_XZ_SLICES, help='Whether or not to automatically determine number of y-planes to generate(default: %(default)s)')
    parser.add_argument('--num_x_points', default=DEFAULT_X_POINTS, help='Default number of x points to project across each ROI in a z-plane when making a XZ plane (default: %(default)s)')


    # Parse arguments
    args = parser.parse_args()
    # Use the parsed arguments or fallback to the defaults if no arguments are provided
    in_file = args.in_file or IN_FILE
    out_file = args.out_file or OUT_FILE
    xz_io_path = args.xz_io_path or XZ_IO_PATH
    # Allow no xz caching
    if xz_io_path and xz_io_path.lower() == "none":
        xz_io_path = None
    # Avoid short circuit with boolean restricted mode
    restricted_mode = str_to_bool(args.restricted_mode) if isinstance(args.restricted_mode,str) else RESTRICTED_MODE
    # Unpack parameters to use
    if isinstance(args.use_flat_centroid, str):
        algo_parameters = {
            'restrict_centroid_distance': str_to_bool(args.use_flat_centroid) or str_to_bool(args.use_perc_centroid),
            'use_percent_centroid_distance': str_to_bool(args.use_perc_centroid),
            'restrict_adjacency': str_to_bool(args.use_z_threshold),
            'restrict_area': str_to_bool(args.use_area_restriction),
            'restrict_multiple_xy_per_vol': str_to_bool(args.use_xy_restriction),
            'restrict_area_change' : str_to_bool(args.use_area_deriv_restriction),
            'roi_projection_n_points': int(args.num_projection_points),
            'centroid_distance_max': float(args.flat_centroid_dist),
            'centroid_distance_perc': float(args.perc_centroid_dist),
            'match_z_threshold': int(args.match_z_threshold),
            'area_delta_perc_threshold': float(args.area_delta_perc),
            'ar_change_perc' : float(args.ar_change_perc),
            'ar_change_num_samples' : int(args.ar_change_num_samples),
            'ar_change_activation_thresh' : int(args.ar_change_activation_thresh),
            'cache_interpolated_xz' : str_to_bool(args.cache_interpolated_xz),
            'ignore_colinear_xz' : str_to_bool(args.ignore_colinear_xz),
            'num_xz_slices' : int(args.num_xz_slices),
            'determine_xz_slices' : str_to_bool(args.determine_xz_slices),
            'num_x_points' : int(args.num_x_points)
        }
    else: # Use defaults
        algo_parameters = {
            'restrict_centroid_distance': RESTRICT_CENTROID_DISTANCE,
            'use_percent_centroid_distance': USE_PERCENT_CENTROID_DISTANCE,
            'restrict_adjacency': RESTRICT_ADJACENCY,
            'restrict_area': RESTRICT_AREA,
            'restrict_multiple_xy_per_vol': RESTRICT_MULTIPLE_XY_PER_VOL,
            'restrict_area_change' : RESTRICT_AREA_CHANGE,
            'roi_projection_n_points': ROI_PROJECTION_N_POINTS,
            'centroid_distance_max': CENTROID_DISTANCE_MAX,
            'centroid_distance_perc': CENTROID_DISTANCE_PERC,
            'match_z_threshold': MATCH_Z_THRESHOLD,
            'area_delta_perc_threshold': AREA_DELTA_PERC_THRESHOLD,
            'ar_change_perc' : AR_CHANGE_PERC,
            'ar_change_num_samples' : AR_CHANGE_NUM_SAMPLES,
            'ar_change_activation_thresh' : AR_CHANGE_ACTIVATION_THRESH,
            'cache_interpolated_xz' : CACHE_INTERPOLATED_XZ,
            'ignore_colinear_xz' : IGNORE_COLINEAR_XZ,
            'num_xz_slices' : NUM_XZ_SLICES,
            'determine_xz_slices' : DETERMINE_XZ_SLICES,
            'num_x_points' : DEFAULT_X_POINTS
        }
    # Get X, Y, Z, ROI_ID points
    points = import_csv(in_file=in_file)
    # Generate XZ ROIs
    # DEBUG: Cache XZ ROIS
    if xz_io_path is not None and os.path.exists(xz_io_path):
        print(f"Importing XZ Planes from: {xz_io_path}")
        xz_roi_points = import_xyz_points(xz_io_path)
        print(f"Imported XZ Planes. Performing XZ Cluster Segmentation.")
    else:
        print("Generating XZ Planes...")
        xz_roi_points = generate_xz_rois(points, parameters=algo_parameters)
        if xz_io_path is not None:
            # Export points
            out_xz_points = []
            for x,y,z in xz_roi_points:
                out_xz_points.append([x, y, z])
            xz_df = pd.DataFrame(out_xz_points, columns=['x', 'y', 'z'])
            
            xz_df.to_csv(xz_io_path, index=False)
            print("Generated and Exported XZ Planes. Performing XZ Cluster Segmentation.")
        else:
            print("Generated XZ Planes. Performing XZ Cluster Segmentation.")

    # Cluster XZ ROI points
    clustered_hulls = cluster_xz_rois_tuned(xz_roi_points, parameters=algo_parameters)
    # visualize_random_clusters(clustered_hulls, num_planes=10) 
    print("Cluster segmentation complete. Performing Volume Segmentation...")
    # Perform volume segmentation using XY ROIs and XZ Regions
    if restricted_mode:
        volumes = segment_volumes_drg(clustered_hulls, points, parameters=algo_parameters)
    else:
        volumes = segment_volumes(clustered_hulls, points, parameters=algo_parameters)
    
    if volumes:
        # DEBUG: EST PRINTING OUTPUT OF VOLUMES
        for volume_id, volume in volumes.items():
            print(f"Volume_id: {volume_id}")
            xz_rois = volume.get_xz_rois()
            xy_rois = volume.get_xy_rois()
            print(f"Num xz_rois: {len(xz_rois)}")
            print(f"Num xy_rois: {len(xy_rois)}")

        # Export segmentation to csv
        export_segmentation(volumes, filename=out_file)


def str_to_bool(value):
    """Convert a string to a boolean."""
    return value.lower() in ('true', '1', 'yes', 'y')

def import_xyz_points(csv_path):
    """
    Imports a CSV file with columns 'x', 'y', 'z' into a list of (x, y, z) tuples.
    
    Args:
        csv_path (str): The path to the CSV file.
    
    Returns:
        list of tuples: List containing (x, y, z) points as tuples.
    """
    xz_points = []
    
    # Open the CSV file and read it
    try:
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Ensure the necessary columns 'x', 'y', 'z' are in the file
            if set(['x', 'y', 'z']).issubset(reader.fieldnames):
                # Convert each row to a tuple and append it to the list
                for row in reader:
                    x = float(row['x'])
                    y = float(row['y'])
                    z = float(row['z'])
                    xz_points.append((x, y, z))
            else:
                print("Error: CSV file must contain 'x', 'y', 'z' columns.")
                return []
    
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
    
    return xz_points


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

# Create a dictionary of {volume id : <Volume> object} where id is the id of a unique volume
def segment_volumes(xz_hulls, xyz_points, parameters = {}):
    # Output dict of {volume id : <Volume> object}
    print("Beginning Caching for Volume Segmentation")
    volumes = {}
    # Make dict of {z: {id: [(x,y)]}} for each z-plane 
    z_planes = {}
    for x, y, z, id in xyz_points:
        if not z_planes.get(z, None):
            z_planes[z] = {}
        if not z_planes[z].get(id, None):
            z_planes[z][id] = []
        z_planes[z][id].append((x, y))
    # Make dict of xy ROIs -> {xy_id : BoundaryRegion of pts}
    xy_rois = {}
    xy_id = 0
    for z, id_dict in z_planes.items():
        for id, points in id_dict.items():
            # Make a boundary region using boundary pts of xy roi
            points_3d = [(x,y,z) for x,y in points]
            xy_rois[xy_id] = BoundaryRegion(points_3d, original_index=id)
            xy_id += 1
        
    # Make dict of xz ROIs -> {xz_id : BoundaryRegion of pts}
    xz_rois = {}
    xz_id = 0
    for y, hull_list in xz_hulls.items():
        for item in hull_list:
            points_3d = []
            if isinstance(item, ConvexHull):
                # Handle ConvexHull objects
                boundary_points = get_hull_boundary_points(item)
                points_3d = [(x, y, z) for x, z in boundary_points]
            elif isinstance(item, np.ndarray):
                for point in item:
                    points_3d.append((point[0], y, point[1]))
            elif isinstance(item, list):
                points_3d.extend(item)
            if len(points_3d) > 0:
                xz_rois[xz_id] = BoundaryRegion(points_3d)
                xz_id += 1
    print(f"Finished Volume Segmentation Caching - beginning algorithm")
    # Iterate through all XY ROIs and determine collisions with XZ ROIs
    current_obj_id = 1 # ID of current object being created
    removed_objects = [] # List of objects removed during merge operations
    num_xy_rois = len(xy_rois) # Debug code used to provide progress
    xy_roi_count = 0 # Count of number of XY rois processed
    for xy_roi_id, xy_roi in xy_rois.items():
        if xy_roi_count % 100 == 0:
            print(f"Evaluating XY ROI {xy_roi_count + 1}/{num_xy_rois}")
        xy_roi_count += 1
        for xz_roi_id, xz_roi in xz_rois.items():
            # Rough check if an intersection is theoretically possible
            if not xy_roi.overlaps_with_region(xz_roi):
                continue # Objects cannot spacially overlap
            # Execution reached here - more detailed check for collision
            intersection = check_region_intersection(xy_roi.get_boundary_points(), xz_roi.get_boundary_points(), parameters=parameters)
            if intersection: # XZ and XY ROIs are part of same object
                # Determine what 3D object id to assign to these two ROIs
                xz_roi_obj = None
                xy_roi_obj = None
                for obj_id, volume in volumes.items():
                    if obj_id in removed_objects:
                        continue # Ignore removed object
                    if xz_roi_id in volume.get_xz_rois():
                        # The XZ slice is already in this volume
                        xz_roi_obj = obj_id
                    if xy_roi_id in volume.get_xy_rois():
                        # The XY slice is already in this volume
                        xy_roi_obj = obj_id
                    # Terminate search if an object is found for both rois
                    if xy_roi_obj and xz_roi_obj:
                        break
                # Case 1 : both rois have an object - merge objects    
                if xy_roi_obj and xz_roi_obj:
                    if xz_roi_obj == xy_roi_obj: # Same obj - do nothing
                        continue
                    keep_obj = min(xy_roi_obj, xz_roi_obj)
                    discard_obj = max(xy_roi_obj, xz_roi_obj)

                    volumes[keep_obj].volume_merge(volumes[discard_obj])
                    # Remove (ignore for now) the discarded volume
                    removed_objects.append(discard_obj)
                # Case 2 : XY roi is in an object - add XZ roi
                elif xy_roi_obj and not xz_roi_obj:
                    volumes[xy_roi_obj].add_xz_roi(xz_roi_id, xz_roi)
                # Case 3 : XZ roi is in an object - add XY roi
                elif xz_roi_obj and not xy_roi_obj:
                    volumes[xz_roi_obj].add_xy_roi(xy_roi_id, xy_roi)
                else:
                    # Create a new volume object
                    new_volume = Volume(current_obj_id)
                    # Add the intersecting XY and XZ ROIs to the object
                    new_volume.add_xy_roi(xy_roi_id, xy_roi)
                    new_volume.add_xz_roi(xz_roi_id, xz_roi)
                    # Add new volume to the volumes dictionary
                    volumes[current_obj_id] = new_volume
                    # Increment the id for the next volume
                    current_obj_id += 1

    # Remove deleted volumes
    for obj in removed_objects:
        volumes.pop(obj)

    # Return volumes dictionary
    return volumes


# Determines if an XY roi and an XZ roi intersect
def check_region_intersection(xy_points, xz_points, parameters):
    # Determine the fixed y at which the ZX plane is situated
    _ ,fixed_y, _ = xz_points[0]
    # Determine the fixed z at which the XY plane is situated 
    _, _, fixed_z = xy_points[0]

    # Check if enough points in xy list to create a polygon
    if len (xy_points) < 4:
        return False 
    
    # Generate a polygon for the XY roi being compared
    poly1 = Polygon(xy_points)

    # Calculate centroid of the xz points
    xz_points_2d = [(x,z) for x,y,z in xz_points]
    # Generate n boundary points around xz roi if this has not already been done
    if not parameters.get('cache_interpolated_xz'):
        cx, cz = find_centroid(xz_points_2d)
        # Sort xz boundary points in predicted order around a polygon
        sorted_points = sort_points_by_angle(xz_points_2d, (cx, cz))
        sorted_points.append(sorted_points[0]) # wrap polygon back around on itself
        line = LineString(sorted_points)
        # Generate n zx points around the boundary of the ROI polygon
        distances = np.linspace(0, line.length, parameters['roi_projection_n_points'])
        points = [line.interpolate(distance) for distance in distances]
        # Update 2d xz points to use new point projection
        xz_points_2d = [(point.x, point.y) for point in points]

    # Generate a list of x points on the XZ roi boundary at fixed y and z sufficiently 
    # close to the z location of the XY plane. Stored as an (x,y) value
    close_points = [Point(x, fixed_y) for x, cz in xz_points_2d if abs(fixed_z - cz) < 0.1]

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


# Extracts the list of boundary points of a 2D convex hull
def get_hull_boundary_points(hull):
    # Extract the vertices of the convex hull
    vertices = hull.points[hull.vertices]
    
    # Convert the vertices to a list of (x, y) tuples
    boundary_points = [(x, y) for x, y in vertices]
    
    return boundary_points

# Exports a segmentation into a csv
def export_segmentation(volumes_dict, filename="algorithmic_segmentation.csv"):
    out_data = []
    print(f"Volume dict: {volumes_dict}")
    for vol_id, volume in volumes_dict.items(): # Iterate through all volumes
        xy_rois = volume.get_xy_rois()  # Get the dict of {ROI_ID : <BoundaryRegion>}
        for roi_id, xy_roi in xy_rois.items():   # For each ROI
            points_3d = xy_roi.get_boundary_points() # Get list of [(x,y,z)] boundary points for this ROI
            original_id = xy_roi.get_original_index() # Get original ROI ID from input dataset
            if original_id is not None:
                for x, y, z in points_3d: # For each x,y,z point, append to output list as a row
                    out_data.append([x, y, z, original_id, vol_id])
            else:
                print("export error!!!")
    # Create dataframe to export to csv
    df = pd.DataFrame(out_data, columns=['x', 'y', 'z', 'ROI_ID', 'VOLUME_ID'])
    # Export to csv
    df.to_csv(filename, index=False)

# Function to import points from a source file
def import_csv(in_file =  IN_FILE):
    with open(in_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        points = [(float(row[0]), float(row[1]), float(row[2]), int(row[3])) for row in reader]
        return points

if __name__ == "__main__":
    main()