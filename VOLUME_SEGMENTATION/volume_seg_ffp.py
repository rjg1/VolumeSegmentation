import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial import ConvexHull
from region import BoundaryRegion
from volume import Volume
import random

from scipy.spatial import distance

# Create a dictionary of {volume id : <Volume> object} where id is the id of a unique volume
def segment_volumes_drg(xz_hulls, xyz_points, parameters = {}):
    # Output dict of {volume id : <Volume> object}
    print("Beginning Caching for Volume Segmentation")
    print(f"Parameters: {parameters}")
    volumes = {}
    # Make dict of {z: {id: [(x,y)]}} for each z-plane 
    z_planes = {}
    for x, y, z, id in xyz_points:
        if not z_planes.get(z, None):
            z_planes[z] = {}
        if not z_planes[z].get(id, None):
            z_planes[z][id] = []
        z_planes[z][id].append((x, y))
    z_planes_sorted = dict(sorted(z_planes.items())) # Sort in ascending z, allowing volumes to be built from the ground up
    # Make dict of xy ROIs -> {xy_id : BoundaryRegion of pts}
    xy_rois = {}
    xy_id = 0
    for z, id_dict in z_planes_sorted.items():
        for id, points in id_dict.items():
            # Make a boundary region using boundary pts of xy roi
            points_3d = [(x,y,z) for x,y in points]
            xy_rois[xy_id] = BoundaryRegion(points_3d, 
                                            original_index=id, 
                                            precalc_area=parameters['restrict_area'], 
                                            precalc_centroid=parameters['restrict_centroid_distance'],
                                            precalc_radius=parameters['use_percent_centroid_distance'])
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
                points_3d.extend([(x, y, z) for x, z in boundary_points])
            elif isinstance(item, np.ndarray):
                for point in item:
                    points_3d.append((point[0], y, point[1]))
            if len(points_3d) > 0:
                xz_rois[xz_id] = BoundaryRegion(points_3d)
                xz_id += 1

    print(f"Finished Volume Segmentation Caching - beginning algorithm")
    # Iterate through all XY ROIs and determine collisions with XZ ROIs
    current_obj_id = 1 # ID of current object being created
    removed_objects = [] # List of objects removed during merge operations
    num_xy_rois = len(xy_rois) # Debug code used to provide progress
    xy_roi_count = 0 # Count of number of XY rois processed
    last_z_level = None # Debug
    xy_untracked = [] # Debug
    for xy_roi_id, xy_roi in xy_rois.items():
        #TEST DEBUG
        xy_roi_zmin = xy_roi.zmin
        xy_roi_zmax = xy_roi.zmax
        if (xy_roi_zmax == xy_roi_zmin and last_z_level != xy_roi_zmin):
            print(f"Processing XY Rois at z={xy_roi_zmin}")
            last_z_level = xy_roi_zmin
        if xy_roi_count % 100 == 0:
            print(f"Evaluating XY ROI {xy_roi_count + 1}/{num_xy_rois}")
        xy_roi_count += 1
        found_link = False
        # END DEBUG
        for xz_roi_id, xz_roi in xz_rois.items():
            # Rough check if an intersection is theoretically possible
            if not xy_roi.overlaps_with_region(xz_roi):
                continue # Objects cannot spacially overlap
            # Execution reached here - more detailed check for collision
            intersection = check_region_intersection(xy_roi.get_boundary_points(), xz_roi.get_boundary_points(), parameters=parameters)
            if intersection: # XZ and XY ROIs are part of same object
                found_link = True # Test code
                # Determine what 3D object id to assign to these two ROIs
                xz_roi_objs = set()# List of possible XZ ROI objs
                xy_roi_obj = None
                for obj_id, volume in volumes.items():
                    if obj_id in removed_objects:
                        continue # Ignore removed object
                    if xz_roi_id in volume.get_xz_rois():
                        # The XZ slice is already in this volume
                        xz_roi_objs.add(obj_id)
                    if xy_roi_id in volume.get_xy_rois():
                        # The XY slice is already in this volume
                        xy_roi_obj = obj_id
                # Add a sentinel value if no volumes with a matching XZ roi exist
                if len(xz_roi_objs) == 0:
                    xz_roi_objs = {None}
                # For all possible XZ roi links
                for xz_roi_obj in xz_roi_objs:
                    if xz_roi_obj in removed_objects or xy_roi_obj in removed_objects:
                        continue
                    # Case 1 : both rois have an object - attempt to merge objects    
                    if xy_roi_obj is not None and xz_roi_obj is not None:
                        if xz_roi_obj == xy_roi_obj: # Same obj - do nothing
                            continue
                        keep_obj = min(xy_roi_obj, xz_roi_obj)
                        discard_obj = max(xy_roi_obj, xz_roi_obj)
                        # Merge only if there are no common z-levels and centroid check passes
                        xy_rois_keep = volumes[keep_obj].get_xy_rois().values()
                        xy_rois_discard = volumes[discard_obj].get_xy_rois().values()
                        restriction_check = check_restrictions(xy_rois_keep, xy_rois_discard, distance_threshold=parameters['centroid_distance_max'], z_distance_threshold=parameters['match_z_threshold'], parameters=parameters)
                        # Perform volume check if required
                        ar_restrict_check = True
                        if restriction_check and parameters['restrict_area_change']:
                            ar_restrict_check = check_area_diff(parameters['ar_change_perc'],
                                                                   volumes[keep_obj], volumes[discard_obj],
                                                                   parameters['ar_change_num_samples'],
                                                                   parameters['ar_change_activation_thresh'])
                        if not restriction_check or not ar_restrict_check:
                            # Cannot merge, but can add XZ roi for future linkages
                            # print(f"Merge failed between vols {keep_obj} and {discard_obj}.")
                            # plot_volumes(volumes[keep_obj], volumes[discard_obj], labels=[f"Volume {keep_obj}", f"Volume {discard_obj}"])
                            volumes[xy_roi_obj].add_xz_roi(xz_roi_id, xz_roi)
                            continue
                        else:
                            # Merge is OK
                            volumes[keep_obj].volume_merge(volumes[discard_obj])
                            # Remove (ignore for now) the discarded volume
                            removed_objects.append(discard_obj)
                            # Update XY Roi obj for this iteration of loop
                            xy_roi_obj = keep_obj
                    # Case 2 : XY roi is in an object, XZ roi is not - add XZ roi
                    elif xy_roi_obj is not None and xz_roi_obj is None:
                        # This should only be reached once, as xz_roi_obj is only None of it has no volume associated
                        volumes[xy_roi_obj].add_xz_roi(xz_roi_id, xz_roi)
                    # Case 3 : XZ roi is in an object and XY is not - attempt to add XY roi
                    elif xz_roi_obj is not None and xy_roi_obj is None:
                        # Ensure volume restrictions are upheld when adding in this XY ROI
                        xz_obj_xy_rois = volumes[xz_roi_obj].get_xy_rois().values()
                        restriction_check = check_restrictions(xz_obj_xy_rois, [xy_roi], distance_threshold=parameters['centroid_distance_max'], z_distance_threshold=parameters['match_z_threshold'], parameters=parameters)
                        # Check volume area difference if required
                        ar_restrict_check = True
                        if restriction_check and parameters['restrict_area_change']:
                            # Make a temporary volume for xy roi_obj
                            xy_vol = Volume(-1, track_avg_area=True)
                            xy_vol.add_xy_roi(xy_roi_id, xy_roi) # Add the single xy roi to this temp volume
                            ar_restrict_check = check_area_diff(parameters['ar_change_perc'],
                                                                   volumes[xz_roi_obj], xy_vol,
                                                                   parameters['ar_change_num_samples'],
                                                                   parameters['ar_change_activation_thresh'])
                        if not restriction_check or not ar_restrict_check:
                            # Cannot add XY roi to XZ's volume, but can create a new volume for them
                            # print(f"XY ROI Addition failed between xz vol {xz_roi_obj} and xy ROI {xy_id}.")
                            # Create a new volume object
                            new_volume = Volume(current_obj_id, track_avg_area=parameters['restrict_area_change'])
                            # Add the intersecting XY and XZ ROIs to the object
                            new_volume.add_xy_roi(xy_roi_id, xy_roi)
                            new_volume.add_xz_roi(xz_roi_id, xz_roi)
                            # Add new volume to the volumes dictionary
                            volumes[current_obj_id] = new_volume
                            # Update the volume for this iteration of the loop
                            xy_roi_obj = current_obj_id
                            # Increment the id for the next volume
                            current_obj_id += 1
                            continue
                        else:
                            # Restrictions passed - add this XY ROI 
                            volumes[xz_roi_obj].add_xy_roi(xy_roi_id, xy_roi)
                            xy_roi_obj = xz_roi_obj # Update object ID in case more XZ ROIs to handle
                    else:
                        # Create a new volume object
                        new_volume = Volume(current_obj_id, track_avg_area=parameters['restrict_area_change'])
                        # Add the intersecting XY and XZ ROIs to the object
                        new_volume.add_xy_roi(xy_roi_id, xy_roi)
                        new_volume.add_xz_roi(xz_roi_id, xz_roi)
                        # Add new volume to the volumes dictionary
                        volumes[current_obj_id] = new_volume
                        # Update the volume for this iteration of the loop
                        xy_roi_obj = current_obj_id
                        # Increment the id for the next volume
                        current_obj_id += 1
        # TEST DEBUG
        if not found_link:
            xy_untracked.append(xy_roi_id)
        # END DEBUG
    # Remove deleted volumes
    for obj in removed_objects:
        volumes.pop(obj)
    # DEBUG
    print(f"{len(xy_untracked)} unlinked xy volumes: {xy_untracked}")
    # END DEBUG
    # Return volumes dictionary
    return volumes

# Extracts the list of boundary points of a 2D convex hull
def get_hull_boundary_points(hull):
    # Extract the vertices of the convex hull
    vertices = hull.points[hull.vertices]
    
    # Convert the vertices to a list of (x, y) tuples
    boundary_points = [(x, y) for x, y in vertices]
    
    return boundary_points

# Determines if volume 1's average area per ROI is changed by perc_diff or more when merged with volume 2
def check_area_diff(perc_diff, volume_1, volume_2, rois_to_compare, min_rois):
    vol_1_num_rois = len(volume_1.xy_rois_per_z)
    vol_2_num_rois = len(volume_2.xy_rois_per_z)
    
    # Check to ensure enough rois exist
    if vol_1_num_rois < min_rois and not (vol_2_num_rois > min_rois and (vol_1_num_rois + vol_2_num_rois) > (min_rois + rois_to_compare)) \
        or vol_2_num_rois < min_rois and not (vol_1_num_rois > min_rois and (vol_1_num_rois + vol_2_num_rois) > (min_rois + rois_to_compare)):
        return True # Not enough ROIs to make a confident prediction - passes check
    
    # Determine current running volume averages (modified in if/else chain below)
    _, v1_running_avg_list = volume_1.get_roi_areas()
    _, v2_running_avg_list = volume_2.get_roi_areas()
    v1_avg = v1_running_avg_list[-1]
    v2_avg = v2_running_avg_list[-1]

    if vol_1_num_rois >= min_rois and vol_2_num_rois < rois_to_compare:
        # Vol 1 is bigger volume, augment volume 2 with enough ROIs
        v1_avg, v2_avg = determine_average_area_per_roi(volume_1, volume_2, rois_to_compare)
    elif vol_2_num_rois >= min_rois and vol_1_num_rois < rois_to_compare:
        # Vol 2 is bigger volume, augment volume 1 with enough ROIs
        v2_avg, v1_avg = determine_average_area_per_roi(volume_2, volume_1, rois_to_compare)
    
    # This occurs when the volumes are overlapping in 3d space (share a z-level for xy rois)
    if v1_avg is None or v2_avg is None:
        return True # This restriction is not able to complete (result is discarded as a pass/true)
    
    return compare_average_area_per_roi(v1_avg, v2_avg) <= perc_diff

# Augments smaller volumes with a subset of the larger volume, and calculates average rois per z
def determine_average_area_per_roi(larger_volume, smaller_volume, rois_to_compare):
    num_smaller_rois = len(smaller_volume.xy_rois_per_z)
    num_to_add = rois_to_compare - num_smaller_rois
    
    # Sort both by z-values (extract the zmin from the first ROI in the tuple list)
    smaller_rois_sorted = sorted(smaller_volume.xy_rois_per_z.items(), key=lambda item: item[0])
    larger_rois_sorted = sorted(larger_volume.xy_rois_per_z.items(), key=lambda item: item[0])
    
    # Get zmin and zmax of smaller volume
    smaller_z_min = smaller_rois_sorted[0][0]
    smaller_z_max = smaller_rois_sorted[-1][0]

    # Check if the larger volume has any ROIs inside the smaller volume's z-range
    for z_val, _ in larger_rois_sorted:
        if smaller_z_min <= z_val <= smaller_z_max:
            return None, None  # Exit early since the larger volume has ROIs inside the smaller volume
    
    # Collect ROIs closest to smaller volume from the larger volume
    rois_to_add = []
    rois_selected = set()  # Track which rois were selected by closest z
    
    # Keep adding ROIs from the larger volume until we reach num_to_add
    while len(rois_to_add) < num_to_add:
        if len(rois_selected) == len(larger_rois_sorted):
            break

        closest_roi = None
        closest_distance = float('inf')

        # Find the closest ROI in z from the larger volume that hasn't been added
        for z_val, roi_list in larger_rois_sorted:
            if z_val in rois_selected:
                continue  # Skip already selected rois
            
            # Compute the z-distance to the smaller volume's boundaries
            distance_to_min = abs(z_val - smaller_z_min)
            distance_to_max = abs(z_val - smaller_z_max)
            
            # Choose the minimum distance
            distance = min(distance_to_min, distance_to_max)
            
            # If this ROI is the closest, mark it for selection
            if distance < closest_distance:
                closest_roi = (z_val, roi_list)
                closest_distance = distance
        
        if closest_roi is not None:
            # Add the closest ROI to the list of ROIs to be added
            for roi_id, roi in roi_list:
                if len(rois_to_add) < num_to_add:  # Ensure we only add the required number of ROIs
                    rois_to_add.append((z_val, (roi_id, roi)))  # Track individual ROIs
                    rois_selected.add(z_val)  # Mark this z_val as selected

    # Retrieve volume info from the larger volume
    l_area_dict = larger_volume.roi_areas
    
    # Use a copy of the smaller volume's volume dictionary
    s_area_dict = smaller_volume.roi_areas.copy()  # Assuming smaller_volume has a volume dictionary
    
    # Update the smaller volume with the selected ROIs from the larger volume
    for z_val, (roi_id, roi) in rois_to_add:
        s_area_dict[roi_id] = l_area_dict[roi_id]
    
    # Recalculate the running average for the smaller volume
    smaller_running_avg = recalculate_running_average(s_area_dict)
    
    # Recalculate the running average for the larger volume (excluding the augmented ROIs)
    l_area_dict_updated = {roi_id: l_area_dict[roi_id] for roi_id in l_area_dict if roi_id not in s_area_dict}
    larger_running_avg = recalculate_running_average(l_area_dict_updated)
    
    return larger_running_avg, smaller_running_avg
    
# Recalculate the running average of volumes from the volume dictionary
def recalculate_running_average(area_dict):
    total_volume = sum(area_dict.values())
    num_rois = len(area_dict)
    return total_volume / num_rois if num_rois > 0 else 0

# Compare symmetric percentage difference in average area per roi in volumes
def compare_average_area_per_roi(v1_avg, v2_avg):
    if v1_avg == 0 and v2_avg == 0:
        return 0  # No difference if both are 0
    if v1_avg == 0 or v2_avg == 0:
        return float('inf')  # If only one is 0, return infinity
    
    avg_of_avgs = (v1_avg + v2_avg) / 2
    percent_change = ((v1_avg - v2_avg) / avg_of_avgs) * 100
    return abs(percent_change)

# Checks two sets of ROIs to ensure they meet the restrictions imposed in the parameters
def check_restrictions(xy_rois_1, xy_rois_2, distance_threshold, z_distance_threshold, parameters):
    # Combine XY ROIs from both keep and discard volumes
    combined_xy_rois = list(xy_rois_1) + list(xy_rois_2)
    rois_by_z = {}
    for roi in combined_xy_rois:
        roi_points = roi.get_boundary_points()
        z_value = roi_points[0][2] # Take first point, and 2nd index for static z-value of xy roi
        if rois_by_z.get(z_value, None) and parameters['restrict_multiple_xy_per_vol']: # Two XY rois at same z-level in same volume detected, and this is not allowed
            # print("Two XY ROIs at same level between volumes")
            return False
        else:
            rois_by_z[z_value] = roi

    # Sort by ascending z
    sorted_z_rois = sorted(rois_by_z.items())  # Sorted list of (z_value, roi) tuples

    # Go through list of XY Rois and check adjacent rois to see if they meet restrictions
    for i in range(len(sorted_z_rois)):
        z, roi = sorted_z_rois[i]
        # Check for the ROI above (next in the sorted list)
        if i < len(sorted_z_rois) - 1:  # There is a next ROI
            z_above, roi_above = sorted_z_rois[i + 1]
            if not enforce_restrictions(z, z_above, roi, roi_above, z_distance_threshold, distance_threshold, parameters):
                return False
        # Check for the ROI below (previous in the sorted list)
        if i > 0:  # There is a previous ROI
            z_below, roi_below = sorted_z_rois[i - 1]
            if not enforce_restrictions(z, z_below, roi, roi_below, z_distance_threshold, distance_threshold, parameters):
                return False
    return True

# Enforce all volume segmentation restrictions set in parameters
def enforce_restrictions(z1, z2, roi1, roi2, z_distance_threshold, distance_threshold, parameters):
    if parameters['restrict_adjacency']:
        z_gap = abs(z1 - z2)
        if z_gap > z_distance_threshold:
            # print(f"Distance check failed: z_gap={z_gap}")
            return False
    if parameters['restrict_centroid_distance']:
        # Get centroids
        roi1_centroid = roi1.get_centroid()
        roi2_centroid = roi2.get_centroid()
        # Calculate XY distance in 2D and check against threshold
        xy_distance = distance.euclidean(roi1_centroid[:2], roi2_centroid[:2])
        # Modify the distance threshold if necessary
        if parameters['use_percent_centroid_distance']:
            # When using a percent centroid distance restriction, the ROI with a larger average radius is used for the comparison
            roi1_radius = roi1.get_radius()
            roi2_radius = roi2.get_radius()
            # avg_radius = int((roi1_radius + roi2_radius) / 2)
            # smaller_radius = min(roi1_radius, roi2_radius)
            larger_radius = max(roi1_radius, roi2_radius)
            # Redefine flat distance threshold as a percentage of the {smaller,larger, avg} radius between rois
            distance_threshold = (parameters['centroid_distance_perc']/100) * larger_radius  # testing different metrics...
        if xy_distance > distance_threshold:
            # print(f"Distance check failed: xy_dist={xy_distance}")
            return False
    if parameters['restrict_area']:
        # Calculate area of roi1
        roi1_area = roi1.get_area()
        # Get the area of the ROI above
        roi2_area = roi2.get_area()
        # larger_area = roi1_area if roi1_area > roi2_area else roi2_area
        # smaller_area = roi1_area if roi1_area < roi2_area else roi2_area
        # # Calculate the percentage change in area from small to big (e.g 300% -> 3x increase)
        # area_delta_perc = (larger_area / smaller_area) * 100
        # if area_delta_perc > parameters['area_delta_perc_threshold']:
        #     # print(f"Area change failed: area_delta_perc={area_delta_perc}%")
        #     return False
            # Handle zero areas to prevent division by zero
        if roi1_area == 0 or roi2_area == 0:
            return False  # Or handle zero-area cases as you see fit

        # Calculate the average area for a more symmetric percentage change
        avg_area = (roi1_area + roi2_area) / 2
        area_delta_perc = (abs(roi1_area - roi2_area) / avg_area) * 100
        # Check if the change in area exceeds the threshold
        if area_delta_perc > parameters['area_delta_perc_threshold']:
            return False
    # All checks passed if execution reached here
    return True
        

def plot_volumes(volume1, volume2, labels, subset_fraction=0.5):
    """
    Plot a subset of points from two volumes.
    
    Parameters:
    - volume1, volume2: Volume objects.
    - labels: List of labels for the two volumes.
    - subset_fraction: Fraction of points to plot (default is 0.5, i.e., 50%).
    """
    def subsample_points(points, fraction):
        """Subsample a fraction of points from the list."""
        sample_size = int(len(points) * fraction)
        return random.sample(points, sample_size) if sample_size > 0 else points
    
    # Create a new 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Collect all points for Volume 1 (XZ and XY separately)
    volume1_xz_points = []
    volume1_xy_points = []
    
    # for xz_roi in volume1.get_xz_rois().values():
    #     xz_points = xz_roi.get_boundary_points()  # Get boundary points of the XZ ROI
    #     if xz_points:
    #         sampled_xz_points = subsample_points(xz_points, subset_fraction)
    #         volume1_xz_points.extend(sampled_xz_points)
    
    for xy_roi in volume1.get_xy_rois().values():
        xy_points = xy_roi.get_boundary_points()  # Get boundary points of the XY ROI
        if xy_points:
            sampled_xy_points = subsample_points(xy_points, subset_fraction)
            volume1_xy_points.extend(sampled_xy_points)

    # Collect all points for Volume 2 (XZ and XY separately)
    volume2_xz_points = []
    volume2_xy_points = []
    
    # for xz_roi in volume2.get_xz_rois().values():
    #     xz_points = xz_roi.get_boundary_points()  # Get boundary points of the XZ ROI
    #     if xz_points:
    #         sampled_xz_points = subsample_points(xz_points, subset_fraction)
    #         volume2_xz_points.extend(sampled_xz_points)
    
    for xy_roi in volume2.get_xy_rois().values():
        xy_points = xy_roi.get_boundary_points()  # Get boundary points of the XY ROI
        if xy_points:
            sampled_xy_points = subsample_points(xy_points, subset_fraction)
            volume2_xy_points.extend(sampled_xy_points)

    # Plot all XZ points for volume1
    if volume1_xz_points:
        xs, ys, zs = zip(*volume1_xz_points)
        ax.scatter(xs, ys, zs, label=labels[0] + " XZ", color='b')

    # Plot all XY points for volume1
    if volume1_xy_points:
        xs, ys, zs = zip(*volume1_xy_points)
        ax.scatter(xs, ys, zs, label=labels[0] + " XY", color='c')

    # Plot all XZ points for volume2
    if volume2_xz_points:
        xs, ys, zs = zip(*volume2_xz_points)
        ax.scatter(xs, ys, zs, label=labels[1] + " XZ", color='r')

    # Plot all XY points for volume2
    if volume2_xy_points:
        xs, ys, zs = zip(*volume2_xy_points)
        ax.scatter(xs, ys, zs, label=labels[1] + " XY", color='m')

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Show the plot
    plt.show()


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