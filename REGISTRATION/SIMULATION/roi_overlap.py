import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.affinity import translate, rotate
from scipy.optimize import linear_sum_assignment

NUM_ELLIPSES = 15
INTENSITY_SELECTION_THRESHOLD = 0.5 # Intensity required for an ROI to be considered as an anchor point
INTENSITY_DELTA = 0.1 # Intensity delta between two ROIs for the match to be considered
ANGLE_DELTA_DEG = 5 # Angle to rotate between tests
ANGLE_ROTATE_MAX = 45 # Max angle to rotate ROIs when comparing
# Base data dimensions
NUM_POINTS = 100
RADIUS_X = 4
RADIUS_Y = 2
# Transformations for second set of data
TRANSFORM_ROTATION = 30
X_SHIFT = 5
Y_SHIFT = -3

# Function to create an ellipse as a polygon
def ellipse_polygon(center_x, center_y, radius_x, radius_y, angle_deg=0, num_points=100):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = radius_x * np.cos(t)
    y = radius_y * np.sin(t)
    points = np.column_stack((x, y))
    
    ellipse = Polygon(points)
    ellipse = rotate(ellipse, angle_deg, origin=(0, 0), use_radians=False)
    ellipse = translate(ellipse, center_x, center_y)
    return ellipse

def main():
    random.seed(10)
    ellipses = []
    ellipse_intensities = []
    transformed_ellipses = []
    transformed_intensities = [] 
    # Create a number of randomly placed ellipses
    for _ in range(NUM_ELLIPSES):
        cx = random.uniform(20, 80)
        cy = random.uniform(20, 80)
        angle = random.uniform(0, 360)
        intensity = random.uniform(0, 1)
        ellipse = ellipse_polygon(center_x=cx, center_y=cy, radius_x=RADIUS_X, radius_y=RADIUS_Y, angle_deg=angle)
        ellipses.append(ellipse)
        ellipse_intensities.append(intensity)
    for i in range(len(ellipses)):
        ellipse = ellipses[i]
        transformed = apply_transformations(ellipse, dx=X_SHIFT, dy=Y_SHIFT, rotation_deg=TRANSFORM_ROTATION)
        transformed_ellipses.append(transformed)
        intensity_modifier = random.uniform(90,100) # Slightly modify intensity
        transformed_intensities.append(ellipse_intensities[i] * (intensity_modifier / 100))

    # Plot initial setup of points
    # Normalize intensities for colormap
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.Greys_r

    fig, ax = plt.subplots()

    # Plot original ellipses
    for ellipse, intensity in zip(ellipses, ellipse_intensities):
        ox, oy = ellipse.exterior.xy
        color = cmap(norm(intensity))
        ax.fill(ox, oy, facecolor=color, edgecolor='black', alpha=0.5)

    # Plot transformed ellipses
    for ellipse, intensity in zip(transformed_ellipses, transformed_intensities):
        tx, ty = ellipse.exterior.xy
        color = cmap(norm(intensity))
        ax.fill(tx, ty, facecolor=color, edgecolor='black', alpha=0.7, linestyle='--')

    ax.set_aspect('equal')
    ax.set_title("Original vs Transformed")
    plt.grid(True)
    plt.show()

    # Align polygons
    max_avg_IoU, best_pair, best_angle, best_matches, aligned_set2 = align_polygons(ellipses, transformed_ellipses, ellipse_intensities, transformed_intensities)
    print(f"Max_avg_IoU: {max_avg_IoU}, Best Pair: {best_pair}, Best Angle: {best_angle}")
    # Show alignment
    rotated_poly_set2 = [apply_transformations(poly, rotation_deg=best_angle) for poly in aligned_set2]
    plot_aligned_polygons(
    set1=ellipses,
    set2=rotated_poly_set2,
    matches=best_matches,
    title=f"Aligned Polygons @ {best_angle}°, Avg IoU = {max_avg_IoU:.2%}"
)

def align_polygons(set1, set2, intensity1, intensity2):
    # Initialize return variables
    max_avg_IoU = 0 # Best percent match
    best_pair = None
    best_angle = None
    best_matches = []
    # Normalize both intensity lists - TODO may need outlier removal in real data
    intensity1_norm = normalize(intensity1)
    intensity2_norm = normalize(intensity2)
    # Determine matching candidates as ROIs with intensity above the designated threshold
    potential_anchors = [i for i, intensity in enumerate(intensity1_norm) if intensity > INTENSITY_SELECTION_THRESHOLD]
    # For each ROI candidate, determine viable ROIs in other set it can be matched with (within threshold)
    # Make dictionary to hold matching Ids
    anchor_matches = {i:[] for i in potential_anchors}
    for i in anchor_matches:
        for j, intensity2 in enumerate(intensity2_norm):
            intensity1 = intensity1_norm[i]
            if abs(intensity1 - intensity2) <= INTENSITY_DELTA:
                anchor_matches[i].append(j)
    # For each viable ROI anchor match, determine pairings of all ROIs based off distance
    anchor_pairings = {(i,j) : {} for i in anchor_matches for j in anchor_matches[i]}
    shift_vectors = [] # Store which shifts have already occurred
    checked_matches = [] # Stores which assignments have already occurred
    for pair in anchor_pairings:
        set1_id, set2_id = pair
        anchor_pairings[pair][set1_id] = set2_id # Trivial match of set1 id to set2 id

        aligned_set2 = []
        shift_vector = np.array(set1[set1_id].centroid.coords[0]) - np.array(set2[set2_id].centroid.coords[0])
        # Test to see if this anchor shift has already occurred
        if is_shift_duplicate(shift_vector, shift_vectors):
            print(f"Pairing {pair} has already been tested using shift vector {shift_vector}")
            continue
        else:
            shift_vectors.append(shift_vector)
        # Transform x/y coords of set2 ellipses such that the pairing are overlaid
        for poly in set2:
            aligned = translate(poly, xoff=shift_vector[0], yoff=shift_vector[1])
            aligned_set2.append(aligned)
        # Store polygons for rotation later
        anchor_pairings[pair]['aligned_set2'] = aligned_set2
        # Match polygons in set1 and aligned set 2 based off centroid distance
        matches = match_by_centroid_distance(set1, aligned_set2)
        # If these matches have been evaluated already, ignore them
        if matches in checked_matches:
            print(f"Match assignments for ROIs has been evaluated already")
            continue
        else:
            checked_matches.append(matches)
        anchor_pairings[pair]['matches'] = matches
        print(matches)
        # For each pairing, generate a set of rotated polygons, each rotated at +- ANGLE_DELTA_DEG up to ANGLE_ROTATE_MAX and determine % degree overlap
        for angle in range(0, ANGLE_ROTATE_MAX + 1, ANGLE_DELTA_DEG):
            for direction in [-1, 1]: # Iterate over positive and negative angles
                signed_angle = angle * direction
                # Create a new set of polygons rotated at the desired angle
                rotated_poly_set2 = [apply_transformations(poly, rotation_deg=signed_angle) for poly in anchor_pairings[pair]['aligned_set2']]
                # Calculate average IoU across this match
                avg_iou = compute_average_iou(set1, rotated_poly_set2, anchor_pairings[pair]['matches'])
                # Debug
                print(f"Aligning {pair} at {signed_angle} degrees shift. Avg IoU: {avg_iou}")
                if avg_iou > max_avg_IoU:
                    max_avg_IoU = avg_iou
                    best_pair = pair
                    best_angle = signed_angle
                    best_matches = matches

    # Return best pairing and max IoU
    return max_avg_IoU, best_pair, best_angle, best_matches, anchor_pairings[best_pair]['aligned_set2']
    

def plot_aligned_polygons(set1, set2, matches, title="Polygon Alignment"):
    fig, ax = plt.subplots(figsize=(10, 8))
    patches = []
    colors = []

    minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')

    for i1, i2 in matches:
        poly1 = set1[i1]
        poly2 = set2[i2]

        # Update bounds
        for poly in [poly1, poly2]:
            bounds = poly.bounds
            minx, miny = min(minx, bounds[0]), min(miny, bounds[1])
            maxx, maxy = max(maxx, bounds[2]), max(maxy, bounds[3])

        # Original in blue
        patches.append(MplPolygon(list(poly1.exterior.coords), closed=True))
        colors.append((0.2, 0.4, 1.0, 0.4))  # blue

        # Transformed in red
        patches.append(MplPolygon(list(poly2.exterior.coords), closed=True))
        colors.append((1.0, 0.2, 0.2, 0.4))  # red

        # Intersection in purple
        inter = poly1.intersection(poly2)
        if inter.is_empty:
            continue
        if isinstance(inter, Polygon):
            patches.append(MplPolygon(list(inter.exterior.coords), closed=True))
            colors.append((0.5, 0.2, 0.8, 0.5))  # purple
        elif isinstance(inter, (MultiPolygon, GeometryCollection)):
            for geom in inter.geoms:
                if isinstance(geom, Polygon):
                    patches.append(MplPolygon(list(geom.exterior.coords), closed=True))
                    colors.append((0.5, 0.2, 0.8, 0.5))  # purple

    if not patches:
        print("No matches or nothing to draw.")
        return

    p = PatchCollection(patches, facecolors=colors, edgecolors='black', linewidths=1)
    ax.add_collection(p)

    ax.set_xlim(minx - 5, maxx + 5)
    ax.set_ylim(miny - 5, maxy + 5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def compute_average_iou(set1, set2, matches):
    ious = []
    for i1, i2 in matches:
        if i1 == -1 or i2 == -1: # Check for case where ROI is unmatched
            ious.append(0)
        else:
            poly1 = set1[i1]
            poly2 = set2[i2]
            intersection = poly1.intersection(poly2)
            union = poly1.union(poly2)

            if not union.is_empty and union.area > 0:
                iou = intersection.area / union.area
                ious.append(iou)

    return sum(ious) / len(ious) if ious else 0

def is_shift_duplicate(new_shift, existing_shifts, tol=0.2):
    for shift in existing_shifts:
        if np.allclose(new_shift, shift, atol=tol):
            return True
    return False

def match_by_centroid_distance(set1, set2):
    n = len(set1)
    m = len(set2)
    size = max(n, m)
    
    # Initialize square cost matrix with large default value (acts as penalty for unmatched)
    cost_matrix = np.full((size, size), fill_value=1e6)
    
    # Fill in actual distances for real polygon comparisons
    for i in range(n):
        c1 = np.array(set1[i].centroid.coords[0])
        for j in range(m):
            c2 = np.array(set2[j].centroid.coords[0])
            cost_matrix[i, j] = np.linalg.norm(c1 - c2)

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Store matches
    matches = []
    matched_set1 = set()
    matched_set2 = set()
    for i, j in zip(row_ind, col_ind):
        if i < n and j < m:
            matches.append((i, j))
            matched_set1.add(i)
            matched_set2.add(j)
    
    # Add unmatched from set1
    for i in range(n):
        if i not in matched_set1:
            matches.append((i, -1))
    
    # Add unmatched from set2
    for j in range(m):
        if j not in matched_set2:
            matches.append((-1, j))

    return matches  # list of (set1_idx, set2_idx)    

def align_by_centroid(base_poly, moving_poly):
    base_centroid = np.array(base_poly.centroid.coords[0])
    moving_centroid = np.array(moving_poly.centroid.coords[0])
    shift = base_centroid - moving_centroid
    return translate(moving_poly, xoff=shift[0], yoff=shift[1])

def normalize(intensities):
    min_val = min(intensities)
    max_val = max(intensities)
    return [(i - min_val) / (max_val - min_val) for i in intensities]

def apply_transformations(polygon, dx=0, dy=0, rotation_deg=0):
    coords = np.array(polygon.exterior.coords[:-1])  # exclude duplicate last point
    centroid = coords.mean(axis=0)  # centroid pre-transform

    # Rotate around original centroid
    theta = np.radians(rotation_deg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    
    coords_centered = coords - centroid
    rotated = coords_centered @ rotation_matrix.T + centroid

    # Then translate the rotated polygon
    rotated += np.array([dx, dy])

    return Polygon(rotated)



if __name__ == "__main__":
    main()




# # Create two elliptical polygons
# ellipse1 = ellipse_polygon(center_x=0, center_y=0, radius_x=4, radius_y=2, angle_deg=0)
# ellipse2 = ellipse_polygon(center_x=2, center_y=0.5, radius_x=4, radius_y=2, angle_deg=30)

# # Compute intersection and IoU
# intersection = ellipse1.intersection(ellipse2)
# union = ellipse1.union(ellipse2)

# iou = intersection.area / union.area * 100

# print(f"Ellipse 1 Area: {ellipse1.area:.2f}")
# print(f"Ellipse 2 Area: {ellipse2.area:.2f}")
# print(f"Intersection Area: {intersection.area:.2f}")
# print(f"Union Area: {union.area:.2f}")
# print(f"IoU: {iou:.2f}%")

# # Plotting
# fig, ax = plt.subplots(figsize=(8, 6))
# x1, y1 = ellipse1.exterior.xy
# x2, y2 = ellipse2.exterior.xy

# # Draw ellipses
# ax.fill(x1, y1, alpha=0.5, fc='lightblue', ec='black', label='Ellipse 1')
# ax.fill(x2, y2, alpha=0.5, fc='salmon', ec='black', label='Ellipse 2')

# # Draw intersection if it exists
# if not intersection.is_empty:
#     x_int, y_int = intersection.exterior.xy
#     ax.fill(x_int, y_int, alpha=0.6, fc='purple', label='Intersection')

# ax.set_aspect('equal')
# ax.legend()
# ax.set_title(f"Elliptical Polygon Overlap\nIoU = {iou:.2f}%")
# ax.grid(True)
# plt.show()
