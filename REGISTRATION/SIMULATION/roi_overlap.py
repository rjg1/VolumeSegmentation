import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.affinity import scale, translate, rotate
from scipy.optimize import linear_sum_assignment

NUM_ELLIPSES = 5
INTENSITY_SELECTION_THRESHOLD = 0.5 # Intensity required for an ROI to be considered as an anchor point
INTENSITY_DELTA_PERC = 0.2 # Intensity delta percent between two ROIs for the match to be considered
ANGLE_DELTA_DEG = 10 # Angle to rotate between tests
ANGLE_ROTATE_MAX = 50 # Max angle to rotate ROIs when comparing
ZOOM_DELTA = 2 # +- zoom range to search TODO - currently assuming it is zoomed in, not really support for zoomed out
ZOOM_INTERVAL = 0.5 # Intervals to search in zoom range
# Base data dimensions
NUM_POINTS = 200
RADIUS_X = 4
RADIUS_Y = 2
# Transformations for second set of data
TRANSFORM_ROTATION = 30
X_SHIFT = 10
Y_SHIFT = -7
ZOOM = 3 # 3x zoom
X_SUBWINDOW = 60 # x coordinate of zoomed window centre point
Y_SUBWINDOW = 30 # y coodrdinate of zoomed window centre point
# Image dimensions
X_MIN = 0
X_MAX = 100
Y_MIN = 0
Y_MAX = 100

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

    # Define subwindow in input space
    window_center = (X_SUBWINDOW, Y_SUBWINDOW)

    # Define target display size
    base_width = X_MAX
    base_height = Y_MAX

    # Create a number of randomly placed ellipses
    for _ in range(NUM_ELLIPSES):
        cx = random.uniform(20, 80)
        cy = random.uniform(20, 80)
        angle = random.uniform(0, 360)
        intensity = random.uniform(0, 1)
        rad_x = random.uniform(2,5)
        rad_y = random.uniform(2,5)
        ellipse = ellipse_polygon(center_x=cx, center_y=cy, radius_x=rad_x, radius_y=rad_y, angle_deg=angle, num_points=NUM_POINTS)
        ellipses.append(ellipse)
        ellipse_intensities.append(intensity)
    for i in range(len(ellipses)):
        ellipse = ellipses[i]
        transformed = apply_transformations(ellipse, dx=X_SHIFT, dy=Y_SHIFT, rotation_deg=TRANSFORM_ROTATION)
        transformed_ellipses.append(transformed)
        intensity_modifier = random.uniform(90,100) # Slightly modify intensity
        transformed_intensities.append(ellipse_intensities[i] * (intensity_modifier / 100))

    # # Plot initial setup of points
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

    ax.set_xlim(0, base_width)
    ax.set_ylim(0, base_height)
    ax.set_aspect('equal')
    ax.set_title("Original vs Transformed")
    plt.grid(True)
    plt.show()

#     # Align polygons
#     max_avg_IoU, best_pair, best_angle, best_matches, aligned_set2, best_zoom = align_polygons(ellipses, transformed_ellipses, ellipse_intensities, transformed_intensities)
#     print(f"Max_avg_IoU: {max_avg_IoU}, Best Pair: {best_pair}, Best Angle: {best_angle}")
#     # Show alignment
#     rotated_poly_set2 = [apply_transformations(poly, rotation_deg=best_angle) for poly in aligned_set2]
#     plot_aligned_polygons(
#     set1=ellipses,
#     set2=rotated_poly_set2,
#     matches=best_matches,
#     title=f"Aligned Polygons @ {best_angle}°, Avg IoU = {max_avg_IoU:.2%}"
# )

    # Apply zoom to generated dataset
    zoomed_polys = zoom_polygons_to_window(
        transformed_ellipses,
        window_center=window_center,
        base_width=X_MAX,
        base_height=Y_MAX,
        xfact=ZOOM,
        yfact=ZOOM
    )

    fig, ax = plt.subplots()
    for poly in zoomed_polys:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.5, edgecolor='black')

    ax.set_xlim(0, base_width)
    ax.set_ylim(0, base_height)
    ax.set_aspect('equal')
    ax.set_title("Zoomed Polygon View")
    plt.grid(True)
    plt.show()

    # Extract polygons within subwindow (re-polygonizing edge polygons)
    extracted_polys, extracted_intensities = extract_polygons(zoomed_polys, transformed_intensities, base_width, base_height)

    # Remove zoom from generated dataset
    unzoomed_polys = unzoom_polygons_from_window(
        extracted_polys,
        window_center=window_center,
        base_width=X_MAX,
        base_height=Y_MAX,
        xfact=ZOOM,
        yfact=ZOOM
    )

    fig, ax = plt.subplots()
    for poly in unzoomed_polys:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.5, edgecolor='black')
    ax.set_aspect('equal')
    ax.set_xlim(0, base_width)
    ax.set_ylim(0, base_height)
    ax.set_title("Zoomed Polygon View")
    plt.grid(True)
    plt.show()

    # Align polygons
    # Generate a zoom guess
    min_zoom = ZOOM - ZOOM_DELTA
    max_zoom = ZOOM + ZOOM_DELTA
    zoom_options = [round(min_zoom + i * ZOOM_INTERVAL, 2) for i in range(int((max_zoom - min_zoom) / ZOOM_INTERVAL) + 1)]
    zoom_guess = random.choice(zoom_options)
    print(f"Guess zoom at: {zoom_guess}")
    max_avg_IoU, best_pair, best_angle, best_matches, aligned_set2, best_zoom = align_polygons(ellipses, 
                                                                                               extracted_polys, 
                                                                                               ellipse_intensities, 
                                                                                               extracted_intensities, 
                                                                                               zoom_guess = zoom_guess)
    # max_avg_IoU, best_pair, best_angle, best_matches, aligned_set2, best_zoom = align_polygons(ellipses, 
    #                                                                                         unzoomed_polys, 
    #                                                                                         ellipse_intensities, 
    #                                                                                         extracted_intensities, 
    #                                                                                         zoom_guess = None)
    print(f"Max_avg_IoU: {max_avg_IoU}, Best Pair: {best_pair}, Best Angle: {best_angle}, Best zoom: {best_zoom}")
    # Show alignment
    rotated_poly_set2 = [apply_transformations(poly, rotation_deg=best_angle) for poly in aligned_set2]
    plot_aligned_polygons(
    set1=ellipses,
    set2=rotated_poly_set2,
    matches=best_matches,
    title=f"Aligned Polygons @ {best_angle}°, Avg IoU = {max_avg_IoU:.2%}"
)


def align_polygons(set1, set2, intensity1, intensity2, zoom_guess = None):
    # Initialize return variables
    max_avg_IoU = 0 # Best percent match
    best_pair = None
    best_angle = None
    best_zoom = ZOOM
    best_matches = []
    # Zoom values to be tested # TODO Fix zooming in a bit more, fractional is a bit weird here (maybe call function as well)
    if zoom_guess is not None:
        zoom_values = np.arange(
                max(zoom_guess - ZOOM_DELTA, ZOOM_INTERVAL),
                zoom_guess + ZOOM_DELTA + 0.001,
                ZOOM_INTERVAL
            )
    else:
        zoom_values = [ZOOM]
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
            if abs(intensity1 - intensity2) <= (INTENSITY_DELTA_PERC * intensity1):
                anchor_matches[i].append(j)
    # For each viable ROI anchor match, determine pairings of all ROIs based off distance
    anchor_pairings = {(i,j) : {} for i in anchor_matches for j in anchor_matches[i]}

    for pair in anchor_pairings:
        # Apply all zoom levels for this anchor pairing, then work out shift
        for zoom in zoom_values:
            anchor_pairings[pair][zoom] = {}
            shift_vectors = [] # Store which shifts have already occurred at this zoom level
            checked_matches = [] # Stores which assignments have already occurred at this zoom level
            aligned_set2 = [] # List of polygons which are aligned to the anchor set 1 polygon
            if zoom_guess is not None:
                set2_unzoomed = unzoom_polygons_from_window(set2, (X_SUBWINDOW, Y_SUBWINDOW), X_MAX, Y_MAX, zoom, zoom)
            else:
                set2_unzoomed = set2

            set1_id, set2_id = pair


            shift_vector = np.array(set1[set1_id].centroid.coords[0]) - np.array(set2_unzoomed[set2_id].centroid.coords[0])
            # Test to see if this anchor shift has already occurred
            if is_shift_duplicate(shift_vector, shift_vectors):
                print(f"Pairing {pair} has already been tested using shift vector {shift_vector}")
                continue
            else:
                shift_vectors.append(shift_vector)
            # Transform x/y coords of set2 ellipses such that the pairing are overlaid
            for poly in set2_unzoomed:
                aligned = translate(poly, xoff=shift_vector[0], yoff=shift_vector[1])
                aligned_set2.append(aligned)
            # Store polygons for rotation later
            anchor_pairings[pair][zoom]['aligned_set2'] = aligned_set2
            # Match polygons in set1 and aligned set 2 based off centroid distance
            matches = match_by_centroid_distance(set1, aligned_set2)
            # If these matches have been evaluated already, ignore them
            if matches in checked_matches:
                print(f"Match assignments for ROIs have been evaluated already")
                continue
            else:
                checked_matches.append(matches)
            anchor_pairings[pair][zoom]['matches'] = matches
            print(matches)

            # For each pairing, generate a set of rotated polygons, each rotated at +- ANGLE_DELTA_DEG up to ANGLE_ROTATE_MAX and determine % degree overlap
            for angle in range(0, ANGLE_ROTATE_MAX + 1, ANGLE_DELTA_DEG):
                for direction in [-1, 1]: # Iterate over positive and negative angles
                    signed_angle = angle * direction
                    # Create a new set of polygons rotated at the desired angle
                    rotated_poly_set2 = [apply_transformations(poly, rotation_deg=signed_angle) for poly in anchor_pairings[pair][zoom]['aligned_set2']]
                    # Calculate average IoU across this match
                    avg_iou = compute_average_iou(set1, rotated_poly_set2, anchor_pairings[pair][zoom]['matches'])
                    if 'iou' not in anchor_pairings[pair][zoom]:
                        anchor_pairings[pair][zoom]['iou'] = {}
                    anchor_pairings[pair][zoom]['iou'][signed_angle] = avg_iou
                    # Debug
                    # print(f"Aligning {pair} at Zoom: {zoom} with {signed_angle} degrees shift. Avg IoU: {avg_iou}")
                    if avg_iou > max_avg_IoU:
                        max_avg_IoU = avg_iou
                        best_pair = pair
                        best_angle = signed_angle
                        best_matches = matches
                        best_zoom = zoom

    # Return best pairing and max IoU
    return max_avg_IoU, best_pair, best_angle, best_matches, anchor_pairings[best_pair][best_zoom]['aligned_set2'], best_zoom
    
# Takes in a list of polygons and returns only those within an area. Polygons partially inside area are
# re-polygonized
def extract_polygons(zoomed_polys, transformed_intensities, base_width, base_height):
    extracted_polys = []
    extracted_intensities = []

    for i in range(len(zoomed_polys)):
        poly = zoomed_polys[i]
        pxmin, pymin, pxmax, pymax = poly.bounds
        if pxmin >= 0 and pymin >= 0 and pxmax <= base_width and pymax <= base_height:  # polygon fully fits already
            extracted_polys.append(poly)
            extracted_intensities.append(transformed_intensities[i])
        else:
            # get points which lie in these bounds
            coords_in_bounds = []
            for coord in list(poly.exterior.coords):
                if coord[0] >= 0 and coord[0] <= base_width and coord[1] >= 0 and coord[1] <= base_height:
                    coords_in_bounds.append(coord)
                    
            if len(coords_in_bounds) > 3:
                extracted_polys.append(Polygon(coords_in_bounds))
                extracted_intensities.append(transformed_intensities[i])

    return extracted_polys, extracted_intensities


def plot_aligned_polygons(set1, set2, matches, title="Polygon Alignment"):
    fig, ax = plt.subplots(figsize=(10, 8))
    blue_patches, blue_colors = [], []
    red_patches, red_colors = [], []
    purple_patches, purple_colors = [], []

    for i1, i2 in matches:
        poly1 = set1[i1]
        poly2 = set2[i2]

        # Blue
        blue_patches.append(MplPolygon(list(poly1.exterior.coords), closed=True))
        blue_colors.append((0.2, 0.4, 1.0, 0.4))

        # Red
        red_patches.append(MplPolygon(list(poly2.exterior.coords), closed=True))
        red_colors.append((1.0, 0.2, 0.2, 0.4))

        # Purple intersection
        inter = poly1.intersection(poly2)
        if not inter.is_empty:
            if isinstance(inter, Polygon):
                purple_patches.append(MplPolygon(list(inter.exterior.coords), closed=True))
                purple_colors.append((0.5, 0.2, 0.8, 0.5))
            elif isinstance(inter, (MultiPolygon, GeometryCollection)):
                for geom in inter.geoms:
                    if isinstance(geom, Polygon):
                        purple_patches.append(MplPolygon(list(geom.exterior.coords), closed=True))
                        purple_colors.append((0.5, 0.2, 0.8, 0.5))

    # Combine in global draw order: blue → red → purple
    patches = blue_patches + red_patches + purple_patches
    colors = blue_colors + red_colors + purple_colors
    if not patches:
        print("No matches or nothing to draw.")
        return

    p = PatchCollection(patches, facecolors=colors, edgecolors='black', linewidths=1)
    ax.add_collection(p)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def compute_average_iou(set1, set2, matches):
    ious = []
    for i1, i2 in matches:
        if i1 == -1 or i2 == -1: # Check for case where ROI is unmatched
            # ious.append(0) # punish unmatched rois
            continue # ignore unmatched rois
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

def apply_transformations(polygon, dx=0, dy=0, rotation_deg=0, rotation_origin=None):
    coords = np.array(polygon.exterior.coords[:-1])  # exclude duplicate last point

    # Set rotation origin
    if rotation_origin is None:
        origin = coords.mean(axis=0)  # fallback to centroid
    else:
        origin = np.array(rotation_origin)

    # Rotate around the specified origin
    theta = np.radians(rotation_deg)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    coords_centered = coords - origin
    rotated = coords_centered @ rotation_matrix.T + origin

    # Translate the rotated polygon
    rotated += np.array([dx, dy])

    return Polygon(rotated)


def zoom_polygons_to_window(polygons, window_center, base_width, base_height, xfact, yfact):
    x0, y0 = window_center

    window_width = base_width / xfact
    window_height = base_height / yfact


    # Move window_center to origin
    translated = [translate(poly, xoff=-x0, yoff=-y0) for poly in polygons]

    scaled = [scale(poly, xfact=xfact, yfact=yfact, origin=(0, 0)) for poly in translated]

    # Shift result into view at (0,0) — move lower-left corner to (0,0)
    xmin = -window_width / 2 * xfact
    ymin = -window_height / 2 * yfact
    shifted = [translate(poly, xoff=-xmin, yoff=-ymin) for poly in scaled]

    return shifted

def unzoom_polygons_from_window(polygons, window_center, base_width, base_height, xfact, yfact):
    """
    Reverses zoom applied by zoom_polygons_to_window.

    Parameters:
        polygons: Transformed polygons
        window_center: (x, y) center of the original zoom window
        window_width, window_height: Size of the original data window
        base_width, base_height: Size of the target display space used for zooming

    Returns:
        List of polygons restored to original data space
    """
    #TODO hacky
    if xfact == 1.0 and yfact == 1.0:
        return polygons
    
    x0, y0 = window_center

    if xfact <= 0:
        xfact = ZOOM_INTERVAL
    if yfact <= 0:
        yfact = ZOOM_INTERVAL

    window_width = base_width / xfact
    window_height = base_height / yfact

    xmin = -window_width / 2 * xfact
    ymin = -window_height / 2 * yfact

    unshifted = [translate(poly, xoff=xmin, yoff=ymin) for poly in polygons]

    # Inverse scale
    inv_scaled = [scale(poly, xfact=1/xfact, yfact=1/yfact, origin=(0, 0)) for poly in unshifted]

    # Translate origin back to window center
    restored = [translate(poly, xoff=x0, yoff=y0) for poly in inv_scaled]

    return restored

if __name__ == "__main__":
    main()