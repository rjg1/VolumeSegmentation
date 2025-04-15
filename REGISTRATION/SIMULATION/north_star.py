import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.affinity import scale, translate, rotate
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linear_sum_assignment

NUM_ELLIPSES = 10
INTENSITY_SELECTION_THRESHOLD = 0.5 # Intensity required for an ROI to be considered as an anchor point
INTENSITY_DELTA_PERC = 0.2 # Intensity delta percent between two ROIs for the match to be considered
ANGLE_DELTA_DEG = 10 # Angle to rotate between tests
ANGLE_ROTATE_MAX = 50 # Max angle to rotate ROIs when comparing
Z_RANGE = 10 # +- z_gap to search
Z_GUESS = 1 # Expected z-level to begin search
XZ_ANGLE_MAX = 30 # Max angle difference between two centroid-difference vectors in the XZ plane
# Base data dimensions
NUM_POINTS = 200
RADIUS_X = 4
RADIUS_Y = 2
# Transformations for second set of data
TRANSFORM_ROTATION = 30
X_SHIFT = 10
Y_SHIFT = -7
ZOOM = 3 # 3x zoom
X_SUBWINDOW = 40 # x coordinate of zoomed window centre point
Y_SUBWINDOW = 60 # y coodrdinate of zoomed window centre point
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
        transformed = apply_transformations(ellipse, dx=0, dy=0, rotation_deg=TRANSFORM_ROTATION, rotation_origin=(0,0))
        transformed_ellipses.append(transformed)
        intensity_modifier = random.uniform(90,100) # Slightly modify intensity
        transformed_intensities.append(ellipse_intensities[i] * (intensity_modifier / 100))

    # Make dict of {z: {id: [(x,y)]}} for each z-plane 
    z_planes = {0:{}}
    for id, poly in enumerate(ellipses):
        if not z_planes[0].get(id, None):
            z_planes[0][id] = {}
            z_planes[0][id]['coords'] = []
        z_planes[0][id]['intensity'] = ellipse_intensities[id]
        for coord in list(poly.exterior.coords):
            z_planes[0][id]['coords'].append(coord)
    z_planes = dict(sorted(z_planes.items())) # Sort in ascending z, allowing volumes to be built from the ground up

    # # Plot initial setup of points
    # # Normalize intensities for colormap
    # norm = mcolors.Normalize(vmin=0, vmax=1)
    # cmap = cm.Greys_r

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot only original ellipses on plane z=0
    # for ellipse, intensity in zip(ellipses, ellipse_intensities):
    #     ox, oy = ellipse.exterior.xy
    #     oz = np.zeros_like(ox)  # z=0 plane
    #     verts = [list(zip(ox, oy, oz))]  # List of 3D vertices for the polygon
    #     face_color = cmap(norm(intensity))
    #     poly = Poly3DCollection(verts, facecolors=face_color, edgecolors='black', alpha=0.6)
    #     ax.add_collection3d(poly)

    # ax.set_xlim(0, base_width)
    # ax.set_ylim(0, base_height)
    # ax.set_zlim(0, 1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title("Generated ellipses on a z-plane")
    # plt.show()

    # Apply zoom to generated dataset
    zoomed_polys = zoom_polygons_to_window(
        transformed_ellipses,
        window_center=window_center,
        base_width=X_MAX,
        base_height=Y_MAX,
        xfact=ZOOM,
        yfact=ZOOM
    )

    # Extract polygons within subwindow (re-polygonizing edge polygons)
    extracted_polys, extracted_intensities = extract_polygons(zoomed_polys, transformed_intensities, base_width, base_height)

    # fig, ax = plt.subplots()
    # for poly in extracted_polys:
    #     x, y = poly.exterior.xy
    #     ax.fill(x, y, alpha=0.5, edgecolor='black')
    # ax.set_aspect('equal')
    # ax.set_xlim(0, base_width)
    # ax.set_ylim(0, base_height)
    # ax.set_title("Zoomed Polygon View")
    # plt.grid(True)
    # plt.show()

    # Modify new plane data to fit initial dataset
    new_plane = {}
    for id, poly in enumerate(zoomed_polys):
        if not new_plane.get(id, None):
            new_plane[id] = {}
            new_plane[id]['coords'] = []
        new_plane[id]['intensity'] = ellipse_intensities[id]
        for coord in list(poly.exterior.coords):
            new_plane[id]['coords'].append(coord)

    # Align polygons
    align_polygons(z_planes, new_plane, z_guess=1)


def align_polygons(z_planes, new_plane, z_guess = None):
    # Initialize return variables
    max_avg_IoU = 0 # Best percent match
    best_pair = None
    best_angle = None
    best_z = None
    yaw = None
    pitch = None
    best_matches = []

    # Normalize both intensity lists - TODO may need outlier removal in real data
    # Normalize each plane in the passed in dataset
    for z in z_planes:
        id_list = list(z_planes[z].keys())
        intensity_list = [z_planes[z][id]["intensity"] for id in id_list]
        normalized = normalize(intensity_list)

        for id, norm_val in zip(id_list, normalized):
            z_planes[z][id]["intensity"] = norm_val
        
    # Normalize new plane intensities
    id_list = list(new_plane.keys())
    intensity_list = [new_plane[id]["intensity"] for id in id_list]
    normalized = normalize(intensity_list)
    
    for id, norm_val in zip(id_list, normalized):
        new_plane[id]["intensity"] = norm_val

    # Determine matching candidates as ROIs with intensity above the designated threshold TODO Optimize for z-restriction/z-guess
    potential_anchors = [(z, id) for z in z_planes for id in z_planes[z] if z_planes[z][id]["intensity"] > INTENSITY_SELECTION_THRESHOLD]
    # For each ROI candidate, determine viable ROIs in other set it can be matched with (within threshold)
    # Make dictionary to hold matching Ids
    anchor_matches = {(z,id):[] for z,id in potential_anchors}
    for anchor in anchor_matches:
        z, id =  anchor
        for id2 in new_plane:
            intensity1 = z_planes[z][id]["intensity"]
            intensity2 = new_plane[id2]["intensity"]
            if abs(intensity1 - intensity2) <= (INTENSITY_DELTA_PERC * intensity1):
                anchor_matches[(z,id)].append(id2)
    # For each viable ROI anchor match, determine pairings of all ROIs based off distance
    anchor_pairings = {(z,id1,id2) : {} for z,id1 in anchor_matches for id2 in anchor_matches[(z,id1)]}

    # For new plane, find its anchor ROIs, alignment ROIs, their angles and magnitudes TODO check plane class with 2d data

    # Find all possible combinations of 2 alignment ROIs with each anchor ROI
    #   For each, make a plane (plane class)
    #   Make sure angles between anchor/alignment ROIs isn't too steep in XZ
    #   Also restrict the search range in Z to a +- threshold
    #   Make planes and find points sufficiently close to also project onto planes
    #   Exclude repeat planes from being made when points are projected (check for plane equivalence - some kind of fuzzy check? two planes might be very close in this region)

    # For the new plane and each 3D plane in the plane list, compare the match in terms of angle and magnitude, noting the best performers

    # All performers over a certain threshold will move on to the IoU check stage

    # Generate a set of transformations for each point in the new 2D plane to project them into the 3D plane, such that the angular offsets are also preserved, and magnitudes are
    # scaled relatively 
    
    # Once all points are transformed, find the IoU of all points projected onto the 3D plane

    # Do this for all of the aforementioned best performers, and take the maximum IoU.

    print(anchor_pairings)

    return

    # Return best pairing and max IoU
    return max_avg_IoU, best_pair, best_angle, best_matches, anchor_pairings[best_pair][best_zoom]['aligned_set2'], best_zoom


def compute_centroid(coords):
    coords = np.array(coords)
    return np.mean(coords, axis=0)


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