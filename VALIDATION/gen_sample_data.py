import pyvista as pv
import numpy as np
import pandas as pd
import csv
import random

# Constants
X_BOUND_MIN, X_BOUND_MAX = 0, 1024
Y_BOUND_MIN, Y_BOUND_MAX = 0, 1024
Z_BOUND_MIN, Z_BOUND_MAX = 0, 149
NUM_ELLIPSOIDS = 600
RADIUS_RANGE = ((10, 25), (10, 25), (5, 35))  # Define ROI radius ranges
RAND_SEED = 10
NUM_Z_SLICES = 150

# Set random seed for reproducibility
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)

# Initialize 3D space with accessible points
ellipsoid_data = {}  # Dictionary to keep track of ellipsoid centers and radii


def create_ellipsoid(center, radius):
    """Creates an ellipsoid with given center and radii."""
    ellipsoid = pv.ParametricEllipsoid(radius[0], radius[1], radius[2], center = center)  # Create the ellipsoid object
    return ellipsoid


def mark_inaccessible_space(temp_space, proposed_radius):
    """Marks space as inaccessible based on boundaries and existing ellipsoids."""
    prx, pry, prz = proposed_radius

    # Convert to integer (rounded up) for array indexing
    intrx, intry, intrz = int(np.ceil(prx)), int(np.ceil(pry)), int(np.ceil(prz))

    # Mark all points that are out of bounds for the proposed ellipsoid size
    temp_space[:intrx, :, :] = False  # All x points too close to the minimum x bound
    temp_space[X_BOUND_MAX - intrx:, :, :] = False  # All x points too close to the maximum x bound
    temp_space[:, :intry, :] = False  # All y points too close to the minimum y bound
    temp_space[:, Y_BOUND_MAX - intry:, :] = False  # All y points too close to the maximum y bound
    temp_space[:, :, :intrz] = False  # All z points too close to the minimum z bound
    temp_space[:, :, Z_BOUND_MAX - intrz:] = False  # All z points too close to the maximum z bound

    # Iterate over existing ellipsoids and mark space as unavailable
    for data in ellipsoid_data.values():
        center = data['center']
        radius = data['radius']
        cx, cy, cz = center
        crx, cry, crz = radius

        # Calculate boundaries considering both existing and proposed ellipsoid radii
        x_min = max(0, int(cx - crx - prx))
        x_max = min(X_BOUND_MAX, int(cx + crx + prx))
        y_min = max(0, int(cy - cry - pry))
        y_max = min(Y_BOUND_MAX, int(cy + cry + pry))
        z_min = max(0, int(cz - crz - prz))
        z_max = min(Z_BOUND_MAX, int(cz + crz + prz))

        # Mark the space as unavailable
        temp_space[x_min:x_max, y_min:y_max, z_min:z_max] = False

    return temp_space


def calculate_available_positions(radius):
    """Calculates available positions for the proposed ellipsoid based on existing ellipsoids."""
    # Initialise space of all True values in each dimension
    temp_space = np.ones((X_BOUND_MAX, Y_BOUND_MAX, Z_BOUND_MAX), dtype=bool)
    if len(ellipsoid_data) > 0:
        # Mark areas as unavailable based on boundaries and the existing ellipsoids
        temp_space = mark_inaccessible_space(temp_space, radius)

    # Return available positions after considering existing ellipsoids and boundaries
    available_positions = np.argwhere(temp_space)

    print(f"Calculated {len(available_positions)} available positions for ellipsoid with radii {radius}")
    return available_positions

def generate_ellipsoids(num_ellipsoids):
    """Generates ellipsoids in 3D space with progress updates."""
    ellipsoids = []
    max_sample_size = 5000  # Limit the number of points sampled for weighting
    global_failures = 0 # Total number of placements failed
    for i in range(num_ellipsoids):
        print(f"\nGenerating ellipsoid {i + 1} of {num_ellipsoids}")
        attempts = 0
        if global_failures > 3: # If 3 attempts failed for 3 generations, terminate generation early
            print(f"Generation failure! Generated {len(ellipsoids)} / {num_ellipsoids} ellipsoids.")
            return ellipsoids
        while attempts < 3:
            # Choose random radii within specified ranges
            rx = np.random.uniform(*RADIUS_RANGE[0])
            ry = np.random.uniform(*RADIUS_RANGE[1])
            rz = np.random.uniform(*RADIUS_RANGE[2])
            print(f"Attempt {attempts + 1}: Selected radii (rx, ry, rz) = ({rx:.2f}, {ry:.2f}, {rz:.2f})")

            # Calculate available positions after determining radii
            available_positions = calculate_available_positions((rx, ry, rz))

            if not len(available_positions):
                print("No available positions for current ellipsoid dimensions.")
                attempts += 1
                continue

            # Sample a subset of available positions to reduce computational load
            sample_size = min(max_sample_size, len(available_positions))
            sampled_points = available_positions[np.random.choice(len(available_positions), size=sample_size, replace=False)]
            print(f"Sampled {sample_size} points for weighting")

            # Weight points to favor proximity to existing ellipsoids
            print(f"Calculating weights for ellipsoid {i + 1}")
            weights = np.zeros(len(sampled_points))

            # Check if there are any existing ellipsoids
            if len(ellipsoid_data) == 0:
                # If no ellipsoids exist, set equal weights for all sampled points
                weights[:] = 1 / len(weights)
            else:
                # Calculate the distance of each sampled point to the nearest ellipsoid center
                for idx, point in enumerate(sampled_points):
                    # Calculate minimum distance to any existing ellipsoid center
                    min_distance = np.min([np.linalg.norm(point - np.array(data['center'])) for data in ellipsoid_data.values()])
                    # Weight is higher for closer distances (invert if necessary to avoid division by zero)
                    weights[idx] = 1 / (min_distance + 1e-3)  # Higher values for closer distances

                # Normalize weights to form a probability distribution
                weights = weights / weights.sum()

            # Choose a random point id based on proximity to existing volumes
            chosen_point = sampled_points[np.random.choice(len(weights), p=weights)]

            center = chosen_point.tolist()

            ellipsoid = create_ellipsoid(center, (rx, ry, rz))
            ellipsoids.append((ellipsoid, center))
            ellipsoid_data[i] = {"center": center, "radius": (rx, ry, rz)}
            print(f"Successfully created ellipsoid. Idx: {i}.")
            break
        if attempts == 3:
            print(f"Failed to place ellipsoid {i + 1} after 3 attempts. Moving to the next ellipsoid.")
            global_failures += 1

    print(f"Successfully generated {len(ellipsoids)} ellipsoids.")
    return ellipsoids


def export_rois_to_csv(ellipsoids, filename="roi_validation_data.csv", volumes = False):
    """Exports the ROI data of ellipsoids to CSV."""
    print("Beginning CSV Export Process...")
    # Slice each ellipsoid in each z-plane
    xy_slice_meshes = {}
    z_planes = np.linspace(Z_BOUND_MIN, Z_BOUND_MAX, num=NUM_Z_SLICES)
    for idx, ellipsoid_tuple in enumerate(ellipsoids):
        ellipsoid, centre = ellipsoid_tuple
        # Determine the z-range of this ellipsoid
        _, _, rz = ellipsoid_data[idx]['radius']
        rz_int = int(np.ceil(rz))
        _, _, cz = centre
       
       # Slice over z-range of this ellipsoid
        print(f"Slicing volume idx: {idx} / {len(ellipsoids) - 1}")
        for z in range(max(0, cz - rz_int), min(Z_BOUND_MAX, cz + rz_int)):
            if z not in xy_slice_meshes:
                xy_slice_meshes[z] = {}
            slice = ellipsoid.slice(origin=(0, 0, z), normal=(0, 0, 1)) # PyVista slice operation
            if slice.n_points > 0:
                # Extract and store all xy points for this z-slice of an ellipsoid
                xy_slice_meshes[z][idx] = [tuple(point) for point in slice.points]

    print("Slicing of all volumes complete!")
    
    # Initialise output list
    out_data = []

    for z, ellipsoid_slices in xy_slice_meshes.items():
        print(f"Processing ROIs on plane {z}.")
        roi_id = 0 # Arbitrary ID for an ellipsoid slice on this z-plane
        for ellipsoid_id, slice_points in ellipsoid_slices.items():
            for x,y,z in slice_points:
                out_data.append([x, y, z, roi_id, ellipsoid_id])
            roi_id += 1

    # Create dfs for each type of output file
    df_volumes = pd.DataFrame(out_data, columns=['x', 'y', 'z', 'ROI_ID', 'VOLUME_ID'])
    df_rois = df_volumes.drop(columns=['VOLUME_ID'])

    if volumes:
        df_volumes.to_csv(filename + '_VOLUMES' + '.csv', index=False)
    df_rois.to_csv(filename + '_ROIS' + '.csv', index=False)
    print(f"Exported ellipsoid data to {filename}")

def visualize_ellipsoids(ellipsoids):
    """Visualizes ellipsoids in a 3D PyVista plot."""
    print("Visualizing ellipsoids...")
    plotter = pv.Plotter()
    plotter.add_axes()

    # Set bounds of the visualization to match the actual space
    plotter.set_scale(xscale=1, yscale=1, zscale=1)

    # Add each ellipsoid to the plot and ensure transformations are applied correctly
    for idx, (ellipsoid, center) in enumerate(ellipsoids):
        plotter.add_mesh(ellipsoid, color='lightblue', smooth_shading=True) 

    # Show bounding box for reference
    plotter.show_bounds(grid="back", location="outer", all_edges=True, bounds=[X_BOUND_MIN,X_BOUND_MAX,Y_BOUND_MIN,Y_BOUND_MAX,Z_BOUND_MIN,Z_BOUND_MAX])
    plotter.show()
    print("Visualization complete.")


def main():
    print("Starting ellipsoid generation process...")
    ellipsoids = generate_ellipsoids(NUM_ELLIPSOIDS)
    # Export data used for validation
    export_rois_to_csv(ellipsoids, filename="roi_validation", volumes=True)
    # View 3D plot of generated volumes
    visualize_ellipsoids(ellipsoids)
    print("Ellipsoid generation process completed.")
    print(ellipsoid_data)


if __name__ == "__main__":
    main()
