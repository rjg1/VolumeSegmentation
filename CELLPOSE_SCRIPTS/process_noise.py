import pandas as pd
from scipy.spatial import ConvexHull, distance
import alphashape
from shapely import LineString
import argparse
import os
import json
import numpy as np

NUM_INTERPOLATION_POINTS = 75 

# Takes output of a manual segmentation using the GUI, and groups XY rois in the same volume and in the same z-plane together
# Removes all XY ROIs not in a volume (treated as noise)
def process_csv(input_csv_path, output_csv_path_1 = None, output_csv_path_2 = None, group = True, drop_noise = True, filtered_ids = []):
    # Load the CSV file
    df = pd.read_csv(input_csv_path)

    # Drop rows where VOLUME_ID == -1
    if drop_noise:
        df = df[df['VOLUME_ID'] != -1]

    # Filter out any IDs if looking for a subset of data
    for id in filtered_ids:
        df = df[df['VOLUME_ID'] != id]

    if group:
        # Group XY ROIs in the same volume of equal z
        df = group_rois(df)

    # Save the first CSV with the filtered data (including VOLUME_ID)
    if output_csv_path_1:
        df.to_csv(output_csv_path_1, index=False)
        print(f"_VOLUMES exported to: {output_csv_path_1}")
    if output_csv_path_2:
        # Drop the VOLUME_ID column from the filtered data
        df_filtered_no_volume = df.drop(columns=['VOLUME_ID'])

        # Save the second CSV with VOLUME_ID dropped
        df_filtered_no_volume.to_csv(output_csv_path_2, index=False)
        print(f"_ROIS exported to: {output_csv_path_2}")

def group_rois(data):
    # Group by VOLUME_ID and z level
    grouped = data.groupby(['VOLUME_ID', 'z'])
    
    new_rows = []
    
    for (volume_id, z_level), group in grouped:
        if volume_id == -1:
            continue

        # Find unique ROI_IDs for this group
        roi_ids = group['ROI_ID'].unique()
        
        if len(roi_ids) > 1:
            # Extract x, y points
            points = group[['x', 'y']].values
            
            # If more than 2 points, generate an alpha shape
            if len(points) > 2:
                alpha_shape = alphashape.alphashape(points, 0.05) 
                
                # Extract boundary points from alpha shape if valid
                if not alpha_shape.is_empty and alpha_shape.geom_type == 'Polygon':
                    boundary_points = np.array(alpha_shape.exterior.coords)
                    
                    # Interpolate boundary to increase density
                    boundary_points = interpolate_boundary(boundary_points, num_interpolation_points=NUM_INTERPOLATION_POINTS)
                else:
                    boundary_points = points  # Fallback to the original points if alpha shape fails
            else:
                boundary_points = points  # Use original points if not enough for alpha shape
            
            # Snap boundary points to nearest original points
            snapped_boundary_points = snap_to_nearest_points(boundary_points, points)
            
            # Get the lowest ROI_ID from the existing ROIs
            new_roi_id = min(roi_ids)
            
            # Create new rows for the snapped boundary points
            for x, y in snapped_boundary_points:
                new_rows.append({'x': x, 'y': y, 'z': z_level, 'ROI_ID': new_roi_id, 'VOLUME_ID': volume_id})
            
            # Remove the existing data for the ROIs on this z level
            data = data[~((data['ROI_ID'].isin(roi_ids)) & (data['z'] == z_level))]

    # Append the new alpha shape rows
    new_data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True)
    
    return new_data

def main():
    # Load parameters if passed
    parser = argparse.ArgumentParser(description="Set parameters for noise process script.")
    parser.add_argument('--perform_grouping', help='Group XY ROIs in same vol on same z-level')
    parser.add_argument('--perform_reduction', help='Remove ROIs without a volume')
    parser.add_argument('--temp_reduction', help='Reduction not operating on main dataset')
    parser.add_argument('--scenarios_path', help = "Path to scenarios json file")
    parser.add_argument('--active_scenario', help = "Name of the currently active scenario")
    parser.add_argument('--roi_out_path', help='Path to export ROIS csv')
    parser.add_argument('--vol_out_path', help='Path to export VOLUMES csv')
    # Parse command-line arguments
    input_args = parser.parse_args()
    scenarios = None
    if input_args.scenarios_path:
        with open(input_args.scenarios_path, 'r') as file:
            scenarios = json.load(file)
    if scenarios:
        print("Beginning process script")
        # Use args parsed to decide action
        group = input_args.perform_grouping.lower() in ["true", "1", "yes"]
        # Reduction modifies input file
        reduce = input_args.perform_reduction.lower() in ["true", "1", "yes"]
        temp_reduction = input_args.temp_reduction.lower() in ["true", "1", "yes"]
        active_scenario = input_args.active_scenario
        has_validation = scenarios[active_scenario]["HAS_VALIDATION"]
        if not has_validation: # Validation dataset required for merging or reduction
            print("Please create a validation dataset to perform merging or validation on")
            return
        input_csv = os.path.join(scenarios[active_scenario]["V_DATA_FOLDER"], scenarios[active_scenario]["V_GT_CSV"])
        # Process the dataset
        process_csv(input_csv, output_csv_path_1=input_args.vol_out_path, 
                    output_csv_path_2=input_args.roi_out_path, group=group, drop_noise=reduce)
        # If main validation dataset is modified, update its status as reduced
        if reduce and not temp_reduction:
            scenarios[active_scenario]["HAS_REDUCED_DATASET"] = True
            # Export result
            with open(input_args.scenarios_path, 'w') as file:
                json.dump(scenarios, file, indent=4)
        print("Process script finished")
    else:
        input_csv = 'test.csv'       
        output_csv_1 = 'real_data_filtered_v0_VOLUMES.csv'  # First output CSV with rows dropped
        output_csv_2 = 'real_data_filtered_v0_ROIS.csv' # Second output CSV with VOLUME_ID column removed

        # Remove noise points
        process_csv(input_csv, output_csv_1, output_csv_2)


# Helper function to snap points to nearest cluster points
def snap_to_nearest_points(boundary_points, original_points):
    snapped_boundary = []
    for point in boundary_points:
        # Compute the distance between the boundary point and all original points
        dists = distance.cdist([point], original_points, 'euclidean')
        # Find the nearest original point
        nearest_point = original_points[dists.argmin()]
        snapped_boundary.append(nearest_point)
    return np.array(snapped_boundary)

# Helper function to interpolate along the alpha shape boundary
def interpolate_boundary(boundary_points, num_interpolation_points=10):
    line = LineString(boundary_points)
    interpolated_points = []
    for i in range(num_interpolation_points):
        interpolated_points.append(line.interpolate(i / (num_interpolation_points - 1), normalized=True).coords[0])
    return np.array(interpolated_points)

if __name__ == "__main__":
    main()