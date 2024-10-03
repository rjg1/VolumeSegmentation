import pandas as pd
from scipy.spatial import ConvexHull

# Takes output of a manual segmentation using the GUI, and groups XY rois in the same volume and in the same z-plane together
# Removes all XY ROIs not in a volume (treated as noise)
def process_csv(input_csv_path, output_csv_path_1, output_csv_path_2):
    # Load the CSV file
    df = pd.read_csv(input_csv_path)

    # Drop rows where VOLUME_ID == -1
    df_filtered = df[df['VOLUME_ID'] != -1]

    # Filter out any IDs if looking for a subset of data
    # filtered_ids = [1,2,3,4,5,6,7,8,9,10]
    filtered_ids = []
    for id in filtered_ids:
        df_filtered = df_filtered[df_filtered['VOLUME_ID'] != id]


    # Group XY ROIs in the same volume of equal z
    df_filtered = group_rois(df_filtered)

    # Save the first CSV with the filtered data (including VOLUME_ID)
    df_filtered.to_csv(output_csv_path_1, index=False)

    # Drop the VOLUME_ID column from the filtered data
    df_filtered_no_volume = df_filtered.drop(columns=['VOLUME_ID'])

    # Save the second CSV with VOLUME_ID dropped
    df_filtered_no_volume.to_csv(output_csv_path_2, index=False)

    print(f"File 1 saved: {output_csv_path_1}")
    print(f"File 2 saved: {output_csv_path_2}")

def group_rois(data):
    # Group by VOLUME_ID and z level
    grouped = data.groupby(['VOLUME_ID', 'z'])
    
    new_rows = []
    
    for (volume_id, z_level), group in grouped:
        # Find unique ROI_IDs for this group
        roi_ids = group['ROI_ID'].unique()
        
        if len(roi_ids) > 1:
            # Extract x, y points
            points = group[['x', 'y']].values
            
            # Create a convex hull from these points
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # Get the lowest ROI_ID from the existing ROIs
            new_roi_id = min(roi_ids)
            
            # Create new rows for the convex hull points, keeping the same VOLUME_ID and z level
            for x, y in hull_points:
                new_rows.append({'x': x, 'y': y, 'z': z_level, 'ROI_ID': new_roi_id, 'VOLUME_ID': volume_id})
            
            # Remove the existing data for the ROIs on this z level
            data = data[~((data['ROI_ID'].isin(roi_ids)) & (data['z'] == z_level))]

    # Append the new convex hull rows
    new_data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True)
    
    return new_data

def main():
    input_csv = 'test.csv'       
    output_csv_1 = 'real_data_filtered_v0_VOLUMES.csv'  # First output CSV with rows dropped
    output_csv_2 = 'real_data_filtered_v0_ROIS.csv' # Second output CSV with VOLUME_ID column removed

    # Remove noise points
    process_csv(input_csv, output_csv_1, output_csv_2)



if __name__ == "__main__":
    main()