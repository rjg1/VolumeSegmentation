import pandas as pd

# Load the CSV files
# csv1_path = "../GUI/cell_data/ROI_coordinates_20241003_x300_800_y0_500_z1_38.csv" 
# csv2_path = "./demo_data/drg_complete/drg_complete_algo_VOLUMES.csv"

csv1_path = "../VOLUME_SEGMENTATION/run_data/real_data_filtered_v0_ROIS.csv" 
csv2_path = "../VALIDATION/validation_runs/drg_subset_1_r/real_data_filtered_algo_VOLUMES.csv"

df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# Create unique XY_ROI IDs in both CSVs (z, ROI_ID defines a unique ROI)
df1['XY_ROI_ID'] = list(zip(df1['z'], df1['ROI_ID']))
df2['XY_ROI_ID'] = list(zip(df2['z'], df2['ROI_ID']))

# Get unique XY ROIs in csv1
unique_rois_csv1 = set(df1['XY_ROI_ID'].unique())

# Check how many of those XY ROIs are in csv2
unique_rois_csv2 = set(df2['XY_ROI_ID'].unique())
rois_in_csv2 = unique_rois_csv1.intersection(unique_rois_csv2)

# Now check how many of those ROIs are in a volume (i.e., have a non-null or non-zero VOLUME_ID)
df2_with_volume = df2[df2['VOLUME_ID'].notnull() & (df2['VOLUME_ID'] != 0)]
rois_in_volume = unique_rois_csv1.intersection(set(df2_with_volume['XY_ROI_ID'].unique()))

# Display results
print(f"Number of unique XY ROIs in csv1: {len(unique_rois_csv1)}")
print(f"Number of those XY ROIs in csv2: {len(rois_in_csv2)}")
print(f"Number of those XY ROIs that are part of a volume: {len(rois_in_volume)}")
