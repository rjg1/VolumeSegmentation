import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import os

# Configuration
DATA_FOLDER = './validation_runs/drg_subset_1_r'
GT_CSV_FILENAME = 'real_data_filtered_v0_VOLUMES.csv'  # Ground truth CSV file path
ALGO_CSV_FILENAME = 'real_data_filtered_algo_VOLUMES.csv'  # Algorithmic CSV file path
MAPPING_FILENAME = 'mapping.csv'
MAX_POINTS = 8000 # Maximum points to sample
DRAW_LINES = False

def load_and_sample_data(filename):
    """Load the CSV data and sample a subset of points for visualization."""
    df = pd.read_csv(filename)
    df = df[df['VOLUME_ID'] != -1] # Filter noise points

    # Display the first few rows to check data
    # print(f"Loaded {filename} Preview:")
    # print(df.head())
    subset_ratio = min(1, MAX_POINTS/len(df))
    if 'VOLUME_ID' not in df.columns:
        raise ValueError("CSV does not contain VOLUME_ID column. Ensure export was done with volumes enabled.")


    # Sample a subset of the data for visualization
    sampled_data = df.groupby('VOLUME_ID').apply(lambda x: x.sample(frac=subset_ratio, random_state=42))
    sampled_data.reset_index(drop=True, inplace=True)

    # print(f"Sampled {len(sampled_data)} points out of {len(df)} total points from {filename}.")
    return sampled_data

def assign_colors_by_volume_id(gt_df, algo_df, volume_mapping):
    """Assign matching colors based on the volume mapping."""
    # Unique ground truth and algorithmic volume IDs
    gt_unique_volumes = gt_df['VOLUME_ID'].unique()
    algo_unique_volumes = algo_df['VOLUME_ID'].unique()

    # Create a color map for ground truth volumes
    num_gt_volumes = len(gt_unique_volumes)
    color_map = {vol_id: [random.random(), random.random(), random.random()] for vol_id in gt_unique_volumes}

    # Add matching colors for algorithmic volumes based on the mapping
    algo_color_map = {}
    unmatched_colors = {}  # To store unique colors for unmatched volumes

    for algo_vol_id in algo_unique_volumes:
        # Find matching ground truth volume ID
        matching_gt_id = next((gt_id for gt_id, algo_id in volume_mapping.items() if algo_id == algo_vol_id), None)
        
        if matching_gt_id is not None:
            # If there is a matching ground truth ID, use its color
            algo_color_map[algo_vol_id] = color_map.get(matching_gt_id)
        else:
            # If no matching ground truth ID, assign a unique random color to this unmatched volume
            if algo_vol_id not in unmatched_colors:
                unmatched_colors[algo_vol_id] = [random.random(), random.random(), random.random()]
            algo_color_map[algo_vol_id] = unmatched_colors[algo_vol_id]

    # Assign colors to the ground truth and algorithmic datasets
    gt_colors = gt_df['VOLUME_ID'].map(color_map)
    algo_colors = algo_df['VOLUME_ID'].map(algo_color_map)

    # Convert to numpy arrays for plotting
    gt_colors_array = np.array(gt_colors.tolist())
    algo_colors_array = np.array(algo_colors.tolist())

    return gt_colors_array, algo_colors_array

def visualize_points_side_by_side(gt_df, algo_df, gt_colors, algo_colors, volume_mapping):
    """Visualize the ground truth and algorithmic points side by side in 3D, with indicators for unmatched ground truth volumes."""

    fig = plt.figure(figsize=(12, 6))

    # Ground truth plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Extract unmatched ground truth volumes
    matched_gt_volumes = set(volume_mapping.keys())
    all_gt_volumes = set(gt_df['VOLUME_ID'].unique())
    unmatched_gt_volumes = all_gt_volumes - matched_gt_volumes  # Ground truth volumes that are not matched
    
    # Plot matched ground truth volumes
    matched_gt_points = gt_df[~gt_df['VOLUME_ID'].isin(unmatched_gt_volumes)]
    ax1.scatter(matched_gt_points['x'], matched_gt_points['y'], matched_gt_points['z'], c=gt_colors[~gt_df['VOLUME_ID'].isin(unmatched_gt_volumes)], s=1, label="Matched GT Volumes")

    # Highlight unmatched ground truth volumes with a distinct color (e.g., red) and marker style (e.g., 'x')
    unmatched_gt_points = gt_df[gt_df['VOLUME_ID'].isin(unmatched_gt_volumes)]
    ax1.scatter(unmatched_gt_points['x'], unmatched_gt_points['y'], unmatched_gt_points['z'], 
                c=gt_colors[gt_df['VOLUME_ID'].isin(unmatched_gt_volumes)], s=10, label="Unmatched GT Volumes")

    # Optional: Add lines to indicate unmatched points (e.g., connecting to the origin or a specific point)
    if DRAW_LINES:
        for i, row in unmatched_gt_points.iterrows():
            ax1.plot([row['x'], 512], [row['y'], 512], [row['z'], 512], color='red', linestyle='--', linewidth=0.5) 

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Ground Truth Segmentation')
    ax1.set_xlim3d(0, 1024)
    ax1.set_ylim3d(0, 1024)
    ax1.set_zlim3d(0, 1024)
    ax1.legend()

    # Filter algorithmic volumes to only plot matched ones
    matched_algo_volumes = set(volume_mapping.values())
    matched_algo_points = algo_df[algo_df['VOLUME_ID'].isin(matched_algo_volumes)]

    # Algorithmic segmentation plot
    ax2 = fig.add_subplot(122, projection='3d')
    # ax2.scatter(algo_df['x'], algo_df['y'], algo_df['z'], c=algo_colors, s=1)
    ax2.scatter(matched_algo_points['x'], matched_algo_points['y'], matched_algo_points['z'], c=algo_colors[algo_df['VOLUME_ID'].isin(matched_algo_volumes)], s=1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Algorithmic Segmentation')
    ax2.set_xlim3d(0, 1024)
    ax2.set_ylim3d(0, 1024)
    ax2.set_zlim3d(0, 1024)

    plt.tight_layout()
    plt.show()


def load_volume_mapping(mapping_filename):
    """
    Load the ground truth to algorithmic segmentation volume ID mapping from a CSV file.
    CSV must contain columns 'GT_ID' and 'ALGO_ID'.
    Returns a dictionary {GT_ID: ALGO_ID}.
    """
    mapping_df = pd.read_csv(mapping_filename)
    volume_mapping = dict(zip(mapping_df['GT_ID'], mapping_df['ALGO_ID']))
    return volume_mapping


def main():
    random.seed(5)
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the overlap calculation and accuracy analysis.")
    
    # Add arguments for file paths
    parser.add_argument('--data_folder', default=DATA_FOLDER, help='Data folder within run folder (default: %(default)s)')
    parser.add_argument('--gt_csv', default=GT_CSV_FILENAME, help='Ground truth CSV file (default: %(default)s)')
    parser.add_argument('--algo_csv', default=ALGO_CSV_FILENAME, help='Algorithmic CSV file (default: %(default)s)')
    parser.add_argument('--mapping_csv', default=MAPPING_FILENAME, help='Output mapping CSV file (default: %(default)s)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use the parsed arguments or fallback to the defaults if no arguments are provided
    data_folder = args.data_folder or DATA_FOLDER
    gt_csv_filename = args.gt_csv or GT_CSV_FILENAME
    algo_csv_filename = args.algo_csv or ALGO_CSV_FILENAME
    mapping_filename = args.mapping_csv or MAPPING_FILENAME

    # Construct file paths
    gt_csv_path = os.path.join(data_folder, gt_csv_filename)
    algo_csv_path = os.path.join(data_folder, algo_csv_filename)
    mapping_csv_path = os.path.join(data_folder, mapping_filename)

    # Load and sample the ground truth and algorithmic data
    gt_sampled_df = load_and_sample_data(gt_csv_path)
    algo_sampled_df = load_and_sample_data(algo_csv_path)

    # Load the volume mapping from CSV
    volume_mapping = load_volume_mapping(mapping_csv_path)

    # Assign matching colors based on volume mapping
    gt_colors, algo_colors = assign_colors_by_volume_id(gt_sampled_df, algo_sampled_df, volume_mapping)

    # Visualize the points side by side
    visualize_points_side_by_side(gt_sampled_df, algo_sampled_df, gt_colors, algo_colors, volume_mapping)


if __name__ == "__main__":
    main()
