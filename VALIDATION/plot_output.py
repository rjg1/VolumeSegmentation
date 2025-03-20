import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os

# Configuration
DATA_FOLDER = './validation_runs/test_1'
CSV_FILENAME = 'real_data_filtered_1_VOLUMES.csv'  # Ground truth CSV file path 
PLOT_TITLE = f'Plot of Ground Truth data for dataset: {DATA_FOLDER}'
MAX_POINTS = 10000 # Maximum points to sample
AX_LIMIT_MAX = 1024

def load_and_sample_data(filename, sample_point_max):
    """Load the CSV data and sample a subset of points for visualization."""
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    df = df[df['VOLUME_ID'] != -1] # Filter noise points
    # Test filter good parts of DRG
    # df = df[df['z'] <= 80]
    # df = df[df['z'] >= 30]

    # Display the first few rows to check data
    # print("Loaded Data Preview:")
    # print(df.head())
    subset_ratio = min(1, sample_point_max/len(df))
    # Check if volume IDs are present
    if 'VOLUME_ID' not in df.columns:
        raise ValueError("CSV does not contain VOLUME_ID column. Ensure export was done with volumes enabled.")

    # Sample a subset of the data for visualization
    sampled_df = df.sample(frac=subset_ratio, random_state=42)  # Randomly sample a subset for plotting
    print(f"Sampled {len(sampled_df)} points out of {len(df)} total points.")

    return sampled_df

def assign_colors_by_volume_id(df):
    """Assign colors to points based on their VOLUME_ID."""
    # Extract unique volume IDs and assign a color for each ID
    unique_volume_ids = df['VOLUME_ID'].unique()
    num_volumes = len(unique_volume_ids)
    print(f"{num_volumes} volumes found")

    # Generate random colors for each volume ID
    color_map = {vol_id: [random.random(), random.random(), random.random()] for vol_id in unique_volume_ids}

    # Map colors to each row in the DataFrame
    colors = df['VOLUME_ID'].map(color_map)

    # Convert to a numpy array for Matplotlib
    colors_array = np.array(colors.tolist())

    return colors_array

def visualize_points(df, colors, title):
    """Visualize the sampled points in 3D space with colors assigned by volume ID."""
    # Create a Matplotlib 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract point coordinates from the DataFrame
    x = df['x']
    y = df['y']
    z = df['z']

    # Plot the points with the assigned colors
    ax.scatter(x, y, z, c=colors, s=1)
    max_val = df[['x', 'y', 'z']].max().max()
    min_val = df[['x', 'y', 'z']].min().min()
    # Set labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(min_val,max_val)
    ax.set_ylim3d(min_val,max_val)
    ax.set_zlim3d(min_val,max_val)
    ax.set_zlim3d(0,80)
    ax.set_title(title)
    ax.grid(False)
    plt.show()

def main():
    random.seed(13)
    # Set up argument parser
    parser = argparse.ArgumentParser(description="View a single set of volumes.")
    
    # Add arguments for file paths
    parser.add_argument('--data_folder', default=DATA_FOLDER, help='Data folder within run folder (default: %(default)s)')
    parser.add_argument('--csv_filename', default=CSV_FILENAME, help='Ground truth CSV file (default: %(default)s)')
    parser.add_argument('--plot_title', default=PLOT_TITLE, help='Title for plot (default: %(default)s)')
    parser.add_argument('--sample_point_max', default=MAX_POINTS, help='Max limit number of points to sample (default: %(default)s)')

    # Parse arguments
    args = parser.parse_args()
    
    # Use the parsed arguments or fallback to the defaults if no arguments are provided
    data_folder = args.data_folder or DATA_FOLDER
    print(f"data folder: {data_folder}")
    csv_filename = args.csv_filename or CSV_FILENAME
    plot_title = args.plot_title or PLOT_TITLE
    csv_path = os.path.join(data_folder, csv_filename)
    # Load and sample the data
    sampled_df = load_and_sample_data(csv_path, int(args.sample_point_max))

    # Assign colors based on volume IDs
    colors = assign_colors_by_volume_id(sampled_df)

    # Visualize the points with assigned colors
    visualize_points(sampled_df, colors, plot_title)

if __name__ == "__main__":
    main()
