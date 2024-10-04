import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os

# Configuration
RUN_FOLDER = './validation_runs'
DATA_FOLDER = 'drg_subset_2'
CSV_FILENAME = 'real_data_filtered_VOLUMES.csv'  # Ground truth CSV file path 
SUBSET_RATIO = 0.1  # Fraction of points to sample for visualization (e.g., 0.01 = 1%)

def load_and_sample_data(filename, subset_ratio):
    """Load the CSV data and sample a subset of points for visualization."""
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # Display the first few rows to check data
    print("Loaded Data Preview:")
    print(df.head())

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

def visualize_points(df, colors):
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
    ax.legend(labels=colors[0])

    # Set labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(0,1024)
    ax.set_ylim3d(0,1024)
    ax.set_zlim3d(0,1024)
    ax.set_title('3D Scatter Plot of Ellipsoid Points by Volume ID')
    plt.show()

def main():
    random.seed(13)
    # Set up argument parser
    parser = argparse.ArgumentParser(description="View a single set of volumes.")
    
    # Add arguments for file paths
    parser.add_argument('--run_folder', default=RUN_FOLDER, help='Directory for validation runs (default: %(default)s)')
    parser.add_argument('--data_folder', default=DATA_FOLDER, help='Data folder within run folder (default: %(default)s)')
    parser.add_argument('--csv_filename', default=CSV_FILENAME, help='Ground truth CSV file (default: %(default)s)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use the parsed arguments or fallback to the defaults if no arguments are provided
    run_folder = args.run_folder or RUN_FOLDER
    data_folder = args.data_folder or DATA_FOLDER
    csv_filename = args.csv_filename or CSV_FILENAME
    csv_path = os.path.join(run_folder, data_folder, csv_filename)
    # Load and sample the data
    sampled_df = load_and_sample_data(csv_path, SUBSET_RATIO)

    # Assign colors based on volume IDs
    colors = assign_colors_by_volume_id(sampled_df)

    # Visualize the points with assigned colors
    visualize_points(sampled_df, colors)

if __name__ == "__main__":
    main()
