import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os

# Configuration
# CSV_FILENAME = "./validation_runs/simple_points/algo_seg.csv"   
# CSV_FILENAME = "./validation_runs/complex_points/xyz_complex_points_algo_VOLUMES.csv"  
# CSV_FILENAME = "./validation_runs/complex_points/xyz_complex_points_VOLUMES.csv"  
# CSV_FILENAME = './validation_runs/generated_mock_cell/roi_validation_VOLUMES.csv'
# CSV_FILENAME = './validation_runs/generated_mock_cell/roi_validation_algo_VOLUMES.csv'
RUN_FOLDER = './validation_runs'
DATA_FOLDER = 'drg_subset_2'
CSV_FILENAME = 'real_data_filtered_VOLUMES.csv'  # Ground truth CSV file path
# CSV_FILENAME = "./validation_runs/simple_points/xyz_points_volumes.csv"   
SUBSET_RATIO = 1  # Fraction of points to sample for visualization (e.g., 0.01 = 1%)

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
    csv_path = os.path.join(RUN_FOLDER, DATA_FOLDER, CSV_FILENAME)
    # Load and sample the data
    sampled_df = load_and_sample_data(csv_path, SUBSET_RATIO)

    # Assign colors based on volume IDs
    colors = assign_colors_by_volume_id(sampled_df)

    # Visualize the points with assigned colors
    visualize_points(sampled_df, colors)

if __name__ == "__main__":
    main()
