import subprocess
import os

# VALIDATION / VISUALISATION SCRIPT PARAMETERS
VALIDATION_DIR = '../VALIDATION/' # Relative path to validation folder
V_SUBFOLDER = 'validation_runs' # Subfolder containing all run data in /VALIDATION_DIR/
V_DATA_FOLDER = 'drg_subset_1_r' # Folder holding the data in /VALIDATION_DIR/V_SUBFOLDER/
CSV_FILENAME = 'real_data_filtered_v0_VOLUMES.csv' # Filename of ground truth segmentation in /VALIDATION_DIR/V_SUBFOLDER/V_DATA_FOLDER

def main():
    v_run_folder = os.path.join(VALIDATION_DIR, V_SUBFOLDER)
    v_vis_script_path = os.path.join(VALIDATION_DIR, 'plot_output.py')
    # Argument list for visualisation script
    v_vis_args = [
        'python', v_vis_script_path,  # Python executable and script
        '--run_folder', v_run_folder,
        '--data_folder', V_DATA_FOLDER,
        '--csv_filename', CSV_FILENAME,
    ]

    try:
        subprocess.run(v_vis_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()