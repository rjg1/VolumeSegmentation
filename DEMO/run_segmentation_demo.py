import subprocess
import os

# SEGMENTATION SCRIPT PARAMETERS
SEGMENTATION_DIR = '../VOLUME_SEGMENTATION/' # Relative path to segmentation script folder
SEG_IN_FILE = 'run_data/real_data_filtered_v0_ROIS.csv' # Input csv with data points of structure (x,y,z,ROI_ID). Path relative to SEGMENTATION_DIR
SEG_XZ_IO_FILE = 'xz_cache/real_data_filtered_v0_ROIS_XZ.csv' # File to import/generate of (x,y,z) points representing XZ projected points. Path relative to SEGMENTATION_DIR
# VALIDATION / VISUALISATION SCRIPT PARAMETERS
VALIDATION_DIR = '../VALIDATION/' # Relative path to validation folder
V_SUBFOLDER = 'validation_runs' # Subfolder containing all run data in /VALIDATION_DIR/
V_DATA_FOLDER = 'drg_subset_1_r' # Folder holding the data in /VALIDATION_DIR/V_SUBFOLDER/
V_GT_CSV = 'real_data_filtered_v0_VOLUMES.csv' # Filename of ground truth segmentation in /VALIDATION_DIR/V_SUBFOLDER/V_DATA_FOLDER
V_ALGO_CSV = 'real_data_filtered_algo_VOLUMES.csv' # Filename of algorithmic segmentation result in /VALIDATION_DIR/V_SUBFOLDER/V_DATA_FOLDER
V_MAPPING_CSV = 'mapping.csv' # Output mapping file to place in /VALIDATION_DIR/V_SUBFOLDER/V_DATA_FOLDER



def main():
    # Segmentation run args
    seg_script_path = os.path.join(SEGMENTATION_DIR, 'volume_segmentation.py')
    seg_in_file = os.path.join(SEGMENTATION_DIR, SEG_IN_FILE)
    seg_xz_cache_file = os.path.join(SEGMENTATION_DIR, SEG_XZ_IO_FILE)
    seg_out_file = os.path.join(VALIDATION_DIR, V_SUBFOLDER, V_DATA_FOLDER, V_ALGO_CSV)
    seg_args = [
        'python', seg_script_path,
        '--in_file', seg_in_file,
        '--xz_io_path', seg_xz_cache_file,
        '--out_file', seg_out_file
    ]

    v_run_folder = os.path.join(VALIDATION_DIR, V_SUBFOLDER)
    v_script_path = os.path.join(VALIDATION_DIR, 'v5.py')
    v_vis_script_path = os.path.join(VALIDATION_DIR, 'plot_comparison.py')
    # Argument list for validation script
    v_args = [
        'python', v_script_path,  # Python executable and script
        '--run_folder', v_run_folder,
        '--data_folder', V_DATA_FOLDER,
        '--gt_csv', V_GT_CSV,
        '--algo_csv', V_ALGO_CSV,
        '--mapping_csv', V_MAPPING_CSV
    ]

    # Argument list for visualisation script
    v_vis_args = [
        'python', v_vis_script_path,  # Python executable and script
        '--run_folder', v_run_folder,
        '--data_folder', V_DATA_FOLDER,
        '--gt_csv', V_GT_CSV,
        '--algo_csv', V_ALGO_CSV,
        '--mapping_csv', V_MAPPING_CSV
    ]

    try:
        # Run segmentation
        subprocess.run(seg_args, check=True)
        # Run validation script
        subprocess.run(v_args, check=True)
        # Run visualisation script
        subprocess.run(v_vis_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()