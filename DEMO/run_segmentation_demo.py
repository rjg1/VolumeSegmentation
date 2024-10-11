import subprocess
import os
import json
import argparse

# DEFAULT JSON VARS
SCENARIO_FILEPATH = './scenarios.json'
ACTIVE_SCENARIO = 'drg_subset_1_3'
# SEGMENTATION SCRIPT DEFAULT PARAMETERS
SEGMENTATION_DIR = '../VOLUME_SEGMENTATION/' # Relative path to segmentation script folder
SEG_IN_FILE = 'run_data/real_data_filtered_v0_ROIS.csv' # Input csv with data points of structure (x,y,z,ROI_ID). Path relative to SEGMENTATION_DIR
SEG_XZ_IO_FILE = 'xz_cache/real_data_filtered_v0_ROIS_XZ.csv' # File to import/generate of (x,y,z) points representing XZ projected points. Path relative to SEGMENTATION_DIR
RESTRICTED_MODE = True # Whether to use DRG segmentation technique
# VALIDATION / VISUALISATION DEFAULT SCRIPT PARAMETERS
VALIDATION_DIR = '../VALIDATION/' # Relative path to validation folder
V_DATA_FOLDER = 'validation_runs/drg_subset_1_r' # Folder holding the data in /VALIDATION_DIR/V_SUBFOLDER/
V_GT_CSV = 'real_data_filtered_v0_VOLUMES.csv' # Filename of ground truth segmentation in /VALIDATION_DIR/V_SUBFOLDER/V_DATA_FOLDER
V_ALGO_CSV = 'real_data_filtered_algo_VOLUMES.csv' # Filename of algorithmic segmentation result in /VALIDATION_DIR/V_SUBFOLDER/V_DATA_FOLDER
V_MAPPING_CSV = 'mapping.csv' # Output mapping file to place in /VALIDATION_DIR/V_SUBFOLDER/V_DATA_FOLDER
# DEFAULT SCRIPT METADATA
HAS_VALIDATION = True # Has a validation volume dataset that may be plotted
HAS_ALGORITHMIC = True # Has an algorithmically generated volume dataset
RUN_SEGMENTATION = True # Run a segmentation on the XY rois
PLOT_TYPE = 'both' # Plot validation vs algorithmic dataset. Other options = {'gt','algo'}
TEMP_REDUCTION = False # Don't operate on the data unless specified

def main():
    parser = argparse.ArgumentParser(description="Run a scenario or set parameters for segmentation and validation.")
    parser.add_argument('--scenarios_file', help='Path to the scenarios JSON file')
    parser.add_argument('--scenario_name', help='Name of the scenario to load')
    args = parser.parse_args()
    # Load parameter set from json file
    scenario_filepath = args.scenarios_file or SCENARIO_FILEPATH
    active_scenario = args.scenario_name or ACTIVE_SCENARIO
    all_scenarios, parameters = load_parameters_from_json(scenario_filepath, active_scenario) or {}
    # SCRIPT OPERATION METADATA
    has_validation = parameters.get("HAS_VALIDATION", HAS_VALIDATION)
    has_algorithmic = parameters.get("HAS_ALGORITHMIC", HAS_ALGORITHMIC)
    run_segmentation = parameters.get("RUN_SEGMENTATION", RUN_SEGMENTATION)
    plot_type = parameters.get("PLOT_TYPE", PLOT_TYPE)
    temp_reduction = parameters.get("SEGMENT_VALIDATED_ONLY") or TEMP_REDUCTION
    # SEGMENTATION SCRIPT PARAMETERS 
    segmentation_dir = parameters.get("SEGMENTATION_DIR", SEGMENTATION_DIR)
    seg_script_path = os.path.join(segmentation_dir, 'volume_segmentation.py')
    seg_in_file = parameters.get("SEG_IN_FILE", SEG_IN_FILE) if not temp_reduction else parameters.get("TEMP_REDUCED_FILE") # Use temporary dataset
    seg_xz_cache_file = parameters.get("SEG_XZ_IO_FILE", SEG_XZ_IO_FILE)
    restricted_mode = parameters.get("RESTRICTED_MODE", RESTRICTED_MODE)
    # VALIDATION / VISUALISATION SCRIPT PARAMETERS
    validation_dir = parameters.get("VALIDATION_DIR", VALIDATION_DIR)
    v_data_folder = parameters.get("V_DATA_FOLDER", V_DATA_FOLDER)
    v_gt_csv = parameters.get("V_GT_CSV", V_GT_CSV) if not temp_reduction else parameters.get("V_TEMP_REDUCED_FILE") # Use temporary dataset
    v_algo_csv = parameters.get("V_ALGO_CSV", V_ALGO_CSV)
    v_mapping_csv = parameters.get("V_MAPPING_CSV", V_MAPPING_CSV)
    
    # Determine output file for segmentation
    if has_algorithmic: # Already has an algorithmic path - use this
        seg_out_filename = v_algo_csv
    else: # Create algorithmic path for demo file
        seg_out_filename = active_scenario + "_algo_VOLUMES.csv"

    seg_out_file = os.path.join(v_data_folder, seg_out_filename)
    seg_args = [
        'python', seg_script_path,
        '--in_file', seg_in_file,
        '--xz_io_path', seg_xz_cache_file,
        '--out_file', seg_out_file,
        '--restricted_mode', str(restricted_mode)
    ]
    # Add in parameters
    out_params = []
    algo_parameters = parameters.get("algo_parameters", {})
    for parameter, value in algo_parameters.items():
        command = "--" + str(parameter)
        out_params.extend([command, str(value)])
    # Add parameter list to seg args
    seg_args.extend(out_params)

    v_script_path = os.path.join(validation_dir, 'v5.py')
    v_vis_script_path = os.path.join(validation_dir, 'plot_comparison.py')
    v_single_script_path = os.path.join(validation_dir, 'plot_output.py')
    
    # Argument list for validation script
    v_args = [
        'python', v_script_path, 
        '--data_folder', v_data_folder,
        '--gt_csv', v_gt_csv,
        '--algo_csv', v_algo_csv,
        '--mapping_csv', v_mapping_csv
    ]

    # Argument list for comparison visualisation script
    v_vis_args = [
        'python', v_vis_script_path,  
        '--data_folder', v_data_folder,
        '--gt_csv', v_gt_csv,
        '--algo_csv', v_algo_csv,
        '--mapping_csv', v_mapping_csv
    ]

    try:
        # Run segmentation if desired for scenario
        if run_segmentation:
            if temp_reduction:
                v_out = os.path.join(v_data_folder, v_gt_csv)
                # Generate temporary dataset if required
                process_args = [
                    'python', '../CELLPOSE_SCRIPTS/process_noise.py',  
                    '--perform_grouping', "True",
                    '--perform_reduction', "True",
                    '--temp_reduction', "True",
                    '--scenarios_path', scenario_filepath,
                    '--active_scenario', active_scenario,
                    '--roi_out_path', seg_in_file, # Write to temp roi file
                    '--vol_out_path', v_out # Write to temp validation file
                ]
                subprocess.run(process_args, check=True)
                # Wait for processing to be complete
                
            print(f"Running seg with: {seg_args}")
            subprocess.run(seg_args, check=True)
            # Update json to point to new algo output csv
            has_algorithmic = True
            all_scenarios[active_scenario]["HAS_ALGORITHMIC"] = True
            all_scenarios[active_scenario]["V_ALGO_CSV"] = seg_out_filename
            # Export result
            with open(scenario_filepath, 'w') as file:
                json.dump(all_scenarios, file, indent=4)
        if has_validation and has_algorithmic and plot_type == "both":
            # Run validation script if remap required
            if all_scenarios[active_scenario].get("REMAP_VOLUMES", None) or run_segmentation:
                subprocess.run(v_args, check=True)
            # Visualise GT and ALGO side by side
            if all_scenarios[active_scenario].get("SAMPLE_POINT_MAX", None):
                sample_point_max = str(all_scenarios[active_scenario]["SAMPLE_POINT_MAX"])
                v_vis_args.extend(["--sample_point_max", sample_point_max])
            subprocess.run(v_vis_args, check=True)
        elif has_validation and plot_type == "gt":
            # Argument list for single visualisation script on gt data
            v_single_args = [
                'python', v_single_script_path,  
                '--data_folder', v_data_folder,
                '--csv_filename', v_gt_csv,
                '--plot_title', f'Plot of Ground Truth Data for Dataset: {v_data_folder}'
            ]
            if all_scenarios[active_scenario].get("SAMPLE_POINT_MAX", None):
                sample_point_max = str(all_scenarios[active_scenario]["SAMPLE_POINT_MAX"])
                v_single_args.extend(["--sample_point_max", sample_point_max])
            subprocess.run(v_single_args, check=True)
        elif has_algorithmic and plot_type == "algo":
            # Argument list for single visualisation script on algo data
            v_single_args = [
                'python', v_single_script_path,  
                '--data_folder', v_data_folder,
                '--csv_filename', v_algo_csv,
                '--plot_title', f'Plot of Algorithmic Data for Dataset: {v_data_folder}'
            ]
            
            if all_scenarios[active_scenario].get("SAMPLE_POINT_MAX", None):
                sample_point_max = str(all_scenarios[active_scenario]["SAMPLE_POINT_MAX"])
                v_single_args.extend(["--sample_point_max", sample_point_max])
            subprocess.run(v_single_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def load_parameters_from_json(json_path, scenario):
    """Loads parameters for a specific scenario from a JSON file."""
    if not json_path or not os.path.exists(json_path):
        print(f"Could not load json file: {json_path}")
        return None
    
    with open(json_path, 'r') as file:
        all_scenarios = json.load(file)
    
    # Check if the scenario exists in the JSON
    if scenario not in all_scenarios:
        print(f"Could not find {scenario} in {all_scenarios.keys()}")
        return None
    
    return all_scenarios, all_scenarios[scenario]

if __name__ == "__main__":
    main()