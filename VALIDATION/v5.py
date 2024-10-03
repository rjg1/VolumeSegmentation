import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np
from math import isclose
import os

RUN_FOLDER = './validation_runs'
DATA_FOLDER = 'drg_subset_1_r'
GT_CSV_FILENAME = 'real_data_filtered_v0_VOLUMES.csv'  # Ground truth CSV file path
ALGO_CSV_FILENAME = 'real_data_filtered_algo_VOLUMES.csv'  # Algorithmic CSV file path
MAPPING_FILENAME = 'mapping.csv'

def load_data(ground_truth_path, algorithmic_path):
    """
    Load ground truth and algorithmic segmentation data from CSV files.
    """
    ground_truth_df = pd.read_csv(ground_truth_path)
    algorithmic_df = pd.read_csv(algorithmic_path)
    return ground_truth_df, algorithmic_df

def create_overlap_matrix(ground_truth_df, algorithmic_df, tolerance=1e-4):
    """
    Create an overlap matrix where each row corresponds to a ground truth volume,
    and each column corresponds to an algorithmic volume. The values in the matrix
    represent the number of overlapping (z, ROI_ID) pairs between each volume pair,
    accounting for floating-point imprecision in z-values.

    Returns:
    - overlap_matrix: 2D numpy array with overlap counts
    - gt_volume_ids: List of ground truth volume IDs
    - alg_volume_ids: List of algorithmic volume IDs
    """
    # Group ROIs by Volume_ID, using a set of tuples with adjusted matching for z-values
    ground_truth_volumes = ground_truth_df.groupby('VOLUME_ID')[['z', 'ROI_ID']].apply(lambda x: set(map(tuple, x.values))).to_dict()
    algorithmic_volumes = algorithmic_df.groupby('VOLUME_ID')[['z', 'ROI_ID']].apply(lambda x: set(map(tuple, x.values))).to_dict()

    # List of ground truth and algorithmic volume IDs
    gt_volume_ids = list(ground_truth_volumes.keys())
    alg_volume_ids = list(algorithmic_volumes.keys())

    # Initialize the overlap matrix
    overlap_matrix = np.zeros((len(gt_volume_ids), len(alg_volume_ids)))

    # Populate the overlap matrix with counts of matching (z, ROI_ID) pairs with tolerance
    for i, gt_id in enumerate(gt_volume_ids):
        for j, alg_id in enumerate(alg_volume_ids):
            matched_rois = match_rois_with_tolerance(ground_truth_volumes[gt_id], algorithmic_volumes[alg_id], tolerance)
            overlap_matrix[i, j] = len(matched_rois)

    return overlap_matrix, gt_volume_ids, alg_volume_ids, ground_truth_volumes, algorithmic_volumes

# Helper function to compare ROI sets with tolerance for z-values
def match_rois_with_tolerance(set1, set2, tol):
    matched = set()
    for z1, roi1 in set1:
        for z2, roi2 in set2:
            if isclose(roi1, roi2, abs_tol=tol) and isclose(z1, z2, abs_tol=tol):
                matched.add((z1, roi1))
                break
    return matched

def adjust_for_unequal_volumes(overlap_matrix):
    """
    Adjust the overlap matrix to handle cases where the number of volumes differs
    by adding dummy rows or columns filled with zero overlap.
    """
    n_rows, n_cols = overlap_matrix.shape
    if n_rows < n_cols:
        # More algorithmic volumes than ground truth volumes, add dummy rows
        padding = np.zeros((n_cols - n_rows, n_cols))
        overlap_matrix = np.vstack((overlap_matrix, padding))
    elif n_cols < n_rows:
        # More ground truth volumes than algorithmic volumes, add dummy columns
        padding = np.zeros((n_rows, n_rows - n_cols))
        overlap_matrix = np.hstack((overlap_matrix, padding))
    return overlap_matrix

def merge_sets_with_tolerance(set1, set2, tol):
    """
    Merge two sets of (z, ROI_ID) pairs considering floating-point tolerance for z values.
    This function creates a union of the sets, accounting for z-value inaccuracies.
    """
    merged = set(set1)
    for z2, roi2 in set2:
        found_match = False
        for z1, roi1 in merged:
            if isclose(roi1, roi2, abs_tol=tol) and isclose(z1, z2, abs_tol=tol):
                found_match = True
                break
        if not found_match:
            merged.add((z2, roi2))
    return merged

def count_mismatched_rois(set1, set2, tol):
    """
    Count mismatched ROIs considering floating-point tolerance for z values.
    """
    mismatched = 0
    for z1, roi1 in set1:
        match_found = False
        for z2, roi2 in set2:
            if isclose(roi1, roi2, abs_tol=tol) and isclose(z1, z2, abs_tol=tol):
                match_found = True
                break
        if not match_found:
            mismatched += 1
    return mismatched

def solve_optimal_assignment(overlap_matrix):
    """
    Solve the optimal assignment problem using the Hungarian algorithm to maximize total overlap.
    
    Returns:
    - row_ind: Index of ground truth volumes in the optimal assignment
    - col_ind: Index of algorithmic volumes in the optimal assignment
    """
    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)  # Negate for maximization
    return row_ind, col_ind

def calculate_accuracy(gt_volume_ids, alg_volume_ids, row_ind, col_ind, ground_truth_volumes, algorithmic_volumes, tolerance=1e-5):
    """
    Calculate the accuracy score based on the optimal assignment of ground truth to algorithmic volumes,
    accounting for floating-point inaccuracies in z-values.
    """
    # Total ROIs include all unique (z, ROI_ID) pairs from both ground truth and algorithmic datasets
    total_rois = set()
    for gt_set in ground_truth_volumes.values():
        total_rois = merge_sets_with_tolerance(total_rois, gt_set, tolerance)
    for alg_set in algorithmic_volumes.values():
        total_rois = merge_sets_with_tolerance(total_rois, alg_set, tolerance)

    correctly_classified_rois = 0
    incorrectly_classified_rois = 0

    resolved_mapping = {}
    unmatched_gt = set()
    unmatched_alg = set()

    # Calculate correctly and incorrectly classified ROIs based on optimal assignment
    for gt_idx, alg_idx in zip(row_ind, col_ind):
        if gt_idx < len(gt_volume_ids) and alg_idx < len(alg_volume_ids):  # Valid match, not with dummy volumes
            gt_volume = gt_volume_ids[gt_idx]
            alg_volume = alg_volume_ids[alg_idx]
            resolved_mapping[gt_volume] = alg_volume
            gt_rois = ground_truth_volumes[gt_volume]
            alg_rois = algorithmic_volumes[alg_volume]
            # Correct ROIs are those in both matched volumes using the matching function
            matched_rois = match_rois_with_tolerance(gt_rois, alg_rois, tolerance)
            correctly_classified_rois += len(matched_rois)
        elif gt_idx < len(gt_volume_ids):  # Ground truth volume matched with a dummy algorithmic volume
            unmatched_gt.add(gt_volume_ids[gt_idx])
        elif alg_idx < len(alg_volume_ids):  # Algorithmic volume matched with a dummy ground truth volume
            unmatched_alg.add(alg_volume_ids[alg_idx])

    # Calculate the accuracy score directly
    accuracy_score = correctly_classified_rois / len(total_rois) if total_rois else 0

    return accuracy_score, resolved_mapping, unmatched_gt, unmatched_alg, correctly_classified_rois, len(total_rois)

def export_mapping_to_csv(mapping_dict, filename="mapping.csv"):
    """
    Export a dictionary mapping ground truth volume IDs to algorithmic volume IDs to a CSV.
    The CSV will have columns 'GT_ID' and 'ALGO_ID'.
    """
    # Convert the dictionary to a pandas DataFrame
    mapping_df = pd.DataFrame(list(mapping_dict.items()), columns=['GT_ID', 'ALGO_ID'])

    # Export the DataFrame to a CSV file
    mapping_df.to_csv(filename, index=False)
    print(f"Mapping exported to {filename}")


def main():
    gt_csv_path = os.path.join(RUN_FOLDER, DATA_FOLDER, GT_CSV_FILENAME)
    algo_csv_path = os.path.join(RUN_FOLDER, DATA_FOLDER, ALGO_CSV_FILENAME)
    mapping_csv_path = os.path.join(RUN_FOLDER, DATA_FOLDER, MAPPING_FILENAME)

    ground_truth_df, algorithmic_df = load_data(gt_csv_path, algo_csv_path)
    overlap_matrix, gt_volume_ids, alg_volume_ids, ground_truth_volumes, algorithmic_volumes = create_overlap_matrix(ground_truth_df, algorithmic_df)
    adjusted_matrix = adjust_for_unequal_volumes(overlap_matrix)
    row_ind, col_ind = solve_optimal_assignment(adjusted_matrix)
    accuracy, resolved_mapping, unmatched_gt, unmatched_alg, correct_rois, total_rois = calculate_accuracy(
        gt_volume_ids, alg_volume_ids, row_ind, col_ind, ground_truth_volumes, algorithmic_volumes
    )
    export_mapping_to_csv(resolved_mapping, filename=mapping_csv_path)
    print(overlap_matrix)
    print(f"Resolved Mapping: {resolved_mapping}")
    print(f"Unmatched Ground Truth Volumes: {unmatched_gt}")
    print(f"Unmatched Algorithmic Volumes: {unmatched_alg}")
    print(f"Correctly Classified ROIs: {correct_rois}")
    print(f"Total ROIs: {total_rois}")
    print(f"Accuracy Score: {accuracy}")

if __name__ == "__main__":
    main()
