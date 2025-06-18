import pandas as pd

def check_missing_traits(csv_path):
    """
    Loads a CSV of plane points and checks for missing traits.
    For anchor points, skips 'tr_angle' and 'tr_magnitude'.
    Prints which traits are missing per point.
    """
    df = pd.read_csv(csv_path)

    trait_columns = df.columns[6:]  # Trait columns start after 'type'
    if len(trait_columns) == 0:
        print("[ERROR] No trait columns found.")
        return

    missing_summary = []

    for idx, row in df.iterrows():
        # For anchors, ignore angle/magnitude traits
        skip_traits = {"tr_angle", "tr_magnitude"} if row["type"] == "anchor" else set()
        missing = [
            col for col in trait_columns
            if pd.isna(row[col]) and col not in skip_traits
        ]
        if missing:
            missing_summary.append({
                "plane_id": row["plane_id"],
                "pid": row["pid"],
                "type": row["type"],
                "missing_traits": missing
            })

    if missing_summary:
        print(f"[INFO] Found {len(missing_summary)} entries with missing traits:")
        for entry in missing_summary:
            print(f"  - Plane {entry['plane_id']} PID {entry['pid']} ({entry['type']}): Missing {entry['missing_traits']}")
    else:
        print("[SUCCESS] No missing traits found.")

def check_missing_traits_in_z_planes(z_planes, required_traits=None):
    """
    Checks for missing traits in all ROIs in the z_planes dictionary.
    
    Parameters:
    - z_planes: dict of format {z: {roi_id: {"coords": [...], "intensity": float, "traits": {...}}}}
    - required_traits: optional list of traits to check; defaults to ['angle', 'area', 'avg_radius', 'circularity', 'magnitude']
    
    Skips 'angle' and 'magnitude' checks for anchor-type ROIs (assumes ROI ID 0 is anchor, or uses a 'type' key if present).
    
    Returns:
    - missing_report: list of tuples (z, roi_id, missing_traits)
    """
    if required_traits is None:
        required_traits = ["area", "avg_radius", "circularity"]
    
    missing_report = []

    for z, rois in z_planes.items():
        for roi_id, roi_data in rois.items():
            traits = roi_data.get("traits", {})
            missing = []
            for trait in required_traits:
                if trait not in traits or traits[trait] is None:
                    missing.append(trait)
            if missing:
                missing_report.append((z, roi_id, missing))
    
    if missing_report:
        print("[MISSING TRAITS]")
        for z, roi_id, traits in missing_report:
            print(f"Z={z}, ROI={roi_id} is missing traits: {traits}")
    else:
        print("No missing traits found.")

    return missing_report


# Example usage:
from zstack import *
STACK_IN_FILE = "real_data_filtered_algo_VOLUMES_g.csv"
PLANES_IN_FILE = "real_data_filtered_algo_VOLUMES_g_planes.csv"
z_stack = ZStack(data=STACK_IN_FILE) 
check_missing_traits_in_z_planes(z_stack.z_planes)
check_missing_traits(PLANES_IN_FILE)
