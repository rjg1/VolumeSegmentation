import pandas as pd
import numpy as np
from collections import defaultdict
from region import BoundaryRegion
from scipy.spatial import ConvexHull
from itertools import combinations
from plane import Plane
from planepoint import PlanePoint
import os
import copy
from param_handling import PLANE_GEN_PARAMS_DEFAULT, create_param_dict

def get_hull_boundary_points(hull):
    """Extract the list of boundary points of a 2D convex hull."""
    vertices = hull.points[hull.vertices]
    boundary_points = [(x, y) for x, y in vertices]
    return boundary_points

class ZStack:
    def __init__(self, data=None):
        self.z_planes = defaultdict(lambda: defaultdict(dict))  # {z: {roi_id: {'coords': [...], 'intensity': ...}}}
        self.xy_rois = {}  # {xy_id: BoundaryRegion}
        self.xz_rois = {}  # {xz_id: BoundaryRegion}
        self.volume_assignments = {}  # {xy_id: volume_id}
        self.has_intensity = False
        self.has_volume = False
        self.planes = []
        # Parse input data
        if isinstance(data, str): # Assume string to csv path in local directory
            self.load_from_csv(data)
        elif isinstance(data, dict): # Assume correctly formed dictionary for z_planes
            self.z_planes = self._from_zplane_dict(data)  


    def _find_edge_rois(self, xmin, xmax, ymin, ymax, margin=5):
        """
        Flags ROIs in the z_planes dict that are near the image border as 'edge' ROIs.
        These ROIs are marked with a new key 'is_edge': True within each ROI's metadata.
        """
        for z, roi_dict in self.z_planes.items():
            for roi_id, roi_data in roi_dict.items():
                coords = roi_data.get("coords", [])
                for x, y in coords:
                    if (x < xmin + margin or x > xmax - margin or
                        y < ymin + margin or y > ymax - margin):
                        roi_data["is_edge"] = True
                        break
                else:
                    roi_data["is_edge"] = False
        return self.z_planes

    def _from_zplane_dict(self, z_planes):
        """
        Validates and returns a deep copy of the z_planes dict.
        Expected format:
            {
                z: {
                    roi_id: {
                        'coords': [(x, y), ...],
                        'intensity': float (optional)
                    }
                }
            }
        """
        for z, roi_dict in z_planes.items():
            if not isinstance(roi_dict, dict):
                raise ValueError(f"Expected dict for z={z}, got {type(roi_dict)}")

            for roi_id, info in roi_dict.items():
                if not isinstance(info, dict):
                    raise ValueError(f"Expected dict for ROI {roi_id} at z={z}, got {type(info)}")

                if "coords" not in info:
                    raise ValueError(f"ROI {roi_id} at z={z} missing 'coords' key")
                if not isinstance(info["coords"], list):
                    raise ValueError(f"'coords' for ROI {roi_id} at z={z} must be a list")
                if len(info["coords"]) == 0:
                    raise ValueError(f"'coords' for ROI {roi_id} at z={z} is empty")

                if "intensity" in info:
                    if not isinstance(info["intensity"], (float, int)):
                        raise ValueError(f"'intensity' for ROI {roi_id} at z={z} must be float or int")
                    self.has_intensity = True
                if "volume" in info:
                    if not isinstance(info["intensity"], int):
                        raise ValueError(f"'volume' for ROI {roi_id} at z={z} must be int")
                    self.has_volume = True

        return copy.deepcopy(z_planes)
    
    def load_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        required_cols = {"x", "y", "z", "ROI_ID"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain at least these columns: {required_cols}")

        self.has_intensity = 'intensity' in df.columns
        self.has_volume = 'VOLUME_ID' in df.columns

        for (z, roi_id), group in df.groupby(["z", "ROI_ID"]):
            coords = list(zip(group["x"], group["y"]))
            self.z_planes[z][roi_id]["coords"] = coords
            if self.has_intensity:
                self.z_planes[z][roi_id]["intensity"] = group["intensity"].iloc[0]
            if self.has_volume:
                self.z_planes[z][roi_id]["volume"] = group["VOLUME_ID"].iloc[0]

        self.z_planes = dict(sorted(self.z_planes.items()))  # Sort z keys

    def _build_xy_rois(self, parameters=None):
        self.xy_rois = {}
        parameters = parameters or {}
        for z, roi_dict in self.z_planes.items():
            for roi_id, roi_data in roi_dict.items():
                coords = roi_data["coords"]
                coords_3d = [(x, y, z) for x, y in coords]
                region = BoundaryRegion(
                    coords_3d,
                    original_index=roi_id,
                    precalc_area=parameters.get("restrict_area", False),
                    precalc_centroid=parameters.get("restrict_centroid_distance", False),
                    precalc_radius=parameters.get("use_percent_centroid_distance", False)
                )
                self.xy_rois[(z, roi_id)] = region
                if self.has_volume:
                    self.volume_assignments[(z, roi_id)] = roi_data["volume"]

    def _build_xz_rois(self, xz_hulls):
        self.xz_rois = {}
        xz_id = 0
        for y, hull_list in xz_hulls.items():
            for item in hull_list:
                points_3d = []
                if isinstance(item, ConvexHull):
                    boundary_points = get_hull_boundary_points(item)
                    points_3d = [(x, y, z) for x, z in boundary_points]
                elif isinstance(item, np.ndarray):
                    points_3d = [(x, y, z) for x, z in item]
                elif isinstance(item, list):
                    points_3d = item
                if points_3d:
                    self.xz_rois[xz_id] = BoundaryRegion(points_3d)
                    xz_id += 1

    def export_to_csv(self, out_path):
        rows = []
        for z, roi_dict in self.z_planes.items():
            for roi_id, roi_data in roi_dict.items():
                for x, y in roi_data["coords"]:
                    row = {
                        "x": x,
                        "y": y,
                        "z": z,
                        "ROI_ID": roi_id
                    }
                    if self.has_volume and "volume" in roi_data:
                        row["VOLUME_ID"] = roi_data["volume"]
                    if self.has_intensity and "intensity" in roi_data:
                        row["intensity"] = roi_data["intensity"]
                    rows.append(row)
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_path, index=False)

    def _build_z_indexed_rois(self, anchor_threshold, align_threshold, z_max = np.inf, z_min = -np.inf):
        self._build_xy_rois()  # Ensure rois are constructed

        anchor_rois_by_z = defaultdict(list)
        align_rois_by_z = defaultdict(list)

        for (z, roi_id), roi in self.xy_rois.items():
            if z > z_max or z < z_min: # exclude ROIs out of search range
                continue

            centroid = roi.get_centroid()
            if centroid is None or roi.get_radius() <= 0:
                continue
           
           # Ignore edge rois as they will have been segmented incorrectly
            if self.z_planes[z][roi_id]["is_edge"]:
                continue

            intensity = self.z_planes[z][roi_id]["intensity"]

            if intensity >= anchor_threshold:
                anchor_rois_by_z[z].append((roi_id, roi))
            if intensity >= align_threshold:
                align_rois_by_z[z].append((roi_id, roi))

        return anchor_rois_by_z, align_rois_by_z

    def generate_planes(
            self,
            plane_gen_params = None
        ):
            params = create_param_dict(PLANE_GEN_PARAMS_DEFAULT, plane_gen_params)

            # Load from CSV if specified
            if params["read_filename"] is not None and not params["regenerate_planes"]:
                read_path = params["read_filename"]
                if os.path.exists(read_path):
                    try:
                        print(f"[INFO] Attempting to load planes from: {read_path}")
                        return self.read_planes_from_csv(read_path)
                    except Exception as e:
                        print(f"[WARN] Failed to parse plane file '{read_path}': {e}")
                else:
                    print(f"[WARN] Plane file '{read_path}' does not exist. Regenerating planes.")

            # If planes already exist and regeneration not required, simply return the existing planes
            if len(self.planes) > 0 and not params['regenerate_planes']:
                print("Using past save of planes...")
                return self.planes

            print("Generating planes...")

            max_tilt_rad = np.radians(params["max_tilt_deg"])
            self.planes = []

            boundaries = params["plane_boundaries"]
            self._find_edge_rois(boundaries[0], boundaries[1], boundaries[2], boundaries[3], params["margin"])

            if not self.has_intensity:
                raise ValueError("Requires intensity to run")
            
            # Transform intensities across planes if applicable
            transform_mode = params.get("transform_intensity", "raw")
            for z, roi_data in self.z_planes.items():
                intensities = [info["intensity"] for info in roi_data.values() if "intensity" in info]
                if not intensities:
                    continue

                if transform_mode == "raw":
                    continue  # No change

                elif transform_mode == "minmax":
                    min_int = min(intensities)
                    max_int = max(intensities)
                    range_int = max_int - min_int

                    for info in roi_data.values():
                        if "intensity" in info:
                            if range_int == 0:
                                info["intensity"] = 1.0
                            else:
                                info["intensity"] = (info["intensity"] - min_int) / range_int

                elif transform_mode == "quantile":
                    sorted_intensities = sorted(intensities)
                    total = len(sorted_intensities)
                    for info in roi_data.values():
                        if "intensity" in info:
                            count_less = sum(1 for val in sorted_intensities if val < info["intensity"])
                            info["intensity"] = count_less / total


            # Build a dictionary of {z: [(id, BoundaryRegion)]} for potential anchor and alignment points
            z_max = np.inf
            z_min = -np.inf
            if params["z_guess"] != -1:
                z_max = params["z_guess"] + params["z_range"]
                z_min = params["z_guess"] - params["z_range"]
            
            anchor_by_z, align_by_z = self._build_z_indexed_rois(params["anchor_intensity_threshold"], params["align_intensity_threshold"], z_max = z_max, z_min = z_min)

            # Search through anchor points in ascending Z
            for z_anchor in sorted(anchor_by_z.keys()): # Iterate through z levels
                # For each anchor point in each z-level
                for anchor_id, anchor_roi in anchor_by_z[z_anchor]:
                    anchor_pos = anchor_roi.get_centroid() # Extract centroid
                    anchor_point = PlanePoint(anchor_id, anchor_pos) # Make a PlanePoint object for plane construction

                    # Collect alignment candidates within z threshold
                    local_alignments = []
                    for dz in range(-params["z_threshold"], params["z_threshold"] + 1):
                        for align_id, align_roi in align_by_z.get(z_anchor + dz, []): # Find any entries on this z-level
                            if align_id != anchor_id: # Exclude anchor point from forming a plane with itself
                                local_alignments.append((align_id, align_roi))

                    for (id1, roi1), (id2, roi2) in combinations(local_alignments, 2): # Iterate through all possible planar combinations
                        p1 = PlanePoint(id1, roi1.get_centroid())
                        p2 = PlanePoint(id2, roi2.get_centroid())

                        # Check to see if these points can form a reasonable plane by restricting angle between all 3 points
                        if not self._is_tilt_valid(anchor_point.position, p1.position, max_tilt_rad):
                            continue
                        if not self._is_tilt_valid(anchor_point.position, p2.position, max_tilt_rad):
                            continue
                        if not self._is_tilt_valid(p1.position, p2.position, max_tilt_rad):
                            continue

                        # Construct a plane object with these points
                        plane = Plane(anchor_point, [p1, p2], max_alignments=params["max_alignments"], fixed_basis=params["fixed_basis"])
                        # Find any extra points to project to this plane to increase accuracy of alignment
                        additional_points = [PlanePoint(align_id, align_roi.get_centroid()) for align_id, align_roi in local_alignments if align_id not in (id1, id2, anchor_id)]
                        plane.project_points(additional_points, threshold=params["projection_dist_thresh"]) # Project all nearby points

                        if not any(existing.is_equivalent_to(plane, match_anchors = params["match_anchors"]) for existing in self.planes):
                            self.planes.append(plane)

            # Write output to file if required
            if params["save_filename"] is not None:
                try:
                    print(f"[INFO] Saving planes to: {params['save_filename']}")
                    self.write_planes_to_csv(params["save_filename"])
                except Exception as e:
                    print(f"[WARN] Could not save planes to '{params['save_filename']}': {e}")

            return self.planes

    # TODO test this
    def _is_tilt_valid(self, p1, p2, max_tilt_rad):
        delta = np.abs(np.array(p2) - np.array(p1))
        dz = delta[2]
        flat_distance = np.linalg.norm(delta[:2])
        angle = np.arctan2(dz, flat_distance) if flat_distance != 0 else np.pi / 2
        return angle <= max_tilt_rad

    # Writes plane points to a csv for reconstruction
    def write_planes_to_csv(self, filename):
        rows = []

        # Step 1: Collect all trait names from all plane points
        all_traits = set()
        for plane in self.planes:
            all_traits.update(plane.anchor_point.traits.keys())
            for _, ppt in list(plane.plane_points.items())[1:]:  # skip anchor
                all_traits.update(ppt.traits.keys())

        sorted_traits = sorted(all_traits)  # consistent column ordering

        # Step 2: Build rows for each plane in plane list of z stack
        for plane_id, plane in enumerate(self.planes):
            # Anchor point
            anchor = plane.anchor_point
            row = {
                "plane_id": plane_id,
                "pid": anchor.id,
                "x": anchor.position[0],
                "y": anchor.position[1],
                "z": anchor.position[2],
                "type": "anchor"
            }
            for tr in sorted_traits:
                row[f"tr_{tr}"] = anchor.traits[tr] if tr in anchor.traits else None
            rows.append(row)

            # Alignment points
            for _, ppt in list(plane.plane_points.items())[1:]: # skips anchor at index 0
                row = {
                    "plane_id": plane_id,
                    "pid": ppt.id,
                    "x": ppt.position[0],
                    "y": ppt.position[1],
                    "z": ppt.position[2],
                    "type": "alignment"
                }
                for tr in sorted_traits:
                    row[f"tr_{tr}"] = ppt.traits[tr] if tr in ppt.traits else None
                rows.append(row)

        # Step 3: Save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)


    # Remakes planes saved from a previous csv
    def read_planes_from_csv(self, filename):
        required_columns = {"plane_id", "pid", "x", "y", "z", "type"}
        try:
            df = pd.read_csv(filename)
        except Exception as e:
            raise IOError(f"Failed to read CSV: {e}")
        if not required_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV missing required columns. Expected at least: {required_columns}")
        
        self.planes = []

        # Step 1: Identify all trait columns
        trait_columns = [col for col in df.columns if col.startswith("tr_")]
        trait_names = [col[3:] for col in trait_columns]  # remove 'tr_' prefix

        # Step 2: Group by plane_id
        for plane_id, group in df.groupby("plane_id"):
            # Extract anchor row
            anchor_row = group[group["type"] == "anchor"].iloc[0]
            anchor_traits = {
                tr: anchor_row[f"tr_{tr}"]
                for tr in trait_names
                if not pd.isna(anchor_row[f"tr_{tr}"])
            }
            anchor = PlanePoint(int(anchor_row.pid), (anchor_row.x, anchor_row.y, anchor_row.z), traits=anchor_traits)

            # Extract alignment rows
            alignment_points = []
            alignment_rows = group[group["type"] == "alignment"]
            for _, row in alignment_rows.iterrows():
                traits = {
                    tr: row[f"tr_{tr}"]
                    for tr in trait_names
                    if not pd.isna(row[f"tr_{tr}"])
                }
                ppt = PlanePoint(int(row.pid), (row.x, row.y, row.z), traits=traits)
                alignment_points.append(ppt)

            # Only add planes with enough alignment points
            if len(alignment_points) >= 2:
                plane = Plane(anchor, alignment_points)
                self.planes.append(plane)

        return self.planes






    