import pandas as pd
import numpy as np
from collections import defaultdict
from region import BoundaryRegion
from scipy.spatial import ConvexHull
from itertools import combinations
from plane import Plane
from planepoint import PlanePoint
import copy

PLANE_GEN_PARAMS_DEFAULT = {
    "anchor_intensity_threshold": 0.8,
    "align_intensity_threshold": 0.4,
    "z_threshold": 5,
    "max_tilt_deg": 30.0,
    "projection_dist_thresh":  0.5,
    "normalize_intensity" : True,
    "xmin" : -np.inf, #image boundaries to clip
    "xmax" : np.inf,
    "ymin" : -np.inf,
    "ymax" : np.inf,
    "margin" : 2, # distance between boundary point and pixel in img to be considered an edge roi
    "match_anchors" : True,
    "fixed_basis" : True,
    "max_alignments" : 500
}


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

    def _build_z_indexed_rois(self, anchor_threshold, align_threshold):
        self._build_xy_rois()  # Ensure rois are constructed

        anchor_rois_by_z = defaultdict(list)
        align_rois_by_z = defaultdict(list)

        for (z, roi_id), roi in self.xy_rois.items():
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
            params = PLANE_GEN_PARAMS_DEFAULT.copy()  # start with defaults
            if plane_gen_params:
                params.update(plane_gen_params)  # override with anything user provides
            

            max_tilt_rad = np.radians(params["max_tilt_deg"])
            self.planes = []

            self._find_edge_rois(params["xmin"], params["xmax"], params["ymin"], params["ymax"], params["margin"])

            if not self.has_intensity:
                raise ValueError("Requires intensity to run")
            
            # Normalize intensitys across planes if applicable
            if params["normalize_intensity"]:
                for z, roi_data in self.z_planes.items():
                    intensities = [info["intensity"] for info in roi_data.values() if "intensity" in info]
                    if not intensities:
                        continue  # Skip if no intensity data

                    min_intensity = min(intensities)
                    max_intensity = max(intensities)
                    range_intensity = max_intensity - min_intensity

                    if range_intensity == 0:
                        # If all intensities are the same, set all to 1.0
                        for info in roi_data.values():
                            if "intensity" in info:
                                info["intensity"] = 1.0
                    else:
                        for info in roi_data.values():
                            if "intensity" in info:
                                info["intensity"] = (info["intensity"] - min_intensity) / range_intensity


            # Build a dictionary of {z: [(id, BoundaryRegion)]} for potential anchor and alignment points
            anchor_by_z, align_by_z = self._build_z_indexed_rois(params["anchor_intensity_threshold"], params["align_intensity_threshold"])

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
        for plane_id, plane in enumerate(self.planes):
            rows.append({
                "plane_id": plane_id,
                "pid" : plane.anchor_point.id,
                "x": plane.anchor_point.position[0],
                "y": plane.anchor_point.position[1],
                "z": plane.anchor_point.position[2],
                "type": "anchor"
            })
            for _, ppt in list(plane.plane_points.items())[1:]:  # skip anchor
                rows.append({
                    "plane_id": plane_id,
                    "pid" : ppt.id,
                    "x": ppt.position[0],
                    "y": ppt.position[1],
                    "z": ppt.position[2],
                    "type": "alignment"
                })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

    # Remakes planes saved from a previous csv
    def read_planes_from_csv(self, filename):
        df = pd.read_csv(filename)
        self.planes = []

        for plane_id, group in df.groupby("plane_id"):
            anchor_row = group[group["type"] == "anchor"].iloc[0]
            anchor = PlanePoint(int(anchor_row.pid), (anchor_row.x, anchor_row.y, anchor_row.z))

            alignment_rows = group[group["type"] == "alignment"]
            alignments = [
                PlanePoint(int(row.pid), (row.x, row.y, row.z))
                for i, row in alignment_rows.iterrows()
            ]

            if len(alignments) >= 2:
                plane = Plane(anchor, alignments)
                self.planes.append(plane)

        return self.planes





    