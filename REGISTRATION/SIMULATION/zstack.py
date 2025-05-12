import pandas as pd
import numpy as np
from collections import defaultdict
from region import BoundaryRegion
from scipy.spatial import ConvexHull
from itertools import combinations
from plane import Plane
from planepoint import PlanePoint
import os
from concurrent.futures import ThreadPoolExecutor
import random
import copy
import cupy as cp
from tqdm import tqdm
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
    
    def generate_random_intensities(self, min_val=0.0, max_val=1.0):
        """
        Assigns a random intensity value between min_val and max_val
        to each ROI in the z_planes dictionary that does not already have one.
        Debug function - shouldn't be necessary in practice.
        """
        if not hasattr(self, 'z_planes'):
            raise AttributeError("This class does not have a 'z_planes' attribute.")

        for z, roi_dict in self.z_planes.items():
            for roi_id, info in roi_dict.items():
                if "intensity" not in info:
                    info["intensity"] = random.uniform(min_val, max_val)

        self.has_intensity = True

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

    def generate_planes(self, plane_gen_params=None):
        params = create_param_dict(PLANE_GEN_PARAMS_DEFAULT, plane_gen_params)

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

        if len(self.planes) > 0 and not params['regenerate_planes']:
            print("Using past save of planes...")
            return self.planes

        max_tilt_rad = np.radians(params["max_tilt_deg"])
        self.planes = []
        boundaries = params["plane_boundaries"]
        self._find_edge_rois(boundaries[0], boundaries[1], boundaries[2], boundaries[3], params["margin"])

        if not self.has_intensity:
            raise ValueError("Requires intensity to run")

        transform_mode = params.get("transform_intensity", "raw")
        for z, roi_data in self.z_planes.items():
            intensities = [info["intensity"] for info in roi_data.values() if "intensity" in info]
            if not intensities:
                continue

            if transform_mode == "minmax":
                min_int = min(intensities)
                max_int = max(intensities)
                range_int = max_int - min_int
                for info in roi_data.values():
                    if "intensity" in info:
                        info["intensity"] = (info["intensity"] - min_int) / range_int if range_int else 1.0

            elif transform_mode == "quantile":
                sorted_intensities = sorted(intensities)
                total = len(sorted_intensities)
                for info in roi_data.values():
                    if "intensity" in info:
                        count_less = sum(1 for val in sorted_intensities if val < info["intensity"])
                        info["intensity"] = count_less / total

        z_max = np.inf
        z_min = -np.inf
        if params["z_guess"] != -1:
            z_max = params["z_guess"] + params["z_range"]
            z_min = params["z_guess"] - params["z_range"]

        anchor_by_z, align_by_z = self._build_z_indexed_rois(params["anchor_intensity_threshold"], params["align_intensity_threshold"], z_max=z_max, z_min=z_min)

        tasks = [
            (z, anchor_id, anchor_roi) for z in sorted(anchor_by_z.keys())
            for anchor_id, anchor_roi in anchor_by_z[z]
        ]

        def is_coplanar(plane, normals_list, angle_thresh_rad=0.017, dist_thresh=1e-3):
            for n in normals_list:
                dot = np.dot(plane.normal, n)
                angle_diff = np.arccos(np.clip(dot, -1.0, 1.0))
                if angle_diff <= angle_thresh_rad:
                    d = -np.dot(n, plane.anchor_point.position)
                    dist = abs(np.dot(n, plane.plane_points[1].position) + d)
                    if dist < dist_thresh:
                        return True
            return False

        def process_anchor(z_anchor, anchor_id, anchor_roi):
            anchor_planes = []
            normals_seen = []
            anchor_pos = anchor_roi.get_centroid()
            anchor_point = PlanePoint(anchor_id, anchor_pos)

            # Precompute local alignments once per z-slice
            local_alignments = [
                (align_id, align_roi)
                for dz in range(-params["z_threshold"], params["z_threshold"] + 1)
                for align_id, align_roi in align_by_z.get(z_anchor + dz, [])
                if align_id != anchor_id
            ]

            tilt_cache = {}
            def tilt_valid(id_a, pos_a, id_b, pos_b):
                key = frozenset((id_a, id_b))
                if key in tilt_cache:
                    return tilt_cache[key]
                result = self._is_tilt_valid(pos_a, pos_b, max_tilt_rad)
                tilt_cache[key] = result
                return result

            for (id1, roi1), (id2, roi2) in combinations(local_alignments, 2):
                p1 = PlanePoint(id1, roi1.get_centroid())
                p2 = PlanePoint(id2, roi2.get_centroid())

                if not tilt_valid(anchor_id, anchor_pos, id1, p1.position):
                    continue
                if not tilt_valid(anchor_id, anchor_pos, id2, p2.position):
                    continue
                if not tilt_valid(id1, p1.position, id2, p2.position):
                    continue

                plane = Plane(anchor_point, [p1, p2], max_alignments=params["max_alignments"], fixed_basis=params["fixed_basis"])

                if is_coplanar(plane, normals_seen):
                    continue

                normals_seen.append(plane.normal)

                additional_points = [
                    PlanePoint(align_id, align_roi.get_centroid())
                    for align_id, align_roi in local_alignments
                    if align_id not in (id1, id2, anchor_id)
                ]
                plane.project_points(additional_points, threshold=params["projection_dist_thresh"])

                anchor_planes.append(plane)

            return anchor_planes

        self.planes = []
        n_threads = params.get("n_threads", 4)
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(process_anchor, *task) for task in tasks]
            with tqdm(total=len(futures), desc="Generating planes", ncols=80) as pbar:
                for future in futures:
                    self.planes.extend(future.result())
                    pbar.update(1)
            # for future in futures:
            #     self.planes.extend(future.result())


        if params["save_filename"] is not None:
            try:
                print(f"[INFO] Saving planes to: {params['save_filename']}")
                self.write_planes_to_csv(params["save_filename"])
            except Exception as e:
                print(f"[WARN] Could not save planes to '{params['save_filename']}': {e}")

        return self.planes


    def generate_planes_gpu(self, plane_gen_params=None):
        params = create_param_dict(PLANE_GEN_PARAMS_DEFAULT, plane_gen_params)

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

        if len(self.planes) > 0 and not params['regenerate_planes']:
            print("Using past save of planes...")
            return self.planes

        max_tilt_rad = np.radians(params["max_tilt_deg"])
        self.planes = []
        boundaries = params["plane_boundaries"]
        self._find_edge_rois(boundaries[0], boundaries[1], boundaries[2], boundaries[3], params["margin"])

        if not self.has_intensity:
            raise ValueError("Requires intensity to run")

        transform_mode = params.get("transform_intensity", "raw")
        for z, roi_data in self.z_planes.items():
            intensities = [info["intensity"] for info in roi_data.values() if "intensity" in info]
            if not intensities:
                continue

            if transform_mode == "minmax":
                min_int = min(intensities)
                max_int = max(intensities)
                range_int = max_int - min_int
                for info in roi_data.values():
                    if "intensity" in info:
                        info["intensity"] = (info["intensity"] - min_int) / range_int if range_int else 1.0

            elif transform_mode == "quantile":
                sorted_intensities = sorted(intensities)
                total = len(sorted_intensities)
                for info in roi_data.values():
                    if "intensity" in info:
                        count_less = sum(1 for val in sorted_intensities if val < info["intensity"])
                        info["intensity"] = count_less / total

        z_max = np.inf
        z_min = -np.inf
        if params["z_guess"] != -1:
            z_max = params["z_guess"] + params["z_range"]
            z_min = params["z_guess"] - params["z_range"]

        anchor_by_z, align_by_z = self._build_z_indexed_rois(params["anchor_intensity_threshold"], params["align_intensity_threshold"], z_max=z_max, z_min=z_min)

        tasks = [
            (z, anchor_id, anchor_roi) for z in sorted(anchor_by_z.keys())
            for anchor_id, anchor_roi in anchor_by_z[z]
        ]

        def gpu_tilt_check(anchor_pos, pair_pts1, pair_pts2, max_angle):
            v1 = pair_pts1 - anchor_pos
            v2 = pair_pts2 - anchor_pos
            cp_v1 = cp.asarray(v1)
            cp_v2 = cp.asarray(v2)
            dot = cp.sum(cp_v1 * cp_v2, axis=1)
            norms = cp.linalg.norm(cp_v1, axis=1) * cp.linalg.norm(cp_v2, axis=1)
            angles = cp.arccos(cp.clip(dot / norms, -1.0, 1.0))
            return (angles <= max_angle).get()

        def is_coplanar(normal, normals_list, angle_thresh_rad=0.017):
            if not normals_list:
                return False
            normals = np.stack(normals_list)
            dots = np.dot(normals, normal)
            angles = np.arccos(np.clip(dots, -1.0, 1.0))
            return np.any(angles <= angle_thresh_rad)

        def gpu_project_points(points, normal, anchor_pos, threshold):
            points_gpu = cp.asarray(points)
            normal_gpu = cp.asarray(normal)
            anchor_gpu = cp.asarray(anchor_pos)

            distances = cp.dot(points_gpu - anchor_gpu, normal_gpu)
            projected = points_gpu - distances[:, cp.newaxis] * normal_gpu
            within_thresh = cp.abs(distances) <= threshold

            return projected.get(), within_thresh.get()

        def process_anchor(z_anchor, anchor_id, anchor_roi):
            anchor_planes = []
            normals_seen = []
            anchor_pos = anchor_roi.get_centroid()
            anchor_point = PlanePoint(anchor_id, anchor_pos)

            local_alignments = [
                (align_id, align_roi.get_centroid())
                for dz in range(-params["z_threshold"], params["z_threshold"] + 1)
                for align_id, align_roi in align_by_z.get(z_anchor + dz, [])
                if align_id != anchor_id
            ]

            point_ids, point_pos = zip(*local_alignments)
            combos = list(combinations(range(len(point_pos)), 2))

            pts1 = np.array([point_pos[i] for i, j in combos])
            pts2 = np.array([point_pos[j] for i, j in combos])

            valid_mask = gpu_tilt_check(np.array(anchor_pos), pts1, pts2, max_tilt_rad)

            for (i, j), valid in zip(combos, valid_mask):
                if not valid:
                    continue

                id1, id2 = point_ids[i], point_ids[j]
                p1 = PlanePoint(id1, point_pos[i])
                p2 = PlanePoint(id2, point_pos[j])

                v1 = p1.position - anchor_pos
                v2 = p2.position - anchor_pos
                normal = np.cross(v1, v2)
                norm_mag = np.linalg.norm(normal)
                if norm_mag == 0:
                    continue

                normal /= norm_mag

                if is_coplanar(normal, normals_seen):
                    continue

                plane = Plane(anchor_point, [p1, p2], max_alignments=params["max_alignments"], fixed_basis=params["fixed_basis"])

                candidate_pts = [
                    (align_id, align_roi.get_centroid())
                    for align_id, align_roi in align_by_z.get(z_anchor, [])
                    if align_id not in (id1, id2, anchor_id)
                ]

                if candidate_pts:
                    ids, positions = zip(*candidate_pts)
                    projected_pts, mask = gpu_project_points(np.array(positions), normal, anchor_pos, params["projection_dist_thresh"])

                    for k, keep in enumerate(mask):
                        if keep:
                            plane.plane_points[len(plane.plane_points)] = PlanePoint(ids[k], projected_pts[k])

                anchor_planes.append(plane)
                normals_seen.append(normal)

            return anchor_planes

        self.planes = []
        n_threads = params.get("n_threads", 4)
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(process_anchor, *task) for task in tasks]
            with tqdm(total=len(futures), desc="Generating planes", ncols=80) as pbar:
                for future in futures:
                    self.planes.extend(future.result())
                    pbar.update(1)


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






   
