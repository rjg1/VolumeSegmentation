import numpy as np
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from planepoint import PlanePoint
from numpy.fft import fft, ifft
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from param_handling import MATCH_PLANE_PARAM_DEFAULTS, PLANE_LIST_PARAM_DEFAULTS, create_param_dict


PLANE_ANCHOR_ID = 0

class Plane:
    def __init__(self, anchor_point, alignment_points, fixed_basis = True, max_alignments = 10, flat = False):
        self.anchor_point = anchor_point
        self.max_alignments = max_alignments
        self.alignment_points = alignment_points[:self.max_alignments]
        self.fixed_basis = fixed_basis
        self.flat = flat
        if len(self.alignment_points) >= 2: # Make a plane from the anchor point and 2 alignment points
            self.normal, self.d = self._plane_from_points(self.anchor_point, *self.alignment_points[:2])
        else:
            raise ValueError("At least two alignment points are required to define a plane.")

        # Save all points input on this plane
        self.plane_points = {PLANE_ANCHOR_ID : self.anchor_point} # internal point id : PlanePoint
        self.plane_points.update({i + 1 : p for i, p in enumerate(self.alignment_points)})

        # Initialize angles and magnitude lists
        self.angles = []
        self.magnitudes = []
        self.angles_and_magnitudes()

        # Ensure consistent plane direction)
        if self.normal[2] < 0:
            self.normal = -self.normal

    def _make_subplane_from_ids(self, keep_ids):
        """
        Build a new Plane with the same anchor & normal, but only a subset of alignments.
        The anchor is preserved as index 0; the rest are re-indexed 1..N in insertion order.
        """
        anchor_pp = self.plane_points[0]
        new_align_pps = []
        # preserve order stable by sorting the selected keys except 0
        for k in sorted(keep_ids):
            if k == 0: 
                continue
            new_align_pps.append(self.plane_points[k])

        # Make a new Plane with identical max_alignments/fixed_basis
        sub = Plane(
            anchor_pp,
            new_align_pps,
            max_alignments=self.max_alignments,
            fixed_basis=self.fixed_basis,
        )
        return sub

    def _plane_align_ids(self):
        """Return list of alignment point keys (exclude anchor index 0)."""
        return [k for k in self.plane_points.keys() if k != 0]

    def get_normalised_magnitudes(self, recompute = True):
        if recompute:
            self.angles_and_magnitudes()
        return self._normalize(self.magnitudes) 

    def _normalize(self, magnitudes):
        min_val = min(magnitudes)
        max_val = max(magnitudes)
        return [(i - min_val) / (max_val - min_val) for i in magnitudes]

    # Builds a list of angles into a set of bins [0,...359] for each degree.
    # Gaussian blurs points into nearby bins for partial matches assuming angles are off
    def _build_angular_signal(self, angle_list, resolution=360, blur_sigma=2.0, radians=True):
        if radians:
            angle_list = [np.degrees(a) for a in angle_list]

        signal = np.zeros(resolution)
        for angle in angle_list:
            if not np.isfinite(angle):  # skips NaN and inf
                print(f"[WARN] Skipping invalid angle: {angle}")
                continue
            idx = int(round(angle % 360)) % resolution
            signal[idx] += 1

        if blur_sigma > 0:
            return gaussian_filter1d(signal, sigma=blur_sigma, mode='wrap')
        else:
            return signal
        
    def _match_circular_signals(self, signal_a, signal_b):
        """
        Proper circular cross-correlation by rotating signal_b through all shifts.

        Returns:
        - best_offset: angle in degrees by which to rotate signal_b to align with signal_a
        - max_score: similarity score at best alignment
        - correlation: list of scores for all shifts
        """
        n = len(signal_a)
        scores = []
        for shift in range(n):
            rolled_b = np.roll(signal_b, shift)
            score = np.dot(signal_a, rolled_b)
            scores.append(score)

        best_offset = np.argmax(scores)
        max_score = scores[best_offset]
        return best_offset, max_score, np.array(scores)

    def _match_circular_signals_fft(self, signal_a, signal_b):
        n = len(signal_a)
        corr = ifft(fft(signal_a) * np.conj(fft(signal_b))).real
        best_offset = np.argmax(corr)
        max_score = corr[best_offset]
        return best_offset, max_score, corr

    def _fuzzy_bin_match(
        self, angles_a, mags_a, angles_b, mags_b, ids_a, ids_b, offset_deg, 
        resolution=360, angle_tolerance=2, radians=True,
        min_matches=2, outlier_thresh=2.0, mse_threshold=1e-3, fixed_scale=None
    ):
        """
        Full fuzzy matcher with 1:1 matching, scaling, outlier removal, early termination,
        and returns angular and magnitude differences per match.

        Returns:
        - best_matches: list of (pida, pidb) ROI id pairs
        - best_scaling: fitted scaling factor
        - best_mse: final mean squared error
        - angular_diffs: per-match angle differences
        - magnitude_diffs: per-match scaled magnitude differences
        """
        if radians:
            angles_a_deg = np.degrees(angles_a)
            angles_b_deg = np.degrees(angles_b)
        else:
            angles_a_deg = angles_a
            angles_b_deg = angles_b

        ids_a = ids_a[1:]  # skip anchor
        ids_b = ids_b[1:]

        angles_b_rot = [(angle + offset_deg) % 360 for angle in angles_b_deg]

        # Bin ROIs
        bucket_a = defaultdict(list)
        bucket_b = defaultdict(list)

        for i, (pida, angle) in enumerate(zip(ids_a, angles_a_deg)):
            idx = int(round(angle * resolution / 360)) % resolution
            bucket_a[idx].append((i, pida))

        for j, (pidb, angle) in enumerate(zip(ids_b, angles_b_rot)):
            idx = int(round(angle * resolution / 360)) % resolution
            bucket_b[idx].append((j, pidb))

        # Build candidate matches
        candidates = []
        for angle_a in bucket_a:
            for delta in range(-angle_tolerance, angle_tolerance + 1):
                angle_b = (angle_a + delta) % resolution
                for i, pida in bucket_a[angle_a]:
                    for j, pidb in bucket_b.get(angle_b, []):
                        candidates.append((i, j, pida, pidb))

        if not candidates:
            return [], None, None, [], []

        # Build options
        options = defaultdict(list)
        for i, j, pida, pidb in candidates:
            options[i].append((j, pida, pidb))

        i_list = sorted(options.keys())

        match_sets = []

        def backtrack(current_set, used_is, used_js, candidates_left):
            if len(candidates_left) == 0 and len(current_set) >= min_matches:
                match_sets.append(list(current_set))  # save any valid partial match set

            if not candidates_left:
                return

            for idx, (i, j, pida, pidb) in enumerate(candidates_left):
                if i not in used_is and j not in used_js:
                    current_set.append((i, j, pida, pidb))
                    used_is.add(i)
                    used_js.add(j)

                    backtrack(
                        current_set,
                        used_is,
                        used_js,
                        candidates_left[idx+1:]  # remaining candidates for this match
                    )

                    used_is.remove(i)
                    used_js.remove(j)
                    current_set.pop()

        # Start backtracking to make match sets
        backtrack([], set(), set(), candidates)

        if not match_sets:
            return [], None, None, [], []

        best_mse = np.inf
        best_matches = []
        best_scaling = None
        best_angular_diffs = []
        best_magnitude_diffs = []

        for match_set in match_sets:
            mags_a_sel = np.array([mags_a[i] for (i, j, pida, pidb) in match_set])
            mags_b_sel = np.array([mags_b[j] for (i, j, pida, pidb) in match_set])

            angles_a_sel = np.array([angles_a_deg[i] for (i, j, pida, pidb) in match_set])
            angles_b_sel = np.array([angles_b_rot[j] for (i, j, pida, pidb) in match_set])

            # Initial scaling
            numerator = np.sum(mags_a_sel * mags_b_sel)
            denominator = np.sum(mags_b_sel ** 2)
            if denominator == 0:
                continue
            if fixed_scale:
                s = fixed_scale
            else:  
                s = numerator / denominator

            residuals = mags_a_sel - s * mags_b_sel
            std = np.std(residuals)

            # Outlier removal
            mask = np.abs(residuals) <= outlier_thresh * std
            if np.sum(mask) < min_matches:
                continue

            mags_a_final = mags_a_sel[mask]
            mags_b_final = mags_b_sel[mask]
            angles_a_final = angles_a_sel[mask]
            angles_b_final = angles_b_sel[mask]
            matches_final = [match for match, keep in zip(match_set, mask) if keep]

            # Refit scaling after outlier removal
            numerator = np.sum(mags_a_final * mags_b_final)
            denominator = np.sum(mags_b_final ** 2)
            if denominator == 0:
                continue

            if fixed_scale:
                s_final = fixed_scale
            else:  
                s_final = numerator / denominator

            final_residuals = mags_a_final - s_final * mags_b_final
            mse = np.mean(final_residuals**2)

            # Compute angular differences
            angular_diffs = []
            for a1, a2 in zip(angles_a_final, angles_b_final):
                diff = abs(a1 - a2) % 360
                angular_diffs.append(min(diff, 360 - diff))

            # Compute magnitude differences after scaling
            magnitude_diffs = np.abs(mags_a_final - s_final * mags_b_final)

            if mse < best_mse:
                best_mse = mse
                best_matches = [(pida, pidb) for (_, _, pida, pidb) in matches_final]
                best_scaling = s_final
                best_angular_diffs = angular_diffs
                best_magnitude_diffs = magnitude_diffs.tolist()

                if best_mse <= mse_threshold:
                    break  # early exit

        if not best_matches:
            return [], None, None, [], []

        return best_matches, best_scaling, best_mse, best_angular_diffs, best_magnitude_diffs


    # Full pipeline for matching planes
    def match_planes(self, plane_b, match_plane_params = None):
        params = create_param_dict(MATCH_PLANE_PARAM_DEFAULTS, match_plane_params)

        angles_a, mags_a = self.angles_and_magnitudes()
        angles_b, mags_b = plane_b.angles_and_magnitudes()

        ids_a, ids_b = list(self.plane_points.keys()), list(plane_b.plane_points.keys())


        # TEST DEBUG
        clean_data = [
            (a, m, id_a, id_b)
            for a, m, id_a, id_b in zip(angles_a, mags_a, ids_a, ids_b)
            if np.isfinite(a) and np.isfinite(m)
        ]

        if len(clean_data) == 0:
            return {"match":False}, None

        angles_a, mags_a, ids_a, ids_b = zip(*clean_data)
        # END TEST DEBUG


        # Calculate best rotational offset
        signal_a = self._build_angular_signal(angles_a, **params["angle_match_params"])
        signal_b = self._build_angular_signal(angles_b, **params["angle_match_params"])



        if params["circular_fft"]:
            offset, ang_score, correlation = self._match_circular_signals_fft(signal_a, signal_b)
        else:
            offset, ang_score, correlation = self._match_circular_signals(signal_a, signal_b)
        
        # Find angle matches and compute angular error + magnitudal error
        matches, scale, magnitude_mse, angular_diffs, magnitude_diffs = self._fuzzy_bin_match(angles_a, mags_a, angles_b, mags_b, ids_a, ids_b, offset, 
                                                                                              **params["bin_match_params"])

        # Convert matches to original id matches
        og_matches = []
        for i, j in matches:
            og_matches.append((self.plane_points[i].id, plane_b.plane_points[j].id))

        # debug
        # Same OG ids and same z-plane makes us expect a perfect match and rotation
        explain = all(id1 == id2 for (id1, id2) in og_matches) and (self.anchor_point.position[2] == plane_b.anchor_point.position[2]) \
                and (self.anchor_point.id == plane_b.anchor_point.id)
        debugOut = {}
        if explain:
            debugOut["og_matches"] = og_matches
            debugOut["A_anch"] = self.anchor_point.position
            debugOut["B_anch"] = plane_b.anchor_point.position
            debugOut["A_points"] = [(self.plane_points[idx].id, self.plane_points[idx].position) for idx in list(self.plane_points.keys())]
            debugOut["B_points"] = [(plane_b.plane_points[idx].id, plane_b.plane_points[idx].position) for idx in list(self.plane_points.keys())]
            debugOut["offset"] = offset
            debugOut["magnitude_mse"] = magnitude_mse
            debugOut["magnitude_diffs"] = magnitude_diffs
            debugOut["angular_diffs"] = angular_diffs
        # end debug

        # Check minimum match count
        if len(matches) < params["bin_match_params"]["min_matches"]:
            return {
                "match": False,
                "reason": f"{len(matches)} matches found, below minimum {params['bin_match_params']['min_matches']}.",
                "og_matches" : og_matches,
                "matches": matches,
                "angular_diffs": angular_diffs,
                "magnitude_diffs": magnitude_diffs,
                "offset": offset,
                "ang_cc_score": ang_score
            }, None

        # Check traits
        traits_passed, trait_values, trait_outcomes, mismatched_traits = self._check_traits(
            matches, plane_b, params["traits"], offset=offset, scale=scale
        )

        if explain:
            debugOut["trait_values"] = trait_values
            debugOut["trait_outcomes"] = trait_outcomes
            debugOut["traits_passed"] = traits_passed

        score = self._compute_trait_score(trait_values, params["traits"], explain=explain, debugOut = debugOut)
        
        if explain:
            return {
                    "match": traits_passed, # True unless a trait exceeded a specified terminate_after threshold
                    "score" : score, # Final score match between planes
                    "reason": "Trait mismatch - see outcomes" if not traits_passed else "Trait matches all passed",
                    "offset": offset,
                    "ang_cc_score": ang_score,
                    "og_matches" : og_matches,
                    "matches": matches,
                    "scale_factor": scale,
                    "trait_values": trait_values,
                    "magnitude_diffs": magnitude_diffs,
                    "trait_outcomes": trait_outcomes,
                    "mismatched_traits": mismatched_traits
            }, debugOut
        else:
            return {
                    "match": traits_passed, # True unless a trait exceeded a specified terminate_after threshold
                    "score" : score, # Final score match between planes
                    "reason": "Trait mismatch - see outcomes" if not traits_passed else "Trait matches all passed",
                    "offset": offset,
                    "ang_cc_score": ang_score,
                    "og_matches" : og_matches,
                    "matches": matches,
                    "scale_factor": scale,
                    "trait_values": trait_values,
                    "magnitude_diffs": magnitude_diffs,
                    "trait_outcomes": trait_outcomes,
                    "mismatched_traits": mismatched_traits
            }, None

    def _compute_trait_score(self, trait_values, trait_params, explain = False, debugOut = None):
        """
        Given trait error values and per-trait param dict, compute normalized and weighted total score.
        """
        total_score = 0.0
        total_weight = 0.0

        score_contributions = {}

        for trait, value in trait_values.items():
            max_val = trait_params.get(trait, {}).get("max_value", 1.0)
            weight = trait_params.get(trait, {}).get("weight", 0.0)

            if max_val == 0:
                continue  # avoid divide-by-zero

            norm_score = 1.0 - (value / max_val)
            norm_score = max(0.0, min(1.0, norm_score))  # clamp to [0, 1]

            total_score += norm_score * weight
            total_weight += weight

            # Debug output to work out the contribution of each score
            score_contributions[trait] = norm_score * weight

        if total_weight == 0:
            return 0.0

        if explain:
            if debugOut:
                debugOut["score_contributions"] = score_contributions
            else:
                print(score_contributions)

        return total_score / total_weight


    def _check_traits(self, matches, plane_b, trait_param_dict, offset=None, scale=None):
        traits_passed = True
        trait_values = {}
        trait_outcomes = {}
        trait_error_values = {}  # {trait : [<error1>, <error2>, ...]}
        mismatched_traits = set()
        matched_traits = set()

        for i, j in matches:
            ppta = self.plane_points[i]
            pptb = plane_b.plane_points[j]

            for trait in ppta.traits:
                if trait not in pptb.traits or trait in mismatched_traits:
                    mismatched_traits.add(trait)
                    continue
                matched_traits.add(trait)

                # Init error list for trait
                if trait not in trait_error_values:
                    trait_error_values[trait] = []

               # Special-case adjusted angle/magnitude values for plane B
                if trait == "angle" and offset is not None:
                    b_val = (pptb.traits[trait] + np.radians(offset)) % (2 * np.pi) # Adjust it by the offset (degrees -> radians)
                elif trait == "magnitude" and scale is not None:
                    b_val = pptb.traits[trait] * scale # Rescale
                else:
                    b_val = pptb.traits[trait]

                a_val = ppta.traits[trait]
                err = np.abs(a_val - b_val)
                trait_error_values[trait].append(err)

            # Catch any traits present only in pptb
            for trait in pptb.traits:
                if trait not in matched_traits:
                    mismatched_traits.add(trait)

        # Filter out invalid matched traits
        matched_traits -= mismatched_traits

        # Compute aggregate errors
        for trait in matched_traits:
            errors = trait_error_values[trait]
            trait_params = trait_param_dict.get(trait, {})
            metric = trait_params.get("metric", "mse")
            threshold = trait_params.get("terminate_after", float("inf"))


            if metric == "mse":
                value = np.mean(np.square(errors))
            elif metric == "rmse":
                value = np.sqrt(np.mean(np.square(errors)))
            elif metric == "mean":
                value = np.mean(errors)
            elif metric == "max":
                value = np.max(errors)
            elif metric == "sum":
                value = np.sum(errors)
            elif metric == "range":
                value = np.ptp(errors)
            elif metric == "std":
                value = np.std(errors)
            else:
                print(f"Warning: Unknown metric '{metric}' for trait '{trait}'")
                continue

            trait_values[trait] = value
            trait_outcomes[trait] = value <= threshold
            if not trait_outcomes[trait]:
                traits_passed = False

        return traits_passed, trait_values, trait_outcomes, mismatched_traits

    def angles_and_magnitudes(self):
        angle_mag_data = []

        for idx, ppt in self.plane_points.items():
            if idx == PLANE_ANCHOR_ID:
                continue

            pt = ppt.position
            proj = self._project_point(pt)
            x, y = self._project_point_2d(proj)
            angle = np.arctan2(y, x) % (2 * np.pi)
            if np.isclose(angle, 2 * np.pi):
                angle = 0.0
            magnitude = np.sqrt(x**2 + y**2)

            # Add traits
            ppt.add_trait("angle", angle)
            ppt.add_trait("magnitude", magnitude)

            angle_mag_data.append((angle, magnitude, idx, ppt))

        # Sort by angle
        angle_mag_data.sort(key=lambda x: x[0])

        # Store results
        self.angles = [a for a, _, _, _ in angle_mag_data]
        self.magnitudes = [m for _, m, _, _ in angle_mag_data]
        sorted_ids = [i for _, _, i, _ in angle_mag_data]
        sorted_ppts = [p for _, _, _, p in angle_mag_data]

        # Insert anchor at front
        sorted_ids.insert(0, PLANE_ANCHOR_ID)
        sorted_ppts.insert(0, self.anchor_point)

        self.plane_points = {sorted_ids[i]: sorted_ppts[i] for i in range(len(sorted_ids))}

        return self.angles, self.magnitudes


    def _plane_from_points(self, pp1, pp2, pp3):
        if self.flat:
            normal = np.array[0,0,1]
            d = -np.dot(normal,pp1.position)
            return normal, d
        p1, p2, p3 = pp1.position, pp2.position, pp3.position # Extract positions from planepoints
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            # Degenerate case: points are colinear or identical
            print(f"[Warning] Degenerate plane: points {pp1.id}, {pp2.id}, {pp3.id} are colinear or identical.")
            normal = np.array([0.0, 0.0, 1.0])  # Default fallback normal
        else:
            normal = normal / norm
        d = -np.dot(normal, p1)
        return normal, d

    def _distance_to_plane(self, point):
        return abs(np.dot(self.normal, point) + self.d) / np.linalg.norm(self.normal)

    def _project_point(self, point):
        if self.flat:
            return point
        point = np.array(point)
        distance = np.dot(self.normal, point) + self.d
        return point - distance * self.normal

    # Expects a list of PlanePoint
    def project_points(self, points, threshold=1.0):
        point_added = False
        idx = len(self.plane_points.keys()) # New index to add points
        for ppt in points:
            if len(self.alignment_points) >= self.max_alignments: # Do not allow any more alignment points
                break
            pt = ppt.position # Extract position from planepoint
            if self._distance_to_plane(pt) <= threshold: # If it is close enough to the plane, project it
                proj = self._project_point(pt)
                new_point = PlanePoint(ppt.id, proj, traits=ppt.traits) # Update position vector to projected
                self.plane_points[idx] = new_point
                self.alignment_points.append(new_point)
                idx += 1 # Increase index
                point_added = True
        # Re-calculate angles list
        if point_added:
            self.angles_and_magnitudes()

    def get_local_2d_coordinates(self):
        if len(self.alignment_points) < 1:
            raise ValueError("Need at least 1 alignment point to define a frame.")
        u, v = self.compute_local_axes()

        self.projected_2d = {}
        for idx, ppt in self.plane_points.items():
            pt = ppt.position
            if self.flat:
                x, y = ppt.position[0], ppt.position[1]
            else: 
                x, y = self._project_point_2d(pt, u=u, v=v)
            self.projected_2d[idx] = (x, y)
        return self.projected_2d
    

    def _project_point_2d(self, pt, u = None, v = None):
        if self.fixed_basis or u is None or v is None:
                u, v = self.compute_local_axes()

        vec = pt - self.anchor_point.position
        x = np.dot(vec, u)
        y = np.dot(vec, v)
        return x, y

    def get_aligned_2d_projection(self, plane_b, offset_deg, scale_factor):
        # Get own 2D projection
        proj_a = self.get_local_2d_coordinates()

        # Get plane B local 2D projection
        proj_b = plane_b.get_local_2d_coordinates()

        # Step 3: Build transform
        offset_rad = np.radians(offset_deg)
        R = np.array([[np.cos(offset_rad), -np.sin(offset_rad)],
                    [np.sin(offset_rad),  np.cos(offset_rad)]])
        
        proj_b_transformed = {}
        # translation = np.array(proj_a[0]) - np.array(proj_b[0])  # match anchor points - incorrect will always evaluate to 0
        
        # Global anchor positions (in x,y)
        anchorA = np.array(self.anchor_point.position[:2])
        anchorB = np.array(plane_b.anchor_point.position[:2])

        # True global translation after applying rotation/scale to plane B
        translation = anchorA - (R @ (scale_factor * anchorB))

        for idx, (x, y) in proj_b.items():
            vec = np.array([x, y])
            rotated = R @ vec
            scaled = rotated * scale_factor
            translated = scaled  #+ translation # translation is 0 in local coords
            proj_b_transformed[idx] = tuple(translated)

        return proj_a, proj_b_transformed, {
            "rotation_deg": offset_deg,
            "scale": scale_factor,
            "translation": tuple(translation)
        }


    # Projects a point onto the plane, projects it into 2d, applies transformations.
    # Can be used for any arbitrary point - used for all ROI points after matching before UoI calculation
    def _project_and_transform(self, point, plane_b, rotation_deg=0, scale=1.0, translation=(0, 0), project = True):
        if project:
            # Project to the plane
            proj = plane_b._project_point(point)
        else:
            proj = point

        # Project into 2D local coordinates of its own plane
        x, y = plane_b._project_point_2d(proj)  # Uses internally computed u, v vectors

        # Apply 2D rotation
        theta = np.radians(rotation_deg)
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)

        # Apply scaling
        x_scaled = x_rot * scale
        y_scaled = y_rot * scale

        # Apply translation
        x_final = x_scaled + translation[0]
        y_final = y_scaled + translation[1]

        return (x_final, y_final)

    # Transforms multiple points by projecting into a 2D plane and applying rotation/scale/translation
    def project_and_transform_points(self, points, plane_b, rotation_deg=0, scale=1.0, translation=(0, 0), project = True):
        return [self._project_and_transform(point, plane_b, rotation_deg=rotation_deg, scale=scale, translation=translation, project = project) for point in points]

    # Align points of B in 3D
    def get_aligned_3d_projection(self, plane_b, matches, scale_factor):
        # Extract anchors
        p0a = self.anchor_point.position
        p0b = plane_b.anchor_point.position

        # Extract matching vectors relative to anchors
        alignment_matches = [
            (i, j) for i, j in matches
            if i != self.anchor_point.id and j != plane_b.anchor_point.id
        ]

        Va = np.stack([self.plane_points[i].position - p0a for i, j in alignment_matches])
        Vb = np.stack([plane_b.plane_points[j].position - p0b for i, j in alignment_matches])

        # Compute optimal rotation (Kabsch algorithm)
        H = Vb.T @ Va
        U, _, Vt = np.linalg.svd(H)
        R_align = Vt.T @ U.T

        if np.linalg.det(R_align) < 0:
            Vt[2, :] *= -1
            R_align = Vt.T @ U.T

        # Compute global translation: where Plane B's anchor should end up (to align to A)
        translation = p0a

        # Apply transformation to each point on Plane B
        aligned_b_points = {}
        for id_b, ppt in plane_b.plane_points.items():
            pt = ppt.position
            aligned = self._apply_3d_transform(
                point=pt,
                rotation_matrix=R_align,
                scale=scale_factor,
                translation=translation,
                b_anchor_point=p0b,
                project=False
            )
            aligned_b_points[id_b] = aligned

        return aligned_b_points, R_align, translation

    def _apply_3d_transform(self, point, rotation_matrix, scale=1.0, translation=np.zeros(3), b_anchor_point=None, project=True):
        # Optionally project the point onto the plane
        if project:
            proj = self._project_point(point)
        else:
            proj = point

        if b_anchor_point is None:
            raise ValueError("You must provide the anchor_point used as origin during transformation.")

        # Convert point into local frame of anchor of plane B
        vec = proj - b_anchor_point

        # Rotate, scale, and translate into target frame
        rotated = rotation_matrix @ vec
        scaled = rotated * scale
        transformed = scaled + translation

        return transformed
    
    # Transforms a list of points in 3D
    def transform_points_3d(self, points, rotation_matrix, b_anchor_point, scale=1.0, translation = np.zeros(3), project = True):
        return [self._apply_3d_transform(point, rotation_matrix, scale=scale, translation=translation, b_anchor_point=b_anchor_point, project=project) for point in points]
    
    
    def is_equivalent_to(self, other_plane, comparison_pts = 3, angle_threshold_deg=1.0, distance_threshold=0.5, match_anchors = False):
        """
        Fast approximate check if two planes are geometrically equal by:
        - Comparing the angle between their normal vectors.
        - Measuring perpendicular distance of some points from self to other_plane.

        Parameters:
            comparison_pts (int) : max number of points to compare
            angle_threshold_deg (float): max allowed angular difference between normals.
            distance_threshold (float): max allowed point-to-plane distance.
        """
        # Check anchor points are different first - assuming anchor points must match for equivalence
        if match_anchors:
            anchor_a = np.array(self.anchor_point.position)
            anchor_b = np.array(other_plane.anchor_point.position)
            if not np.allclose(anchor_a, anchor_b, atol=distance_threshold):
                return False


        # Normalize both normals
        n1 = self.normal / np.linalg.norm(self.normal)
        n2 = other_plane.normal / np.linalg.norm(other_plane.normal)

        # Check angle between normals
        dot_product = np.dot(n1, n2)
        angle_rad = np.arccos(np.clip(abs(dot_product), -1.0, 1.0))  # take abs for flipped normals
        angle_deg = np.degrees(angle_rad)
        if angle_deg > angle_threshold_deg:
            return False

        # Check perpendicular distances of 3 sampled points from self to other_plane
        sampled_points = [self.anchor_point.position] # always sample anchor
        alignments = list(self.alignment_points)
        sampled_points += [pt.position for pt in alignments[:comparison_pts-1]] if len(alignments) >= comparison_pts-1 else [pt.position for pt in alignments]

        # Plane B equation: n·(x - p0) = 0 → distance = |n·(x - p0)|
        p0 = other_plane.anchor_point.position
        n = other_plane.normal / np.linalg.norm(other_plane.normal)

        for pt in sampled_points:
            vec = pt - p0
            dist = np.abs(np.dot(n, vec))
            if dist > distance_threshold:
                return False


        return True
    
    def compute_local_axes(self):
        """
        Compute the local 2D coordinate axes (u, v) for the plane.

        Returns:
            (u, v): A tuple of two orthonormal 3D vectors defining the local plane axes.
        """
        if self.fixed_basis:
            z_axis = np.array([0.0, 0.0, 1.0])
            if np.allclose(self.normal, z_axis, atol=1e-2) or np.allclose(self.normal, -z_axis, atol=1e-2):
                # Flat plane case: preserve x and y directly
                u = np.array([1.0, 0.0, 0.0])
                v = np.array([0.0, 1.0, 0.0])
            else:
                u = np.array([1.0, 0.0, 0.0])
                v = np.cross(self.normal, u)
                v = v / np.linalg.norm(v)
        else:
            v1 = self.alignment_points[0].position - self.anchor_point.position
            u = v1 / np.linalg.norm(v1)
            v = np.cross(self.normal, u)
            v = v / np.linalg.norm(v)
        return u, v


    @staticmethod
    def match_plane_lists(planes_a, planes_b, 
                          plane_list_params = None,
                          match_plane_params = None):
        """
        Matches planes from planes_a to planes_b based on a composite similarity score.
        Each score is calculated as a weighted combination of angle, magnitude, and trait similarities.

        Parameters:
            planes_a, planes_b (list of Plane): Lists of planes to match.

        Returns:
            List of matched tuples (idx_a, idx_b, score, match_result_dict)
        """
        params = create_param_dict(PLANE_LIST_PARAM_DEFAULTS, plane_list_params) # Params for this function
        match_params = create_param_dict(MATCH_PLANE_PARAM_DEFAULTS, match_plane_params) # Params for match_planes function
    
        # Remove planes not in the specified z range
        planes_a = filter_planes_by_z(planes_a, params["z_guess_a"], params["z_range"])
        planes_b = filter_planes_by_z(planes_b, params["z_guess_b"], params["z_range"])

        results = {}
        max_matches_observed = 0

        outer = tqdm(planes_a, desc="Planes A", position=0)
        inner = tqdm(total=len(planes_b), position=1, leave=False, desc="Planes B")

        # Debug
        debugList = []

        for i, plane_a in enumerate(outer):
            inner.set_description(f"Planes B (A[{i}])")
            inner.reset() # Reset plane B progress bar
            for j, plane_b in enumerate(planes_b):
                match_result, debugOut = plane_a.match_planes(plane_b, 
                                                    match_plane_params = match_params)
                if debugOut:
                    debugList.append(debugOut)

                # Set if early termination specified for specific traits, or if less than min_matches between planes
                if not match_result["match"]:
                    continue

                score = match_result["score"]

                if score >= params["min_score"]:
                    results[score] = {
                        "plane_a": plane_a,
                        "plane_b": plane_b,
                        "result": match_result,
                        "plane_a_idx": i,
                        "plane_b_idx": j
                    }

                    num_matches = len(match_result['matches'])
                    if num_matches > max_matches_observed:
                        max_matches_observed = num_matches

                inner.update(1)

        # Re-scale scores based on matches
        scaled_results = {}
        min_matches = match_plane_params["bin_match_params"]["min_matches"]
        max_matches = min(params["max_matches"], max_matches_observed) # clamp to largest number of matches observed
        min_modifier = params["min_score_modifier"]
        max_modifier = params["max_score_modifier"]

        print(f"Max observed matches: {max_matches_observed}, max matches used: {max_matches}")

        if max_matches > min_matches:
            for score, outcome_dict in results.items():
                num_matches = len(outcome_dict['result']['matches'])
                # Clamp match count between min and max
                clamped_matches = max(min(num_matches, max_matches), min_matches)
                
                # Linearly interpolate modifier
                alpha = (max_modifier - min_modifier) / (max_matches - min_matches) # e.g 0.2 / 3 -> number to step up in score modifier per match over min matches
                alpha_steps = clamped_matches - min_matches # Number of matches over min matches
                score_modifier = min_modifier + (alpha * alpha_steps)
                score_adjusted = score * score_modifier
                
                if score_adjusted >= params["min_score"]:
                    scaled_results[score_adjusted] = outcome_dict
        else:
            print(f"no scaling done")
            scaled_results = results # no scaling required


        # write out debug lost
        import json
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return json.JSONEncoder.default(self, obj)
        outfile = "debug.json"
        out = ''
        # for item in debugList:
        #     print(item)
        #     out += json.dump(item, cls=NumpyEncoder) + ",\n"
        with open(outfile, "w") as file:
            json.dump(debugList, file, indent = 4, cls=NumpyEncoder)
        # Re-make dict with scores sorted in ascending order
        return dict(sorted(scaled_results.items(), key=lambda item: item[0], reverse=True))


    def plot_plane_2d_projection(self, show=True, save_path=None):
        """
        Plots the 2D projection of a plane (anchor + alignment points).
        Anchor is plotted in red, alignments in blue.
        """
        # Get 2D projections of all points
        projected = self.get_local_2d_coordinates()  # returns {id: (x, y)}

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("2D Projection of Plane")

        for pid, (x, y) in projected.items():
            if pid == 0:
                ax.scatter(x, y, color='red', s=60, label='Anchor')
                ax.text(x + 0.2, y + 0.2, f"A{self.plane_points[pid].id}", color='red')
            else:
                ax.scatter(x, y, color='blue', s=40)
                ax.text(x + 0.2, y + 0.2, f"P{self.plane_points[pid].id}", color='blue')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()


# Helper function to filter out a list of planes by a z guess. A plane must have at least one anchor or alignment point
# p such that z_guess - z_range <= p.z <= z_guess + z_range
def filter_planes_by_z(planes, z_guess, z_range):
    if z_guess == -1:
        return planes  # no filtering
    z_min = z_guess - z_range
    z_max = z_guess + z_range
    filtered = []

    for plane in planes:
        all_points = [p.position[2] for p in plane.plane_points.values()]
        if any(z_min <= z <= z_max for z in all_points):
            filtered.append(plane)

    return filtered
