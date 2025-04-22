
import numpy as np
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from planepoint import PlanePoint
from numpy.fft import fft, ifft

PLANE_ANCHOR_ID = 0

class Plane:
    def __init__(self, anchor_point, alignment_points, fixed_basis = True):
        self.anchor_point = anchor_point
        self.alignment_points = alignment_points
        self.fixed_basis = fixed_basis
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

        # # TODO some equality checking to ensure duplicate planes aren't created
        # # TODO fix handling of ids that are non integer

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

    def _fuzzy_bin_match(self, angles_a, mags_a, angles_b, mags_b, ids_a, ids_b, 
        offset_deg, resolution=360, window=2, radians = True
    ):
        """
        Fast fuzzy bucket-based ROI matcher using binning and moving window.

        Parameters:
        - angles_a, mags_a: Plane A ROI angles and magnitudes
        - angles_b, mags_b: Plane B ROI angles and magnitudes
        - offset_deg: degrees to rotate Plane B
        - resolution: number of bins (e.g., 360 for 1° resolution)
        - window: angular window size in degrees to allow fuzzy matching

        Returns:
        - matches: list of (i, j) index pairs for matched ROIs
        - angular_diffs: list of angle differences
        - magnitude_diffs: list of magnitude differences
        """
        if radians:
            angles_a_deg = [np.degrees(a) for a in angles_a]
            angles_b_deg = [np.degrees(b) for b in angles_b]
        else:
            angles_a_deg = angles_a
            angles_b_deg = angles_b

        # Exclude anchor ROIs
        ids_a = ids_a[1:]
        ids_b = ids_b[1:]

        # Normalize and rotate Plane B
        angles_b_rot = [(angle + offset_deg) % 360 for angle in angles_b_deg]

        # Bin ROIs
        bucket_a = defaultdict(list) # {angle : [<indices>]}
        bucket_b = defaultdict(list)

        for i, (pida, angle) in enumerate(zip(ids_a, angles_a_deg)):
            idx = int(round(angle * resolution / 360)) % resolution
            bucket_a[idx].append((i, pida))

        for j, (pidb, angle) in enumerate(zip(ids_b, angles_b_rot)):
            idx = int(round(angle * resolution / 360)) % resolution
            bucket_b[idx].append((j, pidb))

        # Match using fuzzy window
        used_a = set()
        used_b = set()
        matches = []
        angular_diffs = []
        magnitude_diffs = []

        for angle_a in bucket_a: # iterate through all angles in plane A
            for delta in range(-window, window + 1): # match over a certain offset window
                angle_b = (angle_a + delta) % resolution # wrap offset window around 360 degrees if necessary
                for i, pida in bucket_a[angle_a]: # for all ROIs at this offset (assumed 1 in most cases)
                    for j, pidb in bucket_b.get(angle_b, []): # for all ROIs at the corresponding offset (assumed 0-1) 
                        if i not in used_a and j not in used_b: # match immediately and break if execution reaches here
                            angle_diff = abs(angles_a_deg[i] - angles_b_rot[j]) % 360 # calculate the angle diff - offszet by 1 as id 1 is first entry in angles list
                            angle_diff = min(angle_diff, 360 - angle_diff) # take the minimum of the two directions in angle
                            mag_diff = abs(mags_a[i] - mags_b[j]) # get the absolute magnitude diff

                            matches.append((pida, pidb)) # append roi ids for matches
                            angular_diffs.append(angle_diff)
                            magnitude_diffs.append(mag_diff)

                            used_a.add(i) # mark as matched
                            used_b.add(j) # mark as matched
                            break  # only one match per roi pair

        return matches, angular_diffs, magnitude_diffs


    # Full pipeline for matching planes
    def match_planes(self, plane_b, blur_sigma = 2, angle_tolerance = 2, min_matches = 2, angle_mse_threshold = 1.0, magnitude_mse_threshold = 1.0, circular_fft = True):
        angles_a, mags_a = self.angles_and_magnitudes()
        angles_b, mags_b = plane_b.angles_and_magnitudes()

        ids_a, ids_b = list(self.plane_points.keys()), list(plane_b.plane_points.keys())

        # Calculate best rotational offset
        signal_a = self._build_angular_signal(angles_a, resolution=360, blur_sigma=blur_sigma)
        signal_b = self._build_angular_signal(angles_b, resolution=360, blur_sigma=blur_sigma)
        if circular_fft:
            offset, score, correlation = self._match_circular_signals_fft(signal_a, signal_b)
        else:
            offset, score, correlation = self._match_circular_signals(signal_a, signal_b)
        
        # Find angle matches and compute angular error + magnitudal error
        matches, angular_diffs, magnitude_diffs = self._fuzzy_bin_match(angles_a, mags_a, angles_b, mags_b, ids_a, ids_b, offset, window = angle_tolerance)

        # Map a points id in the plane_points dictionary to its corresponding index in mags_/angles_ list. Excludes anchor pt.
        id_to_idx_a = {pt_id: idx for idx, pt_id in enumerate(ids_a[1:])}
        id_to_idx_b = {pt_id: idx for idx, pt_id in enumerate(ids_b[1:])}

        # Convert matches to original id matches
        og_matches = []
        for i, j in matches:
            og_matches.append((self.plane_points[i].id, plane_b.plane_points[j].id))

        # Check minimum match count
        if len(matches) < min_matches:
            return {
                "match": False,
                "reason": f"{len(matches)} matches found, below minimum {min_matches}.",
                "og_matches" : og_matches,
                "matches": matches,
                "angular_diffs": angular_diffs,
                "magnitude_diffs": magnitude_diffs,
                "offset": offset,
                "score": score
            }

        # Compute angle MSE
        angle_mse = np.mean(np.square(angular_diffs)) if angular_diffs else float('inf')

        if angle_mse >= angle_mse_threshold:
            return {
                "match": False,
                "reason": "Failed angle MSE threshold",
                "offset": offset,
                "score": score,
                "og_matches" : og_matches,
                "matches": matches,
                "angular_diffs": angular_diffs,
                "magnitude_diffs": magnitude_diffs,
                "angle_mse": angle_mse
            }
        
        # Compute best-fit scale for magnitudes and MSE
        # Extract the magnitudes of all matched vectors
        matched_a = np.array([mags_a[id_to_idx_a[i]] for i, j in matches])
        matched_b = np.array([mags_b[id_to_idx_b[j]] for i, j in matches])
        # Solve the linear least squares problem with the closed form solution
        numerator = np.sum(matched_a * matched_b) # Sum of the product of all elements index wise
        denominator = np.sum(matched_b ** 2)
        scale = numerator / denominator if denominator != 0 else 0.0
        scaled_b = matched_b * scale # Scale the vectors to match
        diff = matched_a - scaled_b # Calculate the difference in magnitudes
        magnitude_mse = np.mean(diff ** 2) # Calculate the MSE
        abs_diffs = np.abs(diff).tolist() # Calculate the differences

        # Check thresholds
        if magnitude_mse > magnitude_mse_threshold:
            return {
                "match": False,
                "reason": "Failed magnitude MSE threshold",
                "offset": offset,
                "score": score,
                "og_matches" : og_matches,
                "matches": matches,
                "angular_diffs": angular_diffs,
                "magnitude_diffs": abs_diffs,
                "angle_mse": angle_mse,
                "magnitude_mse": magnitude_mse,
                "scale_factor": scale
            }
    
        # Check traits
        traits_passed, trait_values, trait_outcomes, mismatched_traits = self._check_traits(matches, plane_b)
        return {
                "match": traits_passed,
                "reason": "Trait missmatch - see outcomes" if not traits_passed else "MSE angle and magnitudes passed. Trait matches all passed",
                "offset": offset,
                "score": score,
                "og_matches" : og_matches,
                "matches": matches,
                "angular_diffs": angular_diffs,
                "magnitude_diffs": abs_diffs,
                "angle_mse": angle_mse,
                "magnitude_mse": magnitude_mse,
                "scale_factor": scale,
                "trait_values": trait_values,
                "trait_outcomes": trait_outcomes,
                "mismatched_traits": mismatched_traits
        }

    def _check_traits(self, matches, plane_b):
        traits_passed = True    # output variable initialisation
        trait_values = {}
        trait_outcomes = {}

        mismatched_traits = set()
        matched_traits = set()
        trait_error_values = {}  # {trait : [<error m1>, <error m2>, ...]}
        trait_metrics = {}
        trait_thresholds = {}
        for match in matches:
            i, j = match
            ppta = self.plane_points[i]
            pptb = plane_b.plane_points[j]

            for trait in ppta.traits:
                if trait not in pptb.traits or trait in mismatched_traits: # unable to gain meaningful information on mismatched traits
                    mismatched_traits.add(trait) # ignore in future
                    continue
                else:
                    matched_traits.add(trait)
                if not trait_error_values.get(trait, None):
                    trait_error_values[trait] = []
                    trait_metrics[trait] = ppta.traits[trait]["metric"] # weird assumption that all points have the same stored metric but should be right
                    trait_thresholds[trait] = ppta.traits[trait]["threshold"] # as above tbh
                
                trait_error_values[trait].append(np.abs(ppta.traits[trait]["value"] - pptb.traits[trait]["value"]))

            # Find any mismatched traits from the matched point on plane B
            mismatched_b_traits = [trait for trait in pptb.traits if trait not in matched_traits]
            mismatched_traits.update(mismatched_b_traits)

        if mismatched_traits.intersection(matched_traits):
            for trait in matched_traits:
                if trait in mismatched_traits:
                    matched_traits.pop(trait)
                    trait_error_values.pop(trait)

        # Calculate error by metric
        for trait in matched_traits:
            metric = trait_metrics[trait]
            errors = trait_error_values[trait]

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
                value = np.max(errors) - np.min(errors)
            elif metric == "std":
                value = np.std(errors)
            else:
                print(f"Warning: Unknown metric '{metric}' for trait '{trait}'")
                continue

            trait_values[trait] = value
            trait_outcomes[trait] = value <= trait_thresholds[trait]
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
        p1, p2, p3 = pp1.position, pp2.position, pp3.position # Extract positions from planepoints
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, p1)
        return normal, d

    def _distance_to_plane(self, point):
        return abs(np.dot(self.normal, point) + self.d) / np.linalg.norm(self.normal)

    def _project_point(self, point):
        point = np.array(point)
        distance = np.dot(self.normal, point) + self.d
        return point - distance * self.normal

    # Expects a list of PlanePoint
    def project_points(self, points, threshold=1.0):
        idx = len(self.plane_points.keys()) # New index to add points
        for ppt in points:
            pt = ppt.position # Extract position from planepoint
            if self._distance_to_plane(pt) <= threshold: # If it is close enough to the plane, project it
                proj = self._project_point(pt)
                self.plane_points[idx] = PlanePoint(ppt.id, proj, traits=ppt.traits) # Update position vector to projected
                idx += 1 # Increase index

    def get_local_2d_coordinates(self):
        if len(self.alignment_points) < 1:
            raise ValueError("Need at least 1 alignment point to define a frame.")
        v1 = self.alignment_points[0].position - self.anchor_point.position
        u = v1 / np.linalg.norm(v1)
        v = np.cross(self.normal, u)

        self.projected_2d = {}
        for idx, ppt in self.plane_points.items():
            pt = ppt.position
            x, y = self._project_point_2d(pt, u=u, v=v)
            self.projected_2d[idx] = (x, y)
        return self.projected_2d
    

    def _project_point_2d(self, pt, u = None, v = None):
        if self.fixed_basis:
                u = np.array([1, 0, 0])
                v = np.cross(self.normal, u)
                v = v / np.linalg.norm(v)
        elif u is None or v is None:
                v1 = self.alignment_points[0].position - self.anchor_point.position
                u = v1 / np.linalg.norm(v1)
                v = np.cross(self.normal, u)
        vec = pt - self.anchor_point.position
        x = np.dot(vec, u)
        y = np.dot(vec, v)
        return x, y

    def get_aligned_2d_projection(self, plane_b, offset_deg, scale_factor):
        # SGet own 2D projection
        proj_a = self.get_local_2d_coordinates()

        # Get B’s local 2D projection
        proj_b = plane_b.get_local_2d_coordinates()

        # Step 3: Build transform
        offset_rad = np.radians(offset_deg)
        R = np.array([[np.cos(offset_rad), -np.sin(offset_rad)],
                    [np.sin(offset_rad),  np.cos(offset_rad)]])
        
        proj_b_transformed = {}
        translation = np.array(proj_a[0]) - np.array(proj_b[0])  # match anchor points

        for idx, (x, y) in proj_b.items():
            vec = np.array([x, y])
            rotated = R @ vec
            scaled = rotated * scale_factor
            translated = scaled + translation
            proj_b_transformed[idx] = tuple(translated)

        return proj_a, proj_b_transformed, {
            "rotation_deg": offset_deg,
            "scale": scale_factor,
            "translation": tuple(translation)
        }

    # Projects a point onto the plane, projects it into 2d, applies transformations.
    # Can be used for any arbitrary point - used for all ROI points after matching before UoI calculation
    def project_and_transform(self, point, rotation_deg=0, scale=1.0, translation=(0, 0)):
        # Project to the plane
        proj = self._project_point(point)

        # Project into 2D local coordinates
        x, y = self._project_point_2d(proj)  # Uses internally computed u, v vectors

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

    # Aligns points of Plane b in 3d
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

        # Compute translation vector
        # transformed_anchor_b = (R_align @ p0b) * scale_factor
        # translation = p0a - transformed_anchor_b
        translation = p0a - p0b

        # Apply transform to all points in Plane B
        aligned_b_points = {}
        for id_b, ppt in plane_b.plane_points.items():
            pt = ppt.position
            aligned = self._apply_3d_transform(
                point=pt,
                rotation_matrix=R_align,
                scale=scale_factor,
                translation=translation,
                project=False
            )
            aligned_b_points[id_b] = aligned

        return aligned_b_points, R_align, translation
    
    # def _apply_3d_transform(self, point, rotation_matrix, scale=1.0, translation=np.zeros(3), project = True):
    #     # Project point onto this plane (if it is not already)
    #     if project:
    #         proj = self._project_point(point)
    #     else:
    #         proj = point
    #     vec = proj - self.anchor_point.position              # Translate into local frame
    #     rotated = rotation_matrix @ vec     # Apply rotation
    #     scaled = rotated * scale            # Apply scaling
    #     transformed_point = scaled + translation
    #     return transformed_point

    def _apply_3d_transform(self, point, rotation_matrix, scale=1.0, translation=np.zeros(3), project = True):
        # Project point onto this plane (if it is not already)
        if project:
            proj = self._project_point(point)
        else:
            proj = point
        
        translated = proj + translation
        rotated = rotation_matrix @ (translated - self.anchor_point.position)
        scaled = rotated * scale
        return scaled