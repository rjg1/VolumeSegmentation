
import numpy as np
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

class Plane:
    def __init__(self, anchor_id, alignment_ids, anchor_point, alignment_points):
        self.anchor_id = anchor_id
        self.alignment_ids = alignment_ids
        self.anchor_point = np.array(anchor_point)
        self.alignment_points = [np.array(p) for p in alignment_points]

        if len(self.alignment_points) >= 2:
            self.normal, self.d = self._plane_from_points(self.anchor_point, *self.alignment_points[:2])
        else:
            raise ValueError("At least two alignment points are required to define a plane.")

        self.projected_points = []
        self.projected_ids = []
        self.project_points([self.anchor_point, *self.alignment_points], [anchor_id, *alignment_ids], threshold=0.1) # Track all vectors on plane

        # Initialize angles and magnitude lists
        self.angles = []
        self.magnitudes = []
        self.relative_angles = []
        self.angles_and_magnitudes()
        self.get_relative_angles()

        # TODO some equality checking to ensure duplicate planes aren't created

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

    def _build_angular_signal(angle_list, resolution=360, blur_sigma=2.0, radians = True):
        """
        Converts a list of angles (in degrees) into a circular signal vector.

        Parameters:
        - angle_list: list of angles in degrees
        - resolution: number of bins in the circular signal
        - blur_sigma: standard deviation for Gaussian blur (in bins)

        Returns:
        - signal: 1D numpy array representing the circular signal
        """
        if radians: # Convert radians to degrees if necessary
            angle_list = [np.degrees(a) for a in angle_list]
        signal = np.zeros(resolution) # Initialise an empty array with a "bucket" for each degree
        for angle in angle_list:
            idx = int(round(angle % 360)) % resolution # Wrap each angle around 360, rounding to the nearest degree
            signal[idx] += 1 # Add a match to that angle
        return gaussian_filter1d(signal, sigma=blur_sigma, mode='wrap') # Blur values between buckets

    def _match_circular_signals(signal_a, signal_b):
        """
        Performs circular cross-correlation to find the best alignment.

        Parameters:
        - signal_a: 1D numpy array (reference)
        - signal_b: 1D numpy array (to be rotated)

        Returns:
        - best_offset: angle in degrees by which to rotate signal_b to align with signal_a
        - max_score: similarity score at best alignment
        - correlation: full cross-correlation array
        """
        correlation = correlate(signal_a, signal_b, mode='full', method='auto')
        n = len(signal_a) # should be 360 - equivalent to signal b
        mid = len(correlation) // 2 # Index with no shift
        circular_corr = correlation[mid:mid+n] # takes shifts of 0-360 degrees, ignoring shifts of -360->0 degrees which are equivalent
        best_offset = np.argmax(circular_corr) # get index of best score
        max_score = circular_corr[best_offset] # get best score
        return best_offset, max_score, circular_corr
    
    def _fuzzy_bin_match(
        angles_a, mags_a, angles_b, mags_b,
        offset_deg, resolution=360, window=2, radians = True
    ):
        """
        Fast fuzzy bucket-based ROI matcher using binning and moving window.

        Parameters:
        - angles_a_deg, mags_a: Plane A ROI angles (degrees) and magnitudes
        - angles_b_deg, mags_b: Plane B ROI angles (degrees) and magnitudes
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

        # Normalize and rotate Plane B
        angles_b_rot = [(a + offset_deg) % 360 for a in angles_b_deg]

        # Bin ROIs
        bucket_a = defaultdict(list)
        bucket_b = defaultdict(list)

        for i, angle in enumerate(angles_a_deg):
            idx = int(round(angle * resolution / 360)) % resolution
            bucket_a[idx].append(i)

        for j, angle in enumerate(angles_b_rot):
            idx = int(round(angle * resolution / 360)) % resolution
            bucket_b[idx].append(j)

        # Match using fuzzy window
        used_a = set()
        used_b = set()
        matches = []
        angular_diffs = []
        magnitude_diffs = []

        for angle_a in bucket_a: # iterate through all angles in plane A
            for delta in range(-window, window + 1): # match over a certain offset window
                angle_b = (angle_b + delta) % resolution # wrap offset window around 360 degrees if necessary
                for i in bucket_a[angle_a]: # for all ROIs at this offset (assumed 1 in most cases)
                    for j in bucket_b.get(angle_b, []): # for all ROIs at the corresponding offset (assumed 0-1) 
                        if i not in used_a and j not in used_b: # match immediately and break if execution reaches here
                            angle_diff = abs(angles_a_deg[i] - angles_b_rot[j]) % 360 # calculate the angle diff
                            angle_diff = min(angle_diff, 360 - angle_diff) # take the minimum of the two directions in angle
                            mag_diff = abs(mags_a[i] - mags_b[j]) # get the absolute magnitude diff

                            matches.append((i, j))
                            angular_diffs.append(angle_diff)
                            magnitude_diffs.append(mag_diff)

                            used_a.add(i) # mark as matched
                            used_b.add(j) # mark as matched
                            break  # only one match per roi pair

        return matches, angular_diffs, magnitude_diffs


    # Full pipeline for matching planes
    def match_planes(self, angles_b, mags_b, blur_sigma = 2, angle_tolerance = 3, min_matches = 2, angle_mse_threshold = 1.0, magnitude_mse_threshold = 1.0):
        mags_a = self.magnitudes
        angles_a = self.angles

        # Calculate best rotational offset
        signal_a = self._build_angular_signal(angles_a, resolution=360, blur_sigma=blur_sigma)
        signal_b = self._build_angular_signal(angles_b, resolution=360, blur_sigma=blur_sigma)
        offset, score, correlation = self._match_circular_signals(signal_a, signal_b)
        
        # Find angle matches and compute angular error + magnitudal error
        matches, angular_diffs, magnitude_diffs = self._fuzzy_bin_match(angles_a, mags_a, angles_b, mags_b, offset, window = angle_tolerance)

        # Check minimum match count
        if len(matches) < min_matches:
            return {
                "match": False,
                "reason": f"Only {len(matches)} matches found, below minimum {min_matches}.",
                "matches": matches,
                "angular_diffs": angular_diffs,
                "magnitude_diffs": magnitude_diffs,
                "offset": offset,
                "score": score
            }

        # Compute angle MSE
        angle_mse = np.mean(np.square(angular_diffs)) if angular_diffs else float('inf')

        if angle_mse <= angle_mse_threshold:
            return {
                "match": False,
                "reason": "Failed angle MSE threshold",
                "offset": offset,
                "score": score,
                "matches": matches,
                "angular_diffs": angular_diffs,
                "magnitude_diffs": abs_diffs,
                "angle_mse": angle_mse,
                "magnitude_mse": magnitude_mse,
            }
        
        # Compute best-fit scale for magnitudes and MSE
        # Extract the magnitudes of all matched vectors
        matched_a = np.array([mags_a[i] for i, j in matches])
        matched_b = np.array([mags_b[j] for i, j in matches])
        # Solve the linear least squares problem with the closed form solution
        numerator = np.sum(matched_a * matched_b) # Sum of the product of all elements index wise
        denominator = np.sum(matched_b ** 2)
        scale = numerator / denominator if denominator != 0 else 0.0
        scaled_b = matched_b * scale # Scale the vectors to match
        diff = matched_a - scaled_b # Calculate the difference in magnitudes
        magnitude_mse = np.mean(diff ** 2) # Calculate the MSE
        abs_diffs = np.abs(diff).tolist() # Calculate the differences

        # Check thresholds
        match = magnitude_mse <= magnitude_mse_threshold

        return {
            "match": match,
            "reason": "Passed angle and magnitude error thresholds" if match else "Failed magnitude MSE threshold",
            "offset": offset,
            "score": score,
            "matches": matches,
            "angular_diffs": angular_diffs,
            "magnitude_diffs": abs_diffs,
            "angle_mse": angle_mse,
            "magnitude_mse": magnitude_mse,
            "scale_factor": scale
        }


    def angles_and_magnitudes(self):
        angles = []
        magnitudes = []

        # Compute anchor point in plane
        anchor_proj = self._project_point(self.anchor_point)

        # Get first alignment vector for u-axis
        align_proj = self._project_point(self.alignment_points[0])
        u = (align_proj - anchor_proj)
        u = u / np.linalg.norm(u)
        v = np.cross(self.normal, u)

        for i, pt in zip(self.projected_ids, self.projected_points):
            if i == self.anchor_id:
                continue
            vec = pt - anchor_proj
            x = np.dot(vec, u)
            y = np.dot(vec, v)
            angle = np.arctan2(y, x) % (2 * np.pi)
            if np.isclose(angle, 2 * np.pi):
                angle = 0.0
            magnitude = np.linalg.norm(vec)
            angles.append(angle)
            magnitudes.append(magnitude)

        sorted_indices = np.argsort(angles)
        angles = [angles[i] for i in sorted_indices]
        magnitudes = [magnitudes[i] for i in sorted_indices]

        self.angles = angles
        self.magnitudes = magnitudes

        return angles, magnitudes
    
    def angles_and_magnitudes_from_2d(self):
        projected_2d = self.get_local_2d_coordinates()
        angles = []
        magnitudes = []
        ids = []

        for id_, (x, y) in projected_2d.items():
            if id_ == self.anchor_id:
                continue
            angle = np.arctan2(y, x) % (2 * np.pi) 
            if np.isclose(angle, 2 *np.pi):
                angle = 0.0
            magnitude = np.sqrt(x**2 + y**2)
            angles.append(angle)
            magnitudes.append(magnitude)
            ids.append(id_)

        sorted_indices = np.argsort(angles)
        angles = [angles[i] for i in sorted_indices]
        magnitudes = [magnitudes[i] for i in sorted_indices]

        self.angles = angles
        self.magnitudes = magnitudes

        return angles, magnitudes

    def _plane_from_points(self, p1, p2, p3):
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

    def project_points(self, points, ids, threshold=1.0):
        if not hasattr(self, "_projected_dict"):
            self._projected_dict = {}

        for pt, id_ in zip(points, ids):
            if id_ not in self._projected_dict and self._distance_to_plane(pt) <= threshold:
                proj = self._project_point(pt)
                self._projected_dict[id_] = proj

        # Update synced attributes
        self.projected_ids = list(self._projected_dict.keys())
        print(self.projected_ids)
        self.projected_points = np.array([self._projected_dict[i] for i in self.projected_ids])
        print(self.projected_points)
        return self.projected_points, self.projected_ids

    def get_local_2d_coordinates(self):
        if len(self.alignment_points) < 1:
            raise ValueError("Need at least 1 alignment point to define a frame.")
        v1 = self.alignment_points[0] - self.anchor_point
        u = v1 / np.linalg.norm(v1)
        v = np.cross(self.normal, u)

        self.projected_2d = {}
        for pt, id_ in zip(self.projected_points, self.projected_ids):
            vec = pt - self.anchor_point
            x = np.dot(vec, u)
            y = np.dot(vec, v)
            self.projected_2d[id_] = (x, y)
        return self.projected_2d
