
import numpy as np
from itertools import combinations, product

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
        # TODO some means to prevent planes at certain angles from being created

    def get_normalised_magnitudes(self, recompute = True):
        if recompute:
            self.angles_and_magnitudes()
        return self._normalize(self.magnitudes) 

    def _normalize(self, magnitudes):
        min_val = min(magnitudes)
        max_val = max(magnitudes)
        return [(i - min_val) / (max_val - min_val) for i in magnitudes]

    # TODO restructure to make it easy for subset/supersets of alignment points to be matched
    # Returns a list of angles of alignment points relative to each other. First entry is angle between 
    # first alignment point and anchor point
    def get_relative_angles(self, recompute = True):
        if recompute:
            self.angles_and_magnitudes()
        self.relative_angles = []
        for i, angle in enumerate(self.angles):
            if i == 0:
                self.relative_angles.append(angle)
            else:
                self.relative_angles.append(angle - self.angles[i - 1])
        # Append the distance between the last alignment point and the first alignment point
        distance = 2*np.pi - self.angles[len(self.angles) - 1] + self.angles[0]
        self.relative_angles.append(distance)
        return self.relative_angles


    def match_relative_angle_arcs(rel_a, rel_b, max_error=0.15, min_k=2):
        """
        Matches arc segments between two circular relative angle lists.
        Only allows arc segments formed by accumulating sequential gaps (preserving order).
        
        Parameters:
        - rel_a: List of relative angles (radians) for Plane A
        - rel_b: List of relative angles (radians) for Plane B
        - max_error: Maximum allowed average arc difference
        - min_k: Minimum number of ROIs (implying at least min_k-1 arcs)

        Returns:
        - best_error: Minimum average error between matched arc segments
        - best_match: ((start_a, end_a), (start_b, end_b)) indicating matched segments
        """
        def arc_lengths(rel_angles):
            n = len(rel_angles)
            arcs = []
            for i in range(n):
                for j in range(i + min_k - 1, n + i):  # ensure at least min_k points
                    idx_range = [(k % n) for k in range(i, j + 1)]
                    arc_len = sum(rel_angles[k % n] for k in range(i, j))
                    arcs.append((arc_len, (idx_range[0], idx_range[-1])))
            return arcs

        arcs_a = arc_lengths(rel_a)
        arcs_b = arc_lengths(rel_b)


        best_error = float('inf')
        best_match = None

        for arc_len_a, (start_a, end_a) in arcs_a:
            for arc_len_b, (start_b, end_b) in arcs_b:
                error = abs(arc_len_a - arc_len_b)
                if error < best_error and error <= max_error:
                    best_error = error
                    best_match = ((start_a, end_a), (start_b, end_b))

        return best_error, best_match

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
