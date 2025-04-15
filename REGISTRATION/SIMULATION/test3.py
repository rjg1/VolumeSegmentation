import numpy as np
from itertools import combinations

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

    print(arcs_a)

    best_error = float('inf')
    best_match = None

    for arc_len_a, (start_a, end_a) in arcs_a:
        for arc_len_b, (start_b, end_b) in arcs_b:
            error = abs(arc_len_a - arc_len_b)
            if error < best_error and error <= max_error:
                best_error = error
                best_match = ((start_a, end_a), (start_b, end_b))

    return best_error, best_match

rel_a = [np.radians(x) for x in [10, 20, 30, 300]]
rel_b = [np.radians(x) for x in [10, 50, 300]]

error, match = match_relative_angle_arcs(rel_a, rel_b)
print("Best match error:", error)
print("Matching arc indices (Plane A, Plane B):", match)