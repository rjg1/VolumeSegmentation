# Re-running due to environment reset
import numpy as np
import matplotlib.pyplot as plt
from plane import Plane
import pprint
from planepoint import PlanePoint
import random

def main():
    # Step 1: Create original plane A
    anchor_a = PlanePoint(0, (0, 0, 0), traits={"avg_radius" :  5.0})
    a_traits = {"avg_radius" : [3, 4, 5, 6, 7]}

    alignment_positions_a = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (1, 2, 0)]
    alignment_points_a = [PlanePoint(i + 1, p) for i, p in enumerate(alignment_positions_a)]
    for trait in a_traits:
        for idx, value in enumerate(a_traits[trait]):
            alignment_points_a[idx].add_trait(trait, value)

    plane_a = Plane(
        anchor_point = anchor_a,
        alignment_points=alignment_points_a
    )
    
    # Step 2: Create rotated, scaled, subset-transformed version (Plane B)
    subset_indices = [0, 2, 3]
    b_traits = {"avg_radius": [a_traits["avg_radius"][i] * random.uniform(0.8, 1.2) for i in subset_indices]}
    subset_points = [np.array(alignment_positions_a[i]) for i in subset_indices]

    # Rotate subset in 2D (XY plane)
    theta_deg = 270
    theta = np.radians(theta_deg)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    rotated = [Rz @ np.array([x, y, 0]) for x, y in [p[:2] for p in subset_points]]

    # Scale uniformly
    scale = 1.5
    scaled = [p * scale for p in rotated]

    # Apply Z and XY translation
    z_offset = 2.0
    xy_offset = np.array([3.0, 2.0, 0.0])
    total_offset = xy_offset + np.array([0, 0, z_offset])
    transformed_points = [p + total_offset for p in scaled]

    # Anchor B: transformed origin (0,0,0) under same transformation
    anchor_b_pos = Rz @ np.array([0, 0, 0]) * scale + total_offset
    anchor_b = PlanePoint(0, anchor_b_pos, traits={"avg_radius": 5.0})

    # Create PlanePoint objects for B
    transformed = [PlanePoint(i + 1, p) for i, p in enumerate(transformed_points)]
    for trait in b_traits:
        for idx, value in enumerate(b_traits[trait]):
            transformed[idx].add_trait(trait, value)

    plane_b = Plane(
        anchor_point=anchor_b,
        alignment_points=transformed
    )

    # Step 3: Match
    match_data = plane_a.match_planes(plane_b)
    import json
    print(match_data)

    ##
    proj_a = plane_a.get_local_2d_coordinates()
    proj_b = plane_b.get_local_2d_coordinates()

    # Plot 1: 3D view of original and transformed planes
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title("1 Original and Transformed Planes in 3D")
    ax1.scatter(*anchor_a.position, color='red', s=100, label='Anchor A')
    for idx, p in enumerate(alignment_positions_a, start=1):
        ax1.scatter(*p, color='blue', label='Plane A ROI' if idx == 1 else None)
    ax1.scatter(*anchor_b.position, color='orange', s=100, label='Anchor B')
    for idx, p in enumerate(transformed_points, start=1):
        ax1.scatter(*p, color='green', label='Plane B ROI' if idx == 1 else None)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend(); ax1.grid(True)
    plt.tight_layout()
    plt.show()
    ##

    ##
    # Plot 2.1, 2.2, 2.3: Projections + Match Overlay
    fig2, (ax2a, ax2b, ax2c) = plt.subplots(1, 3, figsize=(18, 6))

    # 2.1 Plane A projection
    ax2a.set_title("2.1 Plane A - Local Projection")
    for id_, (x, y) in proj_a.items():
        color = 'red' if id_ == 0 else 'blue'
        marker = 'o'
        ax2a.scatter(x, y, color=color, marker=marker)
    ax2a.set_aspect('equal'); ax2a.grid(True); ax2a.set_title("Plane A")

    # 2.2 Plane B projection
    ax2b.set_title("2.2 Plane B - Local Projection")
    for id_, (x, y) in proj_b.items():
        color = 'orange' if id_ == 0 else 'green'
        marker = 'x' if id_ != 0 else 'o'
        ax2b.scatter(x, y, color=color, marker=marker)
    ax2b.set_aspect('equal'); ax2b.grid(True); ax2b.set_title("Plane B")

    # 2.3 Overlayed match alignment
    ax2c.set_title("2.3 Alignment Overlay (2D)")

    # Plane A points
    for id_, (x, y) in proj_a.items():
        color = 'red' if id_ == 0 else 'blue'
        marker = 'o'
        ax2c.scatter(x, y, color=color, marker=marker)
        ax2c.text(x + 0.1, y + 0.1, f"A{id_}", color=color, fontsize=8)

    # Plane B points
    for id_, (x, y) in proj_b.items():
        color = 'orange' if id_ == 0 else 'green'
        marker = 'x' if id_ != 0 else 'o'
        ax2c.scatter(x, y, color=color, marker=marker)
        ax2c.text(x + 0.1, y + 0.1, f"B{id_}", color=color, fontsize=8)

    # Match lines
    for i, j in match_data["matches"]:
        # i, j are already the ROI IDs
        if i in proj_a and j in proj_b:
            xa, ya = proj_a[i]
            xb, yb = proj_b[j]
            ax2c.plot([xa, xb], [ya, yb], 'k--', alpha=0.6)

    ax2c.set_aspect('equal'); ax2c.grid(True)
    plt.tight_layout()
    plt.show()

    ##

    # Step 4: 2d proj
    proj_a, proj_b_aligned, transform_info = plane_a.get_aligned_2d_projection(
        plane_b,
        offset_deg=match_data["offset"],
        scale_factor=match_data["scale_factor"]
    )

    print("Applied Transformations:")
    pprint.pp(transform_info)

    # Step 5: Plot the aligned 2D projections

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Aligned 2D Projections")

    # Plot Plane A
    for idx, (x, y) in proj_a.items():
        color = "blue" if idx != 0 else "red"
        marker = "o"
        ax.scatter(x, y, color=color, marker=marker)
        ax.text(x + 0.1, y + 0.1, f"A{idx}", fontsize=8, color=color)

    # Plot Aligned Plane B
    for idx, (x, y) in proj_b_aligned.items():
        color = "green" if idx != 0 else "orange"
        marker = "x"
        ax.scatter(x, y, color=color, marker=marker)
        ax.text(x + 0.1, y + 0.1, f"B{idx}", fontsize=8, color=color)

    # Draw match lines between matched ROI IDs
    for id_a, id_b in match_data["matches"]:
        if id_a in proj_a and id_b in proj_b_aligned:
            xa, ya = proj_a[id_a]
            xb, yb = proj_b_aligned[id_b]
            ax.plot([xa, xb], [ya, yb], 'k--', alpha=0.5)

    ax.set_aspect('equal')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()