# Re-running due to environment reset
import numpy as np
import matplotlib.pyplot as plt
from plane import Plane
import pprint
from planepoint import PlanePoint
import random

def main():
    anchor_offset = np.array([-1, -0.5])
    anchor_offset_3d = np.array([*anchor_offset, 0])
    # Step 1: Create original plane A
    anchor_a = PlanePoint(0, (0 + anchor_offset[0], 0 + anchor_offset[1], 0), traits={"avg_radius" : {"threshold": 1.0, "metric": "mse", "value": 5.0}})
    a_traits = {"avg_radius" : [3, 4, 5, 6, 7]}

    alignment_positions_a = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (1, 2, 0)]
    alignment_points_a = [PlanePoint(i + 1, p) for i, p in enumerate(alignment_positions_a)]
    for trait in a_traits:
        for idx, value in enumerate(a_traits[trait]):
            alignment_points_a[idx].add_trait(trait, threshold=1.0, metric="mse", value=value)

    plane_a = Plane(
        anchor_point = anchor_a,
        alignment_points=alignment_points_a
    )
    
    # Step 2: Create rotated, scaled, subset-transformed version (Plane B)
    subset_indices = [0, 2, 3]
    b_traits = {"avg_radius" : [a_traits["avg_radius"][i] * random.uniform(0.8, 1.2) for i in subset_indices]}
    subset_points = [
    np.array(alignment_positions_a[i]) - anchor_offset_3d
    for i in subset_indices
]

    
    # Rotate subset in 2D before any tilt
    subset_2d = [p[:2] for p in subset_points]
    theta_deg = 40
    theta = np.radians(theta_deg)
    R2 = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]])
    rotated_2d = [R2 @ p for p in subset_2d]
    rotated_3d = [np.array([x, y, 0]) for x, y in rotated_2d]
    rotated_3d = [p + np.array([*anchor_offset, 0]) for p in rotated_3d]

    # Scale uniformly
    scale = 1.5
    scaled = [p * scale for p in rotated_3d]

    # Apply 3D tilt (around X)
    tilt = np.radians(30)
    Rx = np.array([[1, 0, 0],
                [0, np.cos(tilt), -np.sin(tilt)],
                [0, np.sin(tilt),  np.cos(tilt)]])
    transformed_points = [Rx @ p for p in scaled]
    anchor_b_pos = Rx @ (scale * np.array([0 + anchor_offset[0], 0 + anchor_offset[1], 0]))

    # Offset centres
    offset_vector = np.array([3.0, 2.0, 1.0])
    transformed_points = [p + offset_vector for p in transformed_points]
    transformed = [PlanePoint(i+1, p) for i, p in enumerate(transformed_points)]
    anchor_b = PlanePoint(0, anchor_b_pos + offset_vector, traits={"avg_radius" : {"threshold": 1.0, "metric": "mse", "value": 5.0}})

    for trait in b_traits:
        for idx, value in enumerate(b_traits[trait]):
            transformed[idx].add_trait(trait, threshold=1.0, metric="mse", value=value)

    plane_b = Plane(
        anchor_point=anchor_b,
        alignment_points=transformed
    )

    # Step 3: Match
    match_data = plane_a.match_planes(plane_b)
    pprint.pp(match_data)

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

    # Plot 3: Final overlay in 3D with all ROIs
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.set_title("3 Final Aligned Points in 3D")

    # Plane A anchor and ROIs
    ax3.scatter(*anchor_a.position, color='red', s=100)
    ax3.text(*anchor_a.position, "A0", color='red')

    for id_, ppt in plane_a.plane_points.items():
        pt = ppt.position
        ax3.scatter(*pt, color='blue')
        ax3.text(*pt, f"A{id_}", color='blue', fontsize=8)

    # Plane B anchor and ROIs
    ax3.scatter(*anchor_b.position, color='orange', s=100)
    ax3.text(*anchor_b.position, "B0", color='orange')

    for id_, ppt in plane_b.plane_points.items():
        pt = ppt.position
        ax3.scatter(*pt, color='green', marker='x')
        ax3.text(*pt, f"B{id_}", color='green', fontsize=8)

    # Match lines (use ROI IDs directly)
    for i, j in match_data["matches"]:
        ida = None
        idb = None
        # Find the indices in plane points of each plane which match the index of each point (yeah dont ask)
        for idx, ppta in plane_a.plane_points.items():
            if ppta.id == i:
                ida = idx
        for idx, pptb in plane_b.plane_points.items():
            if pptb.id == j:
                idb = idx
        
        if ida is not None and idb is not None:
            pt_a = plane_a.plane_points[ida].position
            pt_b = plane_b.plane_points[idb].position
            ax3.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]], [pt_a[2], pt_b[2]], 'k--', alpha=0.5)

    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.grid(True)
    plt.tight_layout()
    plt.show()

    ###### FINAL TRANSLATION ##########
    aligned_b_points, R_align, translation = plane_a.get_aligned_3d_projection(
        plane_b,
        matches=match_data["matches"],
        scale_factor=match_data["scale_factor"]
    )
    print(R_align)
    print(translation)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Aligned 3D Projection")

    # Plane A
    for id_, ppt in plane_a.plane_points.items():
        pt = ppt.position
        ax.scatter(*pt, color='blue')
        ax.text(*pt, f"A{id_}", color='blue')

    # Transformed B
    for id_, pt in aligned_b_points.items():
        ax.scatter(*pt, color='green', marker='x')
        ax.text(*pt, f"B{id_}", color='green')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_zlim(-1, 1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()