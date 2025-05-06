import numpy as np
import matplotlib.pyplot as plt
import random
from planepoint import PlanePoint
from plane import Plane
from region import Region, BoundaryRegion
from registration_utils import compute_avg_uoi
import pprint

def generate_circle_3d(center, radius, num_points=50):
    """
    Generates a circle of points around a center in the XY-plane of 3D.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])
    return np.vstack((x, y, z)).T

def main():
    anchor_offset = np.array([-1, -0.5])
    anchor_offset_3d = np.array([*anchor_offset, 0])

    # Step 1: Create original plane A
    anchor_a = PlanePoint(0, (0 + anchor_offset[0], 0 + anchor_offset[1], 0),
                          traits={"avg_radius": {"threshold": 1.0, "metric": "mse", "value": 0.4}})
    a_traits = {"avg_radius": [0.3, 0.45, 0.3, 0.4, 0.5]}
    alignment_positions_a = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (1, 2, 0)]

    alignment_points_a = [PlanePoint(i + 1, p) for i, p in enumerate(alignment_positions_a)]
    for trait in a_traits:
        for idx, value in enumerate(a_traits[trait]):
            alignment_points_a[idx].add_trait(trait, threshold=1.0, metric="mse", value=value)

    plane_a = Plane(anchor_point=anchor_a, alignment_points=alignment_points_a)

    # Step 2: Create transformed Plane B
    subset_indices = [0, 2, 3]
    subset_points = [np.array(alignment_positions_a[i]) - anchor_offset_3d for i in subset_indices]

    # Apply 2D rotation
    theta_deg = 40
    theta = np.radians(theta_deg)
    R2 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    rotated_2d = [R2 @ p[:2] for p in subset_points]
    rotated_3d = [np.array([x, y, 0]) + anchor_offset_3d for x, y in rotated_2d]

    # Apply scaling
    scale = 1.5
    scaled = [p * scale for p in rotated_3d]

    # Apply 3D tilt
    tilt = np.radians(30)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(tilt), -np.sin(tilt)],
                   [0, np.sin(tilt),  np.cos(tilt)]])
    transformed_points = [Rx @ p for p in scaled]
    anchor_b_pos = Rx @ (scale * np.array([0 + anchor_offset[0], 0 + anchor_offset[1], 0]))

    # Apply translation
    offset_vector = np.array([3.0, 2.0, 1.0])
    transformed_points = [p + offset_vector for p in transformed_points]
    anchor_b = PlanePoint(0, anchor_b_pos + offset_vector,
                          traits={"avg_radius": {"threshold": 1.0, "metric": "mse", "value": 0.4}})

    # Plane B trait values
    # b_traits = {"avg_radius": [a_traits["avg_radius"][i] * random.uniform(0.6, 1.4) for i in subset_indices]}
    b_traits = {"avg_radius": [a_traits["avg_radius"][i] for i in subset_indices]}
    transformed = [PlanePoint(i + 1, p) for i, p in enumerate(transformed_points)]
    for trait in b_traits:
        for idx, value in enumerate(b_traits[trait]):
            transformed[idx].add_trait(trait, threshold=1.0, metric="mse", value=value)

    plane_b = Plane(anchor_point=anchor_b, alignment_points=transformed)

    # Step 3: Generate circles for both planes
    circle_points_a = {
        ppt.id: generate_circle_3d(ppt.position, radius=ppt.traits["avg_radius"]["value"])
        for ppt in [anchor_a] + alignment_points_a
    }

    subset_ids_b = [0] + [pt.id for pt in transformed]
    circle_points_b_scaled_tilted = {}
    for pid in subset_ids_b:
        center = plane_b.plane_points[pid].position
        radius = plane_b.plane_points[pid].traits["avg_radius"]["value"]

        # Generate circle points in XY plane centered at origin
        circle = generate_circle_3d(center=[0, 0, 0], radius=radius)

        # Scale the circle (like B points)
        circle_scaled = [pt * scale for pt in circle]

        # Apply 3D tilt
        tilt = np.radians(30)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(tilt), -np.sin(tilt)],
            [0, np.sin(tilt),  np.cos(tilt)]
        ])
        circle_tilted = [Rx @ pt for pt in circle_scaled]

        # Recenter the circle to match the transformed B point center
        circle_shifted = [pt + center for pt in circle_tilted]

        circle_points_b_scaled_tilted[pid] = circle_shifted




    # Step 4: Match & align
    match_data = plane_a.match_planes(plane_b)
    aligned_b_points, R_align, translation = plane_a.get_aligned_3d_projection(
        plane_b, matches=match_data["matches"], scale_factor=match_data["scale_factor"]
    )
    pprint.pprint(match_data)

   # Step 5: Apply transformation to B circles
    aligned_circle_points_b = {}
    for pid, pts in circle_points_b_scaled_tilted.items():
        aligned_circle_points_b[pid] = np.array([
            plane_a._apply_3d_transform(
                point=p,
                rotation_matrix=R_align,
                scale=match_data["scale_factor"],
                translation=translation,
                b_anchor_point=plane_b.anchor_point.position,
                project=False
            )
            for p in pts
        ])

    # Step 6: Plotting original circles and centroids
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original Circles and Points (Before Alignment)")

    # Plane A circles
    for sid, pts in circle_points_a.items():
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='blue')
        ax1.scatter(*plane_a.plane_points[sid].position, color='blue', s=50)
    ax1.scatter(*plane_a.anchor_point.position, color='red', s=80, label='Anchor A')

    # Plane B circles (before transformation)
    for pid, pts in circle_points_b_scaled_tilted.items():
        pts = np.array(pts)
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='orange', linestyle='--', alpha=0.6)
        ax1.scatter(*plane_b.plane_points[pid].position, color='orange', s=50)
    ax1.scatter(*plane_b.anchor_point.position, color='darkred', s=80, label='Anchor B')

    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_xlim(-2, 6); ax1.set_ylim(-2, 6); ax1.set_zlim(-1, 5)
    ax1.legend()

    # Step 7: Plot transformed B circles (aligned)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Transformed B Circles Aligned to A")

    # Plane A
    for sid, pts in circle_points_a.items():
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='blue')
        ax2.scatter(*plane_a.plane_points[sid].position, color='blue', s=50)
    ax2.scatter(*plane_a.anchor_point.position, color='red', s=80, label='Anchor A')

    # Transformed B
    for sid, pts in aligned_circle_points_b.items():
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='green')
        ax2.scatter(*aligned_b_points[sid], color='green', s=50)
    ax2.scatter(*aligned_b_points[0], color='darkgreen', s=80, label='Aligned Anchor B')

    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.set_xlim(-2, 6); ax2.set_ylim(-2, 6); ax2.set_zlim(-1, 5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # 2D example
    proj_a, proj_b_aligned, transform_info = plane_a.get_aligned_2d_projection(
        plane_b,
        offset_deg=match_data["offset"],
        scale_factor=match_data["scale_factor"]
    )

    circle_b_2d = {}
    rotation_2d = transform_info["rotation_deg"]
    scale_2d = transform_info["scale"]
    translation_2d = transform_info["translation"]
    for pid, pts in circle_points_b_scaled_tilted.items(): # apply operations to all pts in plane b
        circle_b_2d [pid] = np.array(
            plane_a.project_and_transform_points(pts, plane_b, rotation_deg=rotation_2d, scale=scale_2d, translation=translation_2d)
            # np.array([plane_a._project_point_2d(pt) for pt in pts])
        )

    circle_a_2d = {}
    for pid, pts in circle_points_a.items():
        circle_a_2d[pid] = np.array([plane_a._project_point_2d(pt) for pt in pts])

    print(transform_info)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Aligned 2D Projections")

    # Plot Plane A circles
    for pid, pts in circle_a_2d.items():
        x, y = pts[:, 0], pts[:, 1]
        ax.plot(x, y, color='blue', label=f"A{pid}" if pid == 0 else None)
        cx, cy = pts.mean(axis=0)
        ax.scatter(cx, cy, color='blue', s=30)
        ax.text(cx + 0.05, cy + 0.05, f"A{pid}", fontsize=8, color='blue')

    # Plot Transformed Plane B circles
    for pid, pts in circle_b_2d.items():
        x, y = pts[:, 0], pts[:, 1]
        ax.plot(x, y, color='green', label=f"B{pid}" if pid == 0 else None)
        cx, cy = pts.mean(axis=0)
        ax.scatter(cx, cy, color='green', s=30)
        ax.text(cx + 0.05, cy + 0.05, f"B{pid}", fontsize=8, color='green')


    ax.set_aspect('equal')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Compute IoU of each region

    # Make region dicts
    regions_a = {pid : BoundaryRegion([(x, y, 0) for x,y in circle_a_2d[pid]]) for pid, pts in circle_a_2d.items()}
    regions_b = {pid : BoundaryRegion([(x, y, 0) for x,y in circle_b_2d[pid]]) for pid, pts in circle_b_2d.items()}

    IoU = compute_avg_uoi(regions_a, regions_b, match_data["og_matches"], plot=True)
    print(IoU)

if __name__ == "__main__":
    main()
