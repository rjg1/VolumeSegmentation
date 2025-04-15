import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plane import Plane 

# --- Generate sample data ---
np.random.seed(42)
num_points = 10
points = np.random.uniform(0, 10, (num_points, 3))
point_ids = list(range(num_points))

# --- Define anchor and alignment ROIs ---
anchor_idx = 0
align_idxs = [1, 2]
anchor_point = points[anchor_idx]
alignment_points = [points[i] for i in align_idxs]

# --- Create plane ---
plane = Plane(
    anchor_id=anchor_idx,
    alignment_ids=align_idxs,
    anchor_point=anchor_point,
    alignment_points=alignment_points
)

# --- Project points onto the plane ---
projected, projected_ids = plane.project_points(points, point_ids, threshold=1)
projected_2d = plane.get_local_2d_coordinates()

# --- Compute angle/magnitude from both methods ---
angles_3d, mags_3d = plane.angles_and_magnitudes()
angles_2d, mags_2d = plane.angles_and_magnitudes_from_2d()

print(angles_3d)
print(angles_2d)

print("===")

print(mags_3d)
print(mags_2d)


# --- Generate Surface for Plane ---
extent = 10
xx, yy = np.meshgrid(np.linspace(0, extent, 10), np.linspace(0, extent, 10))
zz = (-plane.normal[0] * xx - plane.normal[1] * yy - plane.d) / plane.normal[2]

# --- Plotting ---
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("3D View with Plane and Projected Points")

# Original points
ax1.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=50)
ax1.scatter(*anchor_point, color='red', s=100)
for idx in align_idxs:
    ax1.scatter(*points[idx], color='green', s=100)

# Plane surface
ax1.plot_surface(xx, yy, zz, alpha=0.3, color='orange', edgecolor='none')

# Projected points
ax1.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
            color='purple', marker='^', s=60)

# Legend
legend_elements = [
    mpatches.Patch(color='orange', alpha=0.3, label='Fitted Plane'),
    plt.Line2D([0], [0], marker='o', color='w', label='Original Points',
               markerfacecolor='blue', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Anchor Point',
               markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Alignment Points',
               markerfacecolor='green', markersize=10),
    plt.Line2D([0], [0], marker='^', color='w', label='Projected Points',
               markerfacecolor='purple', markersize=10)
]
ax1.legend(handles=legend_elements)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# --- 2D Plot of Projected Points ---
ax2 = fig.add_subplot(122)
ax2.set_title("2D Projection on Local Plane Coordinates")

for id_, (x, y) in projected_2d.items():
    if id_ == anchor_idx:
        ax2.scatter(x, y, color='red', s=100, label='Anchor' if 'Anchor' not in ax2.get_legend_handles_labels()[1] else "")
    elif id_ in align_idxs:
        ax2.scatter(x, y, color='green', s=100, label='Alignment' if 'Alignment' not in ax2.get_legend_handles_labels()[1] else "")
    else:
        ax2.scatter(x, y, color='purple', marker='^')

ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax2.axvline(0, color='gray', linestyle='--', linewidth=0.5)
ax2.set_xlabel('u (plane x-axis)')
ax2.set_ylabel('v (plane y-axis)')
ax2.set_aspect('equal')
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.show()


# --- Compare differences ---
diffs = []
for a3, m3, a2, m2 in zip(angles_3d, mags_3d, angles_2d, mags_2d):
    angle_diff = np.abs(a3 - a2)
    mag_diff = np.abs(m3 - m2)
    diffs.append((angle_diff, mag_diff))

angle_diffs = [d[0] for d in diffs]
mag_diffs = [d[1] for d in diffs]