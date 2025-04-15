# Re-running due to environment reset
import numpy as np
import matplotlib.pyplot as plt
from plane import Plane

# Step 1: Create original plane A
anchor_a = np.array([0, 0, 0])
alignment_points_a = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1, 1, 0]), np.array([2, 1, 0]), np.array([1, 2, 0])]
plane_a_ids = list(range(1, len(alignment_points_a) + 1))

plane_a = Plane(
    anchor_id=0,
    alignment_ids=plane_a_ids,
    anchor_point=anchor_a,
    alignment_points=alignment_points_a
)

# Step 2: Create rotated, scaled, subset-transformed version (Plane B)
subset_indices = [0, 2, 3]
subset_points = [alignment_points_a[i] for i in subset_indices]

# Rotate subset in 2D before any tilt
subset_2d = [p[:2] for p in subset_points]
theta_deg = 40
theta = np.radians(theta_deg)
R2 = np.array([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta),  np.cos(theta)]])
rotated_2d = [R2 @ p for p in subset_2d]
rotated_3d = [np.array([x, y, 0]) for x, y in rotated_2d]

# Scale uniformly
scale = 1.25
scaled = [p * scale for p in rotated_3d]

# Apply 3D tilt (around X)
tilt = np.radians(30)
Rx = np.array([[1, 0, 0],
               [0, np.cos(tilt), -np.sin(tilt)],
               [0, np.sin(tilt),  np.cos(tilt)]])
transformed = [Rx @ p for p in scaled]
anchor_b = Rx @ (scale * np.array([0, 0, 0]))

plane_b = Plane(
    anchor_id=0,
    alignment_ids=list(range(1, len(transformed)+1)),
    anchor_point=anchor_b,
    alignment_points=transformed
)

# Step 3: Match
match_data = plane_a.match_planes(plane_b, angle_tolerance=1)
print(match_data)
proj_a = plane_a.get_local_2d_coordinates()
proj_b = plane_b.get_local_2d_coordinates()

# Plot 1: 3D view of original and transformed planes
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title("1 Original and Transformed Planes in 3D")
ax1.scatter(*anchor_a, color='red', s=100, label='Anchor A')
for idx, p in enumerate(alignment_points_a, start=1):
    ax1.scatter(*p, color='blue', label='Plane A ROI' if idx == 1 else None)
ax1.scatter(*anchor_b, color='orange', s=100, label='Anchor B')
for idx, p in enumerate(transformed, start=1):
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
ax3.scatter(*anchor_a, color='red', s=100)
ax3.text(*anchor_a, "A0", color='red')

for id_, pt in zip(plane_a.projected_ids, plane_a.projected_points):
    ax3.scatter(*pt, color='blue')
    ax3.text(*pt, f"A{id_}", color='blue', fontsize=8)

# Plane B anchor and ROIs
ax3.scatter(*anchor_b, color='orange', s=100)
ax3.text(*anchor_b, "B0", color='orange')

for id_, pt in zip(plane_b.projected_ids, plane_b.projected_points):
    ax3.scatter(*pt, color='green', marker='x')
    ax3.text(*pt, f"B{id_}", color='green', fontsize=8)

# Match lines (use ROI IDs directly)
for i, j in match_data["matches"]:
    if i in plane_a.projected_ids and j in plane_b.projected_ids:
        pt_a = plane_a._projected_dict[i]
        pt_b = plane_b._projected_dict[j]
        ax3.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]], [pt_a[2], pt_b[2]], 'k--', alpha=0.5)

ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")
ax3.grid(True)
plt.tight_layout()
plt.show()

###### FINAL TRANSLATION ##########
from scipy.spatial.transform import Rotation as R

# --- Step 1: Get anchor points ---
anchor_id_a = plane_a.anchor_id
anchor_id_b = plane_b.anchor_id
p0a = plane_a._projected_dict[anchor_id_a]
p0b = plane_b._projected_dict[anchor_id_b]

# --- Step 2: Collect matched vectors (excluding anchors if present) ---
alignment_matches = [(i, j) for i, j in match_data["matches"] if i != anchor_id_a and j != anchor_id_b]

Va = np.stack([plane_a._projected_dict[i] - p0a for i, j in alignment_matches])  # Nx3
Vb = np.stack([plane_b._projected_dict[j] - p0b for i, j in alignment_matches])  # Nx3

# --- Step 3: Compute optimal rotation matrix using Kabsch algorithm ---
H = Vb.T @ Va
U, _, Vt = np.linalg.svd(H)
R_align = Vt.T @ U.T

# Ensure it's a proper rotation (no reflection)
if np.linalg.det(R_align) < 0:
    Vt[2, :] *= -1
    R_align = Vt.T @ U.T

# --- Step 4: Apply alignment (translation + rotation + scale) ---
aligned_pts_b = []
scale = match_data["scale_factor"]

for pt in plane_b.projected_points:
    local_vec = pt - p0b                 # move into local frame
    rotated = R_align @ local_vec        # rotate into Plane A's orientation
    scaled = rotated * scale             # scale vectors to match magnitudes
    final = p0a + scaled                 # move into Plane A's frame
    aligned_pts_b.append(final)

aligned_pts_b = np.array(aligned_pts_b)
aligned_ids_b = plane_b.projected_ids

# Step 5: Plot result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Final 3D Alignment (Direct Point Matching)")

# Plane A points
for id_, pt in plane_a._projected_dict.items():
    ax.scatter(*pt, color='blue')
    ax.text(*pt, f"A{id_}", color='blue', fontsize=8)

# Transformed B points
for pt, id_ in zip(aligned_pts_b, aligned_ids_b):
    ax.scatter(*pt, color='green', marker='x')
    ax.text(*pt, f"B{id_}", color='green', fontsize=8)

# Draw matched lines
for i, j in match_data["matches"]:
    if i in plane_a._projected_dict and j in aligned_ids_b:
        pt_a = plane_a._projected_dict[i]
        pt_b = aligned_pts_b[aligned_ids_b.index(j)]
        ax.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]], [pt_a[2], pt_b[2]], 'k--', alpha=0.5)

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.set_zlim(-1, 1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.grid(True)
plt.tight_layout()
plt.show()
