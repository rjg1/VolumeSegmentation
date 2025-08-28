import csv
import numpy as np
import matplotlib.pyplot as plt
import tifffile

# === PARAMETERS ===
# tif_file = 'file_00001.tif'
# csv_file = 'day0_single_plane.csv'

tif_file = 'file_00001.tif'
csv_file = 'drg_complete_3i_algo_VOLUMES.csv'
target_z = 82  # Set to an integer to select that z-plane, or None for single-plane tif

# === LOAD THE IMAGE STACK OR SINGLE IMAGE ===
image_stack = tifffile.imread(tif_file)
print("Full image shape:", image_stack.shape)

if image_stack.ndim == 2:  
    # Single-plane image
    image = image_stack
    z_plane_index = 0
else:  
    # Multi-plane stack, select plane
    if target_z is None:
        raise ValueError(f"Set 'target_z' (0 to {image_stack.shape[0]-1}) for a multi-plane stack!")
    if target_z < 0 or target_z >= image_stack.shape[0]:
        raise ValueError(f"target_z {target_z} out of range! Must be 0 to {image_stack.shape[0]-1}")
    
    image = image_stack[target_z]
    z_plane_index = target_z

print(f"Using z-plane {z_plane_index}")

# === LOAD CSV DATA ===
x_list, y_list, roi_list = [], [], []

with open(csv_file, newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if abs(float(row['z']) - target_z) < 0.9:
            x_list.append(float(row['x']))
            y_list.append(float(row['y']))
            roi_list.append(int(row['ROI_ID']))

x = np.array(x_list)
y = np.array(y_list)
roi_ids = np.array(roi_list)

# === PLOT THE IMAGE AND OUTLINES ===
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray', origin='upper')

scatter = plt.scatter(x, y, c=roi_ids, cmap='tab20', s=1, alpha=0.8)
plt.colorbar(scatter, label='ROI ID')

plt.title(f"ROI Outlines on Z-plane {z_plane_index}")
plt.axis('off')
plt.tight_layout()
plt.show()
