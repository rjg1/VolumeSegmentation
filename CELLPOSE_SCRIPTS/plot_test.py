import csv
import numpy as np
import matplotlib.pyplot as plt
import tifffile

# === Load the image (single plane .tif) ===
image = tifffile.imread('AVG_file_test.tif')  # shape: (H, W)
print("Image shape:", image.shape)

# === Load CSV Data ===
csv_file = 'roi_coords_single_plane.csv'  # or any CSV in the correct format
x_list, y_list, roi_list = [], [], []

with open(csv_file, newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        x_list.append(float(row['x']))
        y_list.append(float(row['y']))
        roi_list.append(int(row['ROI_ID']))

x = np.array(x_list)
y = np.array(y_list)
roi_ids = np.array(roi_list)

# === Plot the image and overlay the outlines ===
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray', origin='upper')

# Overlay outline points (use small size and color by ROI_ID)
scatter = plt.scatter(x, y, c=roi_ids, cmap='tab20', s=1, alpha=0.8)
plt.colorbar(scatter, label='ROI ID')

plt.title("ROI Outlines Overlaid on Image")
plt.axis('off')
plt.tight_layout()
plt.show()
