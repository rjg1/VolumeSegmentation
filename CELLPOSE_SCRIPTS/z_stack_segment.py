import numpy as np
from skimage import io
from cellpose import models, utils
import tifffile
import csv

# Temp parameter to segment only a single plane
target_z = 82

# Load the 3D image z-stack from a .tif file
# Assume the file 'image_stack.tif' is a 3D stack where the first dimension is the Z-plane
# image_stack = tifffile.imread('AVG_file_test.tif')
image_stack = tifffile.imread('file_00001.tif')
print(image_stack.shape)

# Initialize Cellpose model
# model = models.CellposeModel(pretrained_model='./DRG_xyzmodel_20220705')
model = models.CellposeModel(pretrained_model='./DRG_20250116_111124')

# Placeholder list to store segmentation masks for each Z-plane
output_list = []

# Iterate over each Z-plane in the stack
if image_stack.ndim == 2:
    z_planes = [image_stack]
else:
    if target_z is not None:
        if target_z < 0 or target_z >= image_stack.shape[0]:
            raise ValueError(f"target_z {target_z} out of range! Must be 0 to {image_stack.shape[0]-1}")
        z_planes = [image_stack[target_z, :, :]]
    else:
        z_planes = [(z, image_stack[z, :, :]) for z in range(image_stack.shape[0])]

for z, z_plane in enumerate(z_planes):
    print(f"Segmenting z-plane: {z} of {len(z_planes)-1}")

    # Segment the Z-plane using Cellpose
    masks, flows, styles = model.eval(z_plane, diameter=None, channels=[0, 0], flow_threshold=0.4, cellprob_threshold=-3)
    # masks, flows, styles = model.eval(z_plane, diameter=None, channels=[0, 0], flow_threshold=1.1, cellprob_threshold=-6)

    # Extract avg intensity per roi
    roi_ids = np.unique(masks)
    roi_ids = roi_ids[roi_ids != 0]  # skip background label 0

    mean_intensities = {
        roi: float(z_plane[masks == roi].mean())    # cast to float for JSON/CSV friendliness
        for roi in roi_ids
    }


    # Extract outlines
    outlines = utils.outlines_list(masks)

    # Append the outlines to a list
    for roi_id, outline in enumerate(outlines, start=1):
        if outline.size > 0:  # Check if outline is not empty
            avg_intensity = mean_intensities.get(roi_id, np.nan)
            for point in outline:
                x, y = point
                # Append the (x, y, z, ROI_ID) tuple
                output_list.append((x, y, z, roi_id, avg_intensity))

# Write the list to a CSV
csv_filename = 'day0_single_plane.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['x', 'y', 'z', 'ROI_ID', 'intensity'])
    # Write the ROI coordinates
    writer.writerows(output_list)
