import numpy as np
from skimage import io
from cellpose import models, utils
import tifffile
import csv

# Load the 3D image z-stack from a .tif file
# Assume the file 'image_stack.tif' is a 3D stack where the first dimension is the Z-plane
image_stack = tifffile.imread('file_00001.tif')
print(image_stack.shape)

# Initialize Cellpose model
model = models.CellposeModel(pretrained_model='./DRG_xyzmodel_20220705')

# Placeholder list to store segmentation masks for each Z-plane
output_list = []

# Iterate over each Z-plane in the stack
for z in range(image_stack.shape[0]):
    print(f"Segmenting z-plane: {z} of {image_stack.shape[0]-1}")
    # Extract the Z-plane
    z_plane = image_stack[z, :, :]

    # Segment the Z-plane using Cellpose
    # masks, flows, styles = model.eval(z_plane, diameter=None, channels=[0, 0], flow_threshold=1, cellprob_threshold=-3)
    masks, flows, styles = model.eval(z_plane, diameter=None, channels=[0, 0], flow_threshold=1.1, cellprob_threshold=-6)

    # Extract outlines
    outlines = utils.outlines_list(masks)

    # Append the outlines to a list
    for roi_id, outline in enumerate(outlines, start=1):
        if outline.size > 0:  # Check if outline is not empty
            for point in outline:
                x, y = point
                # Append the (x, y, z, ROI_ID) tuple
                output_list.append((x, y, z, roi_id))

# Write the list to a CSV
csv_filename = 'roi_coordinates_3.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['x', 'y', 'z', 'ROI_ID'])
    # Write the ROI coordinates
    writer.writerows(output_list)
