import numpy as np
import matplotlib.pyplot as plt
import csv
from cellpose import plot, utils, io

# Load the segmentation data from the *_seg.npy file
dat = np.load('file_00001_seg.npy', allow_pickle=True).item()
img = io.imread('file_00001.tif')

masks = dat['masks']

print(img.shape)
print(dat['masks'].shape)

# plot image with masks overlaid
# mask_RGB = plot.mask_overlay(img, dat['masks'],
#                         colors=np.array(dat['colors']))

output_list = []

# for z in range(masks.shape[0]):
#     print(f"Processing Z-plane {z + 1} of {masks.shape[0]}")
    
#     # Get the unique ROI labels in the current plane (excluding the background label 0)
#     unique_labels = np.unique(masks[z, :, :])
#     unique_labels = unique_labels[unique_labels != 0]  # Exclude background label
    
#     # Process each ROI label in the current Z-plane
#     for roi_id in unique_labels:
#         # Find coordinates of the current ROI
#         y_coords, x_coords = np.where(masks[z, :, :] == roi_id)
        
#         # Create tuples of (x, y, z, ROI_ID)
#         roi_tuples = [(x, y, z, roi_id) for x, y in zip(x_coords, y_coords)]
        
#         # Add the tuples to the list
#         output_list.extend(roi_tuples)



# # Iterate over each Z-plane in the masks
# for z in range(masks.shape[0]):
#     print(f"Processing Z-plane {z + 1} of {masks.shape[0]}")
    
#     # Get the outlines of ROIs in the current Z-plane
#     outlines = utils.outlines_list(masks[z, :, :])
    
#     # Process each outline to extract (x, y) points
#     for roi_id, outline in enumerate(outlines, start=1):
#         if outline.size > 0:  # Check if outline is not empty
#             for point in outline:
#                 x, y = point
#                 # Append the (x, y, z, ROI_ID) tuple
#                 output_list.append((x, y, z, roi_id))

# # # Write the list to a CSV
# # # Export the list of tuples to a CSV file
# csv_filename = 'roi_coordinates.csv'
# with open(csv_filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # Write header
#     writer.writerow(['x', 'y', 'z', 'ROI_ID'])
#     # Write the ROI coordinates
#     writer.writerows(output_list)


# # plot image with outlines overlaid in red
for z in range(dat['masks'].shape[0]):
    print(f"showing z-plane {z} of {dat['masks'].shape[0]}")
    outlines = utils.outlines_list(dat['masks'][z, :, :])
    print(outlines)
    plt.imshow(img[z, :, :])
    for o in outlines:
        plt.plot(o[:,0], o[:,1], color='r')

    plt.show()