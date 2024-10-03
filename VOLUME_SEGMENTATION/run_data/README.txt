This folder should contain valid .csv files that can be used to run the volume segmentation in the script `volume_segmentation.py`

csv input files should be all numerical and have the headings 'x','y','z','ROI_ID', with each entry corresponding to a datapoint of an ROI.
ROI_ID is unique per z-plane.

The csv file used in runtime must be set using the `IN_FILE` global variable in `volume_segmentation.py`.