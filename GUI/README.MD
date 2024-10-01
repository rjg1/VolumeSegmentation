# Environment Setup
In order to run the GUIs, a Python environment with the required libraries should be setup:

1. Create the Environment

`py -3.11 -m venv .env` 

2. Activate the Environment

`./.env/scripts/activate`

3. Install the required dependencies

`python -m pip install -r requirements.txt`

4. Ensure your interpreter is set to use this environment (for quick run in VSCode)
In VSCode, `ctrl+shift+p` will allow you to select an interpreter. The interpreter path should be manually defined as: `./.env/scripts/python.exe`

# Cell Merger
## Getting started
In order to run the merger GUI, you can execute

`python ./cell_merger.py` 

in a terminal with the appropriate environment making sure that you are first in the `root/GUI` directory
Once the GUI opens, you must select a CSV file containing ROI boundary points, as well as a 3D .tif file which contains the image data. This can be done using the file menu in the top left.
Samples of each of these files are provided in the Microsoft Teams private channel (Image Segmentation - DRG) file storage: `GUI_data/cell_data/roi_coordinates_2` and `GUI_data/tif_files/file00001.tif` respectively.
Once data is loaded in, the GUI should display the plane z=0 with the ROIs overlaid on top 
## Using the tool
The current ROI being merged is displayed in the top-right of the screen.
### Merging (Merge Mode)
The tool begins in merge mode (hotkey 'm'), in which, clicking on the edges of a cluster will associate it with the current ROI. After the desired number of clusters are selected, you can hit the space key to 'lock in'this roi selection. It should now become visible in the side-bar, while also beginning merging of the next cluster (as evidenced by a higher current ROI count). Clicking on a cluster in the sidebar will zoom and centre on it.
To edit an existing selection, click on the ROI in the side bar, and you may continue to add or remove clusters from the ROI. After doing so, to begin merging a
new ROI, press 'n' (new ROI) or de-select the currently selected ROI tab.

If merging is not working - ensure merge mode is selected as evidenced by the green Merge Mode button at the top (all modes may be de-selected simultaneously)
### Zooming (Zoom Mode)
To begin zooming, you must first enter Zoom Mode, accessible via the 'z' hotkey, or Zoom Mode button at the top. In zoom mode, you may drag a box on the canvas to increase
zoom. You may not merge in zoom mode.

### Trashing (Trash Mode)
Trash mode allows for the selection of a group of noise ROIs to be discarded during export. When trash mode is entered ('t' hotkey), a new sidebar should appear, indicating
the current trashed ROIs for this z-plane. Trashed clusters may be viewed closely by clicking on their entry in the sidebar, and they should all be uniformly coloured black.

### Exporting
When the job is done, you may export the merged product by using the export menu at the top left of the GUI. This may take some time to run, as the data is being operated over
millions of data points. Once you have selected a csv file to save to, and the operation completes successfully, you may re-load in this csv in the file menu to view the result.
Please note this may require a restart of the GUI.

### Adjustable Parameters
Two parameters are easily adjustable that may provide some quick relief, in `cell_merger.py`. These are two global variables declared at the top:

`POINT_PRECISION` and `NUM_INTERPOLATION_POINTS`

Long story short, point precision refers to how precise your click must be near a cluster to select it. I have set it to 2 (high precision), to allow for dense clusters to be selected, however you may find that is too annoying. Num_interpolation_points is a post-processing step to make smooth merged ROIs. Higher = more smooth = more computationally intensive, but less chance for overlap.

## Hotkeys
'z' : Zoom mode enabled

't' : Trashing mode enabled

'm' : Merging mode enabled

'1' : Lower z-plane

'2' : Higher z-plane

'n' : New ROI

# Manual Segmenter
TBD