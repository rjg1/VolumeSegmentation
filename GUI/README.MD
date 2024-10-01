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
`python ./cell_merger.py` in a terminal with the appropriate environment making sure that you are first in the `root/GUI` directory
Once the GUI opens, you must select a CSV file containing ROI boundary points, as well as a 3D .tif file which contains the image data.
Samples of each of these files are provided in the Microsoft Teams private channel (Image Segmentation - DRG) file storage: `GUI_data/cell_data/roi_coordinates_2` and `GUI_data/tif_files/file00001.tif` respectively.
Once data is loaded in, the GUI should display the plane z=0 with the ROIs overlaid on top 
## Using the tool
The 

# Manual Segmenter
TBD