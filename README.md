# VolumeSegmentation
This repository documents my work for my honours thesis project (REIT4841) at the University of Queensland.
The scripts within are written for the paper "Algorithmic Volume Segmentation of Two-photon Microscopy Z-stacks".
- Ryan Gibbons

NOTE: Without necessary data files, the scripts in this repository will NOT run as expected. Cloning this repository is not
sufficient to run segmentations, as segmentations require a 2D segmented z-stack .csv file. 

Contact for questions: s4393450@student.uq.edu.au

# Python Environment
Each sub-folder has requirements to run its own environment, however at the root level a `requirements.txt` file has been provided that will allow all scripts to be run in a single environment.

1. Create the Environment

`py -3.11 -m venv .env` 

2. Activate the Environment

`./.env/scripts/activate`

3. Install the required dependencies

`python -m pip install -r requirements.txt`

4. Ensure your interpreter is set to use this environment (for quick run in VSCode)
In VSCode, `ctrl+shift+p` will allow you to select an interpreter. The interpreter path should be manually defined as: `./.env/scripts/python.exe`

# Demo GUI
In order to run the unified demo GUI, complete the following steps:
1. Ensure a Python 3.11 environment with the required libraries is created and activated following the steps above.
2. Ensure you are in the ./DEMO directory 

`cd demo`

3. Run the GUI script: 

`python ./demo_gui.py`
