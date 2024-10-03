# VolumeSegmentation
Hi, if you are reading this, you are either myself or Clarissa. At this point, nothing really is set up besides the /GUI/ folder which will have its own README.

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
