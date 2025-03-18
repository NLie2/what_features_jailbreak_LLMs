"""
Example code for notebooks to import the paths module.

This is a simple example that you can copy and paste into your notebooks
to import the paths module and use the project paths.
"""

# Copy and paste this code at the top of your notebook:

# ----------------- START COPY HERE -----------------
import sys
from pathlib import Path

# Import the paths module - this works regardless of where your notebook is located
# No need to calculate the project root anymore!
from analysis.paths import (
    PROJECT_ROOT,
    DATASETS_DIR,
    IMAGES_DIR,
    get_project_path,
    set_project_root_as_cwd
)

# Optionally set the working directory to the project root
set_project_root_as_cwd()
print(f"Project root set to: {PROJECT_ROOT}")
# ----------------- END COPY HERE -----------------

# Now you can use the path constants throughout your notebook:
# data_path = DATASETS_DIR / "your_data.pkl"
# image_path = IMAGES_DIR / "your_image.png"
# custom_path = get_project_path("custom/path/to/file.txt") 