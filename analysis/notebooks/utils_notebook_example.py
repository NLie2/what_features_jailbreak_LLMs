"""
Example of how to use the init_paths.py module in notebooks.

Copy and paste this code at the beginning of your notebooks.
"""

# ----------------- START COPY HERE -----------------
# Import the init_paths module
# This will set up all the paths and set the working directory to the project root
from init_paths import (
    PROJECT_ROOT,
    DATASETS_DIR,
    ANALYSIS_DIR,
    EXPERIMENTS_DIR,
    IMAGES_DIR,
    GEN_DATA_DIR,
    LOGS_DIR,
    get_project_path
)
# ----------------- END COPY HERE -----------------

# Now you can use the path constants throughout your notebook:
print(f"Project root: {PROJECT_ROOT}")
print(f"Datasets directory: {DATASETS_DIR}")
print(f"Analysis directory: {ANALYSIS_DIR}")

# Example usage:
data_path = DATASETS_DIR / "your_data.pkl"
print(f"Data path: {data_path}")

image_path = IMAGES_DIR / "your_image.png"
print(f"Image path: {image_path}")

custom_path = get_project_path("custom/path/to/file.txt")
print(f"Custom path: {custom_path}")

# The key advantage of this approach is that it works regardless of:
# 1. Where your notebook is located in the project
# 2. What your current working directory is
# 3. Whether you're running in a notebook or a script
# 4. No hardcoded absolute paths! 