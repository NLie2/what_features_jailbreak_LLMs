"""
Path initialization for notebooks.

This module provides a simple way to set up paths for notebooks.
Import this module at the beginning of your notebooks.
"""

from pathlib import Path
import os
import sys

# Get the path to this file
THIS_FILE = Path(__file__).resolve()

# Get the path to the notebooks directory
NOTEBOOKS_DIR = THIS_FILE.parent

# Get the path to the analysis directory
ANALYSIS_DIR = NOTEBOOKS_DIR.parent

# Get the path to the project root directory
PROJECT_ROOT = ANALYSIS_DIR.parent

# Add the project root to the Python path if it's not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define common paths relative to the project root
DATASETS_DIR = PROJECT_ROOT / "datasets"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
IMAGES_DIR = PROJECT_ROOT / "images"
GEN_DATA_DIR = PROJECT_ROOT / "gen_data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Function to get a path relative to the project root
def get_project_path(relative_path):
    """
    Get an absolute path relative to the project root.
    
    Args:
        relative_path (str or Path): Path relative to the project root
        
    Returns:
        Path: Absolute path
    """
    return PROJECT_ROOT / relative_path

# Function to change the working directory to the project root
def set_project_root_as_cwd():
    """Change the current working directory to the project root."""
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT

# Set the working directory to the project root by default
set_project_root_as_cwd()

# Print a success message
print("Paths initialized successfully. You can now use PROJECT_ROOT, DATASETS_DIR, etc.") 