"""
Path management for the ERA Fellowship project.

This module provides a consistent way to access project paths regardless of where
the code is executed from. Import this module instead of manually calculating paths.
"""

from pathlib import Path
import os
import sys

# Get the path to this file
THIS_FILE = Path(__file__).resolve()

# Get the path to the analysis directory
ANALYSIS_DIR = THIS_FILE.parent

# Get the path to the project root directory
PROJECT_ROOT = ANALYSIS_DIR.parent

# Add the project root to the Python path if it's not already there
# This allows imports to work properly regardless of where the code is executed from
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