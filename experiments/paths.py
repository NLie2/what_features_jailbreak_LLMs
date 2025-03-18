"""
Path management for experiments.

This module provides a consistent way to access project paths for experiments.
"""

from pathlib import Path
import os
import sys

# Get the path to this file
THIS_FILE = Path(__file__).resolve()

# Get the path to the experiments directory
EXPERIMENTS_DIR = THIS_FILE.parent

# Get the path to the project root directory
PROJECT_ROOT = EXPERIMENTS_DIR.parent

# Add the project root to the Python path if it's not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define common paths relative to the project root
DATASETS_DIR = PROJECT_ROOT / "datasets"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
IMAGES_DIR = PROJECT_ROOT / "images"
GEN_DATA_DIR = PROJECT_ROOT / "gen_data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Add any experiment-specific paths here
EXPERIMENT_RESULTS_DIR = EXPERIMENTS_DIR / "results"

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