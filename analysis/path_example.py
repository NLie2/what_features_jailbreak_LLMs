"""
Example script demonstrating how to use the path management system.

This script shows how to use the paths module to access project paths
regardless of where the script is executed from.
"""

# Import the paths module
from analysis.paths import (
    PROJECT_ROOT,
    DATASETS_DIR,
    IMAGES_DIR,
    get_project_path,
    set_project_root_as_cwd
)

def main():
    # Print the project root
    print(f"Project root: {PROJECT_ROOT}")
    
    # Print some common directories
    print(f"Datasets directory: {DATASETS_DIR}")
    print(f"Images directory: {IMAGES_DIR}")
    
    # Get a path relative to the project root
    custom_path = get_project_path("custom/path/to/file.txt")
    print(f"Custom path: {custom_path}")
    
    # Set the working directory to the project root
    set_project_root_as_cwd()
    print(f"Current working directory set to: {PROJECT_ROOT}")
    
    # Example of how to use the paths in your code
    # Instead of:
    # notebook_dir = Path.cwd()
    # project_root = notebook_dir.parent.parent
    # data_path = project_root / "datasets" / "some_data.pkl"
    
    # You can now do:
    data_path = DATASETS_DIR / "some_data.pkl"
    print(f"Data path: {data_path}")
    
    # The key advantage of this approach is that it works regardless of:
    # 1. Where your script is located in the project
    # 2. What your current working directory is
    # 3. Whether you're running in a notebook or a script

if __name__ == "__main__":
    main() 