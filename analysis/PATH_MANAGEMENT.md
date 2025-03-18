# Path Management System

This document explains how to use the path management system for the ERA Fellowship project.

## Overview

The path management system provides a consistent way to access project paths regardless of where your code is executed from. It uses **relative paths** to determine the project root, which means you don't need to hardcode absolute paths or manually calculate the project root relative to your current file or working directory.

## How It Works

The system uses Python's `Path(__file__).resolve()` to get the absolute path to the current file, and then navigates up the directory tree to find the project root. This works regardless of where your code is located in the project.

## How to Use

### In Python Scripts

```python
# Import the paths module
from analysis.paths import (
    PROJECT_ROOT,
    DATASETS_DIR,
    ANALYSIS_DIR,
    EXPERIMENTS_DIR,
    IMAGES_DIR,
    GEN_DATA_DIR,
    LOGS_DIR,
    get_project_path,
    set_project_root_as_cwd
)

# Use the path constants
data_path = DATASETS_DIR / "your_data.pkl"
image_path = IMAGES_DIR / "your_image.png"

# Get a custom path relative to the project root
custom_path = get_project_path("custom/path/to/file.txt")

# Optionally set the working directory to the project root
set_project_root_as_cwd()
```

### In Jupyter Notebooks

Copy and paste this code at the top of your notebook:

```python
# Import the init_paths module
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
```

Then use the path constants throughout your notebook:

```python
data_path = DATASETS_DIR / "your_data.pkl"
image_path = IMAGES_DIR / "your_image.png"
custom_path = get_project_path("custom/path/to/file.txt")
```

## Benefits

1. **No Absolute Paths**: The system doesn't rely on hardcoded absolute paths.
2. **Consistency**: All paths are relative to the project root, which is determined dynamically.
3. **Simplicity**: No need to calculate the project root relative to your current file or working directory.
4. **Reliability**: Works regardless of where your code is executed from.
5. **Portability**: The project can be moved to a different location without breaking the path management.

## Example Files

- `analysis/paths.py`: The main paths module.
- `experiments/paths.py`: The paths module for experiments.
- `analysis/notebooks/init_paths.py`: The paths module for notebooks.
- `analysis/notebooks/notebook_example.py`: An example for notebooks.

## Migration

To migrate your existing code to use the new path management system:

1. Replace code like this:
   ```python
   notebook_dir = Path.cwd()
   project_root = notebook_dir.parent.parent
   os.chdir(project_root)
   data_path = project_root / "datasets" / "your_data.pkl"
   ```

2. With code like this:
   ```python
   from init_paths import DATASETS_DIR
   data_path = DATASETS_DIR / "your_data.pkl"
   ``` 