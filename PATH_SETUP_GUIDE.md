---
noteId: "4ee173209d3911f0871c71d0ae697433"
tags: []

---

# Path Configuration Guide

## Problem Solved

This project had issues with Python import paths when running scripts directly. The error `ModuleNotFoundError: No module named 'app'` occurred because Python couldn't find the project modules when scripts were executed individually.

## Universal Solution

### Method 1: Import `path_config` (Recommended)

Add this single line at the top of ANY Python file in the project:

```python
import path_config  # This automatically configures all paths
```

Then you can import any module from the project:

```python
import path_config  # Always import this first

from app.data_injestion import train_df, test_df
from app.preprocessor.missing_value_analysis import MissingValueAnalyzer
from app.feature_engineering.feature_encoder import FeatureEncoder
# ... any other imports
```

### Method 2: Use the Enhanced App Package

The `app/__init__.py` has been enhanced to automatically configure paths when imported:

```python
import app  # This automatically sets up paths
from app.data_injestion import train_df
```

## How It Works

1. **Automatic Detection**: The system automatically finds the project root by looking for characteristic files (`README.md`, `pyproject.toml`, etc.)

2. **Path Configuration**: Adds the project root to Python's `sys.path` so all modules can be imported

3. **One-Time Setup**: Once imported in a Python session, the configuration persists

## Files Updated

### Core Configuration Files

- `path_config.py` - Universal path configuration utility
- `setup_path.py` - Alternative setup script
- `project_imports.py` - Common imports module
- `app/__init__.py` - Enhanced package initialization

### Updated Scripts

- `app/preprocessor/missing_value_analysis.py` - Now uses universal path config

## Usage Examples

### For New Scripts

```python
#!/usr/bin/env python3
"""
My new analysis script
"""
import path_config  # Always import this first

# Now you can import anything from the project
from app.data_injestion import train_df
from app.preprocessor.missing_value_analysis import MissingValueAnalyzer

def my_analysis():
    analyzer = MissingValueAnalyzer()
    analyzer.analyze(train_df)

if __name__ == "__main__":
    my_analysis()
```

### For Jupyter Notebooks

```python
# First cell
import path_config

# Second cell
from app.data_injestion import train_df, test_df
from app.eda.univariate_analysis import *
# ... continue with your analysis
```

### For Running Scripts Directly

```bash
# From any directory, these will now work:
python app/preprocessor/missing_value_analysis.py
python app/eda/univariate_analysis.py
python app/feature_engineering/feature_encoder.py
```

## Benefits

1. **No More Import Errors**: Eliminates `ModuleNotFoundError` permanently
2. **Consistent Imports**: Same import statements work everywhere
3. **Easy to Use**: Just one line addition to any script
4. **Automatic**: No manual path calculations needed
5. **Flexible**: Works from any directory in the project

## Migration Guide

To update existing scripts in the project:

1. Add `import path_config` at the top
2. Remove any manual `sys.path` manipulation
3. Use standard imports like `from app.module import something`

## Troubleshooting

If you still get import errors:

1. Make sure `path_config.py` is in the project root
2. Check that you're importing `path_config` before any app imports
3. Verify the project structure matches the expected layout

## Next Steps

1. Apply this pattern to all Python files in the project
2. Update any documentation or README files
3. Consider adding this to your project template for future projects

This solution ensures that imports work consistently across the entire project, whether running scripts directly, in Jupyter notebooks, or as part of a larger application.
