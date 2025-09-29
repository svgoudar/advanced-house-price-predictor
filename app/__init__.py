# # This file makes the app directory a Python package
import sys
import os
from pathlib import Path


# Setup project path automatically when app package is imported
# def _setup_project_path():
#     """Add project root to Python path if not already present."""
#     # Get project root (parent of app directory)
#     app_dir = Path(__file__).parent.absolute()
#     project_root = app_dir.parent
#     project_root_str = str(project_root)

#     if project_root_str not in sys.path:
#         sys.path.insert(0, project_root_str)

print(sys.path)
# # Auto-setup when package is imported
# _setup_project_path()

# # # Import data with proper error handling
# # try:
# #     from .data_injestion import train_df, test_df, sample_submission
# # except ImportError:
# #     # Fallback for direct script execution
# #     try:
# #         from data_injestion import train_df, test_df, sample_submission
# #     except ImportError as e:
# #         print(f"Warning: Could not import data_injestion: {e}")
# #         train_df = test_df = sample_submission = None
# print(train_df.head(3))
