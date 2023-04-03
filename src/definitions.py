"""
File containing static definitions to be used by the segmentation project.
These definitions should be constant once the project is installed and running.
"""
import os

# This definitions.py file should be located in ROOT/src so it is
# located 1 directory below the root directory. Hence, we need to navigate
# up one directory from this file's absolute path (__file__) to get the
# root directory. Navigate up one directories with ".."
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

# Specify other paths in project relative to ROOT_DIR
DATA_DIR = os.path.join(ROOT_DIR, "data")
