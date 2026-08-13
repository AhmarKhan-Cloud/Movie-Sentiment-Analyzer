"""
paths.py
--------
Resolve all project paths from this file's location so train.py and app.py
work in Spyder even when the working directory is not the project folder.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def ensure_project_root():
    """Set working directory to the project root."""
    os.chdir(PROJECT_ROOT)


def project_path(*parts):
    """Build an absolute path inside the project directory."""
    return os.path.join(PROJECT_ROOT, *parts)
