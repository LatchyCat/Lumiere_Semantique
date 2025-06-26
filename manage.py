#!/usr/bin/env python
"""
Lumière Sémantique Project - Root Management Utility.

This script acts as a proxy to the real Django manage.py script
located inside the 'backend' directory. This allows you to run
Django commands from the project root.
"""
import os
import sys
from pathlib import Path

def main():
    # 1. Find the project root and the backend directory
    project_root = Path(__file__).resolve().parent
    backend_dir = project_root / 'backend'

    # 2. Add the backend directory to Python's path
    # This is crucial so that Python can find 'lumiere_core.settings'
    sys.path.insert(0, str(backend_dir))

    # 3. Change the current working directory to the backend
    # This ensures that files like 'db.sqlite3' are found correctly
    os.chdir(backend_dir)

    # 4. Set the DJANGO_SETTINGS_MODULE environment variable
    # This is what the original manage.py does.
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lumiere_core.settings')

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # 5. Execute the command that was passed to this script
    execute_from_command_line(sys.argv)

if __name__ == "__main__":
    main()
