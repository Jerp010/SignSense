#!/usr/bin/env python3
"""
SignSense Launcher Script

Run this script to start the SignSense application:
    python run.py

Or run as a module:
    python -m signsense
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from signsense.main import main

if __name__ == "__main__":
    main()
