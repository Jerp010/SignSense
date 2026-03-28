"""
SignSense Build Script
======================
Build standalone .exe using PyInstaller.

Usage:
    python build.py              # Build in release mode
    python build.py --debug      # Build in debug mode
    python build.py --clean      # Clean previous build first
"""

import os
import sys
import subprocess
import shutil

def clean_build():
    """Remove previous build artifacts."""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Removing {dir_name}/...")
            shutil.rmtree(dir_name)
    
    # Clean PyInstaller cache
    if os.path.exists('signsense.spec'):
        # Remove .spec file's build directory
        spec_name = os.path.splitext('signsense.spec')[0]
        if os.path.exists(spec_name):
            print(f"Removing {spec_name}/...")
            shutil.rmtree(spec_name)

def build(debug=False):
    """Build the standalone application."""
    print("=" * 50)
    print("SignSense Standalone Build")
    print("=" * 50)
    
    # Check for required files
    if not os.path.exists('signsense/main.py'):
        print("ERROR: signsense/main.py not found!")
        return False
    
    if not os.path.exists('signsense.spec'):
        print("ERROR: signsense.spec not found!")
        return False
    
    # Build command
    cmd = ['pyinstaller', 'signsense.spec']
    
    if debug:
        cmd.append('--debug=all')
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Build failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("ERROR: PyInstaller not found.")
        print("Install with: pip install pyinstaller")
        return False

def main():
    # Parse arguments
    debug = '--debug' in sys.argv
    clean = '--clean' in sys.argv
    
    if clean:
        print("Cleaning previous build...")
        clean_build()
        print()
    
    # Run build
    success = build(debug=debug)
    
    if success:
        print()
        print("=" * 50)
        print("Build successful!")
        print("Output: dist/SignSense/SignSense.exe")
        print("=" * 50)
    else:
        print()
        print("=" * 50)
        print("Build FAILED")
        print("=" * 50)
        sys.exit(1)

if __name__ == '__main__':
    main()