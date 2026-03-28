# SignSense - ASL Sign Language Learning Application

SignSense is an interactive American Sign Language (ASL) learning application that uses computer vision and machine learning to help users learn and practice ASL signs.

## Features

- **Interactive Learning**: Learn ASL letters and gestures through real-time hand tracking
- **Multiple Modes**: Play mode with stage-by-stage learning, debug mode for testing
- **Dynamic Signs**: Support for dynamic signs like J and Z that require motion
- **Recording Tools**: Record new signs for training custom models
- **Responsive UI**: Resizable window with fullscreen support

## Requirements

- Python 3.9 - 3.12 (MediaPipe does not support Python 3.14+)
- Webcam for hand tracking
- Operating System: Windows, macOS, or Linux

## Installation

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/signsense/signsense.git
cd signsense

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Using pyproject.toml (Modern Python)

```bash
# Clone the repository
git clone https://github.com/signsense/signsense.git
cd signsense

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install with pip (pyproject.toml will be used automatically)
pip install -e .
```

## Running the Application

### Method 1: Using the launcher script

```bash
python run.py
```

### Method 2: As a Python module

```bash
python -m signsense
```

### Method 3: After installation with pip

```bash
signsense
```

## Development

### Installing Development Dependencies

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Linting

```bash
# Check for syntax errors and undefined names
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Check all issues (warnings only)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

## Project Structure

```
signsense/
├── assets/
│   └── models/          # MediaPipe model files
├── config/              # Configuration files
├── detector/            # Hand and face tracking modules
├── ml/                  # Machine learning models and training
├── signs/               # Sign definitions and registry
├── ui/                  # User interface components
├── utils/               # Utility functions and logging
├── __init__.py
├── __main__.py          # Module entry point
└── main.py              # Main application entry point
```

## Usage

1. **Main Menu**: Choose between Play, About, or Quit
2. **Level Select**: Choose between Letters or Gestures
3. **Play Mode**: Follow the on-screen instructions to perform signs
4. **Debug Mode**: Test hand tracking and classification in real-time
5. **Record Mode**: Record new signs for training custom models

### Controls

- **ESC**: Exit current mode or quit application
- **F**: Toggle fullscreen mode
- **Mouse**: Click buttons and interact with menus
- **Mouse Wheel**: Scroll in About menu

## Building Standalone Application

### Prerequisites

- Windows 10/11 (64-bit) for .exe output
- Python 3.10-3.12 (not 3.14+)
- 8GB RAM minimum (more recommended for ML)

### Build Steps

```bash
# Install PyInstaller
pip install pyinstaller

# Build the standalone .exe
python build.py
```

The executable will be created at: `dist/SignSense/SignSense.exe`

### Build Options

```bash
# Clean previous build before building
python build.py --clean

# Debug build (shows console for error output)
python build.py --debug
```

### Distribution

The `dist/SignSense/` folder contains the complete application:
- `SignSense.exe` - Main executable
- All required DLLs and dependencies
- MediaPipe model files
- Configuration files

To distribute, zip the entire `SignSense` folder.

## Troubleshooting

### Camera Not Working

- Ensure your webcam is connected and not in use by another application
- Check that you have granted camera permissions to the application
- Try changing the camera index in `main.py` if you have multiple cameras

### Import Errors

- Make sure you have activated your virtual environment
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (must be 3.9-3.12)

### Performance Issues

- Close other applications that might be using the camera or GPU
- Reduce window size for better performance
- Check that your system meets the minimum requirements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for hand and face tracking
- OpenCV for computer vision capabilities
- PyTorch for machine learning models
