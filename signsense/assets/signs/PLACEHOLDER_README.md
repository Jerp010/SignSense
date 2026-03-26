# SignSense Preview Images

This directory contains preview images for ASL signs and gestures.

## Directory Structure

Place images in the following structure:
```
signsense/assets/signs/
├── A/
│   ├── view_1.png
│   ├── view_2.png
│   └── view_3.png
├── B/
│   └── ...
├── HELLO/
│   └── ...
└── THANK YOU/
    └── ...
```

## Image Requirements

- Format: PNG
- Recommended size: 150x140 pixels
- Maximum: 160x150 pixels
- 3 view images per sign (view_1.png, view_2.png, view_3.png)

## Supported Signs

### Letters (A-Z)
- A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

### Gestures
- HELLO, THANK YOU, NAME, GOOD, HELP, WATER, YES, NO, BAD

## Usage

The PreviewBox in play_mode.py automatically loads these images if they exist.
If no images are found, a styled placeholder is displayed instead.