# ASL Alphabet Directory

This directory contains American Sign Language (ASL) handshape images and videos for each letter of the English alphabet.

## Directory Structure

```
ASL_Alphabet/
├── A/
│   ├── view_1.png
│   ├── view_2.png
│   └── sign.mp4
├── B/
│   ├── view_1.png
│   └── sign.mp4
├── C/
│   └── ...
└── ... (26 folders total, one for each letter A-Z)
```

## Supported File Formats

- **Images:** `.png`, `.jpg`, `.jpeg`
- **Videos:** `.mp4`

## File Naming Convention

For consistency, please use the following naming pattern:

- **Images:** `{LETTER}_view_1.png`, `{LETTER}_view_2.jpg`, `{LETTER}_handshape.png`
- **Videos:** `{LETTER}_sign.mp4`, `{LETTER}_demo.mp4`, `{LETTER}_slow_motion.mp4`

## How to Add Files

1. Navigate to the appropriate letter folder (A, B, C, etc.)
2. Place your image or video files directly in the folder
3. Follow the naming convention above
4. Ensure files are clear and well-lit
5. Recommended image resolution: at least 300x300 pixels
6. Recommended video resolution: 720p or higher

## Preview Box Integration

The SignSense application automatically loads preview images from this directory structure. When you run the program, it will look for images in the format:

```
ASL_Alphabet/{LETTER}/view_{PAGE_NUMBER}.png
```

For example:
- `ASL_Alphabet/A/view_1.png` - First preview page for letter A
- `ASL_Alphabet/A/view_2.png` - Second preview page for letter A
- `ASL_Alphabet/A/view_3.png` - Third preview page for letter A

The preview box supports up to 3 pages per letter, which can be cycled using arrow keys or by clicking the left/right arrows in the preview box.

## Notes

- Each letter folder can contain multiple views of the ASL handshape
- Both static images and dynamic signing videos are supported
- The preview box will display a placeholder if no images are found
- Images should clearly show the handshape for effective learning
