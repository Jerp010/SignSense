# SignSense Model Guide

## Overview

SignSense uses a PyTorch MLP (Multi-Layer Perceptron) to classify American Sign Language (ASL) letters from hand landmarks captured by Google MediaPipe. This guide covers setup, usage, training, and inference.

---

## Quick Start

### 1. Install Dependencies

```bash
cd path/to/SignSense
pip install -r signsense/requirements.txt
```

Key dependencies:
- **torch** (>=2.0.0) - Deep learning framework
- **mediapipe** (>=0.10.31) - Hand landmark detection
- **opencv-python** (>=4.8.0) - Video capture and processing
- **numpy** (>=1.24.0) - Numerical computing

### 2. Run the Application

```bash
cd signsense
python main.py
```

---

## Architecture

### Model: SignMLP

**Input:** 63 normalized coordinates (21 hand landmarks × 3 dimensions: x, y, z)

**Architecture:**
```
Input (63)
  ↓
Linear(63 → 256) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Linear(128 → 64) + BatchNorm + ReLU
  ↓
Linear(64 → num_classes)
  ↓
Output (logits for each letter)
```

### Landmark Normalization

The model uses **normalized landmarks** to handle different hand sizes and distances from camera:

1. **Translation:** Translate all landmarks so wrist (landmark[0]) is at origin
2. **Scaling:** Scale by distance from wrist to middle MCP joint (landmark[0] → landmark[9])
3. **Flattening:** Convert 21 landmarks to 63-element feature vector: `[x₀,y₀,z₀, x₁,y₁,z₁, ..., x₂₀,y₂₀,z₂₀]`

This makes the model **rotation-invariant** and **scale-invariant**.

---

## Workflow: Training Your Own Model

### Step 1: Record Training Data

Record hand landmark data for ASL letters:

```bash
python -m ml.record_landmarks
```

**What this does:**
- Opens your webcam
- Detects hand landmarks using MediaPipe
- Saves normalized landmark frames to `ml/data/` directory
- Creates separate files for each letter

**Key options during recording:**
- Press `c` to start/stop capturing for current letter
- Press `n` to move to next letter
- Press `q` to quit

**Output:** `.npy` files in `ml/data/` containing numpy arrays of normalized landmarks

### Step 2: Train the Model

After collecting data for multiple letters, train the model:

```bash
python -m ml.train
```

**What this does:**
- Loads all `.npy` files from `ml/data/`
- Splits data into training/validation sets (80/20)
- Trains the SignMLP for multiple epochs
- Saves best model to `ml/models/sign_mlp.pt`
- Logs accuracy and loss metrics

**Training details:**
- **Optimizer:** Adam (learning rate 0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** Trains until convergence (typically 50-100 epochs)
- **Batch size:** 32
- **Device:** CPU or GPU (auto-detected)

**Output files:**
- `ml/models/sign_mlp.pt` - PyTorch checkpoint containing:
  - Model weights
  - Landmark normalizer state
  - Letter→index mapping
  - Index→letter mapping

---

## Inference

### Using the Classifier in Code

```python
from detector.asl_classifier_letters import ASLClassifierLetters

# Initialize classifier (loads model automatically)
classifier = ASLClassifierLetters()

# Classify hand landmarks (from MediaPipe)
result = classifier.classify(
    landmarks=hand_landmarks,           # 21 MediaPipe NormalizedLandmark objects
    handedness="Right",                 # Optional: "Left" or "Right"
    target_letter=None                  # Optional: for play-mode scoring
)

# Result format (if confidence > threshold):
# {
#     "letter": "A",
#     "confidence": 0.92,
#     "scores": {
#         "A": 0.92,
#         "B": 0.03,
#         "C": 0.02,
#         ...
#     }
# }

# Returns None if confidence below threshold (default: 0.6)
```

### Confidence Threshold

Adjust confidence threshold for stricter/looser predictions:

```python
classifier.min_confidence = 0.7  # Stricter (default is 0.6)
classifier.min_confidence = 0.5  # Looser
```

### Handling Missing Model

If `sign_mlp.pt` doesn't exist:
- The classifier logs a warning with next steps
- Returns `None` for all frames (graceful degradation)
- App continues running unchanged

```
[ASLClassifierLetters] Warning: Model not found at ml/models/sign_mlp.pt
  Run: python -m ml.record_landmarks  (to collect data)
  Then: python -m ml.train            (to train the model)
  Until then, the classifier will return None (no detection).
```

---

## Application Modes

### 1. **MAIN_MENU**
- Camera OFF
- Shows main menu with options: Play, Debug, Quit

### 2. **LEVEL_SELECT**
- Camera OFF
- Select learning level (currently only "Letters" available)

### 3. **PLAY MODE**
- Camera ON
- Real-time ASL letter recognition
- Stage-by-stage learning progression
- Shows confidence scores and visual feedback
- Press ESC to return to menu

### 4. **DEBUG MODE**
- Camera ON
- Raw classifier output with full score bars
- Shows confidence for all letters in real-time
- Useful for testing model performance
- Press ESC to return to menu

---

## File Structure

```
signsense/
├── main.py                          # Application entry point & state machine
├── detector/
│   ├── hand_tracker.py             # MediaPipe hand landmark detection
│   ├── face_tracker.py             # Face detection (unused currently)
│   └── asl_classifier_letters.py   # MLP inference wrapper ← You interact with this
├── ml/
│   ├── model.py                    # SignMLP architecture & normalizer
│   ├── record_landmarks.py         # Data recording script
│   ├── train.py                    # Training script
│   ├── data/                       # 📁 Stores .npy training files
│   └── models/
│       └── sign_mlp.pt             # 📁 Trained model checkpoint
├── signs/
│   ├── sign_registry.py            # Active letters registry
│   └── dynamic_signs.py            # J, Z sign dynamics (future)
└── utils/
    ├── logger.py                   # Logging system
    └── smoothing.py                # Prediction smoothing
```

---

## Training Tips

### Data Collection Best Practices
1. **Variety:** Record poses from multiple angles and distances
2. **Consistency:** Hold each letter consistently (at least 20-30 frames per letter)
3. **Lighting:** Try different lighting conditions
4. **Handedness:** Collect data for both left and right hands if needed

### Common Issues

**Problem:** Low accuracy (< 80%)
- **Solution:** Collect more training data, especially for confusable letters (B↔D, M↔N)

**Problem:** Model overfits (high training, low validation accuracy)
- **Solution:** Adjust dropout rates in `ml/model.py` or collect more diverse data

**Problem:** Model training is slow
- **Solution:** 
  - Use GPU if available (PyTorch auto-detects CUDA)
  - Install PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
  - Reduce dataset size

**Problem:** "ModuleNotFoundError: No module named 'torch'"
- **Solution:** 
  ```bash
  pip install torch
  # or for GPU support:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

---

## Advanced Usage

### Customize Active Letters

Edit `signs/sign_registry.py` to change which letters are trained/recognized:

```python
# Currently supports A-P (15 static letters)
ACTIVE_SIGNS = {
    "A": SignInfo(...),
    "B": SignInfo(...),
    # ... add or remove letters here
}
```

### Modify Model Architecture

Edit `ml/model.py` to change model capacity:

```python
class SignMLP(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 512),      # ← Change hidden layer sizes
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),         # ← Adjust dropout
            # ... modify as needed
        )
```

### Visualize Training

The training script logs metrics. To visualize them:
- Redirect output to a file: `python -m ml.train > training.log`
- Monitor loss and accuracy curves

---

## Performance Optimization

### For Real-Time Performance
1. Use GPU: Install CUDA-enabled PyTorch
2. Reduce model size: Fewer hidden units or lighter architecture
3. Use quantization: Convert model to 8-bit integers for faster inference

### For Better Accuracy
1. Collect more diverse training data
2. Increase model capacity (more hidden units)
3. Use data augmentation during training
4. Train longer with learning rate scheduling

---

## Troubleshooting

### Application won't start
```bash
py -3.11 -m venv .venv  # Recreate venv with Python 3.11
.venv\Scripts\activate.ps1
pip install -r signsense/requirements.txt
python main.py
```

### MediaPipe errors
- Ensure Python 3.9-3.12 (3.14+ not supported by MediaPipe)
- Reinstall MediaPipe: `pip install --upgrade mediapipe`

### GPU not being used
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- If False, reinstall PyTorch with CUDA support

---

## Next Steps

1. **Quick test:** `python main.py` → Debug mode to see classifier output
2. **Collect data:** `python -m ml.record_landmarks` → Record 5-10 samples per letter
3. **Train model:** `python -m ml.train` → Train for basic accuracy
4. **Iterate:** Collect more data in problem areas, retrain

---

## References

- **MediaPipe Hand Landmark** - https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- **PyTorch MLP Tutorial** - https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- **ASL Alphabet** - Standard 26-letter ASL (static + dynamic signs)

