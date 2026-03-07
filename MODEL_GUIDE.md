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

## Dynamic Signs (Trainable)

### Overview

Dynamic signs like **J** and **Z** involve **multi-stage hand movements**. Instead of hardcoded state machines, SignSense now includes a **trainable LSTM model** that can learn arbitrary dynamic sign sequences.

**Key difference from static signs:**
- **Static (A-D, etc.):** Single hand pose → MLP → letter classification
- **Dynamic (J, Z):** Sequence of poses over time → LSTM → stage recognition → sign completion

### How It Works

1. **Record sequences** - Manually mark stage boundaries while performing the sign
2. **Train LSTM** - Model learns to recognize stage transitions and predict completion
3. **Inference** - Real-time stage tracking and sign detection

### Workflow: Record Dynamic Sign Data

```bash
python -m ml.dynamic_recorder
```

**Controls:**
- **S** - Set sign name (e.g., "J", "Z")
- **0-9** - Set current stage (e.g., Stage 0 = starting pose, Stage 1 = first move, etc.)
- **SPACE** - Start/stop recording for current stage
- **N** - Move to next stage
- **Q** - Save and exit

**Example: Recording J sign**

```
Sign: J (3 stages)
  Stage 0: Hold I handshape (pinky extended, fingers curled)
  Stage 1: Hook pinky downward
  Stage 2: Palm away, hold I position
```

Recording process:
```
1. Press S → Enter "J"
2. Press 0 → Set to Stage 0
3. Press SPACE → Start recording
   ... perform the I handshape hold ...
4. Press SPACE → Stop recording (auto-saves Stage 0 sequence)
5. Press N → Move to Stage 1
6. Press SPACE → Start recording
   ... perform the downward hook ...
7. Press SPACE → Stop recording
8. Repeat for Stage 2
9. Press Q → Save all sequences
```

**Best practices:**
- Record 3-5 complete sequences for each sign
- Perform at different speeds and angles
- Be consistent with stage boundaries
- Hold final stage for ~1 second

**Output:**
- Sequences saved to `ml/data/dynamic/<SIGN>/` as `.npy` files
- Metadata stored in `ml/data/dynamic/<SIGN>/metadata.json`

### Workflow: Train Dynamic Sign Model

After recording sequences, train the LSTM model:

```bash
python -m ml.dynamic_train J      # Train for J sign
python -m ml.dynamic_train Z      # Train for Z sign
```

**What happens:**
1. Loads all sequences from `ml/data/dynamic/<SIGN>/`
2. Groups frames by stage (each frame labeled with its stage)
3. Trains LSTM to predict current stage and detect transitions
4. Uses early stopping (default: patience=15 epochs)
5. Saves checkpoint to `ml/models/dynamic_<SIGN>.pt`

**Training details:**
- **Architecture:** 2-layer LSTM (hidden_size=128) → stage classifier + transition detector
- **Loss:** CrossEntropyLoss on per-frame stage predictions
- **Optimizer:** Adam (lr=0.001)
- **Scheduler:** CosineAnnealingLR
- **Train/Val split:** 80/20

**Output:**
```
============================================================
Training Dynamic Sign: J
============================================================
Loading sequences from ml/data/dynamic/J...
  Loaded 15 sequences for 3 stages
  Stage file distribution:
    Stage 0: 5 sequences
    Stage 1: 5 sequences
    Stage 2: 5 sequences
Max sequence length: 47 frames
Train: 12, Val: 3
Device: cuda

Training...
Epoch   1 | Loss: 0.8934 | Val Acc: 0.6234 ✓ (saved)
Epoch   2 | Loss: 0.6821 | Val Acc: 0.7123 ✓ (saved)
...
Epoch  42 | Loss: 0.1234 | Val Acc: 0.9456 ✓ (saved)
Early stopping at epoch 57

============================================================
Training Complete!
Best validation accuracy: 0.9456
Checkpoint: ml/models/dynamic_J.pt
============================================================
```

### Using Trained Dynamic Sign Models

The trained models are automatically loaded in the app:

```python
from signs.trainable_dynamic_signs import TrainedDynamicDetector

# Create detector for trained sign
j_detector = TrainedDynamicDetector("J")

# Update each frame
if j_detector.update(hand_landmarks, handedness="Right"):
    print("J sign completed!")
    j_detector.reset()

# Access stage info
print(j_detector.stage_label)        # "Stage 1/2 (conf: 0.92)"
print(j_detector.current_stage)       # 1
```

### Comparing Static vs Dynamic Training

| Aspect | Static (A-D) | Dynamic (J, Z) |
|--------|------------|--------------|
| **Data Type** | Single frames | Sequences |
| **Model** | MLP | LSTM |
| **Training Input** | 63-dim vector | Sequence of 63-dim vectors |
| **Output** | Letter classification | Stage sequence + completion |
| **Sample Size** | High (need hundreds) | Medium (need ~15-30 complete sequences) |
| **Training Time** | ~1-2 min | ~1-3 min |
| **Inference Speed** | Very fast (~1ms) | Moderate (~10ms) |

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
- Press letter key (A-Z) to set the active label
- While label is active, every detected hand frame is recorded
- Press `]` to save and exit
- Press `ESC` to exit without saving

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

### Static Sign Training (A-D)
1. **Variety:** Record poses from multiple angles and distances
2. **Consistency:** Hold each letter consistently (at least 20-30 frames per letter)
3. **Lighting:** Try different lighting conditions
4. **Handedness:** Collect data for both left and right hands if needed

### Dynamic Sign Training (J, Z)
1. **Clear stages:** Define distinct stages and hold each ~1 second
2. **Consistent pace:** Perform movements at similar speed across recordings
3. **Multiple takes:** Record 3-5 complete sequences per sign
4. **Clean transitions:** Make stage boundaries clear and recognizable
5. **Variety:** Perform at different angles and distances

---

## Common Issues

### Static Sign Issues

**Problem:** Low accuracy (< 80%)
- **Solution:** Collect more training data, especially for confusable letters (B↔D, M↔N)

**Problem:** Model overfits (high training, low validation accuracy)
- **Solution:** Adjust dropout rates in `ml/model.py` or collect more diverse data

### Dynamic Sign Issues

**Problem:** "Model not found" when using dynamic sign
- **Solution:** 
  ```bash
  python -m ml.dynamic_recorder      # Record sequences
  python -m ml.dynamic_train <SIGN>  # Train the model
  ```

**Problem:** Poor stage recognition or doesn't detect completion
- **Solution:** 
  - Record more complete sequences (target: 10-20 per sign)
  - Make stage boundaries clearer (more distinct poses)
  - Check training accuracy - aim for > 85% validation accuracy

**Problem:** LSTM model training is very slow
- **Solution:**
  - Use GPU: Install CUDA-enabled PyTorch
  - Reduce hidden_size in `DynamicSignLSTM` (default: 128)

### General Issues

**Problem:** Model training is slow (static signs)
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

### Modify Static Sign (MLP) Architecture

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

### Modify Dynamic Sign (LSTM) Architecture

Edit `ml/dynamic_model.py` to customize LSTM behavior:

```python
class DynamicSignLSTM(nn.Module):
    def __init__(self, num_stages: int, input_size: int = 63, hidden_size: int = 128):
        # Tune these parameters:
        self.lstm = nn.LSTM(
            input_size=63,
            hidden_size=256,         # ← Increase for more capacity
            num_layers=3,            # ← Add more layers
            dropout=0.4              # ← Adjust dropout
        )
```

### Add More Dynamic Signs

1. Create directory: `ml/data/dynamic/<SIGN_NAME>/`
2. Record sequences: `python -m ml.dynamic_recorder`
3. Train model: `python -m ml.dynamic_train <SIGN_NAME>`
4. Use in app: `TrainedDynamicDetector("<SIGN_NAME>")`

Example: Training Z sign
```bash
# Record Z movements with 4 stages
python -m ml.dynamic_recorder

# Train model
python -m ml.dynamic_train Z

# Model saved to: ml/models/dynamic_Z.pt
```


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

### Static Sign Training (Quick Start)
1. **Quick test:** `python main.py` → Debug mode to see classifier output
2. **Collect data:** `python -m ml.record_landmarks` → Record 5-10 samples per letter
3. **Train model:** `python -m ml.train` → Train for basic accuracy
4. **Iterate:** Collect more data in problem areas, retrain

### Dynamic Sign Training (Multi-Stage Movements)
1. **Record sequences:** `python -m ml.dynamic_recorder` → Record J or Z with stage markers
2. **Train model:** `python -m ml.dynamic_train J` → Trains LSTM for stage recognition
3. **Test in app:** `python main.py` → Dynamic signs now work!

### Combined Workflow
```bash
# Stage 1: Train static signs (A-D)
python -m ml.record_landmarks       # Record A, B, C, D
python -m ml.train                  # Train MLP

# Stage 2: Train dynamic signs (J, Z)
python -m ml.dynamic_recorder       # Record J sequences
python -m ml.dynamic_train J        # Train LSTM for J
python -m ml.dynamic_recorder       # Record Z sequences
python -m ml.dynamic_train Z        # Train LSTM for Z

# Stage 3: Run full app
python main.py                      # Now all signs work!
```

---

## References

- **MediaPipe Hand Landmark** - https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- **PyTorch MLP Tutorial** - https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- **PyTorch LSTM Tutorial** - https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
- **ASL Alphabet** - Standard 26-letter ASL (static + dynamic signs)

---

## File Reference

### Data Files
- **Static training:** `ml/data/landmarks.npy` - Recorded hand frames per letter
- **Dynamic training:** `ml/data/dynamic/<SIGN>/` - Sequence files for each stage
- **Trained models:** `ml/models/sign_mlp.pt` - MLP for static signs
- **Trained models:** `ml/models/dynamic_<SIGN>.pt` - LSTM for each dynamic sign

### Code Files
- **Static inference:** [detector/asl_classifier_letters.py](detector/asl_classifier_letters.py) - Real-time MLP inference
- **Static training:** [ml/train.py](ml/train.py) - Training loop for MLP
- **Static recording:** [ml/record_landmarks.py](ml/record_landmarks.py) - Data collection GUI

- **Dynamic inference:** [signs/trainable_dynamic_signs.py](signs/trainable_dynamic_signs.py) - Real-time LSTM inference
- **Dynamic training:** [ml/dynamic_train.py](ml/dynamic_train.py) - Training loop for LSTM
- **Dynamic recording:** [ml/dynamic_recorder.py](ml/dynamic_recorder.py) - Sequence recording GUI

---

**Last Updated:** March 7, 2026


