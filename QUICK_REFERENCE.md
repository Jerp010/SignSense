# SignSense Quick Reference

## Getting Started

### Virtual Environment
The project uses a dedicated virtual environment located at **`signsense/.venv`** with all required dependencies installed.

```bash
# Activate the virtual environment (Windows)
cd signsense
.venv\Scripts\activate

# Activate the virtual environment (Linux/macOS)
cd signsense
source .venv/bin/activate

# Verify installation
python --version  # Should be Python 3.9-3.12
pip list          # Should include all dependencies from requirements.txt
```

### Dependencies
All required packages are listed in **`signsense/requirements.txt`** and pre-installed in the virtual environment:
- `mediapipe>=0.10.31` - Hand tracking and landmark detection
- `opencv-python>=4.8.0` - Computer vision operations
- `numpy>=1.24.0` - Numerical computations
- `torch>=2.0.0` - Deep learning framework
- `pillow>=9.0.0` - Image processing
- `pandas>=1.5.0` - Data handling
- `matplotlib>=3.7.0` - Visualization
- `scikit-learn>=1.2.0` - Machine learning utilities
- `pyyaml>=6.0` - Configuration file handling

## Static Signs (MLP) - Single Poses

```bash
# 1. Record hand poses for letters
python -m signsense.ml.record_landmarks

# 2. Train model
python -m signsense.ml.train

# 3. Run app
python -m signsense.main  → Play Mode → Letters shown as they're detected
```

**Controls during recording:**
- Press `A-Z` to change letter label
- Perform the letter pose (hold consistently)
- Press `S` to save and exit
- Press `ESC` to exit without saving

**Result:** `ml/models/sign_mlp.pt` (MLP model for A-D, etc.)

---

## Dynamic Signs (LSTM) - Multi-Stage Movements

```bash
# 1. Record J sign with stage boundaries
python -m signsense.ml.dynamic_recorder

# 2. Train LSTM model
python -m signsense.ml.dynamic_train J

# 3. Record Z sign with stage boundaries
python -m signsense.ml.dynamic_recorder

# 4. Train LSTM model  
python -m signsense.ml.dynamic_train Z

# 5. Run app
python -m signsense.main  → Debug Mode → See stages progress as you perform J or Z
```

**Controls during recording:**
- Press `S` → Enter sign name (e.g., "J")
- Press `0-9` → Set stage number
- Press `SPACE` → Start/stop recording current stage
- Press `N` → Move to next stage
- Press `Q` → Save all stages and exit

**Result:** 
- `ml/models/dynamic_J.pt` (LSTM model for J)
- `ml/models/dynamic_Z.pt` (LSTM model for Z)

---

## File Structure

```
signsense/
├── .venv/                    ← Virtual environment with all dependencies
├── assets/
│   ├── models/              ← MediaPipe landmark detection models
│   ├── icons/               ← UI icon resources
│   └── demo_videos/         ← Example videos
├── config/
│   ├── dynamic_config.py    ← Configuration for dynamic signs
│   └── dynamic_signs.yaml   ← Dynamic sign definitions
├── detector/
│   ├── hand_tracker.py      ← Hand tracking using MediaPipe
│   ├── face_tracker.py      ← Face tracking (experimental)
│   └── asl_classifier_letters.py  ← ASL letter classification
├── ml/
│   ├── data/
│   │   ├── landmarks.npy    ← Static pose data (per letter)
│   │   └── dynamic/         ← Dynamic sequence data per sign
│   ├── models/
│   │   ├── sign_mlp.pt      ← Static sign model
│   │   ├── dynamic_J.pt     ← J dynamic model
│   │   └── dynamic_Z.pt     ← Z dynamic model
│   ├── record_landmarks.py  ← Record static poses
│   ├── dynamic_recorder.py  ← Record dynamic sequences
│   ├── train.py             ← Train static signs
│   ├── dynamic_train.py     ← Train dynamic signs
│   ├── model.py             ← DynamicSignMLP (CNN) architecture
│   └── dynamic_model.py     ← DynamicSignLSTM architecture
├── signs/
│   ├── dynamic_signs.py     ← Dynamic sign implementations
│   ├── dynamic_sign_factory.py  ← Factory for creating dynamic signs
│   ├── sign_registry.py     ← Sign registry for sign management
│   └── trainable_dynamic_signs.py  ← Trainable dynamic sign definitions
├── ui/
│   ├── menu.py              ← Main menu UI
│   ├── play_mode.py         ← Play mode UI
│   └── overlay.py           ← Visual overlay for detections
├── utils/
│   ├── logger.py            ← Logging utilities
│   └── smoothing.py         ← Detection smoothing algorithms
├── requirements.txt         ← Required Python packages
└── main.py                  ← Application entry point
```

---

## Model Architecture

### Static Signs (MLP)
```
Input (63 dims: 21 landmarks × 3 coords)
    ↓
Linear(63 → 256) + Batch Norm + ReLU + Dropout(0.3)
    ↓
Linear(256 → 128) + Batch Norm + ReLU + Dropout(0.2)
    ↓
Linear(128 → 64) + Batch Norm + ReLU
    ↓
Output (num_letters logits)
```

### Dynamic Signs (LSTM)
```
Sequence of frames (variable length)
    ↓
LSTM (63 → 128 hidden, 2 layers)
    ↓
Stage Head: Linear(128 → num_stages)    [predict current stage]
Transition Head: Linear(128 → 1)        [detect stage transition]
    ↓
Output (stage logits + transition confidence)
```

---

## Training Parameters

### Static Signs
- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** Until convergence (typically 50-100)
- **Batch size:** 32
- **Early stopping:** patience=15

### Dynamic Signs  
- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss (per frame)
- **Epochs:** Until convergence (typically 30-60)
- **Batch size:** 4
- **Train/Val split:** 80/20
- **Early stopping:** patience=15

---

## Common Commands

**Important:** Always activate the virtual environment first!

```bash
# Record static signs
python -m signsense.ml.record_landmarks

# Train static signs
python -m signsense.ml.train

# Record dynamic sign (e.g., J)
python -m signsense.ml.dynamic_recorder

# Train dynamic sign
python -m signsense.ml.dynamic_train J

# Test recognition
python -m signsense.main

# Debug mode (see all scores)
# (Launch from main menu)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named X" | Ensure you're using the correct virtual environment (signsense/.venv) |
| "Python was not found" | Activate the virtual environment before running commands |
| "ModuleNotFoundError: No module named 'signsense'" | Run commands from the root directory (d:/Jeff Code/SignSense) |
| Low accuracy (<80%) | Record more training data from different angles and distances |
| Model overfits | Add dropout, collect more varied data, or reduce model size |
| Slow training | Use GPU or reduce model size |
| "Model not found" | Train the model first using the appropriate training command |
| Dynamic signs don't work | Record complete sequences with clear stage boundaries and train the LSTM |

---

## Tips for Better Accuracy

### Static Signs
✓ Record poses from multiple angles  
✓ Vary distances from camera  
✓ Hold each pose consistently (~1 sec)  
✓ Try different lighting conditions  
✓ Use both left and right hands  

### Dynamic Signs
✓ Clearly define stage boundaries  
✓ Hold each stage for ~1 second  
✓ Perform movements at consistent speed  
✓ Record 5-10 complete sequences per sign  
✓ Be consistent with hand orientation  

---

**Need more help?** See [MODEL_GUIDE.md](MODEL_GUIDE.md) for detailed documentation.
