# SignSense Quick Reference

## Getting Started

### Virtual Environment
The project uses a dedicated virtual environment located at **`signsense/.venv`** with all required dependencies installed.

```bash
# Activate the virtual environment (Windows)
cd signsense
.venv\Scripts\activate

# Verify installation
python --version  # Should be Python 3.9-3.12
pip list          # Should include all dependencies from requirements.txt
```

### Dependencies
All required packages are listed in **`signsense/requirements.txt`** and pre-installed in the virtual environment.

---

## Recording Training Data

### Option 1: From UI Menu (Recommended)

```bash
# Run the application
python -m signsense.main

# Navigate: Main Menu → Debug → Record Data
# Choose: Static Landmarks or Dynamic Gestures
```

### Option 2: Direct Command Line

```bash
# Static signs (single poses)
python -m signsense.ml.record_landmarks

# Dynamic signs (multi-stage movements)
python -m signsense.ml.dynamic_recorder
```

---

## Static Signs (MLP) - Single Poses

Record individual hand poses for letters or custom labels.

### Recording Controls
| Key | Action |
|-----|--------|
| `T` | Type custom label name |
| `A-Z` | Quick select letter |
| `SPACE` | Start/stop recording |
| `[` | Save and exit |
| `ESC` | Exit without saving |

### Recording Process
1. Press `T` → type custom label (e.g., "hello", "thank_you") → Enter
2. Or press a letter key (A-Z) to select a letter
3. Hold the pose in front of camera
4. Press `SPACE` to start recording (status shows "●REC")
5. Perform the gesture multiple times
6. Press `SPACE` to stop recording
7. Press `[` to save and exit

### Adding More Samples to Existing Labels
Simply select an existing label (A-Z or custom name) and record more samples. The system automatically appends to existing data.

**Result:** `ml/data/landmarks.csv` (appends new rows)

---

## Dynamic Signs (LSTM) - Multi-Stage Movements

Record sequences of hand movements with stage labels.

### Recording Controls
| Key | Action |
|-----|--------|
| `T` | Type custom gesture name |
| `A-Z` | Quick select letter gesture |
| `S` | Toggle simple/complex mode |
| `0-9` | Set current stage |
| `SPACE` | Start/stop recording |
| `N` | Next stage (complex) / new sequence (simple) |
| `Q` | Save and exit |
| `ESC` | Exit without saving |

### Gesture Types

**Simple Mode** - Single-stage gesture (one continuous motion)
- Best for: Short gestures like "hello", "goodbye", "thanks"
- Press `S` to toggle to Simple mode

**Complex Mode** - Multi-stage gesture (distinct phases)
- Best for: Letters like J, Z that have clear stages
- Press `S` to toggle to Complex mode
- Use `0-9` to set stages

### Recording Process (Complex Mode - e.g., J Sign)

```
Sign: J (3 stages)
  Stage 0: Hold I handshape (pinky extended)
  Stage 1: Hook pinky downward
  Stage 2: Move hand to the right

Steps:
1. Press T → type "J" → Enter
2. Press S to ensure Complex mode
3. Press 0 → Set to Stage 0
4. Press SPACE → Start recording
   ... perform and hold Stage 0 ...
5. Press SPACE → Stop (auto-saves Stage 0)
6. Press N → Move to Stage 1
7. Press SPACE → Start recording
   ... perform Stage 1 motion ...
8. Press SPACE → Stop
9. Repeat for remaining stages
10. Press Q → Save all stages
```

### Adding More Samples
Select an existing gesture by name and record more sequences. The system appends to existing data automatically.

**Result:** `ml/data/dynamic/<GESTURE_NAME>/` with `.npy` files and `metadata.json`

---

## Training Models

### Train Static Sign Model (MLP)

```bash
python -m signsense.ml.train
```

**What happens:**
- Loads all data from `ml/data/landmarks.csv`
- Trains MLP to classify hand poses
- Saves to `ml/models/sign_mlp.pt`

### Train Dynamic Sign Model (LSTM)

```bash
# Train for a specific sign
python -m signsense.ml.dynamic_train J
python -m signsense.ml.dynamic_train Z
python -m signsense.ml.dynamic_train hello  # custom gesture
```

**What happens:**
- Loads sequences from `ml/data/dynamic/<SIGN>/`
- Trains LSTM to recognize stages
- Saves to `ml/models/dynamic_<SIGN>.pt`

---

## Running the Application

```bash
python -m signsense.main
```

### Menu Navigation
```
Main Menu
  ├── PLAY    → Level Select → Play Mode
  ├── DEBUG   → Debug Menu
  │     ├── Record Data → Static / Dynamic Recorder
  │     └── Train Model → (future)
  └── QUIT
```

---

## File Structure

```
signsense/
├── .venv/                    ← Virtual environment
├── assets/
│   ├── models/              ← MediaPipe detection models
│   ├── icons/               ← UI icons
│   └── demo_videos/         ← Example videos
├── config/
│   ├── dynamic_config.py    ← Dynamic sign configuration
│   └── dynamic_signs.yaml   ← Sign definitions
├── detector/
│   ├── hand_tracker.py      ← MediaPipe hand tracking
│   ├── face_tracker.py      ← Face tracking
│   └── asl_classifier_letters.py  ← Classification
├── ml/
│   ├── data/
│   │   ├── landmarks.csv   ← Static pose data (appends)
│   │   └── dynamic/        ← Dynamic sequence data
│   │       └── <GESTURE>/
│   │           ├── *.npy   ← Sequence files
│   │           └── metadata.json
│   ├── models/
│   │   ├── sign_mlp.pt    ← Static sign model
│   │   └── dynamic_*.pt   ← Dynamic sign models
│   ├── config/
│   │   ├── training.yaml  ← Training configuration
│   │   └── config_loader.py
│   ├── utils/
│   │   ├── checkpoint.py  ← Model checkpointing
│   │   ├── experiment_tracker.py
│   │   └── metrics.py
│   ├── record_landmarks.py  ← Static recorder
│   ├── dynamic_recorder.py  ← Dynamic recorder
│   ├── train.py             ← Static trainer
│   ├── dynamic_train.py     ← Dynamic trainer
│   ├── model.py             ← MLP architecture
│   └── dynamic_model.py     ← LSTM architecture
├── signs/
│   ├── dynamic_signs.py
│   ├── dynamic_sign_factory.py
│   ├── sign_registry.py
│   └── trainable_dynamic_signs.py
├── ui/
│   ├── menu.py              ← Menu system
│   ├── play_mode.py
│   └── overlay.py
├── utils/
│   ├── logger.py
│   └── smoothing.py
├── requirements.txt
└── main.py
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
Output (num_classes logits)
```

### Dynamic Signs (LSTM)
```
Sequence of frames (variable length)
    ↓
LSTM (63 → 128 hidden, 2 layers)
    ↓
Stage Head: Linear(128 → num_stages)
Transition Head: Linear(128 → 1)
    ↓
Output (stage + transition detection)
```

---

## Training Parameters

### Static Signs
- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** Until convergence
- **Batch size:** 32
- **Early stopping:** patience=15

### Dynamic Signs  
- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** Until convergence
- **Batch size:** 4
- **Train/Val split:** 80/20
- **Early stopping:** patience=15

---

## Common Commands

**Important:** Always activate the virtual environment first!

```bash
# Launch app with menu
python -m signsense.main

# Record static signs (from command line)
python -m signsense.ml.record_landmarks

# Record dynamic signs (from command line)
python -m signsense.ml.dynamic_recorder

# Train static model
python -m signsense.ml.train

# Train dynamic model
python -m signsense.ml.dynamic_train <SIGN_NAME>
```

---

## Troubleshooting

| Problem | Solution |
|--------|----------|
| "No module named X" | Activate virtual environment first |
| "Python was not found" | Use `.venv\Scripts\python.exe` or activate venv |
| Low accuracy | Record more data from different angles |
| Model overfits | Add dropout, more varied data |
| Slow training | Use GPU or reduce model size |
| "Model not found" | Train the model first |
| Dynamic signs don't work | Record complete sequences with clear stages |

---

## Tips for Better Accuracy

### Static Signs
✓ Record poses from multiple angles  
✓ Vary distances from camera  
✓ Hold each pose consistently (~1 sec)  
✓ Try different lighting conditions  
✓ Use both left and right hands  
✓ Record 100+ samples per label

### Dynamic Signs
✓ Clearly define stage boundaries  
✓ Hold each stage for ~1 second  
✓ Perform movements at consistent speed  
✓ Record 5-10 complete sequences per sign  
✓ Be consistent with hand orientation  
✓ Use Complex mode for multi-stage signs

---

## New Features (v0.8)

- **Custom gesture names** - Not limited to A-Z letters
- **Simple/Complex modes** - Toggle with S key
- **Add to existing** - Seamlessly append samples
- **UI integration** - Record from Debug menu
- **Configuration auto-creation** - New gestures get default config

---

**Need more help?** See [MODEL_GUIDE.md](MODEL_GUIDE.md) for detailed documentation.
