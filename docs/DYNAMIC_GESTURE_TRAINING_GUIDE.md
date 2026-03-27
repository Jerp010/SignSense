# SignSense Dynamic Gesture Training Guide

## Table of Contents
1. [Quick Start](#1-quick-start)
2. [Data Collection](#2-data-collection)
3. [Training Configuration](#3-training-configuration)
4. [Model Architecture](#4-model-architecture)
5. [Training Procedure](#5-training-procedure)
6. [Validation & Testing](#6-validation--testing)
7. [Converting Stage 2 Gestures to Dynamic Motion](#7-converting-stage-2-gestures-to-dynamic-motion)
8. [Advanced Topics](#8-advanced-topics)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Quick Start

### Essential Commands

```bash
# Record new dynamic gesture
python -m ml.dynamic_recorder

# Train a dynamic gesture
python -m ml.dynamic_train <GESTURE_NAME>

# Example: Train the J sign
python -m ml.dynamic_train J

# Example: Train the Z sign
python -m ml.dynamic_train Z
```

### Quick Workflow Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  1. Record Data      │────▶│  2. Train Model     │────▶│  3. Validate        │
│  ml.dynamic_recorder│     │  ml.dynamic_train   │     │  Check accuracy     │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
 ml/data/dynamic/<SIGN>/    ml/models/dynamic_<SIGN>.pt    Performance metrics
 *.npy files                Trained LSTM model              Validation accuracy
```

---

## 2. Data Collection

### 2.1 Recording Prerequisites

| Requirement | Description | Verification |
|-------------|-------------|--------------|
| Python | 3.8+ | `python --version` |
| OpenCV | Camera capture | `pip show opencv-python` |
| MediaPipe | Hand tracking | `pip show mediapipe` |
| Webcam | Camera at index 0 | System check |
| Lighting | Good hand visibility | Visual check |

### 2.2 Data Format Specification

#### File Storage Structure
```
ml/data/dynamic/<GESTURE_NAME>/
├── metadata.json           # Gesture metadata
├── <GESTURE>_s0_0.npy     # Stage 0, sequence 0
├── <GESTURE>_s0_1.npy     # Stage 0, sequence 1
├── <GESTURE>_s1_0.npy     # Stage 1, sequence 0
└── ...
```

#### NPY File Format
- **Shape**: `(num_frames, 63)` where:
  - `num_frames`: Variable (30-150 frames typical)
  - `63`: 21 hand landmarks × 3 coordinates (x, y, z)
- **Data Type**: `float32`
- **Normalization**: Wrist-relative, scale-invariant

#### Metadata JSON Structure
```json
{
  "sign_name": "J",
  "gesture_type": "complex",
  "num_stages": 3,
  "sequences_per_stage": {
    "0": 3,
    "1": 3,
    "2": 3
  },
  "total_sequences": 9,
  "recorded_at": "2026-03-26T04:15:43.121693",
  "description": "Letter J in ASL"
}
```

### 2.3 Minimum Data Requirements

| Gesture Type | Minimum Samples | Recommended Samples | Per Stage |
|--------------|-----------------|---------------------|------------|
| Simple (single-stage) | 10 | 15-20 | 1 |
| Complex (multi-stage) | 15 | 20-30 | 5-10 |

### 2.3.1 What is a Sample?

A **sample** equals **one complete gesture recording** stored as a single `.npy` file.

**Example from existing data:**
- **J sign**: 9 samples → 9 `.npy` files (e.g., `J_s0_0.npy`, `J_s1_0.npy`, etc.)
- **Z sign**: 12 samples → 12 `.npy` files

Each `.npy` file contains a numpy array of shape `(num_frames, 63)` representing one complete gesture execution from start to finish.

### 2.3.2 Applying to Stage 2 Gestures

Yes! The same `dynamic_recorder.py` tool used for J and Z can record all 9 Stage 2 gestures (HELLO, THANK YOU, NAME, GOOD, HELP, WATER, YES, NO, BAD).

**Recommended samples per Stage 2 gesture:** 15-20 samples per gesture

### 2.4 Recording Instructions

1. **Start the recorder**:
   ```bash
   python -m ml.dynamic_recorder
   ```

2. **Set gesture name**:
   - Press `T` to enter text mode
   - Type gesture name (e.g., "HELLO", "THANK_YOU")
   - Press Enter to confirm

3. **Choose gesture type**:
   - Press `S` to toggle simple/complex mode
   - Simple: Single continuous motion
   - Complex: Multiple distinct phases

4. **Record sequences**:
   - Press SPACE to start recording
   - Perform the gesture in front of camera
   - Press SPACE to stop
   - Press `N` to move to next stage (complex) or new sequence (simple)

5. **Save and exit**:
   - Press `Q` to save all recordings and exit

### 2.5 Best Practices for Data Collection

1. **Consistency**: Maintain consistent hand position and movement speed
2. **Variety**: Record multiple takes with slight variations
3. **Lighting**: Ensure good, even lighting on hands
4. **Background**: Use plain, non-cluttered background
5. **Framing**: Keep hand centered in camera frame
6. **Duration**: Each sequence should be 1-4 seconds (30-120 frames)

---

## 3. Training Configuration

### 3.1 Configuration Files

| File | Purpose |
|------|---------|
| `signsense/config/dynamic_signs.yaml` | Per-gesture configuration |
| `signsense/ml/config/training.yaml` | Global training defaults |

### 3.2 YAML Configuration Structure

```yaml
dynamic_signs:
  GESTURE_NAME:
    type: "complex"           # or "simple"
    description: "Description"
    
    detector:               # Detection parameters
      i_hold_frames: 6
      down_threshold: 0.06
      finish_hold_frames: 8
      phase_timeout: 70
    
    training:               # Training parameters
      sequence_length: 120
      batch_size: 8
      epochs: 50
      learning_rate: 0.001
      hidden_size: 128
    
    stages:                 # Stage descriptions
      0: "Stage 1 description"
      1: "Stage 2 description"
      2: "Stage 3 description"
```

### 3.3 Training Parameters Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `sequence_length` | 120 | 30-300 | Maximum frames to pad sequences to |
| `batch_size` | 8 | 1-32 | Training batch size |
| `epochs` | 50 | 10-200 | Maximum training iterations |
| `learning_rate` | 0.001 | 0.0001-0.01 | Optimizer learning rate |
| `hidden_size` | 128 | 64-256 | LSTM hidden dimension |

### 3.4 Early Stopping Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patience` | 15 | Epochs to wait before early stop |
| `min_delta` | 0.001 | Minimum improvement threshold |

---

## 4. Model Architecture

### 4.1 LSTM Architecture Overview

```
Input Sequence (seq_len x 63)
         │
         ▼
┌────────────────────────┐
│   LSTM Layer 1        │  hidden_size=128, dropout=0.3
│   (bidirectional=false)│
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│   LSTM Layer 2        │
└────────┬───────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌──────────┐
│ Stage │ │Transition│
│ Head  │ │  Head    │
└───┬───┘ └────┬─────┘
    ▼          ▼
 Stage      Transition
 Probabilities
```

### 4.2 Model Components

#### Stage Classification Head
- Input: LSTM hidden state (128 dimensions)
- Layers: Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→num_stages)
- Output: Per-frame stage probabilities

#### Transition Detection Head
- Input: LSTM hidden state (128 dimensions)
- Layers: Linear(128→32) → ReLU → Dropout(0.2) → Linear(32→1) → Sigmoid
- Output: Transition confidence (0-1)

### 4.3 Checkpoint Format

```python
{
    "model_state": {...},      # Model weights
    "num_stages": 3,           # Number of stages
    "stage_names": {           # Stage labels
        0: "Stage 0",
        1: "Stage 1",
        2: "Stage 2"
    }
}
```

Saved to: `ml/models/dynamic_<GESTURE>.pt`

---

## 5. Training Procedure

### 5.1 Full Training Command

```bash
# Basic training
python -m ml.dynamic_train J

# With custom parameters
python -m ml.dynamic_train J --epochs 100 --lr 0.0005 --batch-size 16

# With specific data directory
python -m ml.dynamic_train J --data ml/data/dynamic --output ml/models
```

### 5.2 Training Pipeline Steps

```
1. Load sequences from ml/data/dynamic/<SIGN>/
   │
   ▼
2. Group by stage, create per-frame labels
   │
   ▼
3. Pad sequences to equal length, create attention masks
   │
   ▼
4. Train/val split (80/20)
   │
   ▼
5. Train LSTM with cross-entropy loss on frames
   │
   ▼
6. Early stopping on validation accuracy
   │
   ▼
7. Save checkpoint: ml/models/dynamic_<SIGN>.pt
   │
   ▼
8. Print per-stage accuracy
```

### 5.3 Training Output

During training, you will see:
```
Epoch   1/50 | Loss: 1.234 | Val Acc: 0.456 | LR: 0.001000
Epoch   2/50 | Loss: 0.876 | Val Acc: 0.623 | LR: 0.000995
...
Epoch  15/50 | Loss: 0.234 | Val Acc: 0.891 * (saved)
```

### 5.4 Training Artifacts

| File | Location | Description |
|------|----------|-------------|
| Model checkpoint | `ml/models/dynamic_<SIGN>.pt` | Best model weights |
| Training curves | `ml/models/training_curves_<SIGN>.png` | Loss/accuracy plots |

---

## 6. Validation & Testing

### 6.1 Training Success Criteria

| Metric | Target | Acceptable |
|--------|--------|------------|
| Validation Accuracy | >90% | >80% |
| Loss | <0.3 | <0.5 |
| Overfitting gap | <10% | <15% |

### 6.2 Visual Inspection

After training, check:
1. **Training curves**: `ml/models/training_curves_<SIGN>.png`
   - Loss should decrease steadily
   - Validation accuracy should increase and stabilize
   
2. **Checkpoint exists**: `ml/models/dynamic_<SIGN>.pt`

### 6.3 Testing with Play Mode

```bash
# Start SignSense application
python -m signsense.main

# Or use play mode directly
# Navigate to dynamic signs section
# Test recognized gestures in real-time
```

---

## 7. Converting Stage 2 Gestures to Dynamic Motion

### 7.1 Overview

The Stage 2 (Gesture Mode) in SignSense includes 9 ASL gestures that currently use **static hold-based detection**:
- HELLO, THANK YOU, NAME, GOOD, HELP, WATER, YES, NO, BAD

These gestures can be converted to use **dynamic motion recognition** instead, similar to how J and Z signs work in Stage 1 (Letter Mode).

### 7.2 Current vs Dynamic Detection

| Aspect | Current (Static) | Dynamic Motion |
|--------|-----------------|----------------|
| Detection | Single frame hand shape | Sequence of frames |
| Recognition | MLP classifier | LSTM neural network |
| Duration | Instant (hold 3 seconds) | Requires motion completion |
| Movement | Not used | Core feature |

### 7.3 Gesture Motion Patterns

Each Stage 2 gesture has a defined movement pattern that can be captured as a dynamic sequence:

| Gesture | Hand Shape | Motion Pattern | Stages |
|----------|------------|----------------|--------|
| HELLO | FLAT_HAND_5 | touch_forehead_move_out | 2-3 |
| THANK YOU | FLAT_HAND_5 | chin_to_forward | 2 |
| NAME | N_SHAPE | cheek_slight_move | 2 |
| GOOD | G_SHAPE | chin_to_down_out | 2 |
| HELP | OPEN_HAND_5 | body_to_up_out | 2 |
| WATER | W_SHAPE | chin_to_forward | 2 |
| YES | FIST_THUMB_UP | head_nod | 2 |
| NO | INDEX_FINGER_1 | head_side_to_side | 2 |
| BAD | Y_SHAPE | chin_to_down_out | 2 |

### 7.4 Conversion Steps

#### Step 1: Record Motion Data

For each gesture you want to convert:

```bash
# Start recorder
python -m ml.dynamic_recorder

# Set gesture name (e.g., HELLO)
T -> type "HELLO" -> Enter

# Set to complex mode (S key) for multi-stage motion
# Or keep simple for single continuous motion

# Record multiple samples (15-20 recommended)
SPACE -> perform gesture -> SPACE -> N -> repeat

# Save and exit
Q
```

#### Step 2: Configure in YAML

Add the gesture to `signsense/config/dynamic_signs.yaml`:

```yaml
dynamic_signs:
  HELLO:
    type: "complex"  # or "simple"
    description: "Touch forehead then move out"
    training:
      sequence_length: 120
      batch_size: 8
      epochs: 50
      learning_rate: 0.001
      hidden_size: 128
    stages:
      0: "Touch forehead"
      1: "Move hand outward"
```

#### Step 3: Train the Model

```bash
# Train the dynamic gesture
python -m ml.dynamic_train HELLO
```

#### Step 4: Update Play Mode Integration

The play mode currently uses `GestureDetector` for Stage 2. To use dynamic detection:

1. **Option A - Use DynamicSignFactory**: Modify `play_mode.py` to use dynamic detectors for gesture stages

2. **Option B - Hybrid Approach**: Keep static detection as fallback, use dynamic when model exists

Key files to modify:
- [`signsense/ui/play_mode.py`](signsense/ui/play_mode.py) - Update `GestureStage` handling
- [`signsense/signs/dynamic_sign_factory.py`](signsense/signs/dynamic_sign_factory.py) - Add gesture support

### 7.5 Data Collection Requirements for Stage 2 Gestures

| Gesture | Min Samples | Stages | Typical Duration |
|---------|-------------|--------|-------------------|
| HELLO | 15-20 | 2-3 | 1-3 seconds |
| THANK YOU | 15-20 | 2 | 1-2 seconds |
| NAME | 15-20 | 2 | 1-2 seconds |
| GOOD | 15-20 | 2 | 1-2 seconds |
| HELP | 15-20 | 2 | 1-3 seconds |
| WATER | 15-20 | 2 | 1-2 seconds |
| YES | 15-20 | 2 | 1-2 seconds |
| NO | 15-20 | 2 | 1-2 seconds |
| BAD | 15-20 | 2 | 1-2 seconds |

### 7.6 Best Practices for Motion Recording

1. **Capture the full motion**: Start position → end position
2. **Consistent speed**: Maintain similar speed across samples
3. **Clear transitions**: Make stage transitions distinct
4. **Record multiple angles**: Slight variations improve robustness
5. **Include handshape changes**: If the gesture changes handshape mid-motion, capture that

### 7.7 Integration Considerations

When converting Stage 2 gestures to dynamic motion:

1. **Detection timing changes**: From instant to motion-completion based
2. **User feedback**: Update UI to show "performing motion" vs "hold gesture"
3. **Scoring adjustment**: Consider motion accuracy vs completion time

---

## 8. Advanced Topics

### 7.1 Adding Custom Gestures

1. **Record the gesture** using `ml.dynamic_recorder`
2. **Configure in YAML** by adding entry to `signsense/config/dynamic_signs.yaml`
3. **Train** using `python -m ml.dynamic_train <GESTURE_NAME>`

### 7.2 Hyperparameter Tuning

For better performance, experiment with:

| Parameter | Recommended Search Range |
|-----------|--------------------------|
| learning_rate | [0.0005, 0.001, 0.002] |
| hidden_size | [64, 128, 256] |
| batch_size | [4, 8, 16] |
| dropout | [0.2, 0.3, 0.4] |

### 7.3 Adding More Samples

To improve an existing model:
1. Run recorder for the same gesture name
2. Record additional sequences
3. Retrain the model
4. New samples are automatically added to existing data

---

## 8. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "No stage sequences found" | Wrong directory structure | Ensure files named `<SIGN>_s<STAGE>_<INDEX>.npy` |
| Low accuracy | Insufficient samples | Record more sequences (15-20 per stage) |
| Model overfitting | Too few samples or too many epochs | Reduce epochs or add more data |
| Camera not opening | Webcam in use or not available | Check camera index in code |
| Memory errors | Sequence too long | Reduce `sequence_length` in config |

### Data Verification

Check your recorded data:
```bash
# List files
ls ml/data/dynamic/<GESTURE>/

# Check metadata
cat ml/data/dynamic/<GESTURE>/metadata.json

# Inspect numpy file
python -c "import numpy as np; print(np.load('ml/data/dynamic/<GESTURE>/<FILE>.npy').shape)"
```

### Getting Help

1. Check existing documentation:
   - `docs/DYNAMIC_RECORDER_GUIDE.md`
   - `docs/MODEL_GUIDE.md`
   - `docs/QUICK_REFERENCE.md`

2. Review configuration files:
   - `signsense/config/dynamic_signs.yaml`
   - `signsense/ml/config/training.yaml`

---

## Appendix: Example Gestures

### J Sign (3 Stages)
- **Type**: Complex
- **Stages**: 
  0. Hold I (pinky up)
  1. Hook pinky down
  2. Palm away, hold I
- **Data**: `ml/data/dynamic/J/`
- **Samples**: 9 sequences (3 per stage)

### Z Sign (4 Stages)
- **Type**: Complex
- **Stages**:
  0. Start position
  1. Horizontal stroke
  2. Diagonal stroke
  3. End position
- **Data**: `ml/data/dynamic/Z/`
- **Samples**: 12 sequences (2-5 per stage)

---

## Quick Reference Card

```bash
# Recording
python -m ml.dynamic_recorder

# Training
python -m ml.dynamic_train <GESTURE>

# Data location
ml/data/dynamic/<GESTURE>/

# Model location
ml/models/dynamic_<GESTURE>.pt

# Configuration
signsense/config/dynamic_signs.yaml
```

---

*Document Version: 1.0*
*Last Updated: 2026-03-26*
*Project: SignSense Dynamic Gesture Recognition*
