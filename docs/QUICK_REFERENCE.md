# SignSense Quick Reference Guide

A comprehensive command reference for training, recording, and using SignSense ASL recognition models.

---

## Table of Contents

1. [Application Commands](#application-commands)
2. [Data Recording](#data-recording)
3. [Model Training](#model-training)
4. [Evaluation & Inference](#evaluation--inference)
5. [Configuration Options](#configuration-options)
6. [Troubleshooting Flags](#troubleshooting-flags)
7. [Common Usage Examples](#common-usage-examples)

---

## Application Commands

### Run SignSense Application

```bash
# Start the SignSense GUI application
python -m signsense.main
# or
python main.py
```

The application launches with a menu-driven interface:
- **Play** - Enter letter/gesture learning mode
- **About** - View application information  
- **Record** - Access data recording options

### Runtime Controls (Play Mode)

| Key | Action |
|-----|--------|
| `F` | Toggle fullscreen mode |
| `ESC` | Exit current mode to menu |
| Mouse scroll | Navigate menus |
| Mouse click | Select options |

---

## Data Recording

### Static Landmark Recording

Record hand landmarks for static ASL letters.

```bash
python -m signsense.ml.record_landmarks
```

**Controls:**

| Key | Action |
|-----|--------|
| `T` | Enter text mode for custom label |
| `A-Z` | Quick-select letter label |
| `SPACE` | Start/stop recording for current label |
| `[` | Save and exit |
| `ESC` | Exit without saving |
| Window close | Exit without saving |

**Output:**
- Appends to `ml/data/landmarks.csv`
- Format: `label, x0, y0, z0, ..., x20, y20, z20` (64 columns)

### Dynamic Sign Recording

Record motion sequences for dynamic ASL signs (J, Z, gestures).

```bash
python -m signsense.ml.dynamic_recorder
```

**Controls:**

| Key | Action |
|-----|--------|
| `T` | Enter text mode for custom gesture name |
| `A-Z` | Select predefined gesture |
| `0-9` | Set current stage (complex gestures) |
| `S` | Toggle simple/complex mode |
| `SPACE` | Start/stop recording |
| `N` | Next stage (complex) / new sequence (simple) |
| `P` | Previous stage |
| `Q` | Save and exit |
| `ESC` | Exit without saving |

**Output:**
- Saves to `ml/data/dynamic/<GESTURE_NAME>/`
- Creates `.npy` files for each sequence
- Metadata: `ml/data/dynamic/<GESTURE_NAME>/metadata.json`

### Dynamic Sign Recording Modes

```bash
# Record simple gesture (single-stage motion like letter J)
python -m signsense.ml.dynamic_recorder
# Press S to toggle to simple mode, select J, press SPACE to record

# Record complex gesture (multi-stage like letter Z or custom gestures)
python -m signsense.ml.dynamic_recorder  
# Press S to toggle to complex mode, select Z, use 0-9 for stages
```

---

## Model Training

### Static MLP Classifier

Train the ASL letter classifier using landmark data.

```bash
python -m signsense.ml.train
```

**Example with custom parameters:**

```bash
# Train with custom data and parameters
python -m signsense.ml.train --data ml/data/landmarks.csv --epochs 100 --lr 0.001 --batch-size 64

# Train with GPU acceleration
python -m signsense.ml.train --device cuda --epochs 50 --batch-size 128

# Disable data augmentation
python -m signsense.ml.train --no-augmentation
```

**Pipeline:**
1. Load CSV, build label_map from unique labels
2. Data augmentation: noise, x-flip, scale jitter
3. Train/val split: 85/15 stratified
4. Train with AdamW, CosineAnnealingLR scheduler
5. Early stopping: patience=15 on val accuracy
6. Save checkpoint: `ml/models/static_mlp.pt`

### Enhanced Dynamic LSTM Model

Train the enhanced bidirectional LSTM for dynamic sign recognition.

```bash
# Train enhanced model for a specific sign
python -m signsense.ml.dynamic_train_enhanced J
python -m signsense.ml.dynamic_train_enhanced Z

# With custom parameters
python -m signsense.ml.dynamic_train_enhanced J --epochs 100 --lr 0.0005 --batch-size 16

# Use GPU if available
python -m signsense.ml.dynamic_train_enhanced Z --device cuda

# Custom data and output directories
python -m signsense.ml.dynamic_train_enhanced J --data-dir ml/data/dynamic --output-dir ml/models
```

**Pipeline:**
1. Load all sequences from `ml/data/dynamic/<SIGN>/`
2. Group by stage, create labels for each frame
3. Pad sequences to equal length, create attention masks
4. Train/val split (80/20)
5. Train enhanced LSTM with multi-task loss
6. Early stopping on validation accuracy
7. Save checkpoint: `ml/models/dynamic_<SIGN>_enhanced.pt`

### Model Comparison

Compare basic and enhanced dynamic models.

```bash
python -m signsense.ml.compare_models
```

---

## Evaluation & Inference

### Load Enhanced Model in Code

```python
from signsense.ml.dynamic_model_enhanced import load_enhanced_dynamic_model

model, normaliser, num_stages = load_enhanced_dynamic_model(
    checkpoint_path="ml/models/dynamic_J_enhanced.pt",
    sign_name="J",
    device="cpu"
)

# Forward pass returns 3 outputs (vs 2 for basic model)
stage_logits, transition_probs, confidence_scores = model(sequences, mask)
```

---

## Configuration Options

### Common Training Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--model` | str | Model type | `static_mlp` |
| `--epochs` | int | Number of training epochs | From config |
| `--lr` | float | Learning rate | From config |
| `--batch-size` | int | Batch size | From config |
| `--weight-decay` | float | Weight decay for optimizer | From config |
| `--device` | str | Device to use | `auto` |
| `--config` | str | Path to custom config file | Default config |
| `--data` | str | Path to training data | Default data path |
| `--output` | str | Path for model output | Default output path |
| `--seed` | int | Random seed | Random |
| `--no-augmentation` | flag | Disable data augmentation | Enabled |

### Device Options

```bash
# Auto-detect (use GPU if available)
python -m signsense.ml.train --device auto

# Force CPU
python -m signsense.ml.train --device cpu

# Force GPU (CUDA)
python -m signsense.ml.train --device cuda
```

---

## Troubleshooting Flags

### Debug Logging

Enable verbose logging for troubleshooting:

```python
# In code
from signsense.utils.logger import logger
logger.setLevel("DEBUG")
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Camera not accessible | Check camera permissions, try different resolution |
| Low accuracy | Increase training data, adjust learning rate |
| Memory issues | Reduce batch size, use CPU |
| Model not found | Train the model first, check path |
| Overfitting | Reduce epochs, increase dropout, add augmentation |

### Performance Tuning

```bash
# Faster training with larger batch
python -m signsense.ml.train --batch-size 128 --device cuda

# More accurate with smaller batch
python -m signsense.ml.train --batch-size 16 --epochs 200

# Reproducible results
python -m signsense.ml.train --seed 42
```

---

## Common Usage Examples

### Complete Workflow: Train Static Classifier

```bash
# 1. Record training data for letters A-Z
python -m signsense.ml.record_landmarks

# 2. Train the model
python -m signsense.ml.train --epochs 100 --lr 0.001

# 3. Run the application to test
python -m signsense.main
```

### Complete Workflow: Train Dynamic Sign (Letter J)

```bash
# 1. Record motion data
python -m signsense.ml.dynamic_recorder

# 2. Train enhanced model
python -m signsense.ml.dynamic_train_enhanced J

# 3. Run application
python -m signsense.main
```

### Quick Inference Test

```bash
# Test model training worked
python -c "
from signsense.ml.model import SignMLP
import torch
model = SignMLP(num_classes=26)
checkpoint = torch.load('ml/models/static_mlp.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state'])
print('Model loaded successfully!')
"
```

### Data Augmentation Examples

```bash
# Enable augmentation (default)
python -m signsense.ml.train

# Disable augmentation
python -m signsense.ml.train --no-augmentation
```

---

## File Paths Reference

| Path | Description |
|------|-------------|
| `ml/data/landmarks.csv` | Static landmark training data |
| `ml/data/dynamic/<SIGN>/` | Dynamic sign sequences |
| `ml/models/static_mlp.pt` | Trained static classifier |
| `ml/models/dynamic_J_enhanced.pt` | Trained enhanced J model |
| `ml/models/dynamic_Z_enhanced.pt` | Trained enhanced Z model |
| `signsense/config/dynamic_signs.yaml` | Dynamic sign configuration |

---

## Configuration Files

### YAML Configuration

SignSense uses YAML config files located in `signsense/config/`:
- `dynamic_signs.yaml` - Dynamic sign configurations
- `training.yaml` - Training hyperparameters

### Dynamic Sign Configuration Example

```yaml
J:
  type: "simple"
  training:
    sequence_length: 120
    batch_size: 8
    epochs: 50
    learning_rate: 0.001

Z:
  type: "complex"
  stages:
    0: "Step 1/4 — Start position"
    1: "Step 2/4 — Horizontal stroke"
    2: "Step 3/4 — Diagonal stroke"
    3: "Step 4/4 — End position"
```

---

## Keyboard Shortcuts Summary

### Main Application
- `F` - Toggle fullscreen
- `ESC` - Return to menu

### Landmark Recorder
- `A-Z` - Select letter
- `T` - Text input mode
- `SPACE` - Toggle recording
- `[` - Save and quit
- `ESC` - Quit without saving

### Dynamic Recorder
- `A-Z` - Select gesture
- `0-9` - Select stage
- `S` - Toggle simple/complex
- `N` - Next stage/sequence
- `P` - Previous stage
- `SPACE` - Toggle recording
- `Q` - Save and quit
- `ESC` - Quit without saving