# SignSense Dynamic Recorder User Guide

## Table of Contents
1. [Overview](#1-overview)
2. [Sign Types](#2-sign-types-simple-vs-complex)
3. [Decision Criteria](#3-decision-criteria)
4. [Control Reference](#4-control-reference)
5. [Recording Steps](#5-step-by-step-recording-guide)
6. [Best Practices](#6-best-practices)
7. [Troubleshooting](#7-troubleshooting)

---

## Quick Reference Commands

```bash
# Start recorder
python -m ml.dynamic_recorder

# Train a recorded sign
python -m ml.dynamic_train <SIGN_NAME>

# Check recorded data
ls ml/data/dynamic/<SIGN_NAME>/

# Verify model exists
ls ml/models/dynamic_<SIGN_NAME>.pt
```
---

============================================================
DYNAMIC GESTURE RECORDER
============================================================
Controls:
  T         - Type custom gesture name (e.g., "hello", "thank_you")
  A-Z       - Quick select letter gesture (J, Z, etc.)
  S         - Toggle simple/complex mode
  0-9       - Set stage (complex mode only)
  SPACE     - Start/stop recording
  N         - Next stage (complex) / new sequence (simple)
  Q         - Save and exit
  ESC       - Exit without saving
============================================================

## 1. Overview

### Purpose and Capabilities

The SignSense Dynamic Recorder is a specialized tool for capturing motion-based ASL signs that require temporal sequence analysis, unlike static signs which can be recognized from a single frame.

---

### Technical Implementation

#### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| [`DynamicSignRecorder`](signsense/ml/dynamic_recorder.py:50) | `ml/dynamic_recorder.py` | Main interactive recorder class |
| [`HandTracker`](signsense/detector/hand_tracker.py:48) | `detector/hand_tracker.py` | MediaPipe hand detection |
| [`LandmarkNormaliser`](signsense/ml/model.py) | `ml/model.py` | Wrist-relative normalization |
| [`DynamicSignLSTM`](signsense/ml/dynamic_model.py:30) | `ml/dynamic_model.py` | LSTM classifier architecture |

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DynamicSignRecorder                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ HandTracker  │→│  Normaliser  │→│  Frame Buffer         │  │
│  │ (MediaPipe)  │  │ (21→63 feat) │  │ List[np.ndarray]     │  │
│  └──────────────┘  └──────────────┘  └───────────┬───────────┘  │
│                                                   │              │
│                              ┌────────────────────┴───────────┐  │
│                              │      stage_sequences: dict     │  │
│                              │      stage_num → sequences[]   │  │
│                              └────────────────────┬───────────┘  │
│                                                   │              │
│                              ┌────────────────────┴───────────┐  │
│                              │   _save_sequence() / _save_all()│  │
│                              └────────────────────┬───────────┘  │
└───────────────────────────────────────────────────┼──────────────┘
                                                    ↓
                                    ml/data/dynamic/<SIGN>/
                                    └── *.npy (shape: N×63)
```

#### Data Flow

1. **Capture Phase** ([`run()`](signsense/ml/dynamic_recorder.py:380)): Main loop captures frames at ~30fps
   ```python
   # Line 404-418
   ret, frame = cap.read()
   frame = cv2.flip(frame, 1)  # Mirror
   hand_data = self.hand_tracker.process_frame(rgb)
   if self.is_recording and hand_landmarks:
       normalized = self.normaliser.normalise(hand_landmarks)
       self.current_sequence.append(normalized)
   ```

2. **Normalization**: Converts 21 MediaPipe landmarks to 63-dim feature vector
   ```python
   # Wrist-relative, scale-invariant normalization
   # Each landmark: (x - wrist_x) / scale, (y - wrist_y) / scale, (z - wrist_z) / scale
   # 21 landmarks × 3 coords = 63 features per frame
   ```

3. **Storage**: Per-stage sequence storage with automatic indexing
   ```python
   # Lines 157-184: _save_sequence()
   # Filename format: <SIGN>_s<STAGE>_<INDEX>.npy
   # Example: HELLO_s0_5.npy → gesture HELLO, stage 0, index 5
   ```

4. **Metadata**: JSON file tracks gesture metadata
   ```python
   # Lines 186-221: _save_all_sequences()
   {
     "sign_name": "HELLO",
     "gesture_type": "simple",
     "num_stages": 1,
     "sequences_per_stage": {"0": 5},
     "total_sequences": 5
   }
   ```

#### Class Structure

```python
# signsense/ml/dynamic_recorder.py
class DynamicSignRecorder:
    def __init__(self, output_base=None):
        # Lines 61-90: Initialize hand tracker, normalizer, state
        self.normaliser = LandmarkNormaliser()
        self.hand_tracker = HandTracker()
        self.current_sequence: List[np.ndarray] = []
        self.stage_sequences: dict = {}
        
    def run(self) -> None:  # Lines 380-553
        # Main interactive loop with cv2.imshow
        
    def toggle_recording(self) -> None:  # Lines 289-306
        # Start/stop recording, auto-save on stop
        
    def set_stage(self, stage_num: int) -> None:  # Lines 270-287
        # Jump to specific stage (complex mode)
        
    def next_stage(self) -> None:  # Lines 308-322
        # Advance to next stage, auto-save current
        
    def _save_all_sequences(self) -> None:  # Lines 186-221
        # Save all sequences and write metadata.json
```

---

### What Makes It Different from Static Recording

### What Makes It Different from Static Recording

| Feature | Static Signs | Dynamic Signs |
|---------|--------------|---------------|
| Data captured | Single frame (21 landmarks) | Multiple frames over time |
| Processing | MLP classifier | LSTM neural network |
| Duration | Instant recognition | Requires sequence completion |
| Examples | A, B, C, D... | J, Z, hello, thank_you |

### Types of Data Captured

1. **Hand Tracking Coordinates**: 21 MediaPipe landmarks per frame (x, y, z normalized coordinates)
2. **Movement Trajectories**: Temporal changes in landmark positions across frames
3. **Temporal Sequences**: Ordered series of landmark configurations representing the gesture progression

---

### Prerequisites

| Requirement | Description | Check Command |
|-------------|-------------|---------------|
| Python | 3.8+ | `python --version` |
| OpenCV | `cv2` for camera capture | `pip show opencv-python` |
| MediaPipe | `mediapipe` for hand tracking | `pip show mediapipe` |
| PyTorch | For model training | `pip show torch` |
| NumPy | For array manipulation | `pip show numpy` |
| Webcam | Camera access at index 0 | System check |

**Installation**:
```bash
pip install opencv-python mediapipe torch numpy matplotlib scikit-learn pyyaml
```

**Required Assets**:
- `signsense/assets/models/hand_landmarker.task` (MediaPipe model)
  - Auto-downloaded if not present
  - Fallback URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

---

### Overall Workflow

```
Camera Frame → Hand Detection → Landmark Normalization → Frame Buffer
                                                                      ↓
Training Data ← .npy File Save ← Sequence Stacking ← Recording Stop
                                                                      ↓
Model Training ← Data Loading ← Sequence Loading ← Model Inference
```

---

## 2. Sign Types: Simple vs Complex

### Simple Signs

**Definition**: Single-stage gestures that don't require distinct phases to complete.

**Criteria**:
- Motion complexity: Low - continuous movement without distinct stages
- Typical duration: 30-120 frames (1-4 seconds at 30fps)
- Storage: ~5KB per sequence (120 frames × 63 floats × 4 bytes)

**Characteristics**:
- One continuous motion from start to finish
- No handshape changes during execution
- Single recognizable pattern
- Examples: hello, goodbye, thank_you, wave

**Processing**:
- Frame normalization: Same as complex (wrist-relative, scale-invariant)
- Feature extraction: Flattened 63-dim vectors per frame
- Model: Single-stage LSTM (num_stages=1)
- Training samples required: Minimum 10

### Complex Signs

**Definition**: Multi-stage gestures requiring distinct phases to complete.

**Criteria**:
- Motion complexity: High - distinct phases with transitions
- Typical duration: 60-180 frames (2-6 seconds at 30fps)
- Storage: ~15KB per sequence (120 frames × 63 floats × 4 bytes)

**Characteristics**:
- Multiple distinct phases (e.g., J: hold I → hook down → palm away)
- Handshape changes during execution
- Sequential pattern recognition required
- Examples: J (3 stages), Z (4 stages)

**Processing**:
- Frame normalization: Same as simple
- Feature extraction: Per-frame 63-dim vectors with stage labels
- Model: Multi-stage LSTM (num_stages=2-4)
- Training samples required: Minimum 15-20 per stage

### Processing Differences

| Aspect | Simple Signs | Complex Signs |
|--------|--------------|---------------|
| Frame interpolation | Not required | Not required |
| Normalization | LandmarkNormaliser | LandmarkNormaliser |
| Feature extraction | Per-frame 63-dim | Per-frame 63-dim + stage labels |
| Model architecture | Single-stage LSTM | Multi-stage LSTM |
| Inference speed | Faster (1 stage) | Slower (multiple stages) |
| Memory usage | Lower | Higher |

### Performance Implications

- **Inference Speed**: Simple signs are ~2-3x faster due to single-stage classification
- **Memory Usage**: Complex signs require ~50% more memory for model and state tracking
- **Accuracy**: Both achieve 90%+ with sufficient training data; complex signs need more samples

---

## 3. Decision Criteria

### Decision Matrix

| Gesture Characteristic | Simple Sign | Complex Sign |
|------------------------|-------------|--------------|
| Single continuous motion | ✓ | |
| Multiple distinct phases | | ✓ |
| No handshape changes | ✓ | |
| Handshape changes during motion | | ✓ |
| Duration < 2 seconds | ✓ | |
| Duration > 2 seconds | | ✓ |
| Recognition difficulty: Low | ✓ | |
| Recognition difficulty: High | | ✓ |

### Flowchart Logic

```
Start
  │
  ▼
Does the gesture have distinct phases?
  │
  ├─ No ──→ Is it a single continuous motion?
  │           │
  │           ├─ Yes ──→ SIMPLE SIGN
  │           │
  │           └─ No ──→ Consider breaking into parts
  │
  └─ Yes ──→ Does handshape change between phases?
              │
              ├─ Yes ──→ COMPLEX SIGN
              │
              └─ No ──→ Could be either; check duration
                        │
                        ├─ Short (<2s) ──→ SIMPLE SIGN
                        │
                        └─ Long (>2s) ──→ COMPLEX SIGN
```

### Motion Complexity Analysis

**Low Complexity (Simple)**:
- Wave-like motions
- Single direction movements
- No positional requirements between phases

**High Complexity (Complex)**:
- Letter tracing (J, Z)
- Multiple directional changes
- Specific handshape requirements per phase

### Recognition Difficulty Assessment

**Easy (Simple)**:
- Distinctive motion pattern
- Limited variation in execution
- Clear start and end points

**Hard (Complex)**:
- Subtle differences between phases
- Multiple valid execution variations
- Requires precise timing

---

## 4. Control Reference

### Keyboard Shortcuts

#### Recording Controls
| Key | Action | Description |
|-----|--------|-------------|
| SPACE | Start/Stop Recording | Toggle recording state |
| Q | Save and Exit | Save all sequences and quit |
| ESC | Exit Without Saving | Discard recordings and quit |

#### Navigation
| Key | Action | Description |
|-----|--------|-------------|
| N | Next Stage | Advance to next stage (complex mode) or new sequence (simple mode) |
| P | Previous Stage | Go back to previous stage |
| 0-9 | Set Stage | Jump to specific stage (complex mode only) |

#### Configuration
| Key | Action | Description |
|-----|--------|-------------|
| T | Enter Text Mode | Type custom gesture name |
| S | Toggle Mode | Switch between simple/complex |
| A-Z | Quick Select | Select letter gesture (J, Z, etc.) |

### Mouse Controls

The dynamic recorder does not use mouse controls - all interaction is keyboard-based.

### Configuration Options

#### SignConfig Parameters (dynamic_config.py)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| name | str | Any | - | Sign identifier |
| gesture_type | GestureType | SIMPLE/COMPLEX | COMPLEX | Sign classification |
| min_samples | int | 1-100 | 10 | Minimum training samples |
| sequence_length | int | 30-300 | 120 | Max frames for training |
| batch_size | int | 1-32 | 8 | Training batch size |
| epochs | int | 10-200 | 50 | Training iterations |
| learning_rate | float | 0.0001-0.01 | 0.001 | Optimizer learning rate |
| hidden_size | int | 64-256 | 128 | LSTM hidden dimension |

#### Detector Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| i_hold_frames | int | 1-20 | 6 | Frames to hold I shape |
| down_threshold | float | 0.01-0.2 | 0.06 | Pinky hook threshold |
| finish_hold_frames | int | 1-20 | 8 | Final hold frames |
| phase_timeout | int | 20-150 | 70 | Max frames per stage |

### UI Elements

#### Main Display Components

1. **Title Bar** (lines 424-426 in dynamic_recorder.py)
   - Background: Dark gray (30, 30, 40)
   - Height: 60 pixels

2. **Status Text** (lines 427-436)
   - Shows: Gesture name, mode (SIMPLE/COMPLEX), stage, recording indicator
   - Color: Green (0, 255, 0) when recording, white otherwise

3. **Info Panel** (lines 442-484)
   - Existing samples count
   - Current recording frame count
   - Stage sequence count (complex mode)

4. **Instructions** (lines 486-495)
   - Bottom of screen showing available controls

### Hand Tracker Integration

**File**: `signsense/detector/hand_tracker.py`

**MediaPipe Hand Landmarks** (21 points):
```
0: Wrist
1-4: Thumb (CMC, MIP, DIP, TIP)
5-8: Index Finger (MCP, PIP, DIP, TIP)
9-12: Middle Finger (MCP, PIP, DIP, TIP)
13-16: Ring Finger (MCP, PIP, DIP, TIP)
17-20: Pinky (MCP, PIP, DIP, TIP)
```

**Detection Parameters**:
- min_detection_confidence: 0.6
- min_tracking_confidence: 0.6
- max_num_hands: 1
- Running mode: VIDEO (for temporal processing)

---

## 5. Step-by-Step Recording Guide

### Phase 1: Prerequisites & Setup

#### Verify Environment
```bash
python --version
pip list | grep -E "opencv|mediapipe|torch|numpy"
```

1. **Start the Recorder**
   ```bash
   python -m ml.dynamic_recorder
   ```

2. **Camera Configuration**
   - Opens webcam at index 0
   - Captures at ~30fps
   - Mirrored for selfie view (cv2.flip with code 1)

3. **Buffer Allocation** (lines 61-90 in dynamic_recorder.py)
   ```python
   self.current_sequence: List[np.ndarray] = []  # Frame buffer
   self.stage_sequences: dict = {}  # Per-stage storage
   self.normaliser = LandmarkNormaliser()  # Feature normalizer
   self.hand_tracker = HandTracker()  # MediaPipe detector
   ```

4. **Hand Tracking Initialization**
   - Loads MediaPipe hand_landmarker.task model
   - Initializes with default confidence thresholds (0.6)
   - Sets up timestamp tracking for video mode

### Phase 2: Capture

1. **Set Gesture Name**
   - Press `T` to enter text mode
   - Type gesture name (e.g., "hello")
   - Press Enter to confirm
   - Press ESC to cancel

2. **Begin Recording**
   - Press SPACE to start recording
   - Status shows "●REC" indicator
   - Frame counter updates in real-time

3. **Execute Gesture**
   - Perform the sign in front of camera
   - Keep hand within camera frame
   - Maintain consistent speed

4. **Stop Recording**
   - Press SPACE to stop
   - Sequence automatically saved to buffer
   - Shows frame count confirmation

### Phase 3: Validation

**Automatic Checks** (not explicitly implemented, but inherent to system):

1. **Sufficient Frames**
   - Minimum: ~30 frames recommended
   - Less may work but reduces accuracy

2. **Hand Visibility**
   - All 21 landmarks must be detected
   - Gaps in detection cause incomplete sequences

3. **Movement Completeness**
   - System doesn't validate gesture correctness
   - User must ensure proper execution

### Phase 4: Saving

1. **File Format**: `.npy` NumPy array
   - Shape: (num_frames, 63)
   - Data type: float32
   - Storage: ml/data/dynamic/<GESTURE_NAME>/

2. **Naming Convention**: `<GESTURE>_s<STAGE>_<INDEX>.npy`
   - Example: HELLO_s0_0.npy (gesture HELLO, stage 0, index 0)

3. **Metadata Structure** (metadata.json):
   ```json
   {
     "sign_name": "HELLO",
     "gesture_type": "simple",
     "num_stages": 1,
     "sequences_per_stage": {"0": 5},
     "total_sequences": 5,
     "recorded_at": "2026-03-25T12:00:00.000000",
     "description": "Custom gesture: HELLO"
   }
   ```

4. **Storage Paths**:
   - Data: ml/data/dynamic/<GESTURE_NAME>/
   - Models: ml/models/dynamic_<GESTURE>.pt

### Registering a New Sign

1. **Automatic Registration**
   - New gestures are auto-added to dynamic_signs.yaml
   - Created via create_default_config() function

2. **Manual Registration** (sign_registry.py)
   - Add to SIGN_REGISTRY list
   - Set SignType to DYNAMIC
   - Enable/disable as needed

3. **Configuration Update** (dynamic_signs.yaml)
   - Add entry under dynamic_signs
   - Specify type (simple/complex)
   - Configure parameters as needed

---

## 6. Best Practices

### Example: Recording "Hello" Sign

The "hello" wave is an excellent example of a simple sign. Here's how to record it optimally:

### Optimal Lighting Conditions

**Lux Range**: 200-500 lux (typical indoor office lighting)

**Recommendations**:
- Position face toward main light source
- Avoid backlighting (creates silhouette)
- Consistent lighting across entire gesture path
- Avoid harsh shadows on hand

**For "hello" specifically**:
- Ensure face is well-lit for hand tracking
- Arm movement should be in lit area
- Maintain consistent lighting throughout wave motion

### Camera Positioning

**Distance**: 0.5-1.5 meters from camera

**Angle**: Straight-on or slightly above

**Field of View**: Hand and upper arm should be visible

**For "hello" specifically**:
- Full arm movement needs to be in frame
- Keep shoulder to hand visible
- Avoid camera too close (choppy motion)

### Gesture Consistency Requirements

1. **Repeatable Execution**
   - Practice gesture 5-10 times before recording
   - Establish consistent start position
   - Use consistent speed and amplitude

2. **Frame Timing**
   - Maintain 25-30 fps for smooth capture
   - Avoid sudden accelerations
   - Keep movement within 1-3 second range

3. **Body Position**
   - Sit or stand consistently between takes
   - Keep torso relatively still
   - Only move the performing arm

### Multiple Take Recommendations

**Number of Recordings**:
- Simple signs: 10-15 takes minimum
- Complex signs: 20-30 takes minimum (5-10 per stage)

**Selection Criteria**:
- Complete gesture (no cut-off frames)
- Consistent hand visibility
- Smooth motion without jitter
- Representative of typical execution

**Batch Recording Workflow**:
1. Record all takes in one session
2. Review frame counts (should be similar)
3. Remove obvious errors
4. Keep best 10-15 for training

---

## 7. Troubleshooting

### Low-Quality Captures

**Symptoms**: Incomplete landmarks, jittery tracking, dropped frames

**Causes and Solutions**:

| Cause | Diagnosis | Resolution |
|-------|-----------|------------|
| Poor lighting | Check environment lux level | Increase to 200-500 lux |
| Inadequate hand visibility | Verify all 21 landmarks detected | Reposition hand in frame |
| Camera resolution | Check cv2.VideoCapture info | Ensure 640x480 or better |
| System performance | Monitor CPU usage | Close other applications |

**For "hello" wave specifically**:
- Ensure arm stays in frame throughout wave
- Maintain consistent distance from camera
- Avoid moving body during gesture

### Recognition Failures

**Symptoms**: Model fails to recognize, low confidence, incorrect stage detection

**Causes and Solutions**:

| Cause | Diagnosis | Resolution |
|-------|-----------|------------|
| Inconsistent execution | Review recorded sequences | Standardize gesture execution |
| Insufficient training data | Count sequences | Record more samples (10+ for simple) |
| Wrong sign type | Review gesture characteristics | Reclassify as simple/complex |
| Model not trained | Check for .pt file | Run: python -m ml.dynamic_train <SIGN> |

**Diagnostic Steps**:
1. Verify sequences exist: ls ml/data/dynamic/<SIGN>/
2. Check metadata: cat ml/data/dynamic/<SIGN>/metadata.json
3. Confirm model exists: ls ml/models/dynamic_<SIGN>.pt
4. Test with hardcoded detector if available

### Data Validation Errors

**Symptoms**: Training fails, insufficient frames error, malformed sequences

**Causes and Solutions**:

| Cause | Diagnosis | Resolution |
|-------|-----------|------------|
| Insufficient frames | Check sequence length | Record longer sequences (30+ frames) |
| Malformed sequences | np.load and check shape | Re-record with proper hand visibility |
| Stage mismatch | Review stage labels | Ensure correct stage numbering |
| Metadata corruption | Check metadata.json | Delete and re-record |

**Validation Checklist**:
- [ ] Each .npy file has shape (N, 63) where N > 0
- [ ] metadata.json exists and is valid JSON
- [ ] All stages have at least 2 sequences
- [ ] No negative frame counts

---

## File Locations Reference

| Component | Path |
|-----------|------|
| Recorder | signsense/ml/dynamic_recorder.py |
| Hand Tracker | signsense/detector/hand_tracker.py |
| Normalizer | signsense/ml/model.py |
| Model | signsense/ml/dynamic_model.py |
| Training | signsense/ml/dynamic_train.py |
| Config | signsense/config/dynamic_config.py |
| Data | ml/data/dynamic/<SIGN>/ |
| Models | ml/models/ |