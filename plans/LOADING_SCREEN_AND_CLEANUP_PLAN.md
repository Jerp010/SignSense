# Loading Screen Implementation & Codebase Cleanup Plan

## Part 1: Loading Screen Implementation

### Overview
The loading screen will display an animated progress bar with percentage and status messages while the play mode initializes its components (detectors, camera, etc.). This prevents the UI from freezing and provides user feedback.

### Design Specifications

**Visual Elements:**
- Dark background matching app theme (BG: 20, 20, 25)
- Centered animated progress bar (horizontal, rounded corners)
- Percentage text above the bar
- Status message below the bar (e.g., "Loading hand tracker...", "Opening camera...")
- Subtle pulsing accent element for visual interest
- App title "SignSense" displayed at top

**Animation:**
- Progress bar fills smoothly with accent color (70, 130, 220)
- Pulsing glow effect on progress bar
- Status text updates as each component loads

**Loading Steps to Display:**
1. "Initializing detectors..." (0-30%)
2. "Loading hand tracker..." (30-50%)
3. "Loading face tracker..." (50-70%)
4. "Loading classifier..." (70-90%)
5. "Opening camera..." (90-100%)
6. "Ready!" (100%)

### Implementation Files

**New File: `signsense/ui/loading.py`**
- Class: `LoadingScreen`
- Methods:
  - `__init__(W, H)` - Initialize with window dimensions
  - `update_progress(percent, status)` - Update progress and status
  - `render()` - Draw the loading screen frame
  - `reset()` - Reset for reuse

### Integration Points

**File: `signsense/main.py`**
- Import `LoadingScreen` from `ui.loading`
- In `run_play_mode()`:
  - Create LoadingScreen instance before detector initialization
  - Show loading screen with cv2.imshow() after each major step
  - Update progress during initialize_detectors()
  - Update progress when opening camera

---

## Part 2: Codebase Cleanup Analysis

### Files to Remove (Unused/Redundant)

#### ML Training Files (Duplicates/Old Versions)
- `signsense/ml/dynamic_model.py` - Replaced by enhanced version
- `signsense/ml/dynamic_train.py` - Replaced by enhanced version  
- `signsense/ml/model.py` - Not used in current play mode
- `signsense/ml/train.py` - Not used in current play mode
- `signsense/ml/record_landmarks.py` - Standalone utility, not imported
- `signsense/ml/models/sign_mlp.pt` - Legacy model, not used
- `signsense/ml/models/training_curves.png` - Not displayed in app

#### Config Files (Duplicates)
- `signsense/config/dynamic_signs.csv` - Duplicate of YAML version
- `signsense/config/dynamic_config.py` - Standalone, not imported by main

#### Backup Data Files
- `ml/data/backups/` - All backup files are old snapshots

#### Documentation (Old/Superseded)
- `docs/MODEL_GUIDE.md` - Superseded by ENHANCED_MODEL_GUIDE.md
- `docs/SINGLE_STAGE_FIX.md` - One-time fix doc, not ongoing reference
- `docs/LEVEL2_REFACTOR_SUMMARY.md` - Historical refactor notes
- `docs/DYNAMIC_RECORDER_GUIDE.md` - Feature not in current play mode
- `docs/DYNAMIC_GESTURE_TRAINING_GUIDE.md` - Feature not in current play mode
- `docs/QUICK_REFERENCE.md` - May keep, but verify usage

#### Assets (Unused)
- `assets/signs/PLACEHOLDER_README.md` - Placeholder file

### Files to Keep

**Required for Play Mode:**
- `signsense/main.py` - Entry point
- `signsense/ui/menu.py` - Menu system
- `signsense/ui/play_mode.py` - Gameplay
- `signsense/ui/loading.py` - NEW loading screen
- `signsense/detector/*.py` - Detection components
- `signsense/signs/*.py` - Sign definitions
- `signsense/ml/dynamic_model_enhanced.py` - Enhanced ML model
- `signsense/ml/dynamic_train_enhanced.py` - Enhanced training
- `signsense/ml/dynamic_recorder.py` - Recording utility
- `signsense/config/dynamic_signs.yaml` - Active config

**Required Dependencies (requirements.txt):**
- mediapipe - Hand/face tracking
- opencv-python - Display/rendering
- numpy - Array operations
- torch - ML model loading
- pillow - Image loading
- pandas - Data handling (check usage)
- matplotlib - Check if used anywhere
- scikit-learn - Check if used anywhere
- pyyaml - Config loading

### Dependencies to Verify
- `pandas` - Used in config_loader.py
- `matplotlib` - Check if imported anywhere
- `scikit-learn` - Check if imported anywhere

---

## Part 3: Implementation Order

1. Create `signsense/ui/loading.py` with LoadingScreen class
2. Update `signsense/main.py` to use loading screen in run_play_mode()
3. Test loading screen displays correctly
4. Remove unused ML files
5. Remove duplicate config files
6. Remove backup data files
7. Remove unused documentation
8. Verify no import errors after cleanup

---

## Mermaid: Loading Screen Flow

```mermaid
graph TD
    A[User selects Play] --> B[Create LoadingScreen]
    B --> C[Show 'Initializing detectors...' 0%]
    C --> D[Call initialize_detectors]
    D --> E[Update: 'Loading hand tracker...' 30%]
    E --> F[Update: 'Loading face tracker...' 50%]
    F --> G[Update: 'Loading classifier...' 70%]
    G --> H[Call open_camera]
    H --> I[Update: 'Opening camera...' 90%]
    I --> J[Update: 'Ready!' 100%]
    J --> K[Transition to Play Mode]
    
    style A fill:#e1f5fe
    style K fill:#e8f5e8