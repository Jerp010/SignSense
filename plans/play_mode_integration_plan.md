# Play Mode Integration Plan

## Overview
Adapt the current gesture-based play mode to incorporate the old specification's letter-based system with dynamic sign support, while preserving the existing UI layout and state machine.

## Current State Analysis

### Old Specification (old_play_model.txt)
- **Sign System**: Letters A-J from ACTIVE_SIGNS (sign_registry.py)
- **Sign Types**: STATIC (hold for HOLD_SECONDS) and DYNAMIC (detector fires on completion)
- **Detection**: Uses classifier_result with "letter" key
- **Dynamic Support**: Integrated with get_detector() from dynamic_signs.py
- **Registry**: Uses SignEntry, SignType from sign_registry

### Current Implementation (play_mode.py)
- **Sign System**: 9 named ASL gestures (HELLO, THANK YOU, NAME, GOOD, HELP, WATER, YES, NO, BAD)
- **Sign Types**: All treated as hold-based (HOLD_SECONDS)
- **Detection**: Uses DetectionResult with sign_name and confidence
- **Dynamic Support**: None
- **Registry**: Uses GestureClass from gesture_definitions.py

## Key Differences

| Aspect | Old Spec | Current |
|--------|----------|---------|
| Sign Source | ACTIVE_SIGNS (letters A-Z) | GESTURE_REGISTRY (9 named gestures) |
| Sign Type | STATIC / DYNAMIC | All STATIC (hold-based) |
| Detection | classifier_result.letter | DetectionResult.sign_name |
| Dynamic | get_detector() integration | None |
| Stage Class | SignEntry | GestureStage / LetterStage |

## Integration Strategy

### 1. Unified Stage System
Create a base `Stage` class that can represent both letter-based and gesture-based stages:

```python
class Stage:
    """Base class for all stage types."""
    def __init__(self, name: str, description: str, sign_type: str = "STATIC"):
        self.name = name
        self.description = description
        self.sign_type = sign_type  # "STATIC" or "DYNAMIC"
    
    @property
    def display_name(self) -> str:
        return self.name
    
    @property
    def hand_shape(self) -> str:
        return "static" if self.sign_type == "STATIC" else "dynamic"
    
    @property
    def movement(self) -> str:
        return "hold" if self.sign_type == "STATIC" else "motion"

class LetterStage(Stage):
    """Stage for letter-based signs from sign_registry."""
    def __init__(self, sign_entry: SignEntry):
        super().__init__(
            name=sign_entry.letter,
            description=sign_entry.description,
            sign_type=sign_entry.sign_type.name
        )
        self.sign_entry = sign_entry
        self.letter = sign_entry.letter

class GestureStage(Stage):
    """Stage for named gesture-based signs."""
    def __init__(self, gesture: GestureClass):
        super().__init__(
            name=gesture.name,
            description=f"{gesture.hand_shape.value}",
            sign_type="STATIC"  # All current gestures are hold-based
        )
        self.gesture = gesture
```

### 2. Dynamic Sign Detector Integration
Add support for dynamic sign detectors in StageTracker:

```python
def _load_detector(self):
    """Load dynamic detector for current stage if needed."""
    stage = self.current_stage
    if stage and stage.sign_type == "DYNAMIC":
        if hasattr(stage, 'letter'):
            # Letter-based dynamic sign
            from signsense.signs.dynamic_signs import get_detector
            self._dynamic_det = get_detector(stage.letter)
            logger.debug(f"Loaded dynamic detector for {stage.letter}")
    else:
        if self._dynamic_det:
            logger.debug("Clearing dynamic detector")
        self._dynamic_det = None
```

### 3. Detection Result Handling
Update StageTracker to handle both detection formats:

```python
def update(self, detection_result, landmarks=None, handedness=None) -> bool:
    """
    Call every frame with detection result.
    
    Args:
        detection_result: Either DetectionResult (gesture) or dict with "letter" key
        landmarks: Hand landmarks for dynamic detection
        handedness: Hand handedness for dynamic detection
    """
    stage = self.current_stage
    if stage is None:
        self._state = "COMPLETE"
        return False
    
    # Handle dynamic signs
    if stage.sign_type == "DYNAMIC":
        if self._dynamic_det is None:
            return False
        if self._state == "CONFIRMING":
            return False  # waiting for user to press Next
        done = self._dynamic_det.update(landmarks, handedness)
        if done:
            self._state = "CONFIRMING"
            self._confirm_time = time.time()
            logger.info(f"Dynamic sign '{stage.name}' detected, entering CONFIRMING")
            return True
        return False
    
    # Handle static signs (both letter and gesture)
    detected_name = None
    if detection_result:
        if hasattr(detection_result, 'sign_name'):
            # Gesture detection format
            detected_name = detection_result.sign_name
        elif isinstance(detection_result, dict) and "letter" in detection_result:
            # Letter detection format
            detected_name = detection_result["letter"]
    
    correct = (detected_name == stage.name)
    
    # ... rest of state machine logic
```

### 4. PlayMode Controller Updates
Update PlayMode to support both modes:

```python
class PlayMode:
    """Main play mode controller."""
    
    def __init__(self, mode="gesture", hand_tracker=None, face_tracker=None, W=640, H=480):
        """
        Args:
            mode: "gesture" for 9 ASL gestures, "letter" for A-Z letters
        """
        self.mode = mode
        
        if mode == "gesture":
            # Use gesture-based stages
            from signsense.signs.gesture_definitions import GESTURE_REGISTRY, get_all_gestures
            stages = [GestureStage(g) for g in get_all_gestures()]
            self.gesture_detector = GestureDetector(
                hand_tracker=hand_tracker,
                face_tracker=face_tracker,
                confidence_threshold=0.7
            )
        elif mode == "letter":
            # Use letter-based stages from ACTIVE_SIGNS
            from signsense.signs.sign_registry import ACTIVE_SIGNS
            stages = [LetterStage(s) for s in ACTIVE_SIGNS]
            self.gesture_detector = None  # Use classifier instead
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.stage_tracker = StageTracker(stages)
        self.renderer = PlayModeRenderer(W, H)
        self._hand_detected = False
```

### 5. UI Updates
Update PlayModeRenderer to show sign type badges:

```python
def _draw_sign_panel(self, frame, tracker, detection_result):
    # ... existing code ...
    
    # Sign type badge
    stage = tracker.current_stage
    if stage:
        badge = stage.sign_type  # "STATIC" or "DYNAMIC"
        badge_col = ACCENT2 if stage.sign_type == "DYNAMIC" else (60, 180, 100)
        cv2.putText(frame, badge, (px + 8, py + 16), FONT, 0.35, badge_col, 1, cv2.LINE_AA)
        
        # For dynamic signs, show stage hint
        if stage.sign_type == "DYNAMIC" and tracker.state != "CONFIRMING":
            hint = tracker.dynamic_hint if hasattr(tracker, 'dynamic_hint') else "Perform the motion"
            cv2.putText(frame, hint, (px + 6, hint_y), FONT, 0.36, GOLD, 1, cv2.LINE_AA)
```

## Implementation Steps

### Step 1: Create Unified Stage Classes
- Create base `Stage` class
- Create `LetterStage` class for sign_registry integration
- Create `GestureStage` class for gesture_definitions integration
- Update `GESTURE_STAGES` to use new classes

### Step 2: Update StageTracker
- Add `_dynamic_det` attribute for dynamic detector
- Add `_load_detector()` method
- Update `update()` to handle both detection formats
- Update `update()` to handle dynamic sign detection
- Add `dynamic_hint` property

### Step 3: Update PlayModeRenderer
- Update `_draw_sign_panel()` to show sign type badge
- Update state hints for dynamic signs
- Update preview box to handle both letter and gesture names

### Step 4: Update PlayMode Controller
- Add `mode` parameter to `__init__`
- Create stages based on mode
- Initialize appropriate detector (GestureDetector or classifier)
- Update `update()` to pass correct detection format

### Step 5: Integration Testing
- Test gesture mode with 9 ASL gestures
- Test letter mode with ACTIVE_SIGNS (A-J)
- Test dynamic signs (J, Z) with detector
- Test state transitions (WAITING → HOLDING → CONFIRMING)
- Test UI rendering for both modes

## Files to Modify

1. **signsense/ui/play_mode.py**
   - Add unified Stage classes
   - Update StageTracker for dynamic support
   - Update PlayModeRenderer for sign type display
   - Update PlayMode controller for dual mode support

2. **signsense/signs/dynamic_signs.py** (if needed)
   - Ensure get_detector() works for J and Z

3. **signsense/main.py** (if needed)
   - Update PlayMode instantiation to support mode selection

## Preserved Functionality

- ✅ UI layout (camera feed, sign panel, progress bar, preview box)
- ✅ State machine (WAITING → HOLDING → CONFIRMING → COMPLETE)
- ✅ Preview box with 3 pages and arrow navigation
- ✅ Progress bar visualization
- ✅ Confirm flash animation
- ✅ Completion screen
- ✅ FPS and hand status display
- ✅ Stage completion dots

## New Functionality

- ✅ Support for letter-based signs (A-Z) from sign_registry
- ✅ Support for dynamic signs (J, Z) with dedicated detectors
- ✅ Sign type badge display (STATIC/DYNAMIC)
- ✅ Dynamic sign stage hints
- ✅ Dual mode operation (gesture or letter)
- ✅ Backward compatibility with existing gesture system

## Testing Checklist

- [ ] Gesture mode: All 9 gestures detect and progress correctly
- [ ] Letter mode: All ACTIVE_SIGNS (A-J) appear as stages
- [ ] Static signs: Hold for HOLD_SECONDS to complete
- [ ] Dynamic sign J: Detector fires on J hook completion
- [ ] Dynamic sign Z: Detector fires on Z trace completion
- [ ] State transitions: WAITING → HOLDING → CONFIRMING → next stage
- [ ] UI: Sign type badge displays correctly
- [ ] UI: Dynamic sign hints display during detection
- [ ] Preview box: Cycles through 3 pages with arrows
- [ ] Progress bar: Fills during HOLDING state
- [ ] Confirm flash: Shows checkmark on completion
- [ ] Completion screen: Shows after all stages
