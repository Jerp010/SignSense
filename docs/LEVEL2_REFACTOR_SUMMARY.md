# Level 2 Refactor - Architecture Summary

## Current Architecture

### Sign Categories

The system handles three sign categories:

1. **Static Signs (Letters A-Y)** - Single-frame ASL hand shapes
   - Handled by `ASLClassifierLetters` (PyTorch model)
   - Defined in `signs/sign_registry.py` `SIGN_REGISTRY`
   - Excludes J (DYNAMIC) and Z (DYNAMIC, disabled)

2. **Dynamic Signs (J, Z)** - Motion-based detection
   - Require trajectory analysis via `dynamic_signs.py`
   - Configurable via CSV/YAML in `signs/config/`
   - J: I shape + pinky traces J hook
   - Z: Index traces Z (currently disabled)

3. **Gesture Signs (HELLO, YES, NO, etc.)** - Movement patterns with facial expressions
   - Defined in `signs/gesture_definitions.py`
   - Currently 5 gestures retained after Level 2 refactor:
     - HELLO, THANK YOU, NAME, YES, NO
   - Not currently used in gameplay levels

### Level Selection

Hardcoded in `menu.py` lines 393-398:
```python
LEVELS = [
    {"id": "letters", "label": "Level 1 - Letters A-Z",
     "sub": "Learn the ASL alphabet", "enabled": True},
    {"id": "gestures", "label": "Level 2 - ASL Gestures",
     "sub": "5 essential ASL phrases", "enabled": True},
]
```

### Application Flow

```
MAIN_MENU
├── [Play] → LEVEL_SELECT
│   ├── [Letters] → PLAY (letter mode)
│   └── [Gestures] → PLAY (gesture mode)
├── [Debug] → DEBUG_MENU
│   ├── [Record Data] → RECORD_MENU
│   │   ├── [Static Landmarks] → Train letter model
│   │   └── [Dynamic Gestures] → Train gesture model
│   └── [Back] → MAIN_MENU
└── [Quit] → EXIT
```

### Data Flow

**Play Mode (Letter):**
```
Camera → HandTracker → ASLClassifierLetters → StageTracker → Renderer
              ↓
           Letter classified
```

**Play Mode (Gesture):**
```
Camera → HandTracker → GestureDetector → StageTracker → Renderer
              ↓
           Gesture matched from GESTURE_REGISTRY
```

**Dynamic Sign (J/Z):**
```
Camera → HandTracker → TrainedDynamicDetector → StageTracker → Renderer
              ↓
           Motion trajectory analyzed
```

## File Organization

```
signsense/
├── signs/
│   ├── sign_registry.py           # A-Y STATIC, J DYNAMIC, Z DYNAMIC(disabled)
│   └── gesture_definitions.py     # HELLO, YES, NO, THANK YOU, NAME
├── detector/
│   ├── hand_tracker.py
│   ├── face_tracker.py
│   └── gesture_detector.py
├── ui/
│   ├── menu.py                    # Menu UIs
│   └── play_mode.py              # Stage-based renderer
├── ml/
│   ├── signs/                    # Static sign models (PyTorch)
│   └── dynamic_signs.py          # Dynamic gesture detector
└── config/
    ├── dynamic_signs.csv         # Quick sign registry
    └── dynamic_signs.yaml        # Detailed detector config
```

## Refactoring Achievements

✅ Clean separation of concerns
✅ Consistent UI across all levels
✅ Configurable dynamic signs via CSV/YAML
✅ Proper stage-based learning system
✅ Preview image system for both letters and gestures
✅ Face tracking integrated for YES/NO signs

## Known Gaps

1. **Gesture detection methods** in `gesture_detector.py` are currently stubbed (return defaults)
   - Should be retrained using the dynamic recorder
   - Placeholder values prevent accurate detection

2. **Dynamic sign detection** not yet implemented
   - `dynamic_signs.py` requires `TrainedDynamicDetector`
   - Motion trajectory logic needs implementation

3. **Gesture signs not in gameplay**
   - Defined in `gesture_definitions.py` but not used in levels
   - Could create "Level 3 - Common Signs" using gestures

## Next Steps (Level 3 Planning)

The user asked: *"What about level 3, and why not just use the gesture definitions?"*

### Option A: Expand Level 2 to Include Gestures
- Modify `LEVELS` to add gesture signs as a third option
- Keep current A-Z letters as Level 1
- Use gesture definitions directly (no training required initially)

### Option B: Create Level 3 - Common Signs
- Add gesture level after gestures
- Use the existing 5 gesture definitions
- Train gesture models using dynamic recorder
- Progressive difficulty: Letters → Gestures → Common Signs

### Option C: Merge Gestures into Current Levels
- Integrate gesture signs into existing level structure
- Hybrid approach: each level has both letters and gestures
- Simpler flow, but changes current design

### Recommendation

Based on your question about using gesture definitions:
- **Option A** seems cleanest - treat gestures as a complete level
- Use existing gesture definitions directly
- No need to retrain initially (can train later if needed)
- Users can progress: Letters → Gestures → (Advanced) Signs

Please clarify your preference for Level 3 so I can proceed with the implementation.
