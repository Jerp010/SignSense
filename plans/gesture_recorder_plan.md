# SignSense Flexible Gesture Recorder Plan

## Current Limitations

### Dynamic Recorder (`dynamic_recorder.py`)
1. Only accepts A-Z keys for sign names
2. Requires config to exist in YAML for each sign
3. Only works for multi-stage gestures (like J, Z)
4. No support for single-stage/simple gestures
5. No way to add more samples to existing gestures

### Static Recorder (`record_landmarks.py`)
1. Only accepts A-Z keys for labels
2. No way to add more samples to existing labels

## Required Changes

### 1. Dynamic Sign Configuration Schema
Update `dynamic_signs.yaml` to support:
- Custom gesture names (not just letters like "hello", "thank_you")
- Single-stage (simple) gestures
- Multi-stage (complex) gestures
- Auto-creation of config for new gestures

```yaml
dynamic_signs:
  # Existing complex gestures (multi-stage)
  J:
    type: "complex"  # multi-stage motion
    stages:
      0: "Hold I position"
      1: "Hook pinky down"
      2: "Palm away"
    # ... rest of config
  
  # New simple gesture (single stage)
  HELLO:
    type: "simple"  # single pose/gesture
    description: "Wave hello"
    min_samples: 10  # minimum samples for training
    
  THANK_YOU:
    type: "simple"
    description: "Thank you wave"
    min_samples: 10
```

### 2. Dynamic Config Loader
Update `dynamic_config.py` to:
- Support loading config for any gesture (not just predefined)
- Allow creating default config for new gestures
- Support both simple and complex gesture types

### 3. Dynamic Recorder
Refactor to support:
- Custom gesture name input via keyboard (type name + Enter)
- Single-stage mode for simple gestures (auto-record)
- Multi-stage mode for complex gestures (stage-by-stage)
- Adding samples to existing gestures (append to existing data)
- Auto-create config if not exists

### 4. Static Recorder
Update to support:
- Custom label input via keyboard (type label + Enter)
- Adding samples to existing labels

## Implementation Plan

### Phase 1: Update Configuration System
1. Update `dynamic_signs.yaml` with new schema
2. Update `dynamic_config.py` to support:
   - `GestureType` enum (SIMPLE, COMPLEX)
   - Default configs for new gestures
   - Auto-create config for unknown gestures

### Phase 2: Refactor Dynamic Recorder
1. Add text input mode for custom gesture names
2. Add mode selection (simple/complex)
3. Implement append-to-existing functionality
4. Auto-save metadata updates

### Phase 3: Refactor Static Recorder
1. Add text input mode for custom labels
2. Implement append-to-existing functionality

### Phase 4: UI Integration
1. Add recording menu options to main menu
2. Add gesture selection UI

## File Changes

| File | Changes |
|------|---------|
| `signsense/config/dynamic_signs.yaml` | Add new gesture types, support custom names |
| `signsense/config/dynamic_config.py` | Add GestureType, auto-create config |
| `signsense/ml/dynamic_recorder.py` | Full refactor with new features |
| `signsense/ml/record_landmarks.py` | Add text input, append support |

## Key Features

### Custom Gesture Names
- Press 'T' to enter text input mode
- Type gesture name (e.g., "hello", "thank_you")
- Press Enter to confirm
- Creates directory: `ml/data/dynamic/HELLO/`

### Simple vs Complex Gestures
- **Simple**: Single stage, continuous recording (e.g., wave)
- **Complex**: Multiple stages, stage-by-stage recording (e.g., J, Z)

### Adding to Existing Gestures
- Select existing gesture name
- System loads existing metadata
- New recordings append to existing .npy files
- Metadata updates with new sample count
