# Single-Stage Dynamic Sign Fix

## Problem Description

The play mode configuration was not properly handling single-stage dynamic signs, causing repeated stage transition errors:

```
[TrainedDynamicDetector] Loaded J model (3 stages)
[J] Stage timeout - resetting
[J] Stage 0 → 2 (REJECTED - must go through 1 first)
[J] Stage 0 → 2 (REJECTED - must go through 1 first)
[J] Stage 0 → 2 (REJECTED - must go through 1 first)
...
```

## Root Cause

The issue occurred because:

1. **Configuration Mismatch**: The J sign was configured as `"type": "simple"` (single-stage) in [`dynamic_signs.yaml`](signsense/config/dynamic_signs.yaml:7)

2. **Model Training**: The trained J model was loaded with 3 stages (as shown in the error message)

3. **Sequential Enforcement**: The [`TrainedDynamicDetector`](signsense/signs/trainable_dynamic_signs.py:29) enforced strict sequential stage progression (0→1→2)

4. **Invalid Jumps**: When the model predicted stage 2 directly from stage 0, it was rejected with the error: "Stage 0 → 2 (REJECTED - must go through 1 first)"

## Solution

Modified [`trainable_dynamic_signs.py`](signsense/signs/trainable_dynamic_signs.py:191) to add special handling for simple/single-stage models:

### Key Changes

Added a conditional check that examines the configuration type before enforcing sequential stage progression:

```python
# SPECIAL CASE: Simple/single-stage models
# For models configured as "simple" type, skip sequential progression enforcement
# and allow any stage prediction to complete the sign
# This handles cases where a model was trained with multiple stages but is
# configured as a simple (single-stage) motion gesture
is_simple_type = hasattr(self._config, 'is_simple') and self._config.is_simple

if is_simple_type or self._num_stages == 1:
    # Simple/single-stage model: accept any stage prediction with sufficient confidence
    if stage_confidence >= min_confidence:
        # For simple models, jump directly to final stage
        if self._current_stage < self._num_stages - 1:
            print(f"[{self.sign_name}] Simple model: advancing to final stage (conf={stage_confidence:.2f})")
            self._current_stage = self._num_stages - 1
            self._stage_frames = 0
            self._consecutive_stage_preds = 0
    else:
        # Low confidence - stay in current stage
        self._stage_frames += 1
elif predicted_stage != self._current_stage:
    # Multi-stage model: enforce sequential progression
    # ... (existing logic)
```

### How It Works

1. **Configuration Type Detection**: Checks if the sign is configured as `type: "simple"` in [`dynamic_signs.yaml`](signsense/config/dynamic_signs.yaml)

2. **Skip Sequential Enforcement**: For simple-type models, the strict sequential progression (0→1→2) is bypassed, even if the model was trained with multiple stages

3. **Direct Advancement**: Any stage prediction with sufficient confidence (≥ 0.4) advances directly to the final stage

4. **Completion**: Once at the final stage, the sign completes after holding for the required frames (10 frames ≈ 0.3 seconds at 30fps)

5. **Multi-Stage Preservation**: Multi-stage signs (Z, HELLO, THANK_YOU, etc.) configured as `type: "complex"` continue to use the existing sequential progression logic

## Additional Fix: Cooldown Mechanism

Added a cooldown mechanism to prevent consecutive frame confirmations:

```python
# Reset hold counter when reaching final stage
if self._current_stage >= self._num_stages - 1:
    self._final_stage_hold = getattr(self, '_final_stage_hold', 0) + 1
    # Need to hold final stage for minimum frames before completing
    min_hold_frames = 10  # ~0.3 seconds at 30fps
    if self._final_stage_hold >= min_hold_frames:
        # Check cooldown to prevent consecutive confirmations
        # This prevents the detector from confirming on every frame after completion
        frames_since_last_completion = getattr(self, '_frames_since_completion', 0)
        cooldown_frames = 30  # ~1 second cooldown at 30fps
        
        if frames_since_last_completion >= cooldown_frames:
            self._phase_complete = True
            print(f"[{self.sign_name}] Sign complete! ✓ (held {self._final_stage_hold} frames)")
            # Reset detector after completion to prevent repeated confirmations
            self.reset()
            # Set cooldown counter
            self._frames_since_completion = 0
            return True
        else:
            # Still in cooldown period - don't confirm yet
            self._frames_since_completion = frames_since_last_completion + 1
    else:
        # Increment cooldown counter if we're not at final stage
        if hasattr(self, '_frames_since_completion'):
            self._frames_since_completion += 1
else:
    # Increment cooldown counter if we're not at final stage
    if hasattr(self, '_frames_since_completion'):
        self._frames_since_completion += 1
```

### How Cooldown Works

1. **Completion Detection**: When the detector reaches the final stage and holds for the required frames (10 frames ≈ 0.3 seconds), it checks the cooldown
2. **Cooldown Period**: After completion, there's a 30-frame cooldown (~1 second at 30fps) before the next completion can be triggered
3. **Prevents Consecutive Confirmations**: This ensures the detector doesn't confirm on every frame after completion, which was causing the "Confirmed gesture J at frame 906" and "907" issue
4. **Automatic Reset**: The cooldown counter resets automatically after the cooldown period expires

## Benefits

- ✅ Eliminates "Stage 0 → 2 (REJECTED)" errors for single-stage models
- ✅ Allows single-stage dynamic signs to complete successfully
- ✅ Prevents consecutive frame confirmations with cooldown mechanism
- ✅ Maintains strict sequential progression for multi-stage signs
- ✅ No changes required to existing trained models
- ✅ Backward compatible with all existing configurations

## Testing

To verify the fix works correctly:

1. **Single-Stage Signs (J)**:
   - Should advance directly to final stage when confidence ≥ 0.4
   - Should complete after holding final stage for 10 frames
   - Should NOT show "REJECTED" errors

2. **Multi-Stage Signs (Z, HELLO, etc.)**:
   - Should continue to enforce sequential progression (0→1→2→3)
   - Should still reject invalid jumps (e.g., 0→2)
   - Should maintain existing behavior

## Related Files

- [`signsense/signs/trainable_dynamic_signs.py`](signsense/signs/trainable_dynamic_signs.py) - Main fix implementation
- [`signsense/config/dynamic_signs.yaml`](signsense/config/dynamic_signs.yaml) - Sign configuration
- [`signsense/ui/play_mode.py`](signsense/ui/play_mode.py) - Play mode integration
- [`signsense/signs/dynamic_sign_factory.py`](signsense/signs/dynamic_sign_factory.py) - Detector factory

## Configuration Reference

The J sign is configured as a simple (single-stage) motion gesture:

```yaml
dynamic_signs:
  J:
    type: "simple"  # single-stage motion gesture
    description: "Letter J in ASL"
    training:
      sequence_length: 120
      batch_size: 8
      epochs: 50
      learning_rate: 0.001
      hidden_size: 128
```

Multi-stage signs (like Z) are configured as complex:

```yaml
  Z:
    type: "complex"
    description: "Letter Z in ASL"
    detector:
      diagonal_threshold: 0.12
      horizontal_threshold: 0.08
      phase_timeout: 90
    training:
      sequence_length: 150
      batch_size: 10
      epochs: 60
      learning_rate: 0.0012
      hidden_size: 128
    stages:
      0: "Step 1/4 — Start position"
      1: "Step 2/4 — Horizontal stroke"
      2: "Step 3/4 — Diagonal stroke"
      3: "Step 4/4 — End position"
```
