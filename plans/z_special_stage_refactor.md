# Z Special Stage Refactoring Plan

## Current Implementation Analysis

The current `ZSpecialStage` in `signsense/ui/play_mode.py`:
- Uses `_draw_z_special_panel()` method (lines 896-938)
- Draws a panel on the right side at position `(W - pw - 8, 56)` with dimensions `160x180`
- Contains instructional text and a "Finish Game" button
- Preview box is hidden for ZSpecialStage (line 751)

## Task Requirements

1. Remove existing detector component (already done - ZSpecialStage has no detector)
2. Implement simplified interface
3. Center a sign panel on screen with step-by-step ASL Z instructions
4. Position preview panel directly beside it, centered vertically
5. Display image at `ASL_Alphabet/Z/view_1.png` in preview panel
6. Ensure both panels are aligned and visually balanced

## Design: Centered Two-Panel Layout

### Screen Dimensions
- Width: 640px
- Height: 480px

### Panel Dimensions
- **Sign Panel**: 280px wide × 320px tall (larger for better text readability)
- **Preview Panel**: 220px wide × 200px tall (matches existing PreviewBox dimensions)
- **Gap**: 20px between panels

### Positioning Calculation
```
Total width = 280 (sign) + 20 (gap) + 220 (preview) = 520px
Start X = (640 - 520) / 2 = 60px
Vertical center = 480 / 2 = 240px

Sign panel position:
  X = 60px
  Y = 240 - 320/2 = 80px

Preview panel position:
  X = 60 + 280 + 20 = 360px
  Y = 240 - 200/2 = 140px
```

### Visual Layout
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│         ┌──────────────────┐  ┌─────────────────┐          │
│         │   SIGN PANEL     │  │  PREVIEW PANEL  │          │
│         │                  │  │                 │          │
│         │  Letter Z:       │  │   [Z image]     │          │
│         │  Motion-Based    │  │                 │          │
│         │  Sign            │  │                 │          │
│         │                  │  │                 │          │
│         │  STEP 1: ...     │  │                 │          │
│         │  STEP 2: ...     │  │                 │          │
│         │  STEP 3: ...     │  │                 │          │
│         │                  │  │                 │          │
│         │  [Finish Game]   │  │                 │          │
│         └──────────────────┘  └─────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Modify `_draw_z_special_panel()` method
- Change panel dimensions to 280×320
- Calculate centered position based on two-panel layout
- Update text positioning for larger panel
- Keep "Finish Game" button at bottom

### Step 2: Create `_draw_z_preview_panel()` method
- New method to draw preview panel beside sign panel
- Load and display `ASL_Alphabet/Z/view_1.png`
- Use same styling as PreviewBox (border, background)
- Position directly beside sign panel, centered vertically

### Step 3: Update `render()` method
- For ZSpecialStage, call both `_draw_z_special_panel()` and `_draw_z_preview_panel()`
- Remove the preview box hiding logic for ZSpecialStage (line 751)
- Ensure proper event handling for the new layout

### Step 4: Update button bounds tracking
- Store finish button bounds for click detection
- Ensure button is clickable in new position

## Code Changes

### File: `signsense/ui/play_mode.py`

#### 1. Update `_draw_z_special_panel()` (lines 896-938)
```python
def _draw_z_special_panel(self, frame, px, py, pw, ph, tracker):
    """
    Draw the Z special educational panel with instructional text and Finish button.
    Centered on screen with preview panel beside it.
    """
    stage = tracker.current_stage
    if not stage:
        return
    
    # Title
    title = stage.instruction_title
    (tw, th), _ = cv2.getTextSize(title, FONT, 0.5, 1)
    tx = px + (pw - tw) // 2
    cv2.putText(frame, title, (tx, py + 25), FONT, 0.5, GOLD, 1, cv2.LINE_AA)
    
    # Instructional text (multi-line)
    lines = stage.instruction_text.split('\n')
    line_y = py + 50
    for line in lines:
        cv2.putText(frame, line, (px + 10, line_y), FONT, 0.28, WHITE, 1, cv2.LINE_AA)
        line_y += 18
    
    # Draw Finish Game button
    btn_w = pw - 20
    btn_h = 30
    btn_x = px + 10
    btn_y = py + ph - btn_h - 10
    
    # Button background
    overlay = frame.copy()
    cv2.rectangle(overlay, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), GREEN, -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), GREEN, 1)
    
    # Button text
    btn_text = stage.button_text
    (btw, bth), _ = cv2.getTextSize(btn_text, FONT, 0.5, 1)
    btx = btn_x + (btn_w - btw) // 2
    bty = btn_y + (btn_h + bth) // 2 - 2
    cv2.putText(frame, btn_text, (btx, bty), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    
    # Store button bounds for click detection
    self._finish_button_bounds = (btn_x, btn_y, btn_x + btn_w, btn_y + btn_h)
```

#### 2. Add new `_draw_z_preview_panel()` method
```python
def _draw_z_preview_panel(self, frame, px, py, pw, ph):
    """
    Draw preview panel for Z special stage with the Z sign image.
    Positioned directly beside the sign panel, centered vertically.
    """
    # Panel background
    overlay = frame.copy()
    _rr(overlay, px, py, px + pw, py + ph, PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (60, 60, 70), 1)
    
    # Try to load Z image
    img_path = "ASL_Alphabet/Z/view_1.png"
    try:
        real_img = cv2.imread(img_path)
        if real_img is not None:
            # Resize to fit preview panel (with padding)
            resized = cv2.resize(real_img, (pw - 8, ph - 8))
            frame[py+4:py+ph-4, px+4:px+pw-4] = resized
        else:
            # Draw placeholder if image not found
            self._draw_z_placeholder(frame, px, py, pw, ph)
    except Exception:
        # Draw placeholder on error
        self._draw_z_placeholder(frame, px, py, pw, ph)
```

#### 3. Add helper `_draw_z_placeholder()` method
```python
def _draw_z_placeholder(self, frame, px, py, pw, ph):
    """Draw placeholder when Z image is not available."""
    # Grey inner area
    cv2.rectangle(frame, (px+4, py+4), (px+pw-4, py+ph-4), (40, 42, 50), -1)
    # Large Z letter
    display = "Z"
    scale = 3.0
    (tw, th), _ = cv2.getTextSize(display, FONT, scale, 4)
    tx = px + (pw - tw) // 2
    ty = py + (ph + th) // 2
    cv2.putText(frame, display, (tx, ty), FONT, scale, ACCENT, 4, cv2.LINE_AA)
    # "preview" label
    cv2.putText(frame, "preview", (px + pw//2 - 28, py + ph - 10), FONT, 0.35, DIM, 1, cv2.LINE_AA)
```

#### 4. Update `render()` method (lines 713-763)
- Modify the preview box section to handle ZSpecialStage differently
- For ZSpecialStage, draw both sign panel and preview panel centered

```python
# In render() method, replace lines 748-754:
stage = stage_tracker.current_stage
if hasattr(stage, 'sign_type') and stage.sign_type == "SPECIAL":
    # ZSpecialStage: draw centered sign panel and preview panel
    self._draw_z_special_stage_centered(frame, stage_tracker)
else:
    # Regular stages: show preview box
    preview = self.preview_box.render(stage.name if stage else "?")
    frame[by:by + PreviewBox.H, bx:bx + PreviewBox.W] = preview
```

#### 5. Add new `_draw_z_special_stage_centered()` method
```python
def _draw_z_special_stage_centered(self, frame, tracker):
    """
    Draw Z special stage with centered two-panel layout:
    - Sign panel on the left with instructions
    - Preview panel on the right with Z image
    Both panels are centered vertically and aligned horizontally.
    """
    H, W = frame.shape[:2]
    
    # Panel dimensions
    sign_pw, sign_ph = 280, 320  # Sign panel
    prev_pw, prev_ph = 220, 200  # Preview panel
    gap = 20  # Gap between panels
    
    # Calculate total width and starting position
    total_width = sign_pw + gap + prev_pw
    start_x = (W - total_width) // 2
    
    # Vertical center
    center_y = H // 2
    
    # Sign panel position (centered vertically)
    sign_x = start_x
    sign_y = center_y - sign_ph // 2
    
    # Preview panel position (beside sign panel, centered vertically)
    prev_x = start_x + sign_pw + gap
    prev_y = center_y - prev_ph // 2
    
    # Draw sign panel
    self._draw_z_special_panel(frame, sign_x, sign_y, sign_pw, sign_ph, tracker)
    
    # Draw preview panel
    self._draw_z_preview_panel(frame, prev_x, prev_y, prev_pw, prev_ph)
```

## Testing Checklist

- [ ] Sign panel is centered on screen
- [ ] Preview panel is positioned directly beside sign panel
- [ ] Both panels are aligned vertically (same center line)
- [ ] Z image loads and displays correctly in preview panel
- [ ] Placeholder displays if image is missing
- [ ] Finish Game button is clickable
- [ ] Text is readable and well-positioned
- [ ] Visual balance is maintained
- [ ] Layout works at different screen sizes (if applicable)

## Notes

- The ZSpecialStage already has no detector component (sign_type="SPECIAL")
- The simplified interface is achieved by removing complex detection logic
- The centered layout provides better visual balance and focus
- The preview panel uses the same styling as the existing PreviewBox for consistency
