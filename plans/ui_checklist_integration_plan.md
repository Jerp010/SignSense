# SignSense UI Checklist Integration Plan

## Executive Summary

This plan transforms the provided UI checklist into a structured, phased integration strategy for SignSense. Each checklist item is broken down into specific, actionable steps with clear dependencies, verification criteria, and implementation details based on the existing codebase analysis.

**NEW: Phase 6 — ML-Based Dynamic Gesture Recognition System** has been added to replace all heuristic-based gesture recognition with a unified, modular machine learning approach.

---

## Current UI Architecture Analysis

### Files Involved
- **[`signsense/ui/play_mode.py`](signsense/ui/play_mode.py)** — Play mode renderer, sign panel, preview box, progress bar
- **[`signsense/ui/menu.py`](signsense/ui/menu.py)** — Main menu, level select, debug menu, record menu
- **[`signsense/ui/overlay.py`](signsense/ui/overlay.py)** — Debug overlay with hand landmarks, score bars
- **[`signsense/main.py`](signsense/main.py)** — Application state machine, window management

### Current Layout (Play Mode)
```
┌─────────────────────────────────────────┐
│  [camera feed]                          │
│                        ┌─────────────┐  │
│                        │ SIGN PANEL  │  │
│                        │  - Hand type│  │
│                        │  - Sign name│  │
│                        │  - Movement │  │
│                        │  - Desc     │  │
│                        └─────────────┘  │
│ ─── progress bar ─────────────────────  │
│                   ┌──────────────────┐  │
│                   │  PREVIEW BOX     │  │
│                   └──────────────────┘  │
└─────────────────────────────────────────┘
```

### Current Color Palette
- `BG = (15, 12, 20)` — Near-black with purple tint
- `PANEL_BG = (22, 18, 32)` — Dark panel background
- `ACCENT = (0, 210, 255)` — Cyan
- `ACCENT2 = (180, 60, 255)` — Purple
- `GREEN = (80, 220, 120)` — Success color
- `GOLD = (40, 200, 255)` — Highlight color
- `DIM = (110, 100, 130)` — Muted text

---

## Phase 1: Critical UI Fixes (Low Risk, High Impact)

### 1.1 Remove Hand Type from UI in Play Mode

**Current State:**  
The sign panel in [`_draw_sign_panel()`](signsense/ui/play_mode.py:553) displays:
- Line 580: `badge = f"Hand: {stage.hand_shape}"` — Shows "Hand: static" or "Hand: dynamic"
- Line 582: `cv2.putText(frame, badge, (px + 8, py + 16), FONT, 0.35, badge_col, 1, cv2.LINE_AA)`

**Implementation Steps:**
1. Remove lines 579-582 from [`_draw_sign_panel()`](signsense/ui/play_mode.py:553)
2. Remove the `hand_shape` property from [`GestureStage`](signsense/ui/play_mode.py:68) and [`LetterStage`](signsense/ui/play_mode.py:102) classes (optional — keep for internal use)
3. Verify no other UI elements reference `hand_shape`

**Files to Modify:**
- [`signsense/ui/play_mode.py`](signsense/ui/play_mode.py) — Remove badge rendering

**Verification:**
- [ ] Play mode launches without "Hand: ..." text in sign panel
- [ ] Sign name and description still display correctly
- [ ] No visual glitches or layout shifts

---

### 1.2 Move Description Below Sign Label

**Current State:**  
In [`_draw_sign_panel()`](signsense/ui/play_mode.py:553):
- Line 572-577: Sign name rendered at `tly = py + 45` (large, centered)
- Line 598-601: Description rendered at `desc_y = py + 48` (small, left-aligned, overlaps with sign name)

**Implementation Steps:**
1. Adjust sign name vertical position: Change `tly = py + 45` to `tly = py + 35` (move up)
2. Adjust description position: Change `desc_y = py + 48` to `desc_y = py + 60` (move below sign name)
3. Ensure description text is truncated properly (already handled at line 600)
4. Test with long descriptions to ensure no overflow

**Files to Modify:**
- [`signsense/ui/play_mode.py`](signsense/ui/play_mode.py) — Adjust vertical positions

**Verification:**
- [ ] Sign name appears at top of panel
- [ ] Description appears directly below sign name
- [ ] No text overlap between name and description
- [ ] Long descriptions truncate with "..." suffix

---

### 1.3 Dynamic Gestures Label Font Too Big

**Current State:**  
In [`_draw_sign_panel()`](signsense/ui/play_mode.py:553):
- Line 572: `tl_scale = 1.6` — Sign name font scale
- Line 573: `(tlw, tlh), _ = cv2.getTextSize(stage.name, FONT, tl_scale, 4)` — Thickness 4
- Line 577: `cv2.putText(frame, stage.name, (tlx, tly), FONT, tl_scale, color, 4, cv2.LINE_AA)`

For dynamic signs (J, Z), the sign name "J" or "Z" is rendered at scale 1.6 with thickness 4, which is disproportionately large.

**Implementation Steps:**
1. Add conditional scaling based on sign type:
   ```python
   is_dynamic = hasattr(stage, 'sign_entry') and stage.sign_entry.sign_type.name == "DYNAMIC"
   tl_scale = 1.2 if is_dynamic else 1.6
   tl_thickness = 3 if is_dynamic else 4
   ```
2. Update `cv2.getTextSize()` call to use `tl_thickness`
3. Update `cv2.putText()` call to use `tl_thickness`

**Files to Modify:**
- [`signsense/ui/play_mode.py`](signsense/ui/play_mode.py) — Add conditional font scaling

**Verification:**
- [ ] Static signs (A-I, K-Y) display at scale 1.6, thickness 4
- [ ] Dynamic signs (J, Z) display at scale 1.2, thickness 3
- [ ] Sign name remains centered in panel
- [ ] No visual clipping or overflow

---

## Phase 2: Layout Improvements (Medium Risk, High Impact)

### 2.1 Move Preview Box Away from Progress Bar

**Current State:**  
- Progress bar: Lines 633-648 — Positioned at `bar_y = H - 16` (bottom of frame)
- Preview box: Lines 505-512 — Positioned at `(w - PreviewBox.W - 8, h - PreviewBox.H - 8)` (bottom-right corner)
- Overlap: Preview box (150px tall) overlaps with progress bar area

**Implementation Steps:**
1. **Option A: Place preview box beside progress bar (recommended)**
   - Move preview box to right side, vertically centered: `bx = W - PreviewBox.W - 8`, `by = (H - PreviewBox.H) // 2`
   - This keeps preview box visible without overlapping progress bar
   
2. **Option B: Move preview box above progress bar**
   - Adjust preview box origin: `by = H - PreviewBox.H - 24` (above progress bar)
   - Requires increasing progress bar margin

3. **Option C: Make progress bar thinner and reposition**
   - Reduce `bar_h = 8` to `bar_h = 4`
   - Move preview box to `by = H - PreviewBox.H - 12`

**Recommended Approach: Option A**
```python
def get_preview_box_origin(self, frame_w=None, frame_h=None) -> tuple:
    w = frame_w if frame_w is not None else self.W
    h = frame_h if frame_h is not None else self.H
    # Position on right side, vertically centered
    bx = w - PreviewBox.W - 8
    by = (h - PreviewBox.H) // 2
    return (bx, by)
```

**Files to Modify:**
- [`signsense/ui/play_mode.py`](signsense/ui/play_mode.py) — Update `get_preview_box_origin()` method

**Verification:**
- [ ] Preview box appears on right side, vertically centered
- [ ] Progress bar spans full width at bottom without overlap
- [ ] Preview box arrows and page dots still functional
- [ ] Mouse click detection works with new position

---

### 2.2 Add Preview Pictures

**Current State:**  
- [`PreviewBox._draw_placeholder()`](signsense/ui/play_mode.py:253) shows a styled placeholder with gesture name
- Line 228-231: Comment indicates path would be `assets/signs/<GESTURE>/view_{page}.png`
- No actual image loading implemented

**Implementation Steps:**
1. **Create asset directory structure:**
   ```
   signsense/assets/signs/
   ├── A/
   │   ├── view_1.png
   │   ├── view_2.png
   │   └── view_3.png
   ├── B/
   │   └── ...
   └── HELLO/
       └── ...
   ```

2. **Update `PreviewBox.render()` method:**
   ```python
   def render(self, gesture_name: str) -> np.ndarray:
       img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
       img[:] = PANEL_BG
       
       # Try to load a real image
       loaded = False
       img_path = f"signsense/assets/signs/{gesture_name}/view_{self._page + 1}.png"
       try:
           real_img = cv2.imread(img_path)
           if real_img is not None:
               # Resize to fit preview box
               resized = cv2.resize(real_img, (self.W - 4, self.H - 20))
               img[10:self.H - 10, 2:self.W - 2] = resized
               loaded = True
       except Exception:
           pass
       
       if not loaded:
           self._draw_placeholder(img, gesture_name)
       
       # ... rest of rendering (page dots, arrows, border)
       return img
   ```

3. **Add fallback for missing images:**
   - Keep existing placeholder rendering
   - Add "No image" text overlay when image fails to load

4. **Create placeholder images for testing:**
   - Generate simple colored rectangles with sign name for each letter/gesture
   - Use consistent naming convention

**Files to Modify:**
- [`signsense/ui/play_mode.py`](signsense/ui/play_mode.py) — Update `PreviewBox.render()` to load images

**New Files to Create:**
- `signsense/assets/signs/` — Directory structure for preview images
- `signsense/assets/signs/PLACEHOLDER_README.md` — Documentation for image requirements

**Verification:**
- [ ] Preview box displays loaded images when available
- [ ] Placeholder shown when images missing
- [ ] Page navigation (arrows) works with loaded images
- [ ] No crashes when image files are corrupted or missing
- [ ] Images scale correctly to preview box dimensions

---

## Phase 3: Fullscreen Support (Medium Risk, Medium Impact)

### 3.1 Make Camera Able to Full Screen

**Current State:**  
- [`open_camera()`](signsense/main.py:105) opens camera with fixed resolution `W=640, H=480`
- [`get_window_size()`](signsense/main.py:143) retrieves current window size but doesn't enable fullscreen
- No fullscreen toggle mechanism exists

**Implementation Steps:**

1. **Add fullscreen toggle key binding:**
   - In [`run_play_mode()`](signsense/main.py:324), add key handler for 'F' key (or F11)
   - Toggle between windowed and fullscreen modes

2. **Implement fullscreen toggle function:**
   ```python
   def toggle_fullscreen(win_name: str):
       """Toggle between windowed and fullscreen mode."""
       prop = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)
       if prop == cv2.WINDOW_FULLSCREEN:
           cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
       else:
           cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
   ```

3. **Update window creation in `main()`:**
   ```python
   cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)  # Allow resizing
   cv2.resizeWindow(WIN, W, H)
   ```

4. **Handle dynamic resizing:**
   - Update `PlayModeRenderer` to accept dynamic frame dimensions
   - Ensure all UI elements scale proportionally
   - Update preview box positioning on resize

5. **Add fullscreen indicator:**
   - Show "Press F to toggle fullscreen" hint in HUD
   - Update HUD to show current mode (windowed/fullscreen)

**Files to Modify:**
- [`signsense/main.py`](signsense/main.py) — Add fullscreen toggle, update window creation
- [`signsense/ui/play_mode.py`](signsense/ui/play_mode.py) — Ensure responsive layout

**Verification:**
- [ ] Pressing 'F' toggles fullscreen mode
- [ ] All UI elements scale correctly in fullscreen
- [ ] Camera feed fills fullscreen without distortion
- [ ] Preview box repositions correctly on resize
- [ ] Progress bar spans full width in fullscreen
- [ ] Sign panel remains readable in fullscreen
- [ ] ESC exits fullscreen and returns to windowed mode

---

## Phase 4: UI Overhaul — Modern Professional Look (High Risk, High Impact)

### 4.1 Design System Updates

**Current State:**
- Dark tech/arcade aesthetic with purple/cyan accents
- Monospace fonts (FONT_HERSHEY_DUPLEX, FONT_HERSHEY_PLAIN)
- Scanline effects and grid backgrounds
- Rounded rectangles with manual corner drawing

**Proposed Modern Design:**

1. **Color Palette Refinement:**
   ```python
   # Primary colors
   BG_PRIMARY = (18, 18, 24)      # Slightly lighter dark background
   BG_SECONDARY = (28, 28, 36)    # Panel backgrounds
   BG_TERTIARY = (38, 38, 48)     # Hover states
   
   # Accent colors
   ACCENT_PRIMARY = (99, 102, 241)    # Indigo (modern, professional)
   ACCENT_SECONDARY = (139, 92, 246)  # Purple
   ACCENT_SUCCESS = (34, 197, 94)     # Green
   ACCENT_WARNING = (250, 204, 21)    # Yellow
   ACCENT_ERROR = (239, 68, 68)       # Red
   
   # Text colors
   TEXT_PRIMARY = (255, 255, 255)     # White
   TEXT_SECONDARY = (156, 163, 175)   # Gray
   TEXT_MUTED = (107, 114, 128)       # Dark gray
   ```

2. **Typography:**
   - Replace `FONT_HERSHEY_DUPLEX` with `FONT_HERSHEY_SIMPLEX` for cleaner look
   - Use consistent font scales:
     - Title: 1.2
     - Heading: 0.8
     - Body: 0.6
     - Caption: 0.4

3. **Spacing System:**
   ```python
   SPACING_XS = 4
   SPACING_SM = 8
   SPACING_MD = 16
   SPACING_LG = 24
   SPACING_XL = 32
   ```

4. **Component Library:**
   - Create reusable UI components:
     - `ModernButton` — Rounded corners, hover states, shadows
     - `ModernPanel` — Semi-transparent with blur effect
     - `ModernProgressBar` — Smooth gradients, rounded ends
     - `ModernBadge` — Pill-shaped status indicators

**Implementation Steps:**

1. **Create design system module:**
   - New file: `signsense/ui/design_system.py`
   - Define all colors, fonts, spacing constants
   - Create helper functions for common UI patterns

2. **Refactor existing UI components:**
   - Update [`_draw_sign_panel()`](signsense/ui/play_mode.py:553) to use new design system
   - Update [`_draw_progress_bar()`](signsense/ui/play_mode.py:633) with smooth gradients
   - Update [`_draw_hud()`](signsense/ui/play_mode.py:524) with modern styling
   - Update [`PreviewBox`](signsense/ui/play_mode.py:189) with rounded corners and shadows

3. **Update menu system:**
   - Refactor [`MainMenu`](signsense/ui/menu.py:158) with modern buttons
   - Update [`LevelSelect`](signsense/ui/menu.py:420) with card-based layout
   - Update [`DebugMenu`](signsense/ui/menu.py:236) with consistent styling

4. **Add visual polish:**
   - Subtle drop shadows on panels
   - Smooth color transitions on hover
   - Animated progress indicators
   - Micro-interactions on button clicks

**Files to Modify:**
- [`signsense/ui/play_mode.py`](signsense/ui/play_mode.py) — Update all rendering methods
- [`signsense/ui/menu.py`](signsense/ui/menu.py) — Update menu rendering
- [`signsense/ui/overlay.py`](signsense/ui/overlay.py) — Update debug overlay

**New Files to Create:**
- `signsense/ui/design_system.py` — Design system constants and helpers

**Verification:**
- [ ] All UI elements use consistent color palette
- [ ] Typography is consistent across all screens
- [ ] Spacing follows design system
- [ ] Hover states work on all interactive elements
- [ ] No visual glitches or rendering artifacts
- [ ] Performance remains acceptable (no frame drops)

---

### 4.2 UX Improvements

**Implementation Steps:**

1. **Visual Hierarchy:**
   - Make sign name more prominent (larger, brighter)
   - Reduce visual noise in sign panel
   - Use whitespace effectively

2. **Feedback Systems:**
   - Add progress ring around sign name during HOLDING state
   - Add success animation on CONFIRMING
   - Add subtle pulse effect on active elements

3. **Accessibility:**
   - Ensure sufficient color contrast (WCAG AA)
   - Add text shadows for readability over camera feed
   - Provide clear visual feedback for all interactions

4. **Information Architecture:**
   - Group related information together
   - Use visual separators for different sections
   - Prioritize most important information (sign name, progress)

**Files to Modify:**
- [`signsense/ui/play_mode.py`](signsense/ui/play_mode.py) — Update rendering logic
- [`signsense/ui/menu.py`](signsense/ui/menu.py) — Update menu layouts

**Verification:**
- [ ] Sign name is immediately visible and readable
- [ ] Progress is clearly communicated
- [ ] All interactive elements have clear affordances
- [ ] UI is readable over varying camera backgrounds

---

## Phase 5: Integration & Testing

### 5.1 Dependency Graph

```
Phase 1 (Critical Fixes)
├── 1.1 Remove Hand Type ──────────────────┐
├── 1.2 Move Description ─────────────────┤
└── 1.3 Fix Dynamic Font Size ─────────────┤
                                            │
Phase 2 (Layout)                           │
├── 2.1 Move Preview Box ◄─────────────────┤
└── 2.2 Add Preview Pictures ──────────────┤
                                            │
Phase 3 (Fullscreen)                       │
└── 3.1 Fullscreen Support ◄───────────────┘
                                            │
Phase 4 (UI Overhaul)                      │
├── 4.1 Design System ◄────────────────────┘
└── 4.2 UX Improvements
```

### 5.2 Implementation Order

**Recommended Sequence:**
1. **Phase 1.1** — Remove hand type (quick win, no dependencies)
2. **Phase 1.3** — Fix dynamic font size (quick win, no dependencies)
3. **Phase 1.2** — Move description (quick win, no dependencies)
4. **Phase 2.1** — Move preview box (requires Phase 1 completion)
5. **Phase 2.2** — Add preview pictures (requires Phase 2.1)
6. **Phase 3.1** — Fullscreen support (requires Phase 2.1)
7. **Phase 4.1** — Design system (requires all Phase 1-3)
8. **Phase 4.2** — UX improvements (requires Phase 4.1)

### 5.3 Testing Strategy

**Unit Tests:**
- Test each UI component in isolation
- Verify color values are valid BGR tuples
- Verify font scales are within acceptable ranges
- Verify positioning calculations don't overflow

**Integration Tests:**
- Test play mode with all sign types (static, dynamic)
- Test menu navigation with all UI changes
- Test fullscreen toggle with all UI elements
- Test preview box with and without images

**Visual Regression Tests:**
- Capture screenshots of each UI state
- Compare against baseline images
- Flag any visual differences for review

**User Testing:**
- Test with actual users for UX feedback
- Measure task completion time
- Collect satisfaction ratings

---

## Risk Assessment

| Phase | Risk Level | Mitigation Strategy |
|-------|------------|---------------------|
| Phase 1 | Low | Simple text removal/positioning changes |
| Phase 2 | Medium | Preview box repositioning may affect mouse hit-testing |
| Phase 3 | Medium | Fullscreen may expose layout scaling issues |
| Phase 4 | High | Major visual changes may introduce rendering bugs |

**Rollback Plan:**
- Each phase is independently deployable
- Git branches for each phase allow easy rollback
- Feature flags for new UI elements (if needed)

---

## Success Criteria

### Functional Requirements
- [ ] All checklist items implemented and verified
- [ ] No regressions in existing functionality
- [ ] All UI elements render correctly at all supported resolutions
- [ ] Performance remains above 30 FPS on target hardware

### Non-Functional Requirements
- [ ] UI is visually consistent across all screens
- [ ] Color contrast meets WCAG AA standards
- [ ] All interactive elements have clear affordances
- [ ] Code is maintainable and well-documented

### User Experience
- [ ] Users can quickly identify current sign and progress
- [ ] Preview images provide clear reference for expected gesture
- [ ] Fullscreen mode enhances immersion
- [ ] Modern UI feels professional and polished

---

## Appendix A: File Change Summary

| File | Changes | Phase |
|------|---------|-------|
| `signsense/ui/play_mode.py` | Remove hand badge, adjust positions, fix font sizes, update preview box, add image loading, update design | 1, 2, 4 |
| `signsense/ui/menu.py` | Update design system, modernize buttons | 4 |
| `signsense/ui/overlay.py` | Update design system, modernize overlay | 4 |
| `signsense/main.py` | Add fullscreen toggle, update window creation | 3 |
| `signsense/ui/design_system.py` | New file — design constants and helpers | 4 |
| `signsense/assets/signs/` | New directory — preview images | 2 |

---

## Appendix B: Code Snippets

### B.1 Remove Hand Type Badge
```python
# In _draw_sign_panel(), REMOVE these lines:
# badge = f"Hand: {stage.hand_shape}"
# badge_col = ACCENT2
# cv2.putText(frame, badge, (px + 8, py + 16), FONT, 0.35, badge_col, 1, cv2.LINE_AA)
```

### B.2 Adjust Description Position
```python
# In _draw_sign_panel(), CHANGE:
# desc_y = py + 48
# TO:
desc_y = py + 60  # Move below sign name
```

### B.3 Conditional Font Scaling for Dynamic Signs
```python
# In _draw_sign_panel(), ADD:
is_dynamic = hasattr(stage, 'sign_entry') and stage.sign_entry.sign_type.name == "DYNAMIC"
tl_scale = 1.2 if is_dynamic else 1.6
tl_thickness = 3 if is_dynamic else 4

# UPDATE:
(tlw, tlh), _ = cv2.getTextSize(stage.name, FONT, tl_scale, tl_thickness)
cv2.putText(frame, stage.name, (tlx, tly), FONT, tl_scale, color, tl_thickness, cv2.LINE_AA)
```

### B.4 Reposition Preview Box
```python
# In get_preview_box_origin(), CHANGE:
# return (w - PreviewBox.W - 8, h - PreviewBox.H - 8)
# TO:
bx = w - PreviewBox.W - 8
by = (h - PreviewBox.H) // 2  # Vertically centered
return (bx, by)
```

### B.5 Fullscreen Toggle
```python
# In run_play_mode(), ADD key handler:
if key == ord('f') or key == ord('F'):
    toggle_fullscreen(WIN)

# ADD helper function:
def toggle_fullscreen(win_name: str):
    prop = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)
    if prop == cv2.WINDOW_FULLSCREEN:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    else:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
```

---

## Appendix C: Design System Constants

```python
# signsense/ui/design_system.py

# Colors (BGR format)
BG_PRIMARY = (18, 18, 24)
BG_SECONDARY = (28, 28, 36)
BG_TERTIARY = (38, 38, 48)

ACCENT_PRIMARY = (99, 102, 241)    # Indigo
ACCENT_SECONDARY = (139, 92, 246)  # Purple
ACCENT_SUCCESS = (34, 197, 94)     # Green
ACCENT_WARNING = (250, 204, 21)    # Yellow
ACCENT_ERROR = (239, 68, 68)       # Red

TEXT_PRIMARY = (255, 255, 255)
TEXT_SECONDARY = (156, 163, 175)
TEXT_MUTED = (107, 114, 128)

# Typography
FONT_PRIMARY = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_TITLE = 1.2
FONT_SCALE_HEADING = 0.8
FONT_SCALE_BODY = 0.6
FONT_SCALE_CAPTION = 0.4

# Spacing
SPACING_XS = 4
SPACING_SM = 8
SPACING_MD = 16
SPACING_LG = 24
SPACING_XL = 32

# Border radius
RADIUS_SM = 4
RADIUS_MD = 8
RADIUS_LG = 12
RADIUS_XL = 16
```

---

**Document Version:** 1.0
**Last Updated:** 2026-03-26
**Author:** SignSense Development Team
**Status:** Ready for Review

---

## Phase 6: ML-Based Dynamic Gesture Recognition System (High Impact, Architectural Change)

### 6.1 Current Heuristic-Based Recognition Analysis

#### 6.1.1 Identified Heuristic Components

**A. GestureDetector** ([`signsense/detector/gesture_detector.py`](signsense/detector/gesture_detector.py))
- **Purpose**: Detects 9 ASL gestures (HELLO, THANK YOU, NAME, GOOD, HELP, WATER, YES, NO, BAD)
- **Heuristic Methods**:
  - [`_detect_hand_shape()`](signsense/detector/gesture_detector.py:290) — Finger extension detection using landmark position comparisons
  - [`_match_hand_shape()`](signsense/detector/gesture_detector.py:360) — Rule-based matching of detected shapes to expected shapes
  - [`_validate_hand_region()`](signsense/detector/gesture_detector.py:390) — Hardcoded body region validation using y/x coordinate thresholds
  - [`_detect_movement_pattern()`](signsense/detector/gesture_detector.py:430) — Movement detection from position history using simple delta calculations
  - [`_match_movement()`](signsense/detector/gesture_detector.py:474) — Rule-based movement pattern matching
  - [`_detect_head_movement()`](signsense/detector/gesture_detector.py:499) — Head nod/shake detection using face position history
  - [`_calculate_confidence()`](signsense/detector/gesture_detector.py:535) — Confidence calculation based on hand shape and movement matches

**B. JDetector** ([`signsense/signs/dynamic_signs.py`](signsense/signs/dynamic_signs.py))
- **Purpose**: Hardcoded state-machine detector for ASL letter 'J'
- **Heuristic Methods**:
  - [`_is_i_shape()`](signsense/signs/dynamic_signs.py:160) — Finger position comparisons for I shape detection
  - [`_is_palm_away()`](signsense/signs/dynamic_signs.py:168) — Thumb-wrist position comparison for palm orientation
  - State machine logic (lines 80-142) — Phase tracking (hold I → hook down → palm away + re-hold I)

**C. TrainedDynamicDetector** ([`signsense/signs/trainable_dynamic_signs.py`](signsense/signs/trainable_dynamic_signs.py))
- **Purpose**: LSTM-based detector for dynamic signs (J, Z)
- **Current ML Approach**: Uses trained LSTM model with stage prediction
- **Limitation**: Only supports dynamic signs (J, Z), not static gestures

#### 6.1.2 Problems with Heuristic Approach

1. **Fragile Thresholds**: Hardcoded coordinate thresholds (e.g., `0.1 < y < 0.35` for forehead) break with different camera angles, lighting, or user body proportions
2. **Limited Generalization**: Rule-based matching doesn't handle variations in signing style or hand anatomy
3. **Maintenance Burden**: Adding new gestures requires writing new heuristic methods
4. **Inconsistent Confidence**: Confidence calculation is ad-hoc (0.5 base + 0.3 for hand shape + 0.2 for movement)
5. **No Continuous Learning**: System can't improve from user data without manual threshold tuning

### 6.2 Proposed ML Architecture

#### 6.2.1 Unified Gesture Recognition Model

**Architecture**: Multi-Task LSTM with Attention
```
Input: Sequence of hand landmarks (variable length)
  ↓
Feature Extraction: Landmark normalization + temporal features
  ↓
LSTM Encoder: 2-layer LSTM with attention mechanism
  ↓
Multi-Task Heads:
  ├── Static Gesture Classifier (9 ASL gestures + letters)
  ├── Dynamic Gesture Stage Predictor (for J, Z, etc.)
  └── Confidence Estimator (learned confidence, not heuristic)
```

**Key Components**:

1. **UnifiedLandmarkProcessor**: Normalizes landmarks and extracts temporal features
2. **GestureEncoder**: LSTM-based sequence encoder with attention
3. **StaticGestureHead**: Classifier for static gestures (HELLO, THANK YOU, etc.)
4. **DynamicGestureHead**: Stage predictor for dynamic gestures (J, Z)
5. **ConfidenceHead**: Learned confidence estimation

#### 6.2.2 Training Pipeline

**Data Collection**:
- Extend existing [`dynamic_recorder.py`](signsense/ml/dynamic_recorder.py) to support static gesture recording
- Create unified dataset format for both static and dynamic gestures
- Use existing landmark data from [`ml/data/landmarks.csv`](ml/data/landmarks.csv)

**Training Script**:
- New file: `signsense/ml/unified_gesture_train.py`
- Supports training both static and dynamic gesture models
- Multi-task learning with weighted loss

**Model Registry**:
- Extend [`dynamic_signs.yaml`](signsense/config/dynamic_signs.yaml) to include static gestures
- Unified configuration for all gesture types

### 6.3 Implementation Plan

#### Phase 6.1: Create Unified ML Infrastructure

**Step 1: Create Unified Model Architecture**
- New file: `signsense/ml/unified_gesture_model.py`
- Implement `UnifiedGestureLSTM` with multi-task heads
- Implement `UnifiedLandmarkProcessor` for feature extraction

**Step 2: Create Unified Training Script**
- New file: `signsense/ml/unified_gesture_train.py`
- Support for both static and dynamic gesture training
- Multi-task learning with configurable loss weights

**Step 3: Extend Configuration System**
- Update [`dynamic_signs.yaml`](signsense/config/dynamic_signs.yaml) to include static gestures
- Add `gesture_type` field: "static" or "dynamic"
- Add `num_stages` field for dynamic gestures

**Step 4: Create Data Pipeline**
- Extend [`dynamic_recorder.py`](signsense/ml/dynamic_recorder.py) to record static gestures
- Create unified dataset loader for both gesture types
- Support for existing landmark data

#### Phase 6.2: Replace GestureDetector with ML Model

**Step 1: Create ML-Based Static Gesture Detector**
- New file: `signsense/detector/ml_gesture_detector.py`
- Implements same interface as [`GestureDetector`](signsense/detector/gesture_detector.py:54)
- Uses trained LSTM model for detection

**Step 2: Update PlayMode Integration**
- Modify [`PlayMode.__init__()`](signsense/ui/play_mode.py:715) to use ML detector
- Add model loading and initialization
- Maintain backward compatibility with heuristic detector

**Step 3: Update Main Loop**
- Modify [`run_play_mode()`](signsense/main.py:324) to initialize ML detector
- Pass model path and configuration

#### Phase 6.3: Replace JDetector with ML Model

**Step 1: Extend TrainedDynamicDetector**
- Update [`TrainedDynamicDetector`](signsense/signs/trainable_dynamic_signs.py:29) to support all dynamic gestures
- Remove hardcoded J-specific logic
- Use configuration-driven stage detection

**Step 2: Update DynamicSignFactory**
- Modify [`DynamicSignFactory.create_detector()`](signsense/signs/dynamic_sign_factory.py:28) to prefer ML models
- Remove hardcoded J detector fallback
- Add model availability checking

**Step 3: Remove Heuristic JDetector**
- Delete [`JDetector`](signsense/signs/dynamic_signs.py:37) class
- Remove [`_is_i_shape()`](signsense/signs/dynamic_signs.py:160) and [`_is_palm_away()`](signsense/signs/dynamic_signs.py:168) methods
- Update [`get_detector()`](signsense/signs/dynamic_signs.py:179) to only return ML-based detectors

#### Phase 6.4: Training and Validation

**Step 1: Train Static Gesture Model**
- Record training data for 9 ASL gestures
- Train unified model with static gesture head
- Validate accuracy on test set

**Step 2: Train Dynamic Gesture Models**
- Use existing J and Z training data
- Train unified model with dynamic gesture head
- Validate stage detection accuracy

**Step 3: Integration Testing**
- Test play mode with ML-based detection
- Compare accuracy with heuristic approach
- Measure latency and performance

### 6.4 Files to Create

| File | Purpose |
|------|---------|
| `signsense/ml/unified_gesture_model.py` | Unified LSTM model with multi-task heads |
| `signsense/ml/unified_gesture_train.py` | Training script for unified model |
| `signsense/detector/ml_gesture_detector.py` | ML-based static gesture detector |
| `signsense/config/gesture_registry.yaml` | Unified configuration for all gestures |
| `signsense/ml/data/static/` | Directory for static gesture training data |

### 6.5 Files to Modify

| File | Changes |
|------|---------|
| `signsense/detector/gesture_detector.py` | Deprecate in favor of ML detector |
| `signsense/signs/dynamic_signs.py` | Remove JDetector, update get_detector() |
| `signsense/signs/dynamic_sign_factory.py` | Prefer ML models, remove hardcoded fallback |
| `signsense/signs/trainable_dynamic_signs.py` | Extend to support all dynamic gestures |
| `signsense/config/dynamic_signs.yaml` | Add static gesture configurations |
| `signsense/ui/play_mode.py` | Update to use ML detector |
| `signsense/main.py` | Update detector initialization |

### 6.6 Files to Delete

| File | Reason |
|------|--------|
| `signsense/signs/dynamic_signs.py` | Replaced by ML-based detectors |
| `signsense/detector/gesture_detector.py` | Replaced by ML detector |

### 6.7 Verification Criteria

**Functional Requirements**:
- [ ] ML model detects all 9 static gestures with ≥90% accuracy
- [ ] ML model detects dynamic gestures (J, Z) with ≥85% accuracy
- [ ] Confidence scores are calibrated (not heuristic)
- [ ] Latency ≤33ms per frame (30 FPS)
- [ ] Model loads in <2 seconds

**Non-Functional Requirements**:
- [ ] Adding new gesture requires only data collection + training (no code changes)
- [ ] Configuration-driven gesture definitions
- [ ] Clear separation between training and inference
- [ ] Comprehensive logging and error handling

**Backward Compatibility**:
- [ ] Play mode works with ML detector
- [ ] Debug mode shows ML confidence scores
- [ ] Training pipeline supports both static and dynamic gestures

### 6.8 Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model accuracy lower than heuristic | High | Train on diverse dataset, use data augmentation |
| Latency too high for real-time | Medium | Optimize model size, use quantization |
| Training data insufficient | High | Use existing landmark data, collect more if needed |
| Breaking existing functionality | High | Maintain backward compatibility, gradual rollout |

### 6.9 Implementation Order

**Recommended Sequence**:
1. **Phase 6.1** — Create unified ML infrastructure (model, training, config)
2. **Phase 6.2** — Replace GestureDetector with ML model
3. **Phase 6.3** — Replace JDetector with ML model
4. **Phase 6.4** — Train and validate models

**Dependencies**:
- Phase 6.1 must complete before 6.2 and 6.3
- Phase 6.2 and 6.3 can be done in parallel
- Phase 6.4 requires Phase 6.2 and 6.3 to complete

### 6.10 Superior Architectural Proposal

**Instead of replacing heuristics one-by-one, implement a unified gesture recognition system:**

**Architecture: Gesture Recognition Service**
```python
class GestureRecognitionService:
    """Unified service for all gesture recognition."""
    
    def __init__(self, model_path: str, config_path: str):
        self.model = self._load_model(model_path)
        self.config = self._load_config(config_path)
        self.static_detector = StaticGestureDetector(self.model)
        self.dynamic_detector = DynamicGestureDetector(self.model)
    
    def detect(self, landmarks, handedness, frame_history) -> DetectionResult:
        """Unified detection for all gesture types."""
        # Extract features
        features = self.feature_extractor(landmarks, frame_history)
        
        # Run inference
        with torch.no_grad():
            static_logits, dynamic_logits, confidence = self.model(features)
        
        # Post-process
        if self._is_dynamic_gesture(dynamic_logits):
            return self.dynamic_detector.process(dynamic_logits, confidence)
        else:
            return self.static_detector.process(static_logits, confidence)
```

**Benefits**:
1. **Single Source of Truth**: One model handles all gestures
2. **Easy Extension**: Add new gesture by collecting data + training
3. **Consistent Interface**: Same API for static and dynamic gestures
4. **Learned Confidence**: Model learns confidence, not heuristic
5. **Continuous Improvement**: Can retrain with new data

**Migration Path**:
1. Create unified model alongside existing heuristics
2. Train on existing data + collect new data
3. A/B test ML vs heuristic in debug mode
4. Gradually replace heuristic with ML
5. Remove heuristic code once ML is validated

### 6.11 Success Metrics

**Accuracy**:
- Static gestures: ≥90% accuracy on test set
- Dynamic gestures: ≥85% stage detection accuracy
- Overall: ≥88% accuracy across all gestures

**Performance**:
- Inference latency: ≤33ms (30 FPS)
- Model size: ≤10MB
- Memory usage: ≤100MB

**Maintainability**:
- New gesture addition: ≤1 day (data collection + training)
- Configuration changes: No code changes required
- Clear documentation for training pipeline

---

**Document Version:** 1.1
**Last Updated:** 2026-03-26
**Author:** SignSense Development Team
**Status:** Ready for Review
