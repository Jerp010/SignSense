"""
ui/play_mode.py
===============
Play mode for SignSense - Unified version.

Manages per-sign stage progression supporting both:
  - Letter-based signs (A-Z) from sign_registry with STATIC/DYNAMIC types
  - Gesture-based signs (HELLO, THANK YOU, etc.) from gesture_definitions

Static signs: user must hold for HOLD_SECONDS
Dynamic signs: dedicated detector fires on completion

Layout
------
  ┌─────────────────────────────────────────┐
  │  [camera feed]                          │
  │                        ┌─────────────┐  │
  │                        │ SIGN PANEL  │  │
  │                        └─────────────┘  │
  │ ─── progress bar ─────────────────────  │
  │                   ┌──────────────────┐  │
  │                   │  PREVIEW BOX     │  │
  │                   └──────────────────┘  │
  └─────────────────────────────────────────┘

The preview box sits in the bottom-right corner and cycles through up to
3 placeholder images (or real images when ASL_Alphabet/<LETTER>/ is populated).

Supports both letter-based (A-Z) and gesture-based stages.
"""

import cv2
import numpy as np
import math
import time
from typing import Optional, Dict, List, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import logger


# Import gesture detection components
from signsense.signs.gesture_definitions import (
    GESTURE_REGISTRY,
    GestureClass,
    get_all_gestures
)
from signsense.detector.gesture_detector import GestureDetector, DetectionResult

# Import sign registry for letter-based signs
from signsense.signs.sign_registry import (
    ACTIVE_SIGNS,
    SignType,
    SignEntry,
    get_sign
)

# Import dynamic sign detector
from signsense.signs.dynamic_signs import get_detector


# ---------------------------------------------------------------------------
# Gesture stage list - the 5 ASL gestures
# ---------------------------------------------------------------------------

class GestureStage:
    """Represents a gesture stage in the game."""
    
    def __init__(self, gesture: GestureClass):
        self.gesture = gesture
        self.name = gesture.name
        # Shorter description for UI
        self.description = f"{gesture.hand_shape.value}"
    
    @property
    def display_name(self) -> str:
        """Name to display in UI."""
        return self.name
    
    @property
    def sign_type(self) -> str:
        """Return sign type as string: STATIC or DYNAMIC."""
        # All current gestures are hold-based (STATIC)
        return "STATIC"
    
    @property
    def hand_shape(self) -> str:
        return self.gesture.hand_shape.value
    
    @property
    def movement(self) -> str:
        return self.gesture.movement_pattern.value
    
    @property
    def target_region(self) -> str:
        """Target body region for this gesture."""
        return self.gesture.target_region if hasattr(self.gesture, 'target_region') else ""


class LetterStage:
    """Represents a letter stage in the game (Level 1)."""
    
    def __init__(self, sign_entry: SignEntry):
        self.sign_entry = sign_entry
        self.name = sign_entry.letter
        self.description = sign_entry.description
        self.letter = sign_entry.letter
    
    @property
    def display_name(self) -> str:
        """Letter to display in UI."""
        return self.name
    
    @property
    def sign_type(self) -> str:
        """Return sign type as string: STATIC or DYNAMIC."""
        return self.sign_entry.sign_type.name
    
    @property
    def hand_shape(self) -> str:
        return "static" if self.sign_type == "STATIC" else "dynamic"
    
    @property
    def movement(self) -> str:
        return "hold" if self.sign_type == "STATIC" else "motion"


class ZSpecialStage:
    """
    Represents a special Z educational stage that appears after all regular letters.
    This stage contains instructional text and a "Finish Game" button instead of
    actual gesture detection.
    """
    
    def __init__(self):
        self.name = "Z"
        self.letter = "Z"
        self.description = "Special Z Stage - Learn about the motion-based Z sign"
        self.sign_type = "SPECIAL"  # Different from STATIC/DYNAMIC
    
    @property
    def display_name(self) -> str:
        return "Z - Motion Sign"
    
    @property
    def hand_shape(self) -> str:
        return "motion"
    
    @property
    def movement(self) -> str:
        return "motion"
    
    @property
    def instruction_title(self) -> str:
        return "Letter Z: Motion-Based Sign"
    
    @property
    def instruction_text(self) -> str:
        return (
            "The letter Z is unique among ASL letters because it REQUIRES MOTION.\n\n"
            "STEP 1: Start with the 'I' handshape (pinky finger pointing up,\n"
            "        all other fingers curled into palm)\n\n"
            "STEP 2: While holding the 'I' shape, trace a 'Z' in the air:\n"
            "        - Draw diagonal line from top-left to bottom-right\n"
            "        - Then draw horizontal line from right to left\n\n"
            "This motion-based signing is what makes Z special among the\n"
            "ASL alphabet - it's the only letter that requires movement!"
        )
    
    @property
    def button_text(self) -> str:
        return "Finish Game"
    
    @property
    def short_tip(self) -> str:
        return "Trace 'Z' in the air with your pinky finger"


# Create ordered list of gesture stages
GESTURE_STAGES: List[GestureStage] = [
    GestureStage(GESTURE_REGISTRY["HELLO"]),
    GestureStage(GESTURE_REGISTRY["THANK YOU"]),
    GestureStage(GESTURE_REGISTRY["NAME"]),
    GestureStage(GESTURE_REGISTRY["YES"]),
    GestureStage(GESTURE_REGISTRY["NO"]),
]


# ---------------------------------------------------------------------------
# Layout / palette constants
# ---------------------------------------------------------------------------
BG          = (20, 20, 25)
PANEL_BG    = (35, 38, 45)
ACCENT      = (70, 130, 220)
ACCENT2     = (100, 100, 120)
GREEN       = (80, 180, 120)
ORANGE      = (100, 140, 200)
RED_COL     = (120, 90, 100)
WHITE       = (245, 245, 250)
DIM         = (120, 125, 135)
TEXT_DIM    = (130, 135, 145)
GOLD        = (180, 160, 100)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO   = cv2.FONT_HERSHEY_PLAIN

HOLD_SECONDS    = 3.0   # seconds to hold a gesture
CONFIRM_FLASH   = 1.2   # seconds to show the "✓" flash


def _centered_text(frame, txt, cy, scale, color, thick=1):
    (w, h), _ = cv2.getTextSize(txt, FONT, scale, thick)
    W = frame.shape[1]
    cv2.putText(frame, txt, ((W - w)//2, cy), FONT, scale, color, thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

 


def _rr(frame, x1, y1, x2, y2, color, thick=-1, r=6):
    """Minimal rounded rect."""
    cv2.rectangle(frame, (x1+r, y1), (x2-r, y2), color, thick)
    cv2.rectangle(frame, (x1, y1+r), (x2, y2-r), color, thick)
    if thick == -1:
        for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(frame, (cx, cy), r, color, -1)
    else:
        for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.ellipse(frame, (cx,cy), (r,r), 0,   0,  90, color, thick)
            cv2.ellipse(frame, (cx,cy), (r,r), 90,  0,  90, color, thick)
            cv2.ellipse(frame, (cx,cy), (r,r), 180, 0,  90, color, thick)
            cv2.ellipse(frame, (cx,cy), (r,r), 270, 0,  90, color, thick)


# ---------------------------------------------------------------------------
# Preview box  —  bottom-right corner
# ---------------------------------------------------------------------------

class PreviewBox:
    """
    Shows reference images for the current gesture.
    Dynamically adjusts pages based on available preview files.
    Navigation arrows are enabled/disabled based on available files.

    If real images don't exist yet, draws a styled placeholder.
    Supports dragging to reposition on screen.
    """

    W, H    = 220, 200  # Larger preview box for better learning support
    MAX_PAGES = 3  # Maximum number of preview pages supported

    def __init__(self):
        self._page  = 0
        self._mouse = (0, 0)
        self._pos = None  # Custom position (x, y) - when None, uses default position
        self._dragging = False
        self._drag_offset = (0, 0)
        self._available_pages = 1  # Will be updated dynamically
        self._current_gesture = None

    def reset(self):
        self._page = 0

    def _count_available_pages(self, gesture_name: str) -> int:
        """
        Count available preview files for a given gesture/letter.
        
        Checks for both naming conventions:
        - view_1.png, view_2.png, view_3.png
        - {LETTER}_view_1.png, {LETTER}_view_2.png, {LETTER}_view_3.png
        
        Also checks ASL_Gestures directory for gesture signs.
        
        Returns:
            Number of available preview pages (1 to MAX_PAGES)
        """
        count = 0
        for page_num in range(1, self.MAX_PAGES + 1):
            # Check both naming conventions for letters
            path1 = f"ASL_Alphabet/{gesture_name}/view_{page_num}.png"
            path2 = f"ASL_Alphabet/{gesture_name}/{gesture_name}_view_{page_num}.png"
            
            # Check ASL_Gestures directory for gestures
            path3 = f"ASL_Gestures/{gesture_name}/view_{page_num}.png"
            path4 = f"ASL_Gestures/{gesture_name}/{gesture_name}_view_{page_num}.png"
            
            if Path(path1).exists() or Path(path2).exists() or Path(path3).exists() or Path(path4).exists():
                count += 1
            else:
                # Stop counting if a page is missing (no gaps allowed)
                break
        
        return max(1, count)  # At least 1 page (placeholder)

    def set_position(self, x, y):
        """Set custom position for the preview box."""
        self._pos = (x, y)

    def get_position(self, default_x=8, default_y=56):
        """
        Get current position (custom or default).
        
        Default position is top-left corner (x=8, y=56) for levels containing
        sign information. This provides better visibility than bottom-right.
        """
        # Only return custom position if it's been explicitly set and is valid
        if self._pos is not None and self._pos[0] is not None:
            # Validate the position is reasonable (positive coordinates)
            if self._pos[0] >= 0 and self._pos[1] >= 0:
                return self._pos
        # Otherwise return default position (top-left)
        return (default_x, default_y)

    def handle_event(self, event_type, data=None, bx=0, by=0):
        """bx, by = top-left corner of the preview box on screen."""
        rx, ry = self.W, self.H      # local right/bottom
        
        # Only process navigation if multiple pages are available
        if self._available_pages > 1:
            if event_type == "key":
                if data == ord('.') or data == 0x270000:   # right arrow
                    self._page = (self._page + 1) % self._available_pages
                elif data == ord(',') or data == 0x280000:
                    self._page = (self._page - 1) % self._available_pages
            elif event_type == "mouse_click":
                mx, my = data[0], data[1]
                # Calculate local coordinates relative to box position
                local_x = mx - bx
                local_y = my - by
                
                # Check left arrow zone FIRST (priority over drag)
                # Left arrow zone: left 26 pixels, vertically centered
                if 0 <= local_x <= 26 and ry//2 - 16 <= local_y <= ry//2 + 16:
                    self._page = (self._page - 1) % self._available_pages
                    return  # Arrow click takes priority, don't start drag
                
                # Check right arrow zone (priority over drag)
                # Right arrow zone: right 26 pixels, vertically centered
                elif rx - 26 <= local_x <= rx and ry//2 - 16 <= local_y <= ry//2 + 16:
                    self._page = (self._page + 1) % self._available_pages
                    return  # Arrow click takes priority, don't start drag
                
                # Only check for drag if click is in the main body area (not in arrow zones)
                # Main body area: from x=26 to x=rx-26
                elif 26 <= local_x <= rx - 26 and 0 <= local_y <= ry:
                    self._dragging = True
                    self._drag_offset = (mx - bx, my - by)
                    self._pos = (bx, by)  # Start tracking custom position
        
        # Handle drag events regardless of page count
        if event_type == "mouse_release":
            self._dragging = False
        elif event_type == "mouse_move" and self._dragging and data:
            # Update position based on mouse movement
            mx, my = data[0], data[1]
            # New box position = mouse position - offset within box
            new_x = mx - self._drag_offset[0]
            new_y = my - self._drag_offset[1]
            # Clamp position to reasonable bounds
            new_x = max(-PreviewBox.W + 20, min(new_x, 2000))
            new_y = max(-PreviewBox.H + 20, min(new_y, 2000))
            self._pos = (new_x, new_y)

    def render(self, gesture_name: str) -> np.ndarray:
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        img[:] = PANEL_BG

        # Update available pages count when gesture changes
        if self._current_gesture != gesture_name:
            self._current_gesture = gesture_name
            self._available_pages = self._count_available_pages(gesture_name)
            # Reset page if current page is out of bounds
            if self._page >= self._available_pages:
                self._page = 0

        # Try to load a real image (check ASL_Alphabet/<LETTER>/ and ASL_Gestures/<GESTURE>/)
        loaded = False
        # Try both naming conventions: view_1.png and {LETTER}_view_1.png
        img_path = f"ASL_Alphabet/{gesture_name}/view_{self._page + 1}.png"
        if not Path(img_path).exists():
            img_path = f"ASL_Alphabet/{gesture_name}/{gesture_name}_view_{self._page + 1}.png"
        if not Path(img_path).exists():
            img_path = f"ASL_Gestures/{gesture_name}/view_{self._page + 1}.png"
        if not Path(img_path).exists():
            img_path = f"ASL_Gestures/{gesture_name}/{gesture_name}_view_{self._page + 1}.png"
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

        # Page indicator dots (only show for available pages)
        if self._available_pages > 1:
            dot_y = self.H - 10
            for i in range(self._available_pages):
                cx = self.W // 2 + (i - self._available_pages // 2) * 14
                color = ACCENT if i == self._page else DIM
                cv2.circle(img, (cx, dot_y), 4, color, -1)

        # Left / right arrow buttons (only draw if multiple pages available)
        if self._available_pages > 1:
            mid_y = (self.H - 20) // 2 + 10
            for ax, ch in [(10, "<"), (self.W - 10, ">")]:
                cv2.putText(img, ch, (ax - 5, mid_y + 5), FONT, 0.5, DIM, 1, cv2.LINE_AA)

        # Border
        cv2.rectangle(img, (0, 0), (self.W - 1, self.H - 1), ACCENT, 1)

        return img

    def _draw_placeholder(self, img: np.ndarray, gesture_name: str):
        """Placeholder when no image is available."""
        H, W = img.shape[:2]
        # Grey inner area
        cv2.rectangle(img, (4, 18), (W - 4, H - 18), (40, 42, 50), -1)
        # Large gesture name (abbreviated if needed)
        display = gesture_name[:8] if len(gesture_name) > 8 else gesture_name
        scale = 2.0
        (tw, th), _ = cv2.getTextSize(display, FONT, scale, 4)
        tx = (W - tw) // 2
        ty = (H - 20 + th) // 2
        cv2.putText(img, display, (tx, ty), FONT, scale, ACCENT, 4, cv2.LINE_AA)
        # "no image" label
        cv2.putText(img, "preview", (W//2 - 28, H - 22), FONT, 0.35, DIM, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Stage tracker
# ---------------------------------------------------------------------------

class StageTracker:
    """
    Owns progression through the gesture/letter stages.

    States per stage
    ---------------
    WAITING     — no correct sign detected yet
    HOLDING     — correct sign is being held (progress 0→1)
    CONFIRMING  — hold complete, "Next" button / SPACE to advance
    COMPLETE    — all signs done
    """

    def __init__(self, stages):
        self._stages       = stages
        self._idx           = 0
        self._state         = "WAITING"
        self._hold_start    = None
        self._confirm_time  = None
        self._dynamic_det   = None  # Dynamic detector for DYNAMIC signs
        self._skipped       = set()  # Track skipped stages
        self._load_detector()

    # -- public read --------------------------------------------------------

    @property
    def current_stage(self):
        if self._idx < len(self._stages):
            return self._stages[self._idx]
        return None

    @property
    def progress(self) -> float:
        if self._state == "HOLDING" and self._hold_start:
            return min(1.0, (time.time() - self._hold_start) / HOLD_SECONDS)
        if self._state == "CONFIRMING":
            return 1.0
        return 0.0

    @property
    def state(self) -> str:
        return self._state

    @property
    def stage_num(self) -> int:
        return self._idx + 1

    @property
    def total_stages(self) -> int:
        return len(self._stages)

    @property
    def skipped_stages(self) -> set:
        """Return set of skipped stage indices."""
        return self._skipped

    @property
    def is_complete(self) -> bool:
        return self._state == "COMPLETE"

    @property
    def dynamic_hint(self) -> str:
        """Return dynamic detector stage label if available."""
        if self._dynamic_det:
            return self._dynamic_det.stage_label
        return ""

    def _load_detector(self):
        """Load dynamic detector for current stage if needed."""
        stage = self.current_stage
        if stage and hasattr(stage, 'sign_type') and stage.sign_type == "DYNAMIC":
            if hasattr(stage, 'letter'):
                # Letter-based dynamic sign - use DynamicSignFactory for trained/hardcoded support
                from signsense.signs.dynamic_sign_factory import DynamicSignFactory
                self._dynamic_det = DynamicSignFactory.create_detector(stage.letter)
                if self._dynamic_det:
                    logger.debug(f"Loaded dynamic detector for {stage.letter}")
                else:
                    logger.warning(f"No dynamic detector available for {stage.letter}")
        else:
            if self._dynamic_det:
                logger.debug("Clearing dynamic detector")
            self._dynamic_det = None

    # -- main update --------------------------------------------------------

    def update(self, detection_result: Optional[DetectionResult] = None,
               landmarks=None, handedness=None) -> bool:
        """
        Call every frame with detection result.
        
        Args:
            detection_result: Either DetectionResult (gesture) or dict with "letter" key
            landmarks: Hand landmarks for dynamic detection
            handedness: Hand handedness for dynamic detection
            
        Returns True the moment the current sign is confirmed (advance ready).
        """
        try:
            # track state transitions for logging
            prev_state = self._state

            stage = self.current_stage
            if stage is None:
                self._state = "COMPLETE"
                if prev_state != self._state:
                    logger.debug(f"StageTracker state change {prev_state} -> {self._state}")
                return False

            # Handle ZSpecialStage - special educational level with finish button
            if hasattr(stage, 'sign_type') and stage.sign_type == "SPECIAL":
                # ZSpecialStage is always ready for advancement - user clicks Finish button
                if self._state != "CONFIRMING":
                    self._state = "CONFIRMING"
                    self._confirm_time = time.time()
                    logger.info(f"ZSpecialStage entered CONFIRMING - showing Finish button")
                return False  # Don't auto-advance - wait for user to click Finish

            # Handle dynamic signs
            if hasattr(stage, 'sign_type') and stage.sign_type == "DYNAMIC":
                if self._dynamic_det is None:
                    logger.warning(f"Dynamic detector unavailable for '{stage.name}'")
                    return False
                if self._state == "CONFIRMING":
                    return False  # waiting for user to press Next
                done = self._dynamic_det.update(landmarks, handedness)
                if done:
                    self._state        = "CONFIRMING"
                    self._confirm_time = time.time()
                    logger.info(f"Dynamic sign '{stage.name}' detected, entering CONFIRMING")
                    if prev_state != self._state:
                        logger.debug(f"StageTracker state change {prev_state} -> {self._state}")
                    return True
                return False

            # Handle static signs (both letter and gesture)
            detected_name = None
            detected_conf = 0.0
            
            if detection_result:
                if hasattr(detection_result, 'sign_name'):
                    # Gesture detection format
                    detected_name = detection_result.sign_name
                    detected_conf = detection_result.confidence
                elif isinstance(detection_result, dict) and "letter" in detection_result:
                    # Letter detection format
                    detected_name = detection_result["letter"]
                    detected_conf = detection_result.get("confidence", 0.0)

            # Check if detected sign matches current target
            correct = (detected_name == stage.name)

            if self._state == "WAITING":
                if correct:
                    self._state      = "HOLDING"
                    self._hold_start = time.time()
                    logger.info(f"Sign '{stage.name}' detected, starting hold")

            elif self._state == "HOLDING":
                if not correct:
                    # Lost the sign — restart hold
                    self._state      = "WAITING"
                    self._hold_start = None
                elif self.progress >= 1.0:
                    self._state        = "CONFIRMING"
                    self._confirm_time = time.time()
                    logger.info(f"Sign '{stage.name}' confirmed!")
                    return True

            elif self._state == "CONFIRMING":
                pass  # waiting for user to press Next

            # log any state change that occurred during this update
            if self._state != prev_state:
                logger.debug(f"StageTracker state change {prev_state} -> {self._state} (sign={stage.name if stage else None})")
        except Exception as e:
            logger.error(f"StageTracker.update error: {e}")
            import traceback
            traceback.print_exc()
            # Return False to indicate no confirmation - don't crash the app
        
        return False

    def skip(self):
        """Skip the current letter entirely and move to the next one.
        
        This skips at the letter/sign level, not at individual stages within
        a dynamic sign. For dynamic signs like J (which has 3 internal stages),
        pressing skip once moves to the next letter (e.g., K).
        """
        if self._idx < len(self._stages):
            stage = self._stages[self._idx]
            logger.info(f"Skipping letter '{stage.name}' entirely, moving to next")
        
        self._idx += 1
        
        if self._idx >= len(self._stages):
            self._state = "COMPLETE"
            logger.info(f"All letters complete")
        else:
            logger.info(f"Advancing to next letter ({self.stage_num}/{self.total_stages})")
            self._state = "WAITING"
            self._hold_start = None
            self._confirm_time = None
            self._load_detector()

    def back(self):
        """Go back to the previous stage."""
        if self._idx > 0:
            # Remove from skipped if it was skipped
            prev_idx = self._idx - 1
            if prev_idx in self._skipped:
                self._skipped.remove(prev_idx)
            self._idx -= 1
            logger.info(f"Going back to previous sign ({self.stage_num}/{self.total_stages})")
            self._state = "WAITING"
            self._hold_start = None
            self._confirm_time = None
            self._load_detector()

    def advance(self):
        """Move to the next sign. Called when user presses SPACE / Next."""
        self._idx  += 1
        logger.info(f"Advancing to next sign ({self.stage_num}/{self.total_stages})")
        self._state = "WAITING"
        self._hold_start   = None
        self._confirm_time = None
        self._load_detector()


# ---------------------------------------------------------------------------
# Play Mode renderer
# ---------------------------------------------------------------------------

class PlayModeRenderer:
    """
    Composites the full play-mode frame from a camera feed + overlay data.

    Call render() every frame after updating stage_tracker and gesture detector.
    """

    def __init__(self, W=640, H=480, mode="gesture"):
        self.W, self.H   = W, H
        self.mode = mode  # Store mode for completion message
        self.preview_box = PreviewBox()
        self._finish_button_bounds = None  # Bounds for Z special stage finish button

    def _word_wrap_text(self, text: str, max_width: int, font, font_scale: float, thickness: int) -> list:
        """
        Word wrap text to fit within a given width.
        
        Args:
            text: Text to wrap
            max_width: Maximum width in pixels
            font: OpenCV font
            font_scale: Font scale
            thickness: Line thickness
            
        Returns:
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [""]


    def get_preview_box_origin(self, frame_w=None, frame_h=None) -> tuple:
        """
        Get position for the preview box.
        
        If PreviewBox has a custom position set (dragged by user), use that.
        Otherwise, use default position (top-left corner at x=8, y=56).
        Pass the live frame dimensions from render() so the box stays
        glued to the position after a window resize.
        """
        w = frame_w if frame_w is not None else self.W
        h = frame_h if frame_h is not None else self.H
        
        # Check if user has set a custom position (via dragging)
        if self.preview_box._pos is not None and self.preview_box._pos[0] is not None:
            # Use custom position (but clamp to frame bounds)
            bx, by = self.preview_box._pos
        else:
            # Default position: top-left corner (x=8, y=56)
            bx = 8
            by = 56
        
        # Clamp to valid bounds - ensure we can actually display the box
        max_x = max(0, w - PreviewBox.W)
        max_y = max(0, h - PreviewBox.H)
        bx = min(bx, max_x)
        by = min(by, max_y)
        bx = max(0, bx)
        by = max(0, by)
        
        return (bx, by)

    def handle_event(self, event_type, data=None, stage_tracker=None):
        """Process events (keyboard and mouse) for the renderer.
        
        Args:
            event_type: The type of event ('key', 'mouse_click', 'mouse_move', 'mouse_release')
            data: Event data - for mouse events, this is either a tuple (x, y) for mouse_move
                 or a tuple (x, y) for mouse_click
            stage_tracker: The StageTracker instance for handling skip/back/next actions
        """
        # Handle keyboard events (SPACE, S, B)
        if event_type == "key":
            # SPACE (32) - Advance to next sign when confirming
            if data == 32:  # SPACE
                if stage_tracker and stage_tracker.state == "CONFIRMING":
                    stage_tracker.advance()
                    self.preview_box.reset()
            # S key (83) - Skip current stage (works anytime)
            elif data == ord('s') or data == ord('S'):
                if stage_tracker and not stage_tracker.is_complete:
                    stage_tracker.skip()
                    self.preview_box.reset()
            # B key (66) - Go back to previous stage
            elif data == ord('b') or data == ord('B'):
                if stage_tracker and stage_tracker._idx > 0:
                    stage_tracker.back()
                    self.preview_box.reset()
        
        # Handle mouse click for Finish Game button (ZSpecialStage)
        if event_type == "mouse_click" and data and stage_tracker:
            mx, my = data[0], data[1]
            stage = stage_tracker.current_stage
            # Check if we're on ZSpecialStage and button bounds are available
            if hasattr(stage, 'sign_type') and stage.sign_type == "SPECIAL" and self._finish_button_bounds:
                bx1, by1, bx2, by2 = self._finish_button_bounds
                if bx1 <= mx <= bx2 and by1 <= my <= by2:
                    # User clicked the Finish Game button - advance to complete
                    logger.info("Finish Game button clicked!")
                    stage_tracker.advance()
                    self.preview_box.reset()
                    return
        
        # Handle preview box events
        bx, by = self.get_preview_box_origin()
        self.preview_box.handle_event(event_type, data, bx, by)

    def render(self,
               camera_frame: np.ndarray,
               stage_tracker: "StageTracker",
               detection_result: Optional[DetectionResult],
               fps: float,
               hand_detected: bool) -> np.ndarray:

        frame = camera_frame.copy()
        stage = stage_tracker.current_stage

        if stage_tracker.is_complete:
            return self._render_complete(frame)

        # Update animation phase for Z gesture guidance


        # ── Top HUD bar ────────────────────────────────────────────────────
        self._draw_hud(frame, stage_tracker, fps, hand_detected)

        # ── Sign panel (right side) ─────────────────────────────────────────
        # Skip drawing sign panel for ZSpecialStage (uses centered two-panel layout)
        stage = stage_tracker.current_stage
        if not (hasattr(stage, 'sign_type') and stage.sign_type == "SPECIAL"):
            self._draw_sign_panel(frame, stage_tracker, detection_result)

        # ── Full-width progress bar ─────────────────────────────────────────
        self._draw_progress_bar(frame, stage_tracker, detection_result)


        # ── Preview box ────────────────────────────────────────────────────
        # Pass live frame dims so the box stays in position after window resize
        fH, fW = frame.shape[:2]
        bx, by = self.get_preview_box_origin(fW, fH)
        
        # Ensure preview box stays within frame bounds
        bx = max(0, min(bx, fW - PreviewBox.W))
        by = max(0, min(by, fH - PreviewBox.H))
        
        # Show preview box for regular letter/gesture stages (they have sign info)
        # For ZSpecialStage, draw centered two-panel layout instead
        stage = stage_tracker.current_stage
        if hasattr(stage, 'sign_type') and stage.sign_type == "SPECIAL":
            # ZSpecialStage: draw centered sign panel and preview panel
            self._draw_z_special_stage_centered(frame, stage_tracker)
        else:
            # Regular stages: show preview box
            preview = self.preview_box.render(
                stage.name if stage else "?")
            frame[by:by + PreviewBox.H, bx:bx + PreviewBox.W] = preview

        # ── Confirming flash ────────────────────────────────────────────────
        # Skip confirm flash for ZSpecialStage (obstructs panel text)
        if stage_tracker.state == "CONFIRMING":
            if not (hasattr(stage, 'sign_type') and stage.sign_type == "SPECIAL"):
                self._draw_confirm_flash(frame, stage)

        # ── Control buttons ─────────────────────────────────────────────────────
        self._draw_control_buttons(frame, stage_tracker)

        return frame

    # -----------------------------------------------------------------------
    # Sub-renderers
    # -----------------------------------------------------------------------

    def _draw_hud(self, frame, tracker, fps, hand_detected):
        H, W = frame.shape[:2]
        # Dark bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 48), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Stage counter
        stage_txt = f"Stage {tracker.stage_num} / {tracker.total_stages}"
        cv2.putText(frame, stage_txt, (12, 32), FONT, 0.65, ACCENT, 1, cv2.LINE_AA)

        # FPS
        fps_txt = f"FPS {fps:.0f}"
        (fw, _), _ = cv2.getTextSize(fps_txt, FONT, 0.45, 1)
        cv2.putText(frame, fps_txt, (W - fw - 8, 20), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        
        # Fullscreen hint
        fs_hint = "Press F for fullscreen"
        (fh_w, _), _ = cv2.getTextSize(fs_hint, FONT, 0.35, 1)
        cv2.putText(frame, fs_hint, (W - fh_w - 8, 36), FONT, 0.35, DIM, 1, cv2.LINE_AA)

        # Hand status dot
        dot_color = GREEN if hand_detected else RED_COL
        cv2.circle(frame, (W - 14, 34), 6, dot_color, -1)

        # Stage completion dots
        total = tracker.total_stages
        cur = tracker.stage_num
        skipped = tracker.skipped_stages
        for i in range(total):
            # Determine color: red if skipped, accent if completed, dim if pending
            if i in skipped:
                color = RED_COL  # Red for skipped stages
            elif i < cur - 1:
                color = ACCENT  # Accent for completed stages
            else:
                color = DIM    # Dim for pending stages
            cx = 12 + i * 12
            cy = 44
            cv2.circle(frame, (cx, cy), 4, color, -1)

    def _draw_sign_panel(self, frame, tracker, detection_result):
        H, W = frame.shape[:2]
        # Thinner panel for better visual balance with larger preview box
        pw, ph = 160, 180  # Reduced width from 185 to 160
        px = W - pw - 8
        py = 56

        overlay = frame.copy()
        _rr(overlay, px, py, px + pw, py + ph, PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (60, 60, 70), 1)

        stage = tracker.current_stage
        if not stage:
            return
        
        # Handle ZSpecialStage - special educational level with instructional text and button
        if hasattr(stage, 'sign_type') and stage.sign_type == "SPECIAL":
            self._draw_z_special_panel(frame, px, py, pw, ph, tracker)
            return
        
        # Check if current stage is a dynamic sign (define early for use throughout)
        is_dynamic = hasattr(stage, 'sign_entry') and stage.sign_entry.sign_type.name == "DYNAMIC" if hasattr(stage, 'sign_entry') else False
        
        # Target gesture name (large)
        # Use smaller font for dynamic signs (J, Z) and gestures (complete words) to avoid oversized display
        if is_dynamic:
            tl_scale = 1.0
            tl_thickness = 2
        elif hasattr(stage, 'gesture'):
            # Gesture mode - use smaller font for complete words
            tl_scale = 0.9
            tl_thickness = 2
        else:
            # Letter mode - use larger font for single letters
            tl_scale = 1.6
            tl_thickness = 4
        (tlw, tlh), _ = cv2.getTextSize(stage.name, FONT, tl_scale, tl_thickness)
        tlx = px + (pw - tlw) // 2
        tly = py + 43  # Top padding of 8px added
        color = GREEN if tracker.state == "CONFIRMING" else ACCENT
        cv2.putText(frame, stage.name, (tlx, tly), FONT, tl_scale, color, tl_thickness, cv2.LINE_AA)

        # Movement hint (smaller font)
        mvmt_y = py + 56
        mvmt_text = f"Move: {stage.movement}"
        (mw, _), _ = cv2.getTextSize(mvmt_text, FONT, 0.32, 1)
        cv2.putText(frame, mvmt_text, (px + 8, mvmt_y), FONT, 0.32, DIM, 1, cv2.LINE_AA)
        
        # Dynamic sign stage indicator (for J, Z, etc.)
        if is_dynamic and hasattr(stage, 'sign_entry'):
            # Show current stage progress for dynamic signs
            stage_y = py + 68
            # Get stage info from dynamic detector if available
            cv2.putText(frame, f"Stage progress...", (px + 8, stage_y), FONT, 0.28, ACCENT2, 1, cv2.LINE_AA)
            


        # Description hint (for letters) - with word wrapping
        if hasattr(stage, 'description') and stage.description:
            desc_y = py + 78  # Increased from 68 to 78 for more spacing
            # Word wrap description to fit panel width
            desc_lines = self._word_wrap_text(stage.description, pw - 16, FONT, 0.28, 1)
            for i, line in enumerate(desc_lines):
                # Only draw lines that fit within panel (leave space for state hint)
                if desc_y + i * 14 < py + ph - 30:  # Increased line spacing from 12 to 14
                    cv2.putText(frame, line, (px + 8, desc_y + i * 14), FONT, 0.28, DIM, 1, cv2.LINE_AA)

        # State hint - customize for dynamic signs vs static letters
        state = tracker.state
        hint_y = py + ph - 25  # Increased from 20 to 25 for more spacing
        
        if state == "WAITING":
            if is_dynamic:
                hint = "Perform the motion"
            else:
                hint = "Show the gesture above"
        elif state == "HOLDING":
            pct = int(tracker.progress * 100)
            hint = f"Keep going...  {pct}%"
        elif state == "CONFIRMING":
            hint = "SPACE -> Next gesture"
        else:
            hint = ""

        if hint:
            cv2.putText(frame, hint, (px + 6, hint_y),
                        FONT, 0.36, GOLD, 1, cv2.LINE_AA)
        

        # Detected gesture/letter (small, bottom right of panel)
        # Show for both gestures and dynamic signs
        if detection_result and detection_result.sign_name:
            det = detection_result.sign_name
            conf = detection_result.confidence
            det_col = GREEN if det == stage.name else DIM
            cv2.putText(frame, f"Seen: {det} {conf*100:.0f}%", 
                        (px + 6, py + ph - 10), FONT, 0.32, det_col, 1, cv2.LINE_AA)

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
        # Try both naming conventions: view_1.png and Z_view_1.png
        img_path = "ASL_Alphabet/Z/view_1.png"
        if not Path(img_path).exists():
            img_path = "ASL_Alphabet/Z/Z_view_1.png"
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

    def _draw_progress_bar(self, frame, tracker, detection_result):
        H, W = frame.shape[:2]
        bar_h = 8
        bar_y = H - 16

        # Background bar
        cv2.rectangle(frame, (0, bar_y), (W, bar_y + bar_h), (35, 38, 45), -1)

        # Progress fill
        if tracker.progress > 0:
            fill_w = int(W * tracker.progress)
            col = GREEN if tracker.state == "CONFIRMING" else ACCENT
            cv2.rectangle(frame, (0, bar_y), (fill_w, bar_y + bar_h), col, -1)

        # Border
        cv2.rectangle(frame, (0, bar_y), (W, bar_y + bar_h), (55, 60, 70), 1)

    def _draw_control_buttons(self, frame, stage_tracker):
        """Draw controls as text labels (no visual buttons) at the bottom of the screen."""
        H, W = frame.shape[:2]
        
        # Single non-bold text label showing available controls
        controls_text = "SPACE: Next | S: Skip | B: Back"
        
        # Calculate text position - centered at bottom
        (tw, th), _ = cv2.getTextSize(controls_text, FONT, 0.4, 1)
        tx = (W - tw) // 2
        ty = H - 30
        
        # Draw non-bold text label
        cv2.putText(frame, controls_text, (tx, ty), FONT, 0.4, DIM, 1, cv2.LINE_AA)

    def _draw_confirm_flash(self, frame, stage):
        """Draws a checkmark overlay when gesture is confirmed."""
        H, W = frame.shape[:2]
        cx, cy = W // 2, H // 2
        r = 40

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, (40, 120, 80), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Checkmark
        cv2.line(frame, (cx - 15, cy), (cx - 5, cy + 10), WHITE, 4, cv2.LINE_AA)
        cv2.line(frame, (cx - 5, cy + 10), (cx + 20, cy - 15), WHITE, 4, cv2.LINE_AA)

        # Text
        txt = f"{stage.name} Done!"
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.8, 2)
        cv2.putText(frame, txt, ((W - tw)//2, cy + r + 30), 
                    FONT, 0.8, GREEN, 2, cv2.LINE_AA)

    def _render_complete(self, frame):
        """Renders the completion screen."""
        H, W = frame.shape[:2]

        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title and subtitle based on mode
        if self.mode == "letter":
            title = "All Letters Complete!"
            subtitle = "You learned all 26 ASL letters!"
        else:  # gesture mode
            title = "All Gestures Complete!"
            subtitle = "You learned all 5 ASL gestures!"
        
        # Title
        _centered_text(frame, title, H // 2 - 40, 1.0, GOLD, 2)

        # Subtitle
        (sw, _), _ = cv2.getTextSize(subtitle, FONT, 0.5, 1)
        cv2.putText(frame, subtitle, ((W - sw)//2, H // 2 + 10), 
                    FONT, 0.5, WHITE, 1, cv2.LINE_AA)

        # Instruction
        inst = "Press SPACE to restart"
        (iw, _), _ = cv2.getTextSize(inst, FONT, 0.45, 1)
        cv2.putText(frame, inst, ((W - iw)//2, H // 2 + 50),
                    FONT, 0.45, DIM, 1, cv2.LINE_AA)

        return frame


# ---------------------------------------------------------------------------
# Play Mode - Main controller
# ---------------------------------------------------------------------------

class PlayMode:
    """
    Main play mode controller that ties together:
    - GestureDetector for detection (gesture mode)
    - ASLClassifierLetters for detection (letter mode)
    - StageTracker for stage progression
    - PlayModeRenderer for UI
    
    Supports two modes:
    - "gesture": 5 ASL gestures (HELLO, THANK YOU, etc.)
    - "letter": A-Z letters from sign_registry with dynamic support
    """

    def __init__(self, mode="gesture", hand_tracker=None, face_tracker=None, W=640, H=480):
        """
        Args:
            mode: "gesture" for 5 ASL gestures, "letter" for A-Z letters
            hand_tracker: HandTracker instance (optional, will create if None)
            face_tracker: FaceTracker instance (optional, will create if None)
            W: Window width
            H: Window height
        """
        self.mode = mode
        
        if mode == "gesture":
            # Use gesture-based stages
            from signsense.signs.gesture_definitions import GESTURE_REGISTRY, get_all_gestures
            stages = [GestureStage(g) for g in get_all_gestures().values()]
            self.gesture_detector = GestureDetector(
                hand_tracker=hand_tracker,
                face_tracker=face_tracker,
                confidence_threshold=0.7
            )
            self.classifier = None
        elif mode == "letter":
            # Use letter-based stages from ACTIVE_SIGNS (Z is excluded - enabled=False)
            from signsense.signs.sign_registry import ACTIVE_SIGNS
            stages = [LetterStage(s) for s in ACTIVE_SIGNS]
            # Add ZSpecialStage after all regular letters
            stages.append(ZSpecialStage())
            self.gesture_detector = None
            # Classifier will be initialized lazily or passed in
            self.classifier = None
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'gesture' or 'letter'.")
        
        # Initialize stage tracker with appropriate stages
        self.stage_tracker = StageTracker(stages)
        
        # Initialize renderer
        self.renderer = PlayModeRenderer(W, H, mode)
        
        # State
        self._hand_detected = False

    def update(self, frame: np.ndarray, hand_detected: bool = False,
               hand_data: dict = None, classifier_result: dict = None) -> np.ndarray:
        """
        Process a frame and return the rendered UI.
        
        Args:
            frame: Input frame in RGB format
            hand_detected: Whether a hand was detected in the frame
            hand_data: Hand tracking data (for letter mode)
            classifier_result: Classifier result (for letter mode)
            
        Returns:
            Rendered frame with UI overlays
        """
        self._hand_detected = hand_detected
        
        # Convert RGB to BGR for rendering (OpenCV expects BGR)
        # The input frame is in RGB (from main.py), but renderer uses OpenCV
        # which expects BGR color format
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        detection_result = None
        
        if self.mode == "gesture":
            # Gesture mode: use GestureDetector
            # Pass original RGB frame to detector (it expects RGB format)
            if self.gesture_detector:
                self.gesture_detector.update(frame)
                # Get detection result - pass target gesture for focused detection
                target_gesture = self.stage_tracker.current_stage.name if self.stage_tracker.current_stage else None
                detection_result = self.gesture_detector.detect(target_gesture)
        elif self.mode == "letter":
            # Letter mode: use classifier result passed from main loop
            if classifier_result and classifier_result.get('letter'):
                detection_result = type('DetectionResult', (), {
                    'sign_name': classifier_result['letter'],
                    'confidence': classifier_result.get('confidence', 0.7),
                    'is_gesture': False
                })()
        
        # Update stage tracker with detection result
        # For dynamic signs, pass landmarks and handedness
        landmarks = hand_data.get('landmarks', []) if hand_data else None
        handedness = hand_data.get('handedness', None) if hand_data else None
        self.stage_tracker.update(detection_result, landmarks=landmarks, handedness=handedness)
        
        # Render the frame using BGR format
        rendered = self.renderer.render(
            camera_frame=frame_bgr,
            stage_tracker=self.stage_tracker,
            detection_result=detection_result,
            fps=30.0,  # Would be calculated in real app
            hand_detected=hand_detected
        )
        
        return rendered

    def handle_event(self, event_type, data=None):
        """Handle input events."""
        self.renderer.handle_event(event_type, data, self.stage_tracker)

    def reset(self):
        """Reset the play mode to start over."""
        if self.mode == "gesture":
            from signsense.signs.gesture_definitions import get_all_gestures
            stages = [GestureStage(g) for g in get_all_gestures().values()]
        elif self.mode == "letter":
            from signsense.signs.sign_registry import ACTIVE_SIGNS
            stages = [LetterStage(s) for s in ACTIVE_SIGNS]
            # Add ZSpecialStage after all regular letters
            stages.append(ZSpecialStage())
        else:
            stages = GESTURE_STAGES
        
        self.stage_tracker = StageTracker(stages)
        self.renderer.preview_box.reset()