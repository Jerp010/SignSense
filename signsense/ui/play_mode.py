"""
ui/play_mode.py
===============
Play mode for SignSense - Gesture-based version.

Manages per-gesture stage progression using 9 ASL signs:
  HELLO, THANK YOU, NAME, GOOD, HELP, WATER, YES, NO, BAD

User must hold each gesture for HOLD_SECONDS to complete a stage.

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
3 placeholder images (or real images when assets/signs/<GESTURE>/ is populated).

Only the 9 defined gestures appear as stages.
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


# ---------------------------------------------------------------------------
# Gesture stage list - the 9 ASL gestures
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
    
    def __init__(self, sign_entry):
        self.sign_entry = sign_entry
        self.name = sign_entry.letter
        self.description = sign_entry.description
    
    @property
    def display_name(self) -> str:
        """Letter to display in UI."""
        return self.name
    
    @property
    def hand_shape(self) -> str:
        return "static" if self.sign_entry.sign_type.name == "STATIC" else "dynamic"
    
    @property
    def movement(self) -> str:
        return "hold" if self.sign_entry.sign_type.name == "STATIC" else "motion"


# Create ordered list of gesture stages
GESTURE_STAGES: List[GestureStage] = [
    GestureStage(GESTURE_REGISTRY["HELLO"]),
    GestureStage(GESTURE_REGISTRY["THANK YOU"]),
    GestureStage(GESTURE_REGISTRY["NAME"]),
    GestureStage(GESTURE_REGISTRY["GOOD"]),
    GestureStage(GESTURE_REGISTRY["HELP"]),
    GestureStage(GESTURE_REGISTRY["WATER"]),
    GestureStage(GESTURE_REGISTRY["YES"]),
    GestureStage(GESTURE_REGISTRY["NO"]),
    GestureStage(GESTURE_REGISTRY["BAD"]),
]


# ---------------------------------------------------------------------------
# Layout / palette constants
# ---------------------------------------------------------------------------
BG          = (15,  12, 20)
PANEL_BG    = (22, 18, 32)
ACCENT      = (0, 210, 255)
ACCENT2     = (180, 60, 255)
GREEN       = (80, 220, 120)
ORANGE      = (40, 165, 255)
RED_COL     = (60, 60, 200)
WHITE       = (240, 235, 250)
DIM         = (110, 100, 130)
GOLD        = (40, 200, 255)
FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO   = cv2.FONT_HERSHEY_PLAIN

HOLD_SECONDS    = 3.0   # seconds to hold a gesture
CONFIRM_FLASH   = 1.2   # seconds to show the "✓" flash


def _centered_text(frame, txt, cy, scale, color, thick=1):
    (w, h), _ = cv2.getTextSize(txt, FONT, scale, thick)
    W = frame.shape[1]
    cv2.putText(frame, txt, ((W - w)//2, cy), FONT, scale, color, thick, cv2.LINE_AA)


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
    3 pages cycled by clicking left/right arrows or pressing ← →.

    If real images don't exist yet, draws a styled placeholder.
    """

    W, H    = 160, 150
    PAGES   = 3

    def __init__(self):
        self._page  = 0
        self._mouse = (0, 0)

    def reset(self):
        self._page = 0

    def handle_event(self, event_type, data=None, bx=0, by=0):
        """bx, by = top-left corner of the preview box on screen."""
        rx, ry = self.W, self.H      # local right/bottom
        if event_type == "key":
            if data == ord('.') or data == 0x270000:   # right arrow
                self._page = (self._page + 1) % self.PAGES
            elif data == ord(',') or data == 0x280000:
                self._page = (self._page - 1) % self.PAGES
        elif event_type == "mouse_click":
            mx, my = data[0] - bx, data[1] - by
            # left arrow zone
            if 0 <= mx <= 26 and ry//2 - 16 <= my <= ry//2 + 16:
                self._page = (self._page - 1) % self.PAGES
            # right arrow zone
            elif rx - 26 <= mx <= rx and ry//2 - 16 <= my <= ry//2 + 16:
                self._page = (self._page + 1) % self.PAGES

    def render(self, gesture_name: str) -> np.ndarray:
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        img[:] = PANEL_BG

        # Try to load a real image (check assets/signs/<GESTURE>/)
        loaded = False
        # For now, we don't have gesture-specific images, so show placeholder
        # Path would be: f"assets/signs/{gesture_name}/view_{self._page + 1}.png"
        
        if not loaded:
            self._draw_placeholder(img, gesture_name)

        # Page indicator dots
        dot_y = self.H - 10
        for i in range(self.PAGES):
            cx = self.W // 2 + (i - self.PAGES // 2) * 14
            color = ACCENT if i == self._page else DIM
            cv2.circle(img, (cx, dot_y), 4, color, -1)

        # Left / right arrow buttons
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
        cv2.rectangle(img, (4, 18), (W - 4, H - 18), (35, 30, 48), -1)
        # Large gesture name (abbreviated if needed)
        display = gesture_name[:8] if len(gesture_name) > 8 else gesture_name
        scale = 2.0
        (tw, th), _ = cv2.getTextSize(display, FONT, scale, 4)
        tx = (W - tw) // 2
        ty = (H - 20 + th) // 2
        cv2.putText(img, display, (tx, ty), FONT, scale, (40, 180, 220), 4, cv2.LINE_AA)
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
    def is_complete(self) -> bool:
        return self._state == "COMPLETE"

    # -- main update --------------------------------------------------------

    def update(self, detection_result: Optional[DetectionResult]) -> bool:
        """
        Call every frame with gesture detection result.
        Returns True the moment the current gesture is confirmed (advance ready).
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

            # Get detected gesture name
            detected_name = detection_result.sign_name if detection_result else None
            detected_conf = detection_result.confidence if detection_result else 0.0

            # Check if detected gesture matches current target
            correct = (detected_name == stage.name)

            if self._state == "WAITING":
                if correct:
                    self._state      = "HOLDING"
                    self._hold_start = time.time()
                    logger.info(f"Gesture '{stage.name}' detected, starting hold")

            elif self._state == "HOLDING":
                if not correct:
                    # Lost the gesture — restart hold
                    self._state      = "WAITING"
                    self._hold_start = None
                elif self.progress >= 1.0:
                    self._state        = "CONFIRMING"
                    self._confirm_time = time.time()
                    logger.info(f"Gesture '{stage.name}' confirmed!")
                    return True

            elif self._state == "CONFIRMING":
                pass  # waiting for user to press Next

            # log any state change that occurred during this update
            if self._state != prev_state:
                logger.debug(f"StageTracker state change {prev_state} -> {self._state} (gesture={stage.name if stage else None})")
        except Exception as e:
            logger.error(f"StageTracker.update error: {e}")
            import traceback
            traceback.print_exc()
        
        return False

    def advance(self):
        """Move to the next gesture. Called when user presses SPACE / Next."""
        self._idx  += 1
        logger.info(f"Advancing to next gesture ({self.stage_num}/{self.total_stages})")
        self._state = "WAITING"
        self._hold_start   = None
        self._confirm_time = None


# ---------------------------------------------------------------------------
# Play Mode renderer
# ---------------------------------------------------------------------------

class PlayModeRenderer:
    """
    Composites the full play-mode frame from a camera feed + overlay data.

    Call render() every frame after updating stage_tracker and gesture detector.
    """

    def __init__(self, W=640, H=480):
        self.W, self.H   = W, H
        self.preview_box = PreviewBox()

    def get_preview_box_origin(self, frame_w=None, frame_h=None) -> tuple:
        """
        Always anchored to the ACTUAL frame bottom-right corner.
        Pass the live frame dimensions from render() so the box stays
        glued to the corner after a window resize.
        Falls back to stored W/H for mouse hit-testing between frames.
        """
        w = frame_w if frame_w is not None else self.W
        h = frame_h if frame_h is not None else self.H
        return (w - PreviewBox.W - 8, h - PreviewBox.H - 8)

    def handle_event(self, event_type, data=None, stage_tracker=None):
        bx, by = self.get_preview_box_origin()
        self.preview_box.handle_event(event_type, data, bx, by)
        if event_type == "key" and data == 32:  # SPACE
            if stage_tracker and stage_tracker.state == "CONFIRMING":
                stage_tracker.advance()
                self.preview_box.reset()

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

        # ── Top HUD bar ────────────────────────────────────────────────────
        self._draw_hud(frame, stage_tracker, fps, hand_detected)

        # ── Sign panel (right side) ─────────────────────────────────────────
        self._draw_sign_panel(frame, stage_tracker, detection_result)

        # ── Full-width progress bar ─────────────────────────────────────────
        self._draw_progress_bar(frame, stage_tracker, detection_result)


        # ── Preview box (bottom right) ──────────────────────────────────────
        # Pass live frame dims so the box is always in the true bottom-right
        # corner regardless of how the window has been resized.
        fH, fW = frame.shape[:2]
        bx, by = self.get_preview_box_origin(fW, fH)
        preview = self.preview_box.render(
            stage.name if stage else "?")
        frame[by:by + PreviewBox.H, bx:bx + PreviewBox.W] = preview

        # ── Confirming flash ────────────────────────────────────────────────
        if stage_tracker.state == "CONFIRMING":
            self._draw_confirm_flash(frame, stage)

        return frame

    # -----------------------------------------------------------------------
    # Sub-renderers
    # -----------------------------------------------------------------------

    def _draw_hud(self, frame, tracker, fps, hand_detected):
        H, W = frame.shape[:2]
        # Dark bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 48), (12, 10, 18), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Stage counter
        stage_txt = f"Stage {tracker.stage_num} / {tracker.total_stages}"
        cv2.putText(frame, stage_txt, (12, 32), FONT, 0.65, ACCENT, 1, cv2.LINE_AA)

        # FPS
        fps_txt = f"FPS {fps:.0f}"
        (fw, _), _ = cv2.getTextSize(fps_txt, FONT, 0.45, 1)
        cv2.putText(frame, fps_txt, (W - fw - 8, 20), FONT, 0.45, DIM, 1, cv2.LINE_AA)

        # Hand status dot
        dot_color = GREEN if hand_detected else RED_COL
        cv2.circle(frame, (W - 14, 34), 6, dot_color, -1)

        # Stage completion dots
        total = tracker.total_stages
        cur = tracker.stage_num
        for i in range(total):
            color = ACCENT if i < cur - 1 else DIM
            cx = 12 + i * 12
            cy = 44
            cv2.circle(frame, (cx, cy), 4, color, -1)

    def _draw_sign_panel(self, frame, tracker, detection_result):
        H, W = frame.shape[:2]
        pw, ph = 185, 200
        px = W - pw - 8
        py = 56

        overlay = frame.copy()
        _rr(overlay, px, py, px + pw, py + ph, (18, 14, 28), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (60, 55, 80), 1)

        stage = tracker.current_stage
        if not stage:
            return
        
        # Check if current stage is a dynamic sign (define early for use throughout)
        is_dynamic = hasattr(stage, 'sign_entry') and stage.sign_entry.sign_type.name == "DYNAMIC" if hasattr(stage, 'sign_entry') else False
        
        # Target gesture name (large)
        tl_scale = 1.6  # Smaller scale for better fit
        (tlw, tlh), _ = cv2.getTextSize(stage.name, FONT, tl_scale, 4)
        tlx = px + (pw - tlw) // 2
        tly = py + 45  # Moved up to avoid overlap
        color = GREEN if tracker.state == "CONFIRMING" else ACCENT
        cv2.putText(frame, stage.name, (tlx, tly), FONT, tl_scale, color, 4, cv2.LINE_AA)

        # Hand shape badge (smaller font)
        badge = f"Hand: {stage.hand_shape}"
        badge_col = ACCENT2
        cv2.putText(frame, badge, (px + 8, py + 16), FONT, 0.35, badge_col, 1, cv2.LINE_AA)

        # Movement hint (smaller font)
        mvmt_y = py + 32
        mvmt_text = f"Move: {stage.movement}"
        (mw, _), _ = cv2.getTextSize(mvmt_text, FONT, 0.32, 1)
        cv2.putText(frame, mvmt_text, (px + 8, mvmt_y), FONT, 0.32, DIM, 1, cv2.LINE_AA)
        
        # Dynamic sign stage indicator (for J, Z, etc.)
        if is_dynamic and hasattr(stage, 'sign_entry'):
            # Show current stage progress for dynamic signs
            stage_y = py + 56
            # Get stage info from dynamic detector if available
            cv2.putText(frame, f"Stage progress...", (px + 8, stage_y), FONT, 0.28, ACCENT2, 1, cv2.LINE_AA)

        # Description hint (for letters)
        if hasattr(stage, 'description') and stage.description:
            desc_y = py + 48
            desc_text = stage.description[:28] + "..." if len(stage.description) > 28 else stage.description
            cv2.putText(frame, desc_text, (px + 8, desc_y), FONT, 0.28, DIM, 1, cv2.LINE_AA)

        # State hint - customize for dynamic signs vs static letters
        state = tracker.state
        hint_y = py + ph - 28
        
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
            det_col = GREEN if det == stage.name else (180, 100, 80)
            cv2.putText(frame, f"Seen: {det} {conf*100:.0f}%", 
                        (px + 6, py + ph - 10), FONT, 0.32, det_col, 1, cv2.LINE_AA)

    def _draw_progress_bar(self, frame, tracker, detection_result):
        H, W = frame.shape[:2]
        bar_h = 8
        bar_y = H - 16

        # Background bar
        cv2.rectangle(frame, (0, bar_y), (W, bar_y + bar_h), (30, 25, 40), -1)

        # Progress fill
        if tracker.progress > 0:
            fill_w = int(W * tracker.progress)
            col = GREEN if tracker.state == "CONFIRMING" else ACCENT
            cv2.rectangle(frame, (0, bar_y), (fill_w, bar_y + bar_h), col, -1)

        # Border
        cv2.rectangle(frame, (0, bar_y), (W, bar_y + bar_h), (60, 50, 70), 1)

    def _draw_confirm_flash(self, frame, stage):
        """Draws a checkmark overlay when gesture is confirmed."""
        H, W = frame.shape[:2]
        cx, cy = W // 2, H // 2
        r = 40

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, (40, 180, 100), -1)
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
        cv2.rectangle(overlay, (0, 0), (W, H), (10, 8, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        _centered_text(frame, "All Gestures Complete!", H // 2 - 40, 1.0, GOLD, 2)

        # Subtitle
        sub = "You learned all 9 ASL gestures!"
        (sw, _), _ = cv2.getTextSize(sub, FONT, 0.5, 1)
        cv2.putText(frame, sub, ((W - sw)//2, H // 2 + 10), 
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
    - GestureDetector for detection
    - StageTracker for stage progression
    - PlayModeRenderer for UI
    """

    def __init__(self, hand_tracker=None, face_tracker=None, W=640, H=480):
        # Initialize gesture detector
        self.gesture_detector = GestureDetector(
            hand_tracker=hand_tracker,
            face_tracker=face_tracker,
            confidence_threshold=0.7
        )
        
        # Initialize stage tracker with gesture stages
        self.stage_tracker = StageTracker(GESTURE_STAGES)
        
        # Initialize renderer
        self.renderer = PlayModeRenderer(W, H)
        
        # State
        self._hand_detected = False

    def update(self, frame: np.ndarray, hand_detected: bool = False) -> np.ndarray:
        """
        Process a frame and return the rendered UI.
        
        Args:
            frame: Input frame in RGB format
            hand_detected: Whether a hand was detected in the frame
            
        Returns:
            Rendered frame with UI overlays
        """
        self._hand_detected = hand_detected
        
        # Update gesture detector with new frame
        self.gesture_detector.update(frame)
        
        # Get detection result - pass target gesture for focused detection
        target_gesture = self.stage_tracker.current_stage.name if self.stage_tracker.current_stage else None
        detection_result = self.gesture_detector.detect(target_gesture)
        
        # Update stage tracker
        self.stage_tracker.update(detection_result)
        
        # Render the frame
        rendered = self.renderer.render(
            camera_frame=frame,
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
        self.stage_tracker = StageTracker(GESTURE_STAGES)
        self.renderer.preview_box.reset()