"""
ui/menu.py
=========
OpenCV-drawn menus for SignSense.

States
------
  MAIN_MENU       → show Play / About / Quit
  LEVEL_SELECT    → show available levels 
  RECORD_MENU     → show recording options
  ABOUT_MENU      → show about information

All menus are rendered purely with cv2 — no camera required.
Each render() call returns the frame to display.
Mouse clicks and key presses are handled by handle_event().
"""

import cv2
import numpy as np
import math
import time
from typing import Optional, Tuple

from utils.logger import logger


# ---------------------------------------------------------------------------
# Palette  — dark tech / arcade aesthetic
# ---------------------------------------------------------------------------
BG          = (20, 20, 25)     # dark charcoal (neutral, not purple-tinted)
PANEL       = (35, 38, 45)     # slightly lighter panel background
ACCENT      = (70, 130, 220)   # professional blue
ACCENT2     = (100, 100, 120)  # muted gray-blue
TEXT_WHITE  = (245, 245, 250)  # off-white
TEXT_DIM    = (130, 135, 145)  # muted gray text
TEXT_WARN   = (100, 150, 200)  # light blue accent
GREEN       = (80, 180, 120)   # muted teal-green
RED         = (120, 90, 100)   # muted rose
GOLD        = (180, 160, 100)  # muted gold

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO   = cv2.FONT_HERSHEY_PLAIN


def _fill(frame, color):
    frame[:] = color


def _rect(frame, x1, y1, x2, y2, color, thick=-1, radius=8):
    """Rounded rectangle via corner circles + rects."""
    if thick == -1:  # filled
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(frame, (cx, cy), radius, color, -1)
    else:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y1), color, thick)
        cv2.rectangle(frame, (x1 + radius, y2), (x2 - radius, y2), color, thick)
        cv2.rectangle(frame, (x1, y1 + radius), (x1, y2 - radius), color, thick)
        cv2.rectangle(frame, (x2, y1 + radius), (x2, y2 - radius), color, thick)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 0, 0, color, thick)
            cv2.ellipse(frame, (cx, cy), (radius, radius), 90, 0, 0, color, thick)
            cv2.ellipse(frame, (cx, cy), (radius, radius), 180, 0, 0, color, thick)
            cv2.ellipse(frame, (cx, cy), (radius, radius), 270, 0, 0, color, thick)


def _text(frame, txt, x, y, scale, color, thick=1, font=None):
    f = font or FONT
    cv2.putText(frame, txt, (x, y), f, scale, color, thick, cv2.LINE_AA)


def _text_centered(frame, txt, cy, scale, color, thick=1, font=None):
    f = font or FONT
    (w, _), _ = cv2.getTextSize(txt, f, scale, thick)
    W = frame.shape[1]
    _text(frame, txt, (W - w) // 2, cy, scale, color, thick, font=f)


def _scanlines(frame, alpha=0.0):
    """Disabled - scanlines removed for modern clean look."""
    pass


def _grid_bg(frame):
    """Subtle grid for depth - modern clean version."""
    h, w = frame.shape[:2]
    color = (30, 30, 40)
    step = 60
    for x in range(0, w, step):
        cv2.line(frame, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(frame, (0, y), (w, y), color, 1)


def _glow_text(frame, txt, x, y, scale, color, thick=2):
    """Text with a subtle professional shadow."""
    shadow = tuple(min(255, c + 60) for c in color)
    cv2.putText(frame, txt, (x+2, y+2), FONT, scale, shadow, thick, cv2.LINE_AA)
    cv2.putText(frame, txt, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Button helper
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, x1, y1, x2, y2, label, value,
                 color=ACCENT, disabled=False):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.label   = label
        self.value   = value
        self.color   = color
        self.disabled = disabled
        self._hover  = False

    def contains(self, mx, my) -> bool:
        return self.x1 <= mx <= self.x2 and self.y1 <= my <= self.y2

    def set_hover(self, mx, my):
        self._hover = self.contains(mx, my) and not self.disabled

    def draw(self, frame):
        if self.disabled:
            bg = (40, 36, 52)
            tc = TEXT_DIM
        elif self._hover:
            bg = tuple(min(255, c + 40) for c in self.color)
            tc = BG
        else:
            bg = self.color
            tc = BG if self._hover else TEXT_WHITE

        _rect(frame, self.x1, self.y1, self.x2, self.y2, bg, -1)
        _rect(frame, self.x1, self.y1, self.x2, self.y2,
              tuple(min(255, c + 60) for c in bg), 1)

        # Label
        fs = 0.65
        (tw, th), _ = cv2.getTextSize(self.label, FONT, fs, 1)
        tx = self.x1 + (self.x2 - self.x1 - tw) // 2
        ty = self.y1 + (self.y2 - self.y1 + th) // 2
        cv2.putText(frame, self.label, (tx, ty), FONT, fs,
                    BG if not self.disabled else TEXT_DIM, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main Menu
# ---------------------------------------------------------------------------

class MainMenu:
    """
    Renders the main menu.

    Returns
    -------
    handle_event(event) → str | None
        "play"  — user clicked Play
        "about" — user clicked About
        "quit"  — user clicked Quit / pressed ESC
    """

    def __init__(self, W=640, H=480):
        self.W, self.H = W, H
        self._t0     = time.time()
        self._mouse  = (0, 0)

        bw, bh = 220, 52
        cx = W // 2
        self._buttons = [
            Button(cx - bw//2, 200, cx + bw//2, 200+bh, "PLAY",   "play",  ACCENT),
            Button(cx - bw//2, 268, cx + bw//2, 268+bh, "ABOUT",  "about", (60, 140, 200)),
            Button(cx - bw//2, 336, cx + bw//2, 336+bh, "QUIT",   "quit",  (80, 60, 160)),
        ]

    def handle_event(self, event_type: str, data=None) -> Optional[str]:
        if event_type == "mouse_move":
            self._mouse = data
            for b in self._buttons:
                b.set_hover(*data)
        elif event_type == "mouse_click":
            for b in self._buttons:
                if b.contains(*data) and not b.disabled:
                    logger.info(f"Main menu click: {b.value}")
                    return b.value
        elif event_type == "key":
            if data == 27:   # ESC
                logger.info("Main menu action: quit (ESC)")
                return "quit"
        return None

    def render(self) -> np.ndarray:
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        frame[:] = BG

        t = time.time() - self._t0

        # Clean top accent bar
        bar_x = int((math.sin(t * 0.5) * 0.5 + 0.5) * self.W)
        cv2.rectangle(frame, (0, 0), (bar_x, 4), ACCENT, -1)
        cv2.rectangle(frame, (bar_x, 0), (self.W, 4), ACCENT2, -1)

        # Logo area - clean text
        cv2.putText(frame, "SignSense",
                   (self.W // 2 - 100, 90), FONT, 1.4, TEXT_WHITE, 2, cv2.LINE_AA)
        _text_centered(frame, "ASL Learning System",
                       130, 0.5, TEXT_DIM, 1)

        # Subtle indicator
        r = int(25 + 3 * math.sin(t * 1.5))
        cv2.circle(frame, (self.W // 2, 155), r, (*ACCENT[:2], 40), 1)

        # Version tag
        _text(frame, "v1.0", 8, self.H - 12, 0.35, TEXT_DIM)

        # Buttons
        for b in self._buttons:
            b.draw(frame)

        return frame


# ---------------------------------------------------------------------------
# About Menu
# ---------------------------------------------------------------------------

class AboutMenu:
    """
    Renders the About menu with scrollable content.
    
    Displays information about the developer, SignSense,
    current scope, known limitations, and future developments.
    
    Returns
    -------
    handle_event(event) → str | None
        "back" — user clicked Back / pressed ESC
    """
    
    def __init__(self, W=640, H=480):
        self.W, self.H = W, H
        self._t0     = time.time()
        self._mouse  = (0, 0)
        self._scroll_y = 0
        self._max_scroll = 0
        
        # Scroll interaction state
        self._scroll_dragging = False
        self._scroll_drag_start_y = 0
        self._scroll_drag_start_scroll = 0
        self._scroll_indicator_rect = None  # (x1, y1, x2, y2)
        self._scroll_track_rect = None  # (x1, y1, x2, y2)
        self._is_middle_mouse_drag = False  # Middle mouse drag state
        self._indicator_hover = False  # Hover state for visual feedback
        
        # Smooth scrolling
        self._target_scroll_y = 0
        self._scroll_velocity = 0
        self._last_wheel_time = 0
        self._wheel_accumulator = 0
        
        # Back button
        self._back = Button(20, self.H - 60, 130, self.H - 20,
                            "BACK", "back", (70, 60, 100))
        
        # Content sections
        self._content = [
            ("", 0.4, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("SignSense is an interactive American Sign Language (ASL)", 0.45, TEXT_DIM, False),
            ("learning application that uses computer vision and machine", 0.45, TEXT_DIM, False),
            ("learning to help users learn and practice ASL signs.", 0.45, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("DEVELOPER", 0.8, ACCENT, True),
            ("", 0.4, TEXT_DIM, False),
            ("Developed by Jeff", 0.45, TEXT_WHITE, False),
            ("A passionate developer focused on accessibility", 0.45, TEXT_DIM, False),
            ("and educational technology.", 0.45, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("CURRENT SCOPE", 0.8, ACCENT, True),
            ("", 0.4, TEXT_DIM, False),
            ("Level 1: ASL Alphabet (Letters A-Z)", 0.45, TEXT_WHITE, False),
            ("  - Static hand sign recognition using MediaPipe", 0.40, TEXT_DIM, False),
            ("  - Real-time classification with confidence scores", 0.40, TEXT_DIM, False),
            ("  - Stage-by-stage learning progression", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("Level 2: ASL Gestures (Work in progress)", 0.45, TEXT_WHITE, False),
            ("  - 5 essential ASL phrases", 0.40, TEXT_DIM, False),
            ("  - Motion tracking with temporal analysis", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("Data Collection & Training", 0.45, TEXT_WHITE, False),
            ("  - Record static landmarks for new signs", 0.40, TEXT_DIM, False),
            ("  - Record dynamic gestures (WIP) for motion-based signs", 0.40, TEXT_DIM, False),
            ("  - Train custom models on collected data", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("KNOWN LIMITATIONS", 0.8, GOLD, True),
            ("", 0.4, TEXT_DIM, False),
            ("Camera sensitivity to lighting conditions", 0.45, TEXT_WHITE, False),
            ("  - Performance varies with ambient lighting", 0.40, TEXT_DIM, False),
            ("  - Requires consistent hand visibility", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("Limited gesture vocabulary", 0.45, TEXT_WHITE, False),
            ("  - Currently supports A-Z letters and 5 phrases", 0.40, TEXT_DIM, False),
            ("  - Full ASL vocabulary not yet implemented", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("Single-user design", 0.45, TEXT_WHITE, False),
            ("  - No multi-user profiles or progress tracking", 0.40, TEXT_DIM, False),
            ("  - Limited personalization features", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("Platform limitations", 0.45, TEXT_WHITE, False),
            ("  - Desktop only (Windows/macOS/Linux)", 0.40, TEXT_DIM, False),
            ("  - No mobile or web version available", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("FUTURE DEVELOPMENTS", 0.8, GREEN, True),
            ("", 0.4, TEXT_DIM, False),
            ("Expanded ASL vocabulary", 0.45, TEXT_WHITE, False),
            ("  - Common words and phrases", 0.40, TEXT_DIM, False),
            ("  - Numbers and common expressions", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("Enhanced learning features", 0.45, TEXT_WHITE, False),
            ("  - Progress tracking and statistics", 0.40, TEXT_DIM, False),
            ("  - Personalized learning paths", 0.40, TEXT_DIM, False),
            ("  - Quiz and assessment modes", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("Improved recognition", 0.45, TEXT_WHITE, False),
            ("  - Better handling of lighting variations", 0.40, TEXT_DIM, False),
            ("  - Support for different hand sizes", 0.40, TEXT_DIM, False),
            ("  - Two-handed sign recognition", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("Accessibility improvements", 0.45, TEXT_WHITE, False),
            ("  - Audio feedback and instructions", 0.40, TEXT_DIM, False),
            ("  - Visual aids and tutorials", 0.40, TEXT_DIM, False),
            ("  - Multi-language support", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("Platform expansion", 0.45, TEXT_WHITE, False),
            ("  - Mobile applications (iOS/Android)", 0.40, TEXT_DIM, False),
            ("  - Web-based version", 0.40, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
            ("VERSION: 1.0", 0.5, TEXT_DIM, False),
            ("", 0.4, TEXT_DIM, False),
        ]
        
        # Calculate total content height
        self._line_height = 22
        self._total_height = len(self._content) * self._line_height
        self._visible_height = 0  # Calculated dynamically in render
        self._max_scroll = 0  # Calculated dynamically in render
    
    def handle_event(self, event_type: str, data=None) -> Optional[str]:
        if event_type == "mouse_move":
            self._mouse = data
            self._back.set_hover(*data)
            
            mx, my = data
            
            # Handle left scroll indicator dragging
            if self._scroll_dragging and self._scroll_indicator_rect:
                # Calculate new scroll position based on drag
                track_x1, track_y1, track_x2, track_y2 = self._scroll_track_rect
                track_height = track_y2 - track_y1
                indicator_height = self._scroll_indicator_rect[3] - self._scroll_indicator_rect[1]
                
                # Calculate scroll based on mouse position relative to track
                drag_offset = my - self._scroll_drag_start_y
                scroll_range = self._max_scroll
                track_range = track_height - indicator_height
                
                if track_range > 0:
                    scroll_delta = (drag_offset / track_range) * scroll_range
                    self._scroll_y = max(0, min(self._max_scroll,
                                               self._scroll_drag_start_scroll + scroll_delta))
                    self._target_scroll_y = self._scroll_y
            
            # Handle middle mouse drag scrolling
            elif self._is_middle_mouse_drag and self._scroll_track_rect:
                track_x1, track_y1, track_x2, track_y2 = self._scroll_track_rect
                track_height = track_y2 - track_y1
                
                drag_offset = my - self._scroll_drag_start_y
                scroll_range = self._max_scroll
                track_range = track_height
                
                if track_range > 0:
                    scroll_delta = (drag_offset / track_range) * scroll_range * 2
                    self._scroll_y = max(0, min(self._max_scroll,
                                               self._scroll_drag_start_scroll + scroll_delta))
                    self._target_scroll_y = self._scroll_y
            
            # Update scroll indicator hover state for visual feedback
            if self._scroll_indicator_rect and self._scroll_track_rect:
                ix1, iy1, ix2, iy2 = self._scroll_indicator_rect
                self._indicator_hover = ix1 <= mx <= ix2 and iy1 <= my <= iy2
            else:
                self._indicator_hover = False
                
        elif event_type == "mouse_click":
            mx, my = data
            
            # Check if click is on back button
            if self._back.contains(*data):
                logger.info("About menu: back")
                return "back"
            
            # Stop any active scrolling when clicking anywhere (allows clean re-selection)
            self._scroll_dragging = False
            self._is_middle_mouse_drag = False
            
            # Check if click is on scroll indicator
            if self._scroll_indicator_rect:
                ix1, iy1, ix2, iy2 = self._scroll_indicator_rect
                if ix1 <= mx <= ix2 and iy1 <= my <= iy2:
                    self._scroll_dragging = True
                    self._scroll_drag_start_y = my
                    self._scroll_drag_start_scroll = self._scroll_y
                    return None
            
            # Check if click is on scroll track (page up/down)
            if self._scroll_track_rect:
                tx1, ty1, tx2, ty2 = self._scroll_track_rect
                if tx1 <= mx <= tx2 and ty1 <= my <= ty2:
                    # Click above indicator = page up, below = page down
                    if self._scroll_indicator_rect:
                        ix1, iy1, ix2, iy2 = self._scroll_indicator_rect
                        if my < iy1:
                            # Page up
                            self._target_scroll_y = max(0, self._scroll_y - self._visible_height)
                        elif my > iy2:
                            # Page down
                            self._target_scroll_y = min(self._max_scroll,
                                                       self._scroll_y + self._visible_height)
            # Otherwise click is in content area - scrolling is already stopped above
                    
        elif event_type == "mouse_release":
            # Stop dragging
            self._scroll_dragging = False
            
        elif event_type == "mouse_middle_click":
            mx, my = data
            # Reset any previous scroll drag state
            self._scroll_dragging = False
            self._is_middle_mouse_drag = False
            
            # Middle click on scroll track for page up/down
            if self._scroll_track_rect:
                tx1, ty1, tx2, ty2 = self._scroll_track_rect
                if tx1 <= mx <= tx2 and ty1 <= my <= ty2:
                    if self._scroll_indicator_rect:
                        ix1, iy1, ix2, iy2 = self._scroll_indicator_rect
                        if my < iy1:
                            # Page up
                            self._target_scroll_y = max(0, self._scroll_y - self._visible_height)
                        elif my > iy2:
                            # Page down
                            self._target_scroll_y = min(self._max_scroll,
                                                       self._scroll_y + self._visible_height)
            # Start middle mouse drag
            self._scroll_dragging = True
            self._scroll_drag_start_y = my
            self._scroll_drag_start_scroll = self._scroll_y
            self._is_middle_mouse_drag = True
            
        elif event_type == "mouse_middle_release":
            # Stop middle mouse dragging
            self._scroll_dragging = False
            self._is_middle_mouse_drag = False
            
        elif event_type == "mouse_middle_drag":
            # Handle middle mouse drag scrolling
            mx, my = data
            if self._scroll_track_rect and self._is_middle_mouse_drag:
                track_x1, track_y1, track_x2, track_y2 = self._scroll_track_rect
                track_height = track_y2 - track_y1
                
                drag_offset = my - self._scroll_drag_start_y
                scroll_range = self._max_scroll
                track_range = track_height
                
                if track_range > 0:
                    scroll_delta = (drag_offset / track_range) * scroll_range * 2
                    self._scroll_y = max(0, min(self._max_scroll,
                                               self._scroll_drag_start_scroll + scroll_delta))
                    self._target_scroll_y = self._scroll_y
            
        elif event_type == "mouse_wheel":
            # data is (delta, x, y)
            delta = data[0] if isinstance(data, tuple) else data
            current_time = time.time()
            
            # Accumulate wheel events for smooth scrolling
            time_delta = current_time - self._last_wheel_time
            self._last_wheel_time = current_time
            
            # Reset accumulator if too much time passed
            if time_delta > 0.1:
                self._wheel_accumulator = 0
            
            # Accumulate and apply with momentum
            self._wheel_accumulator += delta
            scroll_amount = self._wheel_accumulator * 25
            
            # Apply scroll with smoothing
            self._target_scroll_y = max(0, min(self._max_scroll,
                                              self._scroll_y - scroll_amount))
            
            # Decay accumulator
            self._wheel_accumulator *= 0.7
            
        elif event_type == "key":
            if data == 27:  # ESC
                logger.info("About menu: back (ESC)")
                return "back"
            elif data == 82:  # Up arrow
                self._target_scroll_y = max(0, self._scroll_y - 40)
            elif data == 84:  # Down arrow
                self._target_scroll_y = min(self._max_scroll, self._scroll_y + 40)
            elif data == 81:  # Page Down
                self._target_scroll_y = min(self._max_scroll,
                                           self._scroll_y + self._visible_height)
            elif data == 73:  # Page Up
                self._target_scroll_y = max(0, self._scroll_y - self._visible_height)
            elif data == 80:  # Home
                self._target_scroll_y = 0
            elif data == 87:  # End
                self._target_scroll_y = self._max_scroll
                
        return None
    
    def render(self) -> np.ndarray:
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        frame[:] = BG
        _grid_bg(frame)
        
        t = time.time() - self._t0
        
        # Smooth scrolling animation
        scroll_diff = self._target_scroll_y - self._scroll_y
        if abs(scroll_diff) > 0.5:
            # Apply easing for smooth animation
            self._scroll_y += scroll_diff * 0.2
        else:
            self._scroll_y = self._target_scroll_y
        
        # Top bar
        cv2.rectangle(frame, (0, 0), (self.W, 50), PANEL, -1)
        _glow_text(frame, "ABOUT SIGNSENSE", 20, 33, 0.9, ACCENT, 2)
        
        # Content area bounds
        content_top = 55
        content_bottom = self.H - 65  # Leave room for back button
        content_height = content_bottom - content_top
        
        # Recalculate scroll bounds based on current window size
        self._visible_height = content_height
        self._max_scroll = max(0, self._total_height - self._visible_height)
        
        # Clamp scroll position
        self._scroll_y = max(0, min(self._max_scroll, self._scroll_y))
        self._target_scroll_y = max(0, min(self._max_scroll, self._target_scroll_y))
        
        # Scroll indicator and track
        if self._max_scroll > 0:
            # Track dimensions
            track_x1 = self.W - 16
            track_x2 = self.W - 4
            track_y1 = content_top
            track_y2 = content_bottom
            self._scroll_track_rect = (track_x1, track_y1, track_x2, track_y2)
            
            # Draw track background
            cv2.rectangle(frame, (track_x1, track_y1), (track_x2, track_y2),
                         (40, 42, 50), -1)
            
            # Indicator dimensions
            if self._max_scroll > 0:
                scroll_pct = self._scroll_y / self._max_scroll
            else:
                scroll_pct = 0.0
            indicator_height = max(40, int(content_height * (content_height / self._total_height)))
            indicator_y = int(track_y1 + scroll_pct * (content_height - indicator_height))
            indicator_x1 = track_x1 + 2
            indicator_x2 = track_x2 - 2
            
            self._scroll_indicator_rect = (indicator_x1, indicator_y, indicator_x2, indicator_y + indicator_height)
            
            # Check if mouse is hovering over indicator
            mx, my = self._mouse
            is_hovering = (indicator_x1 <= mx <= indicator_x2 and
                          indicator_y <= my <= indicator_y + indicator_height)
            
            # Draw indicator with hover effect
            if self._scroll_dragging:
                # Active dragging state - brightest
                indicator_color = tuple(min(255, c + 60) for c in ACCENT)
            elif is_hovering:
                # Hover state - lighter
                indicator_color = tuple(min(255, c + 40) for c in ACCENT)
            else:
                # Normal state
                indicator_color = ACCENT
            
            # Draw indicator with rounded corners effect
            cv2.rectangle(frame, (indicator_x1, indicator_y),
                         (indicator_x2, indicator_y + indicator_height),
                         indicator_color, -1)
            
            # Add subtle highlight on left edge for depth
            highlight_color = tuple(min(255, c + 30) for c in indicator_color)
            cv2.line(frame, (indicator_x1 + 1, indicator_y + 2),
                    (indicator_x1 + 1, indicator_y + indicator_height - 2),
                    highlight_color, 1)
            
            # Draw scroll position indicator lines
            if indicator_height > 50:
                center_y = indicator_y + indicator_height // 2
                line_width = 6
                line_color = tuple(max(0, c - 40) for c in indicator_color)
                for offset in [-4, 0, 4]:
                    cv2.line(frame,
                            (indicator_x1 + 3, center_y + offset),
                            (indicator_x1 + 3 + line_width, center_y + offset),
                            line_color, 1)
        else:
            self._scroll_track_rect = None
            self._scroll_indicator_rect = None
        
        # Render content with clipping
        for i, (text, scale, color, is_header) in enumerate(self._content):
            y_pos = content_top + i * self._line_height - int(self._scroll_y)
            
            # Skip if outside visible area (including negative positions)
            if y_pos < content_top - self._line_height or y_pos > content_bottom or y_pos < 0:
                continue
            
            if is_header:
                _glow_text(frame, text, 30, y_pos + 15, scale, color, 2)
            else:
                _text(frame, text, 30, y_pos + 15, scale, color)
        
        # Bottom fade effect
        for i in range(15):
            alpha = i / 15.0
            fade_color = tuple(int(c * (1 - alpha)) for c in BG)
            cv2.rectangle(frame, (0, content_bottom - 15 + i), (self.W, content_bottom - 14 + i), fade_color, -1)
        
        # Top fade effect (for content scrolling up)
        for i in range(10):
            alpha = i / 10.0
            fade_color = tuple(int(c * (1 - alpha)) for c in BG)
            cv2.rectangle(frame, (0, content_top + i), (self.W, content_top + i + 1), fade_color, -1)
        
        self._back.draw(frame)
        
        return frame


# ---------------------------------------------------------------------------
# Debug Menu
# ---------------------------------------------------------------------------

class DebugMenu:
    """
    Renders the debug/experimental options menu.

    Returns
    -------
    handle_event(event) → str | None
        "record"    — go to record menu
        "train"     — go to training menu
        "back"      — return to main menu
    """

    def __init__(self, W=640, H=480):
        self.W, self.H = W, H
        self._t0     = time.time()
        self._mouse  = (0, 0)

        bw, bh = 260, 52
        cx = W // 2
        self._buttons = [
            Button(cx - bw//2, 160, cx + bw//2, 160+bh, 
                   "RECORD DATA", "record", (0, 180, 200)),
            Button(cx - bw//2, 228, cx + bw//2, 228+bh, 
                   "TRAIN MODEL", "train", (180, 100, 255)),
        ]
        
        # Back button
        self._back = Button(20, self.H - 60, 130, self.H - 20,
                            "BACK", "back", (70, 60, 100))

    def handle_event(self, event_type: str, data=None) -> Optional[str]:
        if event_type == "mouse_move":
            self._mouse = data
            for b in self._buttons + [self._back]:
                b.set_hover(*data)
        elif event_type == "mouse_click":
            if self._back.contains(*data):
                logger.info("Debug menu: back")
                return "back"
            for b in self._buttons:
                if b.contains(*data) and not b.disabled:
                    logger.info(f"Debug menu: {b.value}")
                    return b.value
        elif event_type == "key":
            if data == 27:  # ESC
                logger.info("Debug menu: back (ESC)")
                return "back"
        return None

    def render(self) -> np.ndarray:
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        frame[:] = BG
        _grid_bg(frame)

        t = time.time() - self._t0

        # Top bar
        cv2.rectangle(frame, (0, 0), (self.W, 50), PANEL, -1)
        _glow_text(frame, "DEBUG / EXPERIMENTAL", 20, 33, 0.8, ACCENT2, 2)

        # Sub-label
        _text_centered(frame, "Data collection and model training", 90, 0.5, TEXT_DIM)

        # Description
        _text_centered(frame, "Record new training data or train existing models", 
                        120, 0.40, TEXT_DIM)

        # Buttons
        for b in self._buttons:
            b.draw(frame)
        
        self._back.draw(frame)

        return frame


# ---------------------------------------------------------------------------
# Recording Menu - Data Collection Options
# ---------------------------------------------------------------------------

class RecordMenu:
    """
    Renders the recording options menu for data collection.

    Returns
    -------
    handle_event(event) → str | None
        "record_static"  — record static landmarks (letters)
        "record_dynamic" — (WIP) disabled
        "back"          — return to main menu
    """

    def __init__(self, W=640, H=480):
        self.W, self.H = W, H
        self._t0     = time.time()
        self._mouse  = (0, 0)

        bw, bh = 320, 56
        cx = W // 2
        self._buttons = [
            Button(cx - bw//2, 140, cx + bw//2, 140+bh, 
                   "STATIC LANDMARKS", "record_static", ACCENT),
            Button(cx - bw//2, 220, cx + bw//2, 220+bh, 
                   "DYNAMIC GESTURES (WIP)", "record_dynamic", ACCENT2, disabled=True),
        ]
        # Back button
        self._back = Button(20, self.H - 60, 130, self.H - 20,
                            "BACK", "back", (70, 60, 100))

    def handle_event(self, event_type: str, data=None) -> Optional[str]:
        if event_type == "mouse_move":
            self._mouse = data
            for b in self._buttons + [self._back]:
                b.set_hover(*data)
        elif event_type == "mouse_click":
            if self._back.contains(*data):
                logger.info("Record menu: back")
                return "back"
            for b in self._buttons:
                if b.contains(*data) and not b.disabled:
                    logger.info(f"Record menu: {b.value}")
                    return b.value
        elif event_type == "key":
            if data == 27:  # ESC
                logger.info("Record menu: back (ESC)")
                return "back"
        return None

    def render(self) -> np.ndarray:
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        frame[:] = BG
        _grid_bg(frame)

        t = time.time() - self._t0

        # Top bar
        cv2.rectangle(frame, (0, 0), (self.W, 50), PANEL, -1)
        _glow_text(frame, "RECORD DATA", 20, 33, 0.9, ACCENT, 2)

        # Sub-label
        _text_centered(frame, "Collect training data for new signs", 90, 0.5, TEXT_DIM)

        # Description texts
        descriptions = [
            "Record static hand positions (letters A-Z, custom labels)",
            "Dynamic gestures - Work in progress"
        ]

        # Buttons
        for i, btn in enumerate(self._buttons):
            btn.draw(frame)
            # Description below button
            sub_y = btn.y2 + 18
            _text_centered(frame, descriptions[i], sub_y, 0.40, TEXT_DIM)

        self._back.draw(frame)

        return frame


# ---------------------------------------------------------------------------
# Level Select
# ---------------------------------------------------------------------------

LEVELS = [
    {"id": "letters", "label": "Level 1 - Letters A-Z",
     "sub": "Learn the ASL alphabet", "enabled": True},
    {"id": "gestures", "label": "Level 2 - ASL Gestures",
     "sub": "Work in progress", "enabled": False},
]


class LevelSelect:
    """
    Renders the level selection screen.

    handle_event "back" | level_id str | None
    """

    def __init__(self, W=640, H=480):
        self.W, self.H = W, H
        self._t0     = time.time()
        self._mouse  = (0, 0)

        bw, bh = 360, 64
        cx = W // 2
        self._buttons = []
        for i, lv in enumerate(LEVELS):
            y1 = 180 + i * 90
            color = ACCENT if lv["enabled"] else (50, 45, 70)
            self._buttons.append(
                Button(cx - bw//2, y1, cx + bw//2, y1 + bh,
                       lv["label"], lv["id"],
                       color=color, disabled=not lv["enabled"])
            )
        # Back button
        self._back = Button(20, self.H - 60, 130, self.H - 20,
                            "BACK", "back", (70, 60, 100))

    def handle_event(self, event_type, data=None) -> Optional[str]:
        if event_type == "mouse_move":
            self._mouse = data
            for b in self._buttons + [self._back]:
                b.set_hover(*data)
        elif event_type == "mouse_click":
            if self._back.contains(*data):
                logger.info("Level select: back")
                return "back"
            for b in self._buttons:
                if b.contains(*data) and not b.disabled:
                    logger.info(f"Level select choice: {b.value}")
                    return b.value
        elif event_type == "key":
            if data == 27:
                logger.info("Level select: back (ESC)")
                return "back"
        return None

    def render(self) -> np.ndarray:
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        frame[:] = BG
        _grid_bg(frame)

        t = time.time() - self._t0

        # Top bar
        cv2.rectangle(frame, (0, 0), (self.W, 50), PANEL, -1)
        _glow_text(frame, "SELECT LEVEL", 20, 33, 0.9, ACCENT, 2)

        # Sub-label
        _text_centered(frame, "Choose your challenge", 90, 0.55, TEXT_DIM)

        # Level buttons + sub-labels
        for i, (btn, lv) in enumerate(zip(self._buttons, LEVELS)):
            btn.draw(frame)
            # Sub-label below button
            sub_y = btn.y2 + 14
            sc = TEXT_DIM if not lv["enabled"] else TEXT_WARN
            _text_centered(frame, lv["sub"], sub_y, 0.40, sc)

        self._back.draw(frame)

        # Animated bottom accent
        prog = (math.sin(t) * 0.5 + 0.5)
        w2 = int(self.W * prog)
        cv2.line(frame, (0, self.H - 3), (w2, self.H - 3), ACCENT2, 2)

        return frame
