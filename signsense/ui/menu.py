"""
ui/menu.py
=========
OpenCV-drawn menus for SignSense.

States
------
  MAIN_MENU       → show Play / Debug / Quit
  LEVEL_SELECT    → show available levels 
  RECORD_MENU     → show recording options
  DEBUG_MENU      → show debug options

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
        "debug" — user clicked Debug
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
            Button(cx - bw//2, 268, cx + bw//2, 268+bh, "DEBUG",  "debug", (60, 140, 200)),
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
        _text(frame, "v0.8", 8, self.H - 12, 0.35, TEXT_DIM)

        # Buttons
        for b in self._buttons:
            b.draw(frame)

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
        "record_dynamic" — record dynamic signs (gestures)
        "back"          — return to debug menu
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
                   "DYNAMIC GESTURES", "record_dynamic", ACCENT2),
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
            "Record dynamic gestures with motion (words, phrases)"
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
     "sub": "5 essential ASL phrases", "enabled": True},
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
