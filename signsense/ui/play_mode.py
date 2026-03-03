"""
ui/play_mode.py
===============
Play mode for SignSense.

Manages per-sign stage progression:
  • Static signs  → user must hold the sign for HOLD_SECONDS
  • Dynamic signs → dedicated detector fires on completion

Layout
------
  ┌─────────────────────────────────────────┐
  │  [camera feed]                          │
  │                        ┌─────────────┐  │
  │                        │ SIGN PANEL  │  │
  │                        └─────────────┘  │
  │ ─── progress bar ─────────────────────  │
  │ ─── score bars (debug strip) ────────── │
  │                   ┌──────────────────┐  │
  │                   │  PREVIEW BOX     │  │
  │                   └──────────────────┘  │
  └─────────────────────────────────────────┘

The preview box sits in the bottom-right corner and cycles through up to
3 placeholder images (or real images when assets/signs/<LETTER>/ is populated).

Only letters in ACTIVE_SIGNS (sign_registry.py) appear as stages.
Currently stops at J.
"""

import cv2
import numpy as np
import math
import time
from typing import Optional, Dict, List, Any

from utils.logger import logger

from signs.sign_registry import ACTIVE_SIGNS, SignType, SignEntry
from signs.dynamic_signs  import get_detector


# ---------------------------------------------------------------------------
# Layout / palette constants
# ---------------------------------------------------------------------------
BG          = (15,  12,  20)
PANEL_BG    = (22,  18,  32)
ACCENT      = (0,  210, 255)
ACCENT2     = (180,  60, 255)
GREEN       = (80,  220, 120)
ORANGE      = (40,  165, 255)
RED_COL     = (60,   60, 200)
WHITE       = (240, 235, 250)
DIM         = (110, 100, 130)
GOLD        = (40,  200, 255)
FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO   = cv2.FONT_HERSHEY_PLAIN

HOLD_SECONDS    = 3.0   # seconds to hold a static sign
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
    Shows reference images for the current sign.
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

    def render(self, letter: str, sign_entry: Optional[SignEntry]) -> np.ndarray:
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        img[:] = PANEL_BG

        # Try to load a real image
        loaded = False
        if sign_entry and sign_entry.preview_dir:
            path = f"{sign_entry.preview_dir}/view_{self._page + 1}.png"
            raw = cv2.imread(path)
            if raw is not None:
                raw = cv2.resize(raw, (self.W - 2, self.H - 40))
                img[20:20 + raw.shape[0], 1:1 + raw.shape[1]] = raw
                loaded = True

        if not loaded:
            self._draw_placeholder(img, letter)

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

    def _draw_placeholder(self, img: np.ndarray, letter: str):
        """Placeholder when no image is available."""
        H, W = img.shape[:2]
        # Grey inner area
        cv2.rectangle(img, (4, 18), (W - 4, H - 18), (35, 30, 48), -1)
        # Large letter
        scale = 2.8
        (tw, th), _ = cv2.getTextSize(letter, FONT, scale, 4)
        tx = (W - tw) // 2
        ty = (H - 20 + th) // 2
        cv2.putText(img, letter, (tx, ty), FONT, scale, (40, 180, 220), 4, cv2.LINE_AA)
        # "no image" label
        cv2.putText(img, "preview", (W//2 - 28, H - 22), FONT, 0.35, DIM, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Stage tracker
# ---------------------------------------------------------------------------

class StageTracker:
    """
    Owns progression through the sign list.

    States per sign
    ---------------
    WAITING     — no correct sign detected yet
    HOLDING     — correct static sign is being held (progress 0→1)
    CONFIRMING  — hold complete, "Next" button / SPACE to advance
    COMPLETE    — all signs done
    """

    def __init__(self, signs: List[SignEntry]):
        self._signs         = signs
        self._idx           = 0
        self._state         = "WAITING"
        self._hold_start    = None
        self._confirm_time  = None
        self._dynamic_det   = None
        self._load_detector()

    # -- public read --------------------------------------------------------

    @property
    def current_sign(self) -> Optional[SignEntry]:
        if self._idx < len(self._signs):
            return self._signs[self._idx]
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
        return len(self._signs)

    @property
    def is_complete(self) -> bool:
        return self._state == "COMPLETE"

    @property
    def dynamic_hint(self) -> str:
        if self._dynamic_det:
            return self._dynamic_det.stage_label
        return ""

    # -- main update --------------------------------------------------------

    def update(self, classifier_result: Optional[Dict],
               landmarks=None, handedness=None) -> bool:
        """
        Call every frame.
        Returns True the moment the current sign is confirmed (advance ready).
        """
        # track state transitions for logging
        prev_state = self._state

        sign = self.current_sign
        if sign is None:
            self._state = "COMPLETE"
            if prev_state != self._state:
                logger.debug(f"StageTracker state change {prev_state} -> {self._state}")
            return False

        letter = classifier_result.get("letter") if classifier_result else None

        # ── DYNAMIC sign ────────────────────────────────────────────────────
        if sign.sign_type == SignType.DYNAMIC:
            if self._dynamic_det is None:
                return False
            if self._state == "CONFIRMING":
                return False   # waiting for user to press Next
            done = self._dynamic_det.update(landmarks, handedness)
            if done:
                self._state        = "CONFIRMING"
                self._confirm_time = time.time()
                logger.info(f"Dynamic sign '{sign.letter}' detected, entering CONFIRMING")
                if prev_state != self._state:
                    logger.debug(f"StageTracker state change {prev_state} -> {self._state}")
                return True
            return False

        # ── STATIC sign ─────────────────────────────────────────────────────
        correct = (letter == sign.letter)

        if self._state == "WAITING":
            if correct:
                self._state      = "HOLDING"
                self._hold_start = time.time()

        elif self._state == "HOLDING":
            if not correct:
                # Lost the sign — restart hold
                self._state      = "WAITING"
                self._hold_start = None
            elif self.progress >= 1.0:
                self._state        = "CONFIRMING"
                self._confirm_time = time.time()
                return True

        elif self._state == "CONFIRMING":
            pass  # waiting for user to press Next

        # log any state change that occurred during this update
        if self._state != prev_state:
            logger.debug(f"StageTracker state change {prev_state} -> {self._state} (sign={sign.letter if sign else None})")

        return False

    def advance(self):
        """Move to the next sign. Called when user presses SPACE / Next."""
        self._idx  += 1
        logger.info(f"Advancing to next sign ({self.stage_num}/{self.total_stages})")
        self._state = "WAITING"
        self._hold_start   = None
        self._confirm_time = None
        if self._dynamic_det:
            self._dynamic_det.reset()
        self._load_detector()

    def _load_detector(self):
        sign = self.current_sign
        if sign and sign.sign_type == SignType.DYNAMIC:
            self._dynamic_det = get_detector(sign.letter)
            logger.debug(f"Loaded dynamic detector for {sign.letter}")
        else:
            if self._dynamic_det:
                logger.debug("Clearing dynamic detector")
            self._dynamic_det = None


# ---------------------------------------------------------------------------
# Play Mode renderer
# ---------------------------------------------------------------------------

class PlayModeRenderer:
    """
    Composites the full play-mode frame from a camera feed + overlay data.

    Call render() every frame after updating stage_tracker and classifier.
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
               classifier_result: Optional[Dict],
               fps: float,
               hand_detected: bool) -> np.ndarray:

        frame = camera_frame.copy()
        sign  = stage_tracker.current_sign

        if stage_tracker.is_complete:
            return self._render_complete(frame)

        # ── Top HUD bar ────────────────────────────────────────────────────
        self._draw_hud(frame, stage_tracker, fps, hand_detected)

        # ── Sign panel (right side) ─────────────────────────────────────────
        self._draw_sign_panel(frame, stage_tracker, classifier_result)

        # ── Full-width progress bar ─────────────────────────────────────────
        self._draw_progress_bar(frame, stage_tracker)


        # ── Preview box (bottom right) ──────────────────────────────────────
        # Pass live frame dims so the box is always in the true bottom-right
        # corner regardless of how the window has been resized.
        fH, fW = frame.shape[:2]
        bx, by = self.get_preview_box_origin(fW, fH)
        preview = self.preview_box.render(
            sign.letter if sign else "?", sign)
        frame[by:by + PreviewBox.H, bx:bx + PreviewBox.W] = preview

        # ── Confirming flash ────────────────────────────────────────────────
        if stage_tracker.state == "CONFIRMING":
            self._draw_confirm_flash(frame, sign)

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

    def _draw_sign_panel(self, frame, tracker, classifier_result):
        H, W = frame.shape[:2]
        pw, ph = 185, 200
        px = W - pw - 8
        py = 56

        overlay = frame.copy()
        _rr(overlay, px, py, px + pw, py + ph, (18, 14, 28), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (60, 55, 80), 1)

        sign = tracker.current_sign
        if not sign:
            return

        # Target letter (large)
        tl_scale = 3.2
        (tlw, tlh), _ = cv2.getTextSize(sign.letter, FONT, tl_scale, 4)
        tlx = px + (pw - tlw) // 2
        tly = py + 80
        color = GREEN if tracker.state == "CONFIRMING" else ACCENT
        cv2.putText(frame, sign.letter, (tlx, tly), FONT, tl_scale, color, 4, cv2.LINE_AA)

        # Sign type badge
        badge = "DYNAMIC" if sign.sign_type == SignType.DYNAMIC else "STATIC"
        badge_col = ACCENT2 if sign.sign_type == SignType.DYNAMIC else (60, 180, 100)
        cv2.putText(frame, badge, (px + 8, py + 16), FONT, 0.35, badge_col, 1, cv2.LINE_AA)

        # Description
        desc_y = py + 105
        # Wrap if needed
        words = sign.description.split()
        line, lines = "", []
        for w in words:
            test = line + " " + w if line else w
            (tw, _), _ = cv2.getTextSize(test, FONT, 0.38, 1)
            if tw > pw - 16:
                lines.append(line); line = w
            else:
                line = test
        if line:
            lines.append(line)
        for li, l in enumerate(lines[:3]):
            cv2.putText(frame, l, (px + 8, desc_y + li * 18),
                        FONT, 0.38, DIM, 1, cv2.LINE_AA)

        # State hint
        state = tracker.state
        hint_y = py + ph - 28

        if sign.sign_type == SignType.DYNAMIC and state != "CONFIRMING":
            hint = tracker.dynamic_hint
        elif state == "WAITING":
            hint = "Show the sign above"
        elif state == "HOLDING":
            pct = int(tracker.progress * 100)
            hint = f"Hold...  {pct}%"
        elif state == "CONFIRMING":
            hint = "SPACE -> Next sign"
        else:
            hint = ""

        if hint:
            cv2.putText(frame, hint, (px + 6, hint_y),
                        FONT, 0.36, GOLD, 1, cv2.LINE_AA)

        # Detected letter (small, bottom right of panel)
        if classifier_result:
            det = classifier_result.get("letter", "")
            conf = classifier_result.get("confidence", 0)
            det_col = GREEN if det == sign.letter else (180, 100, 80)
            cv2.putText(frame, f"Seen: {det} {conf*100:.0f}%", 
                        (px + 6, py + ph - 10), FONT, 0.32, det_col, 1, cv2.LINE_AA)

    def _draw_progress_bar(self, frame, tracker):
        H, W = frame.shape[:2]
        bar_h = 12
        pad = 8
        # Use only 40% of the available width so the bar is less dominant
        full_w = W - pad * 2
        bar_w = max(40, int(full_w * 0.4))
        # Center the bar horizontally
        bar_x = (W - bar_w) // 2
        bar_y = H - bar_h - pad  # pinned to bottom of frame
        prog = tracker.progress

        # Background track (narrow centered bar)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 30, 48), -1)
        # Fill (clamped to the centered bar width)
        if prog > 0:
            fill = int(bar_w * prog)
            # guard against any out-of-range values
            fill = max(0, min(bar_w, fill))
            col = GREEN if prog >= 1.0 else ORANGE
            x_end = bar_x + fill
            x_end = min(x_end, bar_x + bar_w, W)
            cv2.rectangle(frame, (bar_x, bar_y), (x_end, bar_y + bar_h), col, -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 65, 90), 1)
        # Label centered above the bar (over the bar, not full window)
        label = "Hold progress"
        (lw, _), _ = cv2.getTextSize(label, FONT, 0.32, 1)
        label_x = bar_x + (bar_w - lw) // 2
        cv2.putText(frame, label, (label_x, bar_y - 4), FONT, 0.32, DIM, 1, cv2.LINE_AA)

        letters = list(scores.keys())
        n       = len(letters)
        if n == 0:
            return
        cell_w  = strip_w // n
        bar_mh  = 18

        for i, ltr in enumerate(letters):
            s   = scores.get(ltr, 0.0)
            bx  = strip_x + i * cell_w + 2
            by  = strip_y + strip_h - 4
            bh  = int(bar_mh * s)

            is_target = (ltr == target)
            col = GREEN if is_target and s >= 0.6 else (ACCENT if is_target else DIM)

            if bh > 0:
                cv2.rectangle(frame, (bx, by - bh), (bx + cell_w - 4, by), col, -1)

            lc = WHITE if is_target else DIM
            cv2.putText(frame, ltr, (bx + 1, strip_y + 12),
                        FONT, 0.32, lc, 1, cv2.LINE_AA)

    def _draw_confirm_flash(self, frame, sign):
        if not sign:
            return
        H, W = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 60, 30), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Big tick + letter
        msg = f"{sign.letter}  Correct!"
        (mw, mh), _ = cv2.getTextSize(msg, FONT, 1.4, 3)
        cv2.putText(frame, msg, ((W - mw) // 2, H // 2 - 10),
                    FONT, 1.4, GREEN, 3, cv2.LINE_AA)
        cv2.putText(frame, "Press SPACE to continue",
                    ((W - 260) // 2, H // 2 + 40),
                    FONT, 0.55, GOLD, 1, cv2.LINE_AA)

    def _render_complete(self, frame) -> np.ndarray:
        H, W = frame.shape[:2]
        overlay = frame.copy()
        overlay[:] = (10, 8, 16)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        _centered_text(frame, "Level Complete!", H // 2 - 40, 1.6, GREEN, 3)
        _centered_text(frame, "All signs recognised", H // 2 + 20, 0.7, ACCENT, 1)
        _centered_text(frame, "Press ESC to return to menu", H // 2 + 60, 0.5, DIM, 1)
        return frame