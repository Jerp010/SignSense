"""
dynamic_signs.py
================
State-machine detectors for motion-based ASL signs.

Each sign that requires movement (J, Z, …) gets a detector class that
implements a simple update(landmarks, handedness) → bool interface.
When update() returns True the sign has been completed.

The play-mode loop calls the appropriate detector only when the current
stage letter is dynamic, so there is no interference from other letters.
"""

from typing import Optional, Dict, Any, List
import math


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DynamicDetector:
    """
    Minimal interface every dynamic sign detector must implement.

    update(landmarks, handedness) → bool
        Call once per frame.  Returns True exactly once when the sign is
        successfully completed, then auto-resets.

    reset()
        Force-reset all internal state (e.g. when play mode moves on).

    stage_label → str
        Human-readable description of what the detector is currently
        waiting for (used for on-screen hints).
    """

    @property
    def stage_label(self) -> str:
        return ""

    def update(self, landmarks, handedness: Optional[str]) -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# J  —  I handshape + pinky traces a J arc (down then hook up)
# ---------------------------------------------------------------------------

class JDetector(DynamicDetector):
    """
    Detects ASL 'J'.

    Phases
    ------
    0  WAIT_I   : Wait for the hand to hold the I handshape (pinky up, others
                  curled) for I_HOLD_FRAMES consecutive frames.
    1  DOWN     : Track the pinky tip moving downward by at least
                  DOWN_THRESHOLD * scale.
    2  HOOK_UP  : After the descent, the pinky tip must rise by at least
                  UP_THRESHOLD * scale to confirm the hook.

    On success returns True once, then resets automatically.
    On timeout (too many frames in phase 1 or 2) resets without firing.
    """

    I_HOLD_FRAMES  = 5     # frames of I shape before motion tracking starts
    DOWN_THRESHOLD = 0.07  # fraction of hand-scale the pinky must drop
    UP_THRESHOLD   = 0.04  # fraction of hand-scale the pinky must rise after drop
    PHASE_TIMEOUT  = 50    # max frames per motion phase

    _PHASE_LABELS = {
        0: "Hold  I  shape (pinky up)",
        1: "Move pinky DOWN",
        2: "Hook pinky back UP",
    }

    def __init__(self) -> None:
        self.reset()

    # -- public interface ---------------------------------------------------

    @property
    def stage_label(self) -> str:
        return self._PHASE_LABELS.get(self._phase, "")

    def reset(self) -> None:
        self._phase        = 0
        self._i_count      = 0
        self._phase_frames = 0
        self._start_y      = 0.0
        self._peak_y       = 0.0   # largest y seen (= lowest point on screen)

    def update(self, landmarks, handedness: Optional[str]) -> bool:
        if landmarks is None or len(landmarks) < 21:
            self.reset()
            return False

        lm    = self._maybe_mirror(landmarks, handedness)
        scale = math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) or 0.15
        py    = lm[20].y   # pinky tip y

        in_i = self._is_i_shape(lm)

        # ── phase 0 : wait for stable I hold ──────────────────────────────
        if self._phase == 0:
            self._i_count = self._i_count + 1 if in_i else 0
            if self._i_count >= self.I_HOLD_FRAMES:
                self._phase        = 1
                self._phase_frames = 0
                self._start_y      = py
                self._peak_y       = py
            return False

        # ── phase 1 : pinky moves DOWN ────────────────────────────────────
        if self._phase == 1:
            self._phase_frames += 1
            if py > self._peak_y:
                self._peak_y = py          # track lowest point

            if self._peak_y - self._start_y > self.DOWN_THRESHOLD * scale:
                self._phase        = 2
                self._phase_frames = 0
                return False

            if self._phase_frames > self.PHASE_TIMEOUT:
                self.reset()
            return False

        # ── phase 2 : pinky hooks back UP ─────────────────────────────────
        if self._phase == 2:
            self._phase_frames += 1
            up_travel = self._peak_y - py      # positive = rising

            if up_travel > self.UP_THRESHOLD * scale:
                self.reset()
                return True                    # ✓ J detected

            if self._phase_frames > self.PHASE_TIMEOUT:
                self.reset()
            return False

        return False

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _maybe_mirror(landmarks, handedness):
        if handedness != "Right":
            return landmarks
        mirrored = []
        for pt in landmarks:
            m   = type(pt)()
            m.x = 1.0 - pt.x
            m.y = pt.y
            m.z = getattr(pt, "z", 0)
            mirrored.append(m)
        return mirrored

    @staticmethod
    def _is_i_shape(lm) -> bool:
        """True when pinky is up and index/middle/ring are all curled."""
        pinky_up   = lm[20].y < lm[18].y   # tip above PIP
        index_down = lm[8].y  > lm[6].y    # tip below PIP
        mid_down   = lm[12].y > lm[10].y
        ring_down  = lm[16].y > lm[14].y
        return pinky_up and index_down and mid_down and ring_down


# ---------------------------------------------------------------------------
# Registry  —  map letter → detector instance
# ---------------------------------------------------------------------------
# Import this dict in play_mode to get the right detector for each letter.

DYNAMIC_DETECTORS: Dict[str, DynamicDetector] = {
    "J": JDetector(),
    # "Z": ZDetector(),   ← add here when implemented
}


def get_detector(letter: str) -> Optional[DynamicDetector]:
    """Return the detector for a dynamic sign, or None if not registered."""
    return DYNAMIC_DETECTORS.get(letter.upper())