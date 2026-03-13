"""
dynamic_signs.py
================
State-machine detectors for motion-based ASL signs.

Uses configuration from config/dynamic_signs.yaml for flexible
parameterization of detectors without modifying code.
"""

from typing import Optional, Dict
import math

# Import configuration system - adjust path for module resolution
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from signsense.config.dynamic_config import get_config, SignConfig


class DynamicDetector:
    @property
    def stage_label(self) -> str:
        return ""

    @property
    def phase_complete(self) -> bool:
        return False

    def update(self, landmarks, handedness: Optional[str]) -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class JDetector(DynamicDetector):
    """
    Detects ASL 'J' in three sequential phases using configuration from YAML.

    Phase 0 — HOLD I
        Hold the I handshape (pinky up, others curled) for configured frames.

    Phase 1 — HOOK DOWN
        Move the pinky tip downward by configured distance relative to scale.

    Phase 2 — FINISH: PALM AWAY + I HOLD
        Rotate so the back of the palm faces the camera, then
        re-hold the I shape for configured frames.
    """

    def __init__(self) -> None:
        self._config = get_config("J")
        if self._config is None:
            raise ValueError("Configuration for sign 'J' not found in dynamic_signs.yaml")
        
        self._phase_complete = False
        self.reset()

    @property
    def stage_label(self) -> str:
        """Get stage label from configuration or fallback to default."""
        return self._config.stages.get(self._phase, f"Stage {self._phase}")

    @property
    def phase_complete(self) -> bool:
        """True for exactly one frame when a phase transition fires."""
        return self._phase_complete

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._phase          = 0
        self._i_count        = 0
        self._phase_frames   = 0
        self._start_y        = 0.0
        self._peak_y         = 0.0
        self._finish_count   = 0
        self._phase_complete = False

    def update(self, landmarks, handedness: Optional[str]) -> bool:
        """Update detector state with current landmarks and handedness."""
        self._phase_complete = False   # clear every frame

        if landmarks is None or len(landmarks) < 21:
            if self._phase == 0:
                self.reset()
            return False

        lm    = self._maybe_mirror(landmarks, handedness)
        scale = math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) or 0.15
        py    = lm[20].y   # pinky tip y (increases downward)

        in_i      = self._is_i_shape(lm)
        palm_away = self._is_palm_away(lm)

        # ── Phase 0: hold I shape ─────────────────────────────────────────
        if self._phase == 0:
            self._i_count = self._i_count + 1 if in_i else 0
            if self._i_count >= self._config.detector.i_hold_frames:
                self._phase          = 1
                self._phase_frames   = 0
                self._start_y        = py
                self._peak_y         = py
                self._phase_complete = True   # ✓ Step 1 done
            return False

        # ── Phase 1: pinky moves DOWN ─────────────────────────────────────
        if self._phase == 1:
            self._phase_frames += 1
            if py > self._peak_y:
                self._peak_y = py

            if self._peak_y - self._start_y > self._config.detector.down_threshold * scale:
                self._phase          = 2
                self._phase_frames   = 0
                self._finish_count   = 0
                self._phase_complete = True   # ✓ Step 2 done
                return False

            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
            return False

        # ── Phase 2: palm away + re-hold I ───────────────────────────────
        if self._phase == 2:
            self._phase_frames += 1

            if in_i and palm_away:
                self._finish_count += 1
            else:
                # Forgive a few bad frames rather than hard-resetting
                self._finish_count = max(0, self._finish_count - 1)

            if self._finish_count >= self._config.detector.finish_hold_frames:
                self.reset()
                return True    # ✓ J complete

            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
            return False

        return False

    # -- helpers ---------------------------------------------------------------

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
        """Pinky up, index/middle/ring curled."""
        return (lm[20].y < lm[18].y and   # pinky tip above pinky PIP
                lm[8].y  > lm[6].y  and   # index tip below index PIP
                lm[12].y > lm[10].y and   # middle tip below middle PIP
                lm[16].y > lm[14].y)      # ring tip below ring PIP

    @staticmethod
    def _is_palm_away(lm) -> bool:
        """
        Proxy for back-of-hand facing camera.

        After mirroring, a palm-facing-camera hand has its thumb to the
        LEFT of the wrist. When the hand rotates palm-away, the thumb
        moves to the RIGHT of the wrist (after your-view flipping).
        """
        return lm[4].x > lm[0].x  # thumb to right of wrist


def get_detector(letter: str) -> Optional[DynamicDetector]:
    """
    Factory function to instantiate the appropriate dynamic detector
    for a given ASL letter.
    
    Args:
        letter: Single character ASL letter (e.g., 'J')
        
    Returns:
        DynamicDetector subclass instance, or None if no dynamic
        detector exists for this letter.
    """
    if letter.upper() == 'J':
        return JDetector()
    return None