"""Dynamic ASL sign definitions for multi-state gesture recognition."""

from typing import Dict, List, Any, Optional


class DynamicSignTracker:
    """Tracks progression through a series of sign stages.

    Each stage is governed by a check function that receives the
    current landmarks, classifier_result, and the tracker itself (for
    state storage). When a check returns True the tracker advances to
    the next stage.  Users can query the current stage index and
    description for UI feedback.
    """

    def __init__(self, name: str, stage_descriptions: List[str], check_funcs: List[Any]):
        self.name = name
        self.stage_descriptions = stage_descriptions
        self.check_funcs = check_funcs
        self.current = 0
        self._state_data: Dict[str, Any] = {}

    def update(self, landmarks: Optional[List[Any]], classifier_result: Optional[Dict]) -> int:
        """Call once per frame; returns the current stage index (0-based)."""
        if self.current < len(self.check_funcs):
            try:
                if self.check_funcs[self.current](landmarks, classifier_result, self):
                    self.current += 1
            except Exception:
                pass
        return self.current

    def is_complete(self) -> bool:
        """Returns True when all stages have been passed."""
        return self.current >= len(self.check_funcs)

    def reset(self) -> None:
        """Reset tracker to initial stage."""
        self.current = 0
        self._state_data.clear()

    def stage_description(self) -> str:
        if self.current < len(self.stage_descriptions):
            return self.stage_descriptions[self.current]
        return "(complete)"


# --- J sign implementation -----------------------------------------------------------

def _j_stage1(landmarks, classifier_result, tracker) -> bool:
    """Stage 1: confirm the I handshape is held."""
    return (
        classifier_result is not None and
        classifier_result.get("letter") == "I"
    )


def _j_stage2(landmarks, classifier_result, tracker) -> bool:
    """
    Stage 2: detect the J hook — pinky tip drops then rises.

    Tracks the raw pinky y position and looks for:
      1. A downward excursion of at least DOWN_THRESHOLD from the starting y.
      2. Followed by an upward recovery of at least UP_THRESHOLD from the peak.

    Uses tracker._state_data for persistence across frames.
    """
    if landmarks is None or len(landmarks) < 21:
        return False

    import math
    pinky_y = landmarks[20].y
    # Use hand scale for adaptive thresholds
    scale   = math.hypot(landmarks[0].x - landmarks[9].x,
                         landmarks[0].y - landmarks[9].y) or 0.15

    data = tracker._state_data

    # ── Initialise on first call into stage 2 ──────────────────────────────
    if "j2_start_y" not in data:
        data["j2_start_y"]   = pinky_y   # y when stage 2 first entered
        data["j2_peak_y"]    = pinky_y   # lowest y reached (highest on screen)
        data["j2_descended"] = False
        data["j2_frames"]    = 0

    data["j2_frames"] += 1

    # Timeout — give up after ~60 frames (~2 s at 30 fps)
    if data["j2_frames"] > 60:
        # Reset stage-2 state so a fresh attempt can start
        for k in list(data.keys()):
            if k.startswith("j2_"):
                del data[k]
        return False

    # Track the lowest point of the pinky (largest y = lowest on screen)
    if pinky_y > data["j2_peak_y"]:
        data["j2_peak_y"] = pinky_y

    down_travel = data["j2_peak_y"] - data["j2_start_y"]

    # Phase A: wait for sufficient downward travel
    if not data["j2_descended"]:
        if down_travel > 0.06 * scale:          # pinky moved down enough
            data["j2_descended"] = True
        return False

    # Phase B: after descent, watch for the hook upward
    up_travel = data["j2_peak_y"] - pinky_y    # positive = moving up from peak
    if up_travel > 0.04 * scale:               # hook confirmed
        return True

    return False


def create_j_tracker() -> DynamicSignTracker:
    """Factory returning a fresh tracker configured for the letter J."""
    descs  = [s["description"] for s in DYNAMIC_SIGNS["J"]["states"]]
    checks = [_j_stage1, _j_stage2]
    return DynamicSignTracker("J", descs, checks)


# -------------------------------------------------------------------------------------

DYNAMIC_SIGNS: Dict[str, Dict[str, Any]] = {
    "HELLO": {
        "description": "Wave greeting gesture",
        "states": [
            {"description": "Hand near forehead", "conditions": []},
            {"description": "Hand away from forehead", "conditions": []},
        ],
    },
    "THANK_YOU": {
        "description": "Hand moving from chin forward",
        "states": [
            {"description": "Fingertips touching chin", "conditions": []},
            {"description": "Hand extended forward", "conditions": []},
        ],
    },
    "PLEASE": {
        "description": "Circular motion with flat hand",
        "states": [
            {"description": "Flat hand at chest", "conditions": []},
            {"description": "Circular motion", "conditions": []},
        ],
    },
    "YES": {
        "description": "Fist moving up and down",
        "states": [
            {"description": "Fist at lower position", "conditions": []},
            {"description": "Fist at upper position", "conditions": []},
        ],
    },
    "NO": {
        "description": "Index and middle fingers snapping together",
        "states": [
            {"description": "Fingers extended apart", "conditions": []},
            {"description": "Fingers snapped together", "conditions": []},
        ],
    },
    "GOODBYE": {
        "description": "Open hand waving side to side",
        "states": [
            {"description": "Hand at center", "conditions": []},
            {"description": "Hand moved to side", "conditions": []},
        ],
    },
    "J": {
        "description": "Dynamic J motion starting from I handshape",
        "states": [
            {"description": "Hold I handshape (pinky up, others curled)"},
            {"description": "Trace J: pinky down then hook up"},
        ],
    },
}