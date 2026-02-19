"""Rule-based ASL letter classifier for static signs A-F."""

import math
from typing import Optional, List, Tuple

# Landmark indices (MediaPipe Hand)
WRIST = 0
THUMB_CMC, THUMB_IP, THUMB_TIP = 1, 2, 3
THUMB_TIP_IDX = 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Finger groups (tip, pip)
FINGERS = [
    (INDEX_TIP, INDEX_PIP, "index"),
    (MIDDLE_TIP, MIDDLE_PIP, "middle"),
    (RING_TIP, RING_PIP, "ring"),
    (PINKY_TIP, PINKY_PIP, "pinky"),
]

TOUCH_THRESHOLD = 0.08
CURLED_Y_THRESHOLD = 0.02
EXTENDED_Y_THRESHOLD = -0.02
C_ANGLE_MIN = 70
C_ANGLE_MAX = 140


def _dist(lm1, lm2: object) -> float:
    """Normalized Euclidean distance between two landmarks."""
    dx = lm2.x - lm1.x
    dy = lm2.y - lm1.y
    dz = getattr(lm2, "z", 0) - getattr(lm1, "z", 0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _pt(lm) -> Tuple[float, float]:
    """(x, y) from landmark."""
    return (lm.x, lm.y)


def is_finger_extended(landmarks: List, tip_idx: int, pip_idx: int) -> bool:
    """
    Finger extended = tip is above (smaller y) PIP in mirror/flipped view.
    In normalized coords, smaller y = higher on screen.
    """
    if tip_idx >= len(landmarks) or pip_idx >= len(landmarks):
        return False
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    return tip.y < pip.y - EXTENDED_Y_THRESHOLD


def is_finger_curled(landmarks: List, tip_idx: int, pip_idx: int) -> bool:
    """Finger curled = tip is at or below PIP."""
    if tip_idx >= len(landmarks) or pip_idx >= len(landmarks):
        return False
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    return tip.y >= pip.y - CURLED_Y_THRESHOLD


def is_thumb_touching(landmarks: List, target_idx: int) -> bool:
    """Thumb tip (4) touching target landmark (normalized distance)."""
    if len(landmarks) <= max(THUMB_TIP_IDX, target_idx):
        return False
    return _dist(landmarks[THUMB_TIP_IDX], landmarks[target_idx]) < TOUCH_THRESHOLD


def _angle_at(a_idx: int, b_idx: int, c_idx: int, landmarks: List) -> float:
    """Angle at landmark b between a-b and b-c."""
    if max(a_idx, b_idx, c_idx) >= len(landmarks):
        return 0.0
    from core.landmark_utils import calculate_angle
    a = _pt(landmarks[a_idx])
    b = _pt(landmarks[b_idx])
    c = _pt(landmarks[c_idx])
    return calculate_angle(a, b, c)


class ASLClassifier:
    """
    Rule-based classifier for ASL letters A-F.
    Uses landmark geometry only; no ML models.
    """

    def __init__(self) -> None:
        """Initialize classifier with default thresholds."""
        self.touch_threshold = TOUCH_THRESHOLD

    def classify(self, landmarks) -> Optional[str]:
        """
        Classify hand landmarks into ASL letter A-F.

        Args:
            landmarks: List of 21 hand landmarks (MediaPipe format)

        Returns:
            'A', 'B', 'C', 'D', 'E', 'F', or None
        """
        if landmarks is None or len(landmarks) < 21:
            return None

        # Letter A: All fingers curled, thumb beside index
        if self._is_a(landmarks):
            return "A"
        if self._is_b(landmarks):
            return "B"
        if self._is_c(landmarks):
            return "C"
        if self._is_d(landmarks):
            return "D"
        if self._is_e(landmarks):
            return "E"
        if self._is_f(landmarks):
            return "F"
        return None

    def _is_a(self, lm: List) -> bool:
        """A: Closed fist, thumb resting beside index."""
        index_curled = is_finger_curled(lm, INDEX_TIP, INDEX_PIP)
        middle_curled = is_finger_curled(lm, MIDDLE_TIP, MIDDLE_PIP)
        ring_curled = is_finger_curled(lm, RING_TIP, RING_PIP)
        pinky_curled = is_finger_curled(lm, PINKY_TIP, PINKY_PIP)
        thumb_out = lm[THUMB_TIP_IDX].x < lm[INDEX_MCP].x or abs(lm[THUMB_TIP_IDX].y - lm[INDEX_MCP].y) < 0.1
        return index_curled and middle_curled and ring_curled and pinky_curled and thumb_out

    def _is_b(self, lm: List) -> bool:
        """B: Four fingers extended, thumb folded across palm."""
        all_ext = (
            is_finger_extended(lm, INDEX_TIP, INDEX_PIP)
            and is_finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
            and is_finger_extended(lm, RING_TIP, RING_PIP)
            and is_finger_extended(lm, PINKY_TIP, PINKY_PIP)
        )
        thumb_folded = lm[THUMB_TIP_IDX].y > lm[INDEX_PIP].y
        return all_ext and thumb_folded

    def _is_c(self, lm: List) -> bool:
        """C: Fingers curved like letter C, thumb extended outward."""
        angles_curved = True
        for tip, pip, mcp in [
            (INDEX_TIP, INDEX_PIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP),
            (PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ]:
            a = _angle_at(tip, pip, mcp, lm)
            if not (C_ANGLE_MIN <= a <= C_ANGLE_MAX):
                angles_curved = False
                break
        thumb_out = is_finger_extended(lm, THUMB_TIP_IDX, THUMB_IP) or _dist(lm[THUMB_TIP_IDX], lm[WRIST]) > 0.15
        return angles_curved and thumb_out

    def _is_d(self, lm: List) -> bool:
        """D: Index extended, others curled, thumb touching middle."""
        index_ext = is_finger_extended(lm, INDEX_TIP, INDEX_PIP)
        middle_curled = is_finger_curled(lm, MIDDLE_TIP, MIDDLE_PIP)
        ring_curled = is_finger_curled(lm, RING_TIP, RING_PIP)
        pinky_curled = is_finger_curled(lm, PINKY_TIP, PINKY_PIP)
        thumb_touch_mid = is_thumb_touching(lm, MIDDLE_TIP) or is_thumb_touching(lm, MIDDLE_PIP)
        return index_ext and middle_curled and ring_curled and pinky_curled and thumb_touch_mid

    def _is_e(self, lm: List) -> bool:
        """E: All fingers curled tightly, thumb touching fingertips."""
        all_curled = (
            is_finger_curled(lm, INDEX_TIP, INDEX_PIP)
            and is_finger_curled(lm, MIDDLE_TIP, MIDDLE_PIP)
            and is_finger_curled(lm, RING_TIP, RING_PIP)
            and is_finger_curled(lm, PINKY_TIP, PINKY_PIP)
        )
        thumb_touch = (
            is_thumb_touching(lm, INDEX_TIP)
            or is_thumb_touching(lm, MIDDLE_TIP)
            or _dist(lm[THUMB_TIP_IDX], lm[INDEX_TIP]) < TOUCH_THRESHOLD * 1.5
        )
        return all_curled and thumb_touch

    def _is_f(self, lm: List) -> bool:
        """F: Thumb and index touching, middle/ring/pinky extended."""
        thumb_index_touch = is_thumb_touching(lm, INDEX_TIP)
        mid_ext = is_finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
        ring_ext = is_finger_extended(lm, RING_TIP, RING_PIP)
        pinky_ext = is_finger_extended(lm, PINKY_TIP, PINKY_PIP)
        return thumb_index_touch and mid_ext and ring_ext and pinky_ext
