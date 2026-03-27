"""
dynamic_signs.py
================
State-machine detectors for motion-based ASL signs.

Uses configuration from config/dynamic_signs.yaml for flexible
parameterization of detectors without modifying code.
"""

from typing import Optional, Dict, Tuple, List
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
    if letter.upper() == 'Z':
        return ZDetector()
    return None


class ZDetector(DynamicDetector):
    """
    Simplified waypoint-based detector for ASL 'Z'.
    
    Uses only 3 essential waypoints to detect the Z stroke pattern:
    
    1. START - Top left corner (initial position)
    2. MIDDLE - Bottom right of first diagonal stroke
    3. END - Bottom left corner (final position)
    
    The Z gesture pattern:
    - Start at top-left of the drawing area
    - Move diagonally down-right to bottom-right (first diagonal)
    - Move horizontally left to bottom-left (horizontal stroke)
    - The diagonal back to bottom-left is implicit in the endpoint
    
    This simplified detector bypasses the trained LSTM model and uses
    geometric rules for reliable, fast detection without ML overhead.
    """

    def __init__(self) -> None:
        self._config = get_config("Z")
        if self._config is None:
            raise ValueError("Configuration for sign 'Z' not found in dynamic_signs.yaml")
        
        # Phase tracking: 0=waiting for start, 1=tracking first diagonal,
        #                2=tracking horizontal, 3=detected
        self._phase = 0
        self._phase_complete = False
        
        # Waypoints
        self._start_point: Optional[Tuple[float, float]] = None
        self._middle_point: Optional[Tuple[float, float]] = None
        self._end_point: Optional[Tuple[float, float]] = None
        
        # Timing and validation
        self._frame_count = 0
        self._min_diagonal_frames = 3  # Minimum frames for diagonal stroke
        self._min_horizontal_frames = 2  # Minimum frames for horizontal stroke
        
        # Distance thresholds (normalized 0-1 coordinates)
        self._diagonal_threshold = 0.10  # Minimum diagonal movement
        self._horizontal_threshold = 0.08  # Minimum horizontal movement
        self._corner_tolerance = 0.15  # Tolerance for corner positions
        
        # Scale factor for distance calculations
        self._scale = 0.15

    @property
    def stage_label(self) -> str:
        """Get current stage label for display."""
        labels = {
            0: "Draw Z - Start at top left",
            1: "Draw diagonal down-right",
            2: "Draw horizontal to left",
            3: "Z detected!"
        }
        return labels.get(self._phase, f"Stage {self._phase}")

    @property
    def phase_complete(self) -> bool:
        """True for exactly one frame when Z is detected."""
        return self._phase_complete

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._phase = 0
        self._phase_complete = False
        self._start_point = None
        self._middle_point = None
        self._end_point = None
        self._frame_count = 0

    def update(self, landmarks, handedness: Optional[str]) -> bool:
        """
        Update detector state with current hand landmarks.
        
        Args:
            landmarks: List of 21 hand landmarks from MediaPipe
            handedness: "Left" or "Right" hand
        
        Returns:
            True if Z gesture is complete, False otherwise
        """
        self._phase_complete = False
        self._frame_count += 1
        
        if landmarks is None or len(landmarks) < 21:
            # No hand detected - reset if we haven't completed
            if self._phase < 3:
                self.reset()
            return False
        
        # Use index finger tip for drawing detection
        lm = self._maybe_mirror(landmarks, handedness)
        index_tip = (lm[8].x, lm[8].y)
        
        # Calculate scale from hand size
        self._scale = math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) or 0.15
        
        # Phase 0: Wait for start at top-left corner
        if self._phase == 0:
            # Check if finger is at top-left position (high y = top, low x = left)
            # In MediaPipe: y=0 is top, y=1 is bottom; x=0 is left, x=1 is right
            if index_tip[1] < 0.5 and index_tip[0] < 0.4:  # Top-left area
                self._start_point = index_tip
                self._phase = 1
                self._phase_complete = True  # Signal phase transition
            return False
        
        # Phase 1: Track diagonal down-right stroke
        if self._phase == 1:
            if self._start_point is None:
                self._phase = 0
                return False
            
            # Check if we've moved sufficiently down-right
            dx = index_tip[0] - self._start_point[0]
            dy = index_tip[1] - self._start_point[1]
            
            # Diagonal down-right: dx > 0 (right), dy > 0 (down)
            if dx > self._diagonal_threshold * 0.5 and dy > self._diagonal_threshold * 0.3:
                # Validate diagonal direction (roughly 45 degrees)
                if abs(dx - dy) < self._diagonal_threshold * 2:
                    self._middle_point = index_tip
                    self._phase = 2
                    self._phase_complete = True
            
            # Timeout - reset if taking too long
            if self._frame_count > 60:
                self.reset()
            return False
        
        # Phase 2: Track horizontal left stroke
        if self._phase == 2:
            if self._middle_point is None:
                self._phase = 1
                return False
            
            # Check if we've moved left enough from middle point
            dx = self._middle_point[0] - index_tip[0]  # Positive = moved left
            dy = index_tip[1] - self._middle_point[1]  # Check not moving much vertically
            
            # Horizontal left: significant x movement left, minimal y movement
            if dx > self._horizontal_threshold and abs(dy) < self._horizontal_threshold * 1.5:
                # Verify we're at bottom-left (y should be > 0.5 for bottom)
                if index_tip[1] > 0.5:
                    self._end_point = index_tip
                    self._phase = 3
                    self._phase_complete = True
                    return True  # Z gesture complete!
            
            # Timeout - reset if taking too long
            if self._frame_count > 100:
                self.reset()
            return False
        
        return False

    @staticmethod
    def _maybe_mirror(landmarks, handedness):
        """Mirror landmarks if needed for left-handed users."""
        if handedness != "Right":
            return landmarks
        mirrored = []
        for pt in landmarks:
            m = type(pt)()
            m.x = 1.0 - pt.x
            m.y = pt.y
            m.z = getattr(pt, "z", 0)
            mirrored.append(m)
        return mirrored

    def get_waypoints(self) -> List[Tuple[float, float]]:
        """
        Get the detected waypoints for visualization.

        Returns:
            List of (x, y) tuples representing the Z path waypoints
        """
        points = []
        if self._start_point:
            points.append(self._start_point)
        if self._middle_point:
            points.append(self._middle_point)
        if self._end_point:
            points.append(self._end_point)
        return points


# =================== Gesture Detector Classes ===================
# These detectors implement multi-stage motion patterns for ASL gestures
# Each gesture follows a pattern: Stage 0 (start) -> Stage 1 (motion) -> Stage 2 (finish)


class GestureDetector(DynamicDetector):
    """Base class for gesture detectors with common helper methods."""

    @staticmethod
    def _maybe_mirror(landmarks, handedness):
        """Mirror landmarks if needed for left-handed users."""
        if handedness != "Right":
            return landmarks
        mirrored = []
        for pt in landmarks:
            m = type(pt)()
            m.x = 1.0 - pt.x
            m.y = pt.y
            m.z = getattr(pt, "z", 0)
            mirrored.append(m)
        return mirrored


class HELLODetector(GestureDetector):
    """
    Detects ASL 'HELLO' gesture in three stages:

    Stage 0 — HOLD PREP: Hold hand in preparation position near face
    Stage 1 — TOUCH: Bring hand to forehead (touch gesture)
    Stage 2 — MOVE OUT: Move hand outward to complete hello
    """

    def __init__(self) -> None:
        self._config = get_config("HELLO")
        if self._config is None:
            raise ValueError("Configuration for gesture 'HELLO' not found")

        self._phase = 0
        self._phase_complete = False
        self._phase_frames = 0
        self._start_y = 0.0
        self._finish_count = 0
        self.reset()

    @property
    def stage_label(self) -> str:
        return self._config.stages.get(self._phase, f"Stage {self._phase}")

    @property
    def phase_complete(self) -> bool:
        return self._phase_complete

    def reset(self) -> None:
        self._phase = 0
        self._phase_complete = False
        self._phase_frames = 0
        self._start_y = 0.0
        self._finish_count = 0

    def update(self, landmarks, handedness: Optional[str]) -> bool:
        self._phase_complete = False

        if landmarks is None or len(landmarks) < 21:
            if self._phase == 0:
                self.reset()
            return False

        lm = self._maybe_mirror(landmarks, handedness)

        # Stage 0: Hold preparation position near face
        if self._phase == 0:
            self._phase_frames += 1
            # Check if hand is near face (forehead region)
            # Face is typically in upper portion of frame
            face_region = lm[5].y < 0.35  # Wrist in upper 35% of frame
            flat_hand = self._is_flat_hand(lm)

            if face_region and flat_hand:
                self._phase = 1
                self._phase_frames = 0
                self._start_y = lm[20].y
                self._phase_complete = True  # Ready for touch
                return False

        # Stage 1: Touch forehead
        if self._phase == 1:
            self._phase_frames += 1
            # Check if hand is touching or near forehead
            forehead_touch = lm[5].y < 0.3 and lm[20].y < 0.35
            flat_hand = self._is_flat_hand(lm)

            if forehead_touch and flat_hand:
                self._phase_frames = 0  # Reset timer while touching
            else:
                # Check if we've moved out (complete gesture)
                if lm[20].y - self._start_y > self._config.detector.move_out_distance:
                    self._phase = 2
                    self._finish_count = 0
                    self._phase_complete = True  # Ready for final hold
                    return False

            # Timeout
            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
                return False

        # Stage 2: Move hand outward
        if self._phase == 2:
            self._phase_frames += 1
            flat_hand = self._is_flat_hand(lm)

            if flat_hand:
                self._finish_count += 1
            else:
                self._finish_count = max(0, self._finish_count - 1)

            if self._finish_count >= self._config.detector.stage_2_hold_frames:
                self.reset()
                return True  # HELLO complete

            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
                return False

        return False

    @staticmethod
    def _is_flat_hand(lm) -> bool:
        """Check for flat open hand (all fingers extended)."""
        # Simple heuristic: fingers not curled
        return (lm[8].y > lm[6].y and      # index
                lm[12].y > lm[10].y and     # middle
                lm[16].y > lm[14].y and     # ring
                lm[20].y > lm[18])          # pinky


class THANK_YOUDetector(GestureDetector):
    """
    Detects ASL 'THANK YOU' gesture in three stages:

    Stage 0 — HOLD PREP: Hold hand near chin in preparation
    Stage 1 — TOUCH CHIN: Place flat hand on chin
    Stage 2 — MOVE FORWARD: Move hand forward to complete
    """

    def __init__(self) -> None:
        self._config = get_config("THANK_YOU")
        if self._config is None:
            raise ValueError("Configuration for gesture 'THANK_YOU' not found")

        self._phase = 0
        self._phase_complete = False
        self._phase_frames = 0
        self._start_y = 0.0
        self._finish_count = 0
        self.reset()

    @property
    def stage_label(self) -> str:
        return self._config.stages.get(self._phase, f"Stage {self._phase}")

    @property
    def phase_complete(self) -> bool:
        return self._phase_complete

    def reset(self) -> None:
        self._phase = 0
        self._phase_complete = False
        self._phase_frames = 0
        self._start_y = 0.0
        self._finish_count = 0

    def update(self, landmarks, handedness: Optional[str]) -> bool:
        self._phase_complete = False

        if landmarks is None or len(landmarks) < 21:
            if self._phase == 0:
                self.reset()
            return False

        lm = self._maybe_mirror(landmarks, handedness)

        # Stage 0: Hold preparation position
        if self._phase == 0:
            self._phase_frames += 1
            chin_region = lm[5].y < 0.45  # Wrist in upper portion
            flat_hand = self._is_flat_hand(lm)

            if chin_region and flat_hand:
                self._phase = 1
                self._phase_frames = 0
                self._start_y = lm[20].y
                self._phase_complete = True  # Ready for touch
                return False

        # Stage 1: Touch chin
        if self._phase == 1:
            self._phase_frames += 1
            chin_touch = lm[5].y < 0.42 and lm[20].y < 0.45
            flat_hand = self._is_flat_hand(lm)

            if chin_touch and flat_hand:
                self._phase_frames = 0
            else:
                if lm[20].y - self._start_y > self._config.detector.move_forward_distance:
                    self._phase = 2
                    self._finish_count = 0
                    self._phase_complete = True  # Ready for final hold
                    return False

            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
                return False

        # Stage 2: Move hand forward
        if self._phase == 2:
            self._phase_frames += 1
            flat_hand = self._is_flat_hand(lm)

            if flat_hand:
                self._finish_count += 1
            else:
                self._finish_count = max(0, self._finish_count - 1)

            if self._finish_count >= self._config.detector.stage_2_hold_frames:
                self.reset()
                return True  # THANK YOU complete

            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
                return False

        return False


class NAMEDetector(GestureDetector):
    """
    Detects ASL 'NAME' gesture in three stages:

    Stage 0 — PREP N-SHAPE: Form N-handshape (index + middle up)
    Stage 1 — NEAR CHEEK: Move hand near cheek
    Stage 2 — COMPLETE: Hold completed gesture
    """

    def __init__(self) -> None:
        self._config = get_config("NAME")
        if self._config is None:
            raise ValueError("Configuration for gesture 'NAME' not found")

        self._phase = 0
        self._phase_complete = False
        self._phase_frames = 0
        self._start_y = 0.0
        self._finish_count = 0
        self.reset()

    @property
    def stage_label(self) -> str:
        return self._config.stages.get(self._phase, f"Stage {self._phase}")

    @property
    def phase_complete(self) -> bool:
        return self._phase_complete

    def reset(self) -> None:
        self._phase = 0
        self._phase_complete = False
        self._phase_frames = 0
        self._start_y = 0.0
        self._finish_count = 0

    def update(self, landmarks, handedness: Optional[str]) -> bool:
        self._phase_complete = False

        if landmarks is None or len(landmarks) < 21:
            if self._phase == 0:
                self.reset()
            return False

        lm = self._maybe_mirror(landmarks, handedness)

        # Stage 0: Prepare N-shape
        if self._phase == 0:
            self._phase_frames += 1
            n_shape = self._is_n_shape(lm)
            cheek_region = lm[5].y < 0.4  # Near cheek

            if n_shape and cheek_region:
                self._phase = 1
                self._phase_frames = 0
                self._start_y = lm[20].y
                self._phase_complete = True  # Ready for cheek proximity
                return False

        # Stage 1: Near cheek
        if self._phase == 1:
            self._phase_frames += 1
            n_shape = self._is_n_shape(lm)
            cheek_proximity = lm[5].y < 0.38 and lm[20].y < 0.4

            if n_shape and cheek_proximity:
                self._phase_frames = 0
            else:
                if lm[20].y - self._start_y > self._config.detector.cheek_proximity:
                    self._phase = 2
                    self._finish_count = 0
                    self._phase_complete = True  # Ready for final hold
                    return False

            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
                return False

        # Stage 2: Complete gesture
        if self._phase == 2:
            self._phase_frames += 1
            n_shape = self._is_n_shape(lm)

            if n_shape:
                self._finish_count += 1
            else:
                self._finish_count = max(0, self._finish_count - 1)

            if self._finish_count >= self._config.detector.stage_2_hold_frames:
                self.reset()
                return True  # NAME complete

            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
                return False

        return False

    @staticmethod
    def _is_n_shape(lm) -> bool:
        """Check for N-handshape (index + middle fingers up, others curled)."""
        return (lm[8].y < lm[6] and      # index up (lower y = higher in image)
                lm[12].y < lm[10] and     # middle up
                lm[16].y > lm[14] and     # ring curled down
                lm[20].y > lm[18])        # pinky curled down


class YESDetector(GestureDetector):
    """
    Detects ASL 'YES' gesture in three stages:

    Stage 0 — READY: Thumb-up handshape (fist with thumb extended)
    Stage 1 — UPWARD MOTION: Move hand upward (nod gesture)
    Stage 2 — COMPLETE: Hold upward position
    """

    def __init__(self) -> None:
        self._config = get_config("YES")
        if self._config is None:
            raise ValueError("Configuration for gesture 'YES' not found")

        self._phase = 0
        self._phase_complete = False
        self._phase_frames = 0
        self._start_y = 0.0
        self._finish_count = 0
        self.reset()

    @property
    def stage_label(self) -> str:
        return self._config.stages.get(self._phase, f"Stage {self._phase}")

    @property
    def phase_complete(self) -> bool:
        return self._phase_complete

    def reset(self) -> None:
        self._phase = 0
        self._phase_complete = False
        self._phase_frames = 0
        self._start_y = 0.0
        self._finish_count = 0

    def update(self, landmarks, handedness: Optional[str]) -> bool:
        self._phase_complete = False

        if landmarks is None or len(landmarks) < 21:
            if self._phase == 0:
                self.reset()
            return False

        lm = self._maybe_mirror(landmarks, handedness)

        # Stage 0: Ready position with thumb-up
        if self._phase == 0:
            self._phase_frames += 1
            thumb_up = self._is_thumb_up(lm)

            if thumb_up:
                self._phase = 1
                self._phase_frames = 0
                self._start_y = lm[20].y
                self._phase_complete = True  # Ready for upward motion
                return False

        # Stage 1: Upward motion
        if self._phase == 1:
            self._phase_frames += 1
            thumb_up = self._is_thumb_up(lm)

            if thumb_up:
                # Check if moved upward
                if self._start_y - lm[20].y > self._config.detector.nod_distance:
                    self._phase = 2
                    self._finish_count = 0
                    self._phase_complete = True  # Ready for final hold
                    return False
            else:
                self._finish_count = max(0, self._finish_count - 1)

            if self._finish_count >= self._config.detector.stage_2_hold_frames:
                self.reset()
                return True  # YES complete

            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
                return False

        # Stage 2: Complete gesture
        if self._phase == 2:
            self._phase_frames += 1
            thumb_up = self._is_thumb_up(lm)

            if thumb_up:
                self._finish_count += 1
            else:
                self._finish_count = max(0, self._finish_count - 1)

            if self._finish_count >= self._config.detector.stage_2_hold_frames:
                self.reset()
                return True  # YES complete

            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
                return False

        return False

    @staticmethod
    def _is_thumb_up(lm) -> bool:
        """Check for thumb-up handshape (fist with thumb extended)."""
        # Index curled: tip below PIP joint
        # Thumb extended: tip above PIP joint (lower y value)
        return (lm[8].y > lm[6] and           # index curled
                lm[12].y > lm[10] and          # middle curled
                lm[16].y > lm[14] and          # ring curled
                lm[20].y > lm[18] and          # pinky curled
                lm[2].y < lm[4])               # thumb extended


class NO_detector(GestureDetector):
    """
    Detects ASL 'NO' gesture in three stages:

    Stage 0 — READY: Index finger extended or head position
    Stage 1 — SIDE MOTION: Move hand sideways
    Stage 2 — COMPLETE: Hold side position
    """

    def __init__(self) -> None:
        self._config = get_config("NO")
        if self._config is None:
            raise ValueError("Configuration for gesture 'NO' not found")

        self._phase = 0
        self._phase_complete = False
        self._phase_frames = 0
        self._start_x = 0.0
        self._finish_count = 0
        self.reset()

    @property
    def stage_label(self) -> str:
        return self._config.stages.get(self._phase, f"Stage {self._phase}")

    @property
    def phase_complete(self) -> bool:
        return self._phase_complete

    def reset(self) -> None:
        self._phase = 0
        self._phase_complete = False
        self._phase_frames = 0
        self._start_x = 0.0
        self._finish_count = 0

    def update(self, landmarks, handedness: Optional[str]) -> bool:
        self._phase_complete = False

        if landmarks is None or len(landmarks) < 21:
            if self._phase == 0:
                self.reset()
            return False

        lm = self._maybe_mirror(landmarks, handedness)

        # Stage 0: Ready position with index finger
        if self._phase == 0:
            self._phase_frames += 1
            index_shape = self._is_index_shape(lm)

            if index_shape:
                self._phase = 1
                self._phase_frames = 0
                self._start_x = lm[8].x
                self._phase_complete = True  # Ready for side motion
                return False

        # Stage 1: Side motion
        if self._phase == 1:
            self._phase_frames += 1
            index_shape = self._is_index_shape(lm)

            if index_shape:
                # Check if moved sideways
                if abs(lm[8].x - self._start_x) > self._config.detector.shake_distance:
                    self._phase = 2
                    self._finish_count = 0
                    self._phase_complete = True  # Ready for final hold
                    return False
            else:
                self._finish_count = max(0, self._finish_count - 1)

            if self._finish_count >= self._config.detector.stage_2_hold_frames:
                self.reset()
                return True  # NO complete

            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
                return False

        # Stage 2: Complete gesture
        if self._phase == 2:
            self._phase_frames += 1
            index_shape = self._is_index_shape(lm)

            if index_shape:
                self._finish_count += 1
            else:
                self._finish_count = max(0, self._finish_count - 1)

            if self._finish_count >= self._config.detector.stage_2_hold_frames:
                self.reset()
                return True  # NO complete

            if self._phase_frames > self._config.detector.phase_timeout:
                self.reset()
                return False

        return False

    @staticmethod
    def _is_index_shape(lm) -> bool:
        """Check for index finger shape (only index extended)."""
        # Index up: tip above PIP joint (lower y)
        # Others curled down
        return (lm[8].y < lm[6] and              # index up
                lm[12].y > lm[10] and            # middle curled
                lm[16].y > lm[14] and            # ring curled
                lm[20].y > lm[18])               # pinky curled


# =================================================================
# Update factory function to include gesture detectors
# =================================================================

def get_detector(letter: str, gesture: Optional[str] = None) -> Optional[DynamicDetector]:
    """
    Factory function to instantiate the appropriate dynamic detector
    for a given ASL letter or gesture.

    Args:
        letter: Single character ASL letter (e.g., 'J') or gesture name
        gesture: Gesture name (e.g., 'HELLO', 'THANK YOU', 'NAME', 'YES', 'NO')

    Returns:
        DynamicDetector subclass instance or None
    """
    if gesture:
        # Handle gestures
        gesture = gesture.upper()
        if gesture == "HELLO":
            return HELLODetector()
        elif gesture == "THANK YOU":
            return THANK_YOUDetector()
        elif gesture == "NAME":
            return NAMEDetector()
        elif gesture == "YES":
            return YESDetector()
        elif gesture == "NO":
            return NODetector()
        return None

    # Handle letters
    if letter.upper() == 'J':
        return JDetector()
    if letter.upper() == 'Z':
        return ZDetector()
    return None