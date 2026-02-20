from typing import List, Optional
import math


class ASLClassifier:

    def __init__(self) -> None:
        self.touch_threshold = 0.06

    # ---------------------------------------------------
    # Utility Functions
    # ---------------------------------------------------

    def _distance(self, p1, p2):
        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2
        )

    def _mirror_landmarks_x(self, landmarks: List) -> List:
        mirrored = []
        for lm in landmarks:
            new_lm = type(lm)()
            new_lm.x = 1.0 - lm.x
            new_lm.y = lm.y
            new_lm.z = getattr(lm, "z", 0)
            mirrored.append(new_lm)
        return mirrored

    def _fingers_curled(self, lm):
        return (
            lm[8].y > lm[6].y and
            lm[12].y > lm[10].y and
            lm[16].y > lm[14].y and
            lm[20].y > lm[18].y
        )

    def _fingers_extended(self, lm):
        return (
            lm[8].y < lm[6].y and
            lm[12].y < lm[10].y and
            lm[16].y < lm[14].y and
            lm[20].y < lm[18].y
        )

    # ---------------------------------------------------
    # Letter Detection
    # ---------------------------------------------------

    def _is_a(self, lm):
        """
        A = fingers curled, thumb resting beside index
        """
        if not self._fingers_curled(lm):
            return False

        thumb_tip = lm[4]
        index_mcp = lm[5]

        # Thumb should be to side of index
        if thumb_tip.x > index_mcp.x:
            return False

        return True

    def _is_b(self, lm):
        """
        B = fingers extended, thumb across palm
        """
        if not self._fingers_extended(lm):
            return False

        thumb_tip = lm[4]
        palm = lm[9]

        if self._distance(thumb_tip, palm) > 0.12:
            return False

        return True

    def _is_c(self, lm):
        """
        C = curved shape (not fully curled, not fully extended)
        """
        # Fingers should not be fully curled or fully extended
        if self._fingers_curled(lm) or self._fingers_extended(lm):
            return False

        thumb_tip = lm[4]
        index_tip = lm[8]

        dist = self._distance(thumb_tip, index_tip)

        if 0.05 < dist < 0.20:
            return True

        return False

    def _is_d(self, lm):
        """
        D = index extended, others curled, thumb touching middle
        """
        if not (
            lm[8].y < lm[6].y and
            lm[12].y > lm[10].y and
            lm[16].y > lm[14].y and
            lm[20].y > lm[18].y
        ):
            return False

        if self._distance(lm[4], lm[12]) > self.touch_threshold:
            return False

        return True

    def _is_e(self, lm):
        """
        E = fingers curled, fingertips near thumb
        """
        if not self._fingers_curled(lm):
            return False

        thumb_tip = lm[4]
        tips = [lm[8], lm[12], lm[16], lm[20]]

        for tip in tips:
            if self._distance(tip, thumb_tip) > 0.07:
                return False

        return True

    def _is_f(self, lm: List) -> bool:
        """
        F = Thumb and Index touching (forming a circle), 
        other fingers extended.
        """
        # 1. Check if Thumb (4) and Index (8) are touching
        # We use a slightly looser threshold for the 'OK' sign shape
        if self._distance(lm[4], lm[8]) > 0.08:
            return False

        # 2. Check if the OTHER fingers are extended (Middle, Ring, Pinky)
        # We cannot use self._fingers_extended() because that requires 
        # the Index finger to be open too, which is not true for 'F'.
        
        # Logic: Tip.y < PIP.y (assuming Y=0 is top of screen)
        middle_extended = lm[12].y < lm[10].y
        ring_extended   = lm[16].y < lm[14].y
        pinky_extended  = lm[20].y < lm[18].y

        if not (middle_extended and ring_extended and pinky_extended):
            return False

        return True

    # ---------------------------------------------------
    # Main Classifier
    # ---------------------------------------------------

    def classify(self, landmarks, handedness: Optional[str] = None) -> Optional[str]:

        if landmarks is None or len(landmarks) < 21:
            return None

        # Normalize right hand
        if handedness == "Right":
            landmarks = self._mirror_landmarks_x(landmarks)

        # Order matters
        if self._is_d(landmarks):
            return "D"

        if self._is_f(landmarks):
            return "F"

        if self._is_e(landmarks):
            return "E"

        if self._is_c(landmarks):
            return "C"

        if self._is_b(landmarks):
            return "B"

        if self._is_a(landmarks):
            return "A"

        return None