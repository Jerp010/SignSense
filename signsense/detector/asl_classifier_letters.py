"""
asl_classifier_letters.py
=========================
Scoring-based static ASL letter classifier.

Active letters:  A  B  C  D  E  F  G  H  I  K  L  M  N  O  P
Dynamic letters (J, Z, ...) live in signs/dynamic_signs.py and are NOT
handled here.  The play-mode loop chooses the right tool per sign using
signs/sign_registry.py.

HOW THE CLASSIFIER WORKS
-------------------------
1. MediaPipe gives us 21 hand landmarks, each with (x, y, z) in
   normalised frame coordinates:
     x  0 = left edge of frame,  1 = right edge
     y  0 = TOP  of frame,       1 = BOTTOM          <- inverted!
   So  tip.y < pip.y  means the tip is HIGHER on screen (pointing up).

2. We compute a hand scale = distance from wrist (lm[0]) to middle MCP
   (lm[9]).  Every distance threshold is multiplied by this scale so the
   same thresholds work whether the hand is close or far from the camera.

3. A FingerState dataclass is pre-computed once per frame.  It stores
   simple boolean facts about each finger (extended? deeply curled?).
   Scorers read these flags instead of re-comparing landmarks each time.

4. Each letter has a _score_X() function that returns a float 0.0-1.0.
   The score is built from:
     POSITIVE checks  - things that SHOULD be true for this sign
     PENALTY  checks  - things that should NOT be true; subtract score

5. classify() runs all scorers, picks the highest score, and returns it
   if it clears min_confidence (default 0.60).

   In "targeted mode" (play mode passing target_letter=) only one scorer
   runs, so no other letter can accidentally out-score the target.

MEDIAPIPE LANDMARK INDEX REFERENCE
------------------------------------
  0  = Wrist
  1  = Thumb CMC (base knuckle)
  2  = Thumb MCP
  3  = Thumb IP  (middle joint)
  4  = Thumb tip
  5  = Index MCP   6  = Index PIP   7  = Index DIP   8  = Index tip
  9  = Middle MCP  10 = Middle PIP  11 = Middle DIP  12 = Middle tip
  13 = Ring MCP    14 = Ring PIP    15 = Ring DIP    16 = Ring tip
  17 = Pinky MCP   18 = Pinky PIP   19 = Pinky DIP   20 = Pinky tip

  MCP = knuckle at the palm
  PIP = first finger joint (middle of finger)
  DIP = second finger joint (near tip)

DEBUGGING TIPS
--------------
Uncomment the print() line at the bottom of classify() to log all
scores every frame.  Read the output like this:

  "Nothing detected" -> the right letter scores ~0.45-0.58.
      Lower min_confidence temporarily to 0.3 to confirm, then raise
      the POSITIVE weight for the distinguishing feature.

  "Wrong letter dominates" -> that letter's POSITIVE checks are too easy
      to satisfy by accident.  Add a PENALTY to it that fires when the
      feature your target sign has (e.g. a gap, a curl) is present.

  "Letter flickers between two" -> the two letters share high scores.
      Find ONE geometry that only the target sign has and make it a
      large POSITIVE for the target and a large PENALTY for the rival.
"""

from typing import List, Optional, Dict
import math
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# FingerState  -  pre-computed per-frame booleans
# -----------------------------------------------------------------------------

@dataclass
class FingerState:
    """
    Holds simple True/False facts about each finger, computed once per frame.

    Why pre-compute?
      Each _score_X function needs many of the same comparisons.  Computing
      them here avoids repeating lm[8].y < lm[6].y a dozen times per frame.

    Extension vs deep-curl
      'extended'    means tip is ABOVE  its PIP joint -> finger is pointing up
      'deep curled' means tip is BELOW  its MCP joint -> finger is fully folded
                    (stronger signal than just checking PIP -- confirms a real fist)
    """
    # True = finger tip is ABOVE (lower y than) its PIP -> finger is extended/up.
    # Only detects vertical extension. Use *_side_ext below for sideways fingers.
    index_ext:        bool
    middle_ext:       bool
    ring_ext:         bool
    pinky_ext:        bool

    # True = finger tip has clear horizontal travel (>12% of scale) from its MCP.
    # Lets G and H detect sideways-pointing fingers without requiring a vertical tilt.
    index_side_ext:   bool
    middle_side_ext:  bool

    # True = finger tip is BELOW (higher y than) its MCP -> deeply folded into palm
    index_deep_curl:  bool
    middle_deep_curl: bool
    ring_deep_curl:   bool
    pinky_deep_curl:  bool

    # True = thumb tip is to the LEFT of thumb base (lm[2]) by a clear margin.
    # On a mirrored-for-right-hand frame this reliably flags a sideways-extended thumb.
    thumb_side:       bool

    # The wrist-to-middle-MCP distance used to normalise all thresholds.
    scale:            float


# -----------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------

class ASLClassifierLetters:

    def __init__(self) -> None:
        # A score must reach this threshold to be reported.
        # 0.6 means "at least 60% of the positive evidence is present".
        # Lower it (e.g. 0.45) if valid signs are being missed;
        # raise it (e.g. 0.75) if false positives are too frequent.
        self.min_confidence = 0.6

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    def _distance(self, p1, p2) -> float:
        """Euclidean 2-D distance between two MediaPipe landmarks."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _hand_scale(self, lm) -> float:
        """
        Reference length for normalising all distance thresholds.
        Uses wrist (lm[0]) -> middle MCP (lm[9]).
        Multiplying a threshold by scale makes it camera-distance-independent.
        e.g.  threshold = 0.30 * scale  means "30% of the palm length".
        """
        return self._distance(lm[0], lm[9])

    def _mirror_landmarks_x(self, landmarks: List) -> List:
        """
        Flip all x-coordinates so a right hand looks like a left hand.

        Why?  All scorers are written assuming the thumb is on the LEFT side
        of the hand (i.e. a left hand facing the camera, or a right hand
        with its back to the camera).  By mirroring right-hand landmarks we
        can use one set of rules for both hands.
        """
        mirrored = []
        for lm in landmarks:
            n   = type(lm)()   # create a new landmark of the same type
            n.x = 1.0 - lm.x  # flip horizontally
            n.y = lm.y
            n.z = getattr(lm, "z", 0)
            mirrored.append(n)
        return mirrored

    def _is_extended(self, tip, pip) -> bool:
        """
        Returns True when a finger is extended (pointing upward).
        In MediaPipe, y=0 is the TOP of the frame, so a tip that is
        ABOVE its PIP joint has a SMALLER y value.
        """
        return tip.y < pip.y

    def _is_deeply_curled(self, tip, mcp) -> bool:
        """
        Returns True when a finger tip has curled all the way below its MCP
        (the large knuckle at the palm).  This is a stronger fist signal than
        just checking PIP, because it means the finger is fully folded in.
        """
        return tip.y > mcp.y

    def _get_finger_states(self, lm, scale: float) -> FingerState:
        """
        Pre-compute all per-finger states once, before any scorer runs.
        Each finger uses its own tip and joint landmarks (see reference above).
        """
        # Sideways extension: tip has moved horizontally from its MCP.
        # Threshold lowered to 0.07 * scale so a truly level (horizontal) finger
        # registers even when the hand isn't tilted at all — a level finger has
        # very little x-travel relative to its MCP but still clears 7% of palm length.
        # Only index and middle are needed (G and H are the only sideways-finger signs).
        def _side_ext(tip, mcp):
            return abs(tip.x - mcp.x) > 0.07 * scale

        return FingerState(
            # Extended = tip above PIP (tip.y < pip.y)
            index_ext        = self._is_extended(lm[8],  lm[6]),   # index tip vs index PIP
            middle_ext       = self._is_extended(lm[12], lm[10]),  # middle tip vs middle PIP
            ring_ext         = self._is_extended(lm[16], lm[14]),  # ring tip vs ring PIP
            pinky_ext        = self._is_extended(lm[20], lm[18]),  # pinky tip vs pinky PIP

            # Sideways extension (index and middle only)
            index_side_ext   = _side_ext(lm[8],  lm[5]),   # index tip vs index MCP
            middle_side_ext  = _side_ext(lm[12], lm[9]),   # middle tip vs middle MCP

            # Deeply curled = tip below MCP (tip.y > mcp.y)
            index_deep_curl  = self._is_deeply_curled(lm[8],  lm[5]),  # index tip vs index MCP
            middle_deep_curl = self._is_deeply_curled(lm[12], lm[9]),  # middle tip vs middle MCP
            ring_deep_curl   = self._is_deeply_curled(lm[16], lm[13]), # ring tip vs ring MCP
            pinky_deep_curl  = self._is_deeply_curled(lm[20], lm[17]), # pinky tip vs pinky MCP

            # Thumb extended sideways: tip is clearly to the left of thumb base
            thumb_side       = (lm[4].x < lm[2].x - 0.03),

            scale            = scale,
        )

    def _palm_center(self, lm):
        """
        Approximate center of the palm by averaging the four MCP knuckles
        (index=5, middle=9, ring=13, pinky=17).
        Used to measure how close the thumb is resting to the palm.
        """
        cx = (lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 4
        cy = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4
        return cx, cy

    def _count_extended(self, fs: FingerState) -> int:
        """How many of the 4 fingers (not thumb) are currently extended."""
        return sum([fs.index_ext, fs.middle_ext, fs.ring_ext, fs.pinky_ext])

    def _count_deep_curled(self, fs: FingerState) -> int:
        """How many of the 4 fingers are fully folded into a fist position."""
        return sum([fs.index_deep_curl, fs.middle_deep_curl,
                    fs.ring_deep_curl,  fs.pinky_deep_curl])

    def _angle_3pts(self, a, b, c) -> float:
        """
        Angle at point B formed by the rays B->A and B->C, in degrees.
        180 = perfectly straight line.  ~90 = right angle.  <90 = bent.
        Useful for measuring joint bend (e.g. thumb angle at IP joint).
        """
        ab  = (a.x - b.x, a.y - b.y)
        cb  = (c.x - b.x, c.y - b.y)
        dot = ab[0]*cb[0] + ab[1]*cb[1]
        mag = math.hypot(*ab) * math.hypot(*cb)
        if mag == 0:
            return 180.0  # degenerate case - treat as straight
        # clamp to [-1, 1] before acos to avoid floating-point domain errors
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))

    def _finger_points_at_camera(self, tip, mcp) -> bool:
        """
        Detect when a finger is pointing roughly toward the camera using z-depth.
        MediaPipe z is negative when the landmark is closer to the camera than
        the wrist origin.  A finger pointing AT the camera will have its tip
        at a more negative z than its MCP base.
        Used only by _score_p to detect the downward-forward P orientation.
        NOT used by G or H — those rely purely on x/y travel and side_ext flags.
        """
        tip_z = getattr(tip, 'z', 0.0)
        mcp_z = getattr(mcp, 'z', 0.0)
        return (mcp_z - tip_z) > 0.05

    # -------------------------------------------------------------------------
    # SCORING FUNCTIONS
    # -------------------------------------------------------------------------
    #
    # Pattern for every scorer
    # -------------------------
    # 1. HARD GATES first - if a critical condition is outright false, return
    #    0.0 immediately so we don't waste time on the rest.
    #
    # 2. POSITIVE checks - add score for each feature that SHOULD be present.
    #    Weights are chosen so that a correct sign scores ~0.85-1.0 and a
    #    clearly wrong hand position scores < 0.3.
    #
    # 3. PENALTY checks - subtract score for features that should NOT be
    #    present.  Penalties exist primarily to separate signs that share many
    #    positive features (e.g. A and E are both closed fists - only the thumb
    #    position tells them apart).
    #
    # 4. Clamp to [0.0, 1.0] and return.
    #
    # Landmark shorthand used in comments:
    #   lm[4]  = thumb tip      lm[5]  = index MCP    lm[6]  = index PIP
    #   lm[8]  = index tip      lm[9]  = middle MCP   lm[12] = middle tip
    #   lm[16] = ring tip       lm[17] = pinky MCP    lm[20] = pinky tip
    # -------------------------------------------------------------------------

    def _score_a(self, lm, scale, fs):
        """
        A - Closed fist with thumb raised alongside index finger.

        Hand shape:
          - All 4 fingers curled tightly into the palm
          - Thumb tip sits BESIDE the index MCP, pointing UPWARD
          - Thumb does NOT touch the fingertips

        Key differences from similar signs:
          vs E : E has thumb TUCKED LOW under the fingertips
                 (thumb.y > index MCP.y).  A has thumb HIGH beside
                 the fist (thumb.y <= index MCP.y).
          vs S : S wraps the thumb OVER the front of the fist.
          vs M/N: M and N have thumb UNDER the fist (thumb.y > thumb MCP.y)
        """
        score = 0.0

        # HARD GATES
        # If index or middle finger is sticking up, this is not a fist at all.
        if fs.index_ext or fs.middle_ext:
            return 0.0

        # POSITIVE: All 4 fingers are curled (none are extended upward).
        all_curled = not any([fs.index_ext, fs.middle_ext, fs.ring_ext, fs.pinky_ext])
        score += 0.30 * all_curled

        # POSITIVE: Compact fist - index tip to pinky tip distance is small.
        score += 0.20 * (self._distance(lm[8], lm[20]) < 0.45 * scale)

        # POSITIVE: Thumb is HIGH - at or slightly above the index MCP (lm[5]).
        score += 0.15 * (lm[4].y <= lm[5].y + 0.02)

        # POSITIVE: Thumb is beside the fist horizontally.
        score += 0.25 * (abs(lm[4].x - lm[5].x) < 0.12)

        # POSITIVE: Thumb tip is NOT close to any fingertip.
        thumb_clear = all(self._distance(lm[4], lm[t]) > 0.18 * scale
                          for t in [8, 12, 16, 20])
        score += 0.30 * thumb_clear

        # POSITIVE: Deep curl confirmation.
        score += 0.20 * (self._count_deep_curled(fs) >= 3)

        # PENALTY: Thumb is tucked LOW (well below index MCP) -> E or M/N, not A.
        score -= 0.50 * (lm[4].y > lm[5].y + 0.06)

        # PENALTY: Thumb close to ANY fingertip -> might be M or N, not pure A.
        tips_near_thumb = sum(
            1 for t in [lm[8], lm[12], lm[16], lm[20]]
            if self._distance(t, lm[4]) < 0.22 * scale
        )
        score -= 0.35 * (tips_near_thumb >= 1)

        # PENALTY: Thumb tip ABOVE the index fingertip -> S sign, not A.
        score -= 0.30 * (lm[4].y < lm[8].y - 0.01)

        # PENALTY: Thumb tip has curled BELOW its own MCP -> thumb is tucked under
        # the fist like M/N/S, not sitting beside the fist like A.
        score -= 0.60 * (lm[4].y > lm[2].y + 0.03)

        return max(0.0, min(1.0, score))

    def _score_b(self, lm, scale, fs):
        """
        B - Four fingers extended straight up, tight together.
            Thumb is tucked across the palm.

        Hand shape:
          - All 4 fingers point straight upward
          - Fingers are CLOSE together (not spread)
          - Thumb folds across the palm (not sticking out)

        Key differences:
          vs open hand : thumb must be tucked, not extended out
          vs W         : W has 3 fingers spread; B has 4 tight ones
          vs H         : H has only 2 fingers and they point sideways
        """
        score = 0.0

        # POSITIVE: All 4 fingers must be extended upward.
        score += 0.40 * all([fs.index_ext, fs.middle_ext, fs.ring_ext, fs.pinky_ext])

        # POSITIVE: Finger tips are all above their respective MCPs.
        score += 0.20 * (lm[8].y  < lm[5].y and
                         lm[12].y < lm[9].y and
                         lm[20].y < lm[17].y)

        # POSITIVE: Fingers are tightly grouped.
        spread = abs(lm[8].x - lm[20].x)
        score += 0.20 * (spread < 0.10)

        # POSITIVE: Thumb is tucked - thumb tip below the index PIP.
        score += 0.20 * (lm[4].y > lm[6].y)

        # PENALTY: Thumb sticking out sideways.
        score -= 0.30 * fs.thumb_side

        # PENALTY: Fingers spread wide.
        score -= 0.20 * (spread > 0.14)

        # PENALTY: Thumb far from palm center.
        cx, cy = self._palm_center(lm)
        thumb_dist_from_palm = math.hypot(lm[4].x - cx, lm[4].y - cy)
        score -= 0.30 * (thumb_dist_from_palm > 0.20 * scale)

        return max(0.0, min(1.0, score))

    def _score_c(self, lm, scale, fs):
        """
        C - Curved open hand forming a C shape.
            (like holding a cylinder or drinking glass)

        Hand shape:
          - All fingers partially bent - NOT straight, NOT fisted
          - Thumb curves outward to face the fingertips
          - Medium gap between thumb tip and index tip
          - The C opening faces to the side

        Key differences:
          vs B : B has straight extended fingers; C has curved ones
          vs O : O has thumb and index close/touching; C has a clear gap
          vs A : A is a closed fist; C is open with visible space
          vs M/N: M and N are closed fists - ALL fingertips deeply curled,
                  NO open gap between thumb and fingers
        """
        score = 0.0

        # HARD GATE: If 3+ fingers are deeply curled this is a fist (A/E/M/N/S),
        # not the open C shape. This single gate prevents M/N from scoring as C.
        if self._count_deep_curled(fs) >= 3:
            return 0.0

        # HARD GATE: If ALL fingertips are very close to the thumb tip (< 0.22 scale),
        # this is O or a tight fist, not C which has a clear open gap.
        tips_near_thumb = sum(
            1 for t in [lm[8], lm[12], lm[16], lm[20]]
            if self._distance(t, lm[4]) < 0.22 * scale
        )
        if tips_near_thumb >= 3:
            return 0.0

        # POSITIVE: Count how many fingers are in the "partial curve zone".
        partial_curve_count = sum(
            1 for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]
            if lm[tip_idx].y > lm[pip_idx].y - 0.02
        )
        score += 0.45 * (partial_curve_count / 4)

        # POSITIVE: The gap between thumb tip and index tip should be medium-sized.
        # Wide enough to be clearly open (not O), narrow enough to be a C (not open hand).
        dist_thumb_index = self._distance(lm[4], lm[8])
        score += 0.35 * (0.28 * scale < dist_thumb_index < 0.70 * scale)

        # POSITIVE: Thumb and index tip are at a similar height.
        score += 0.20 * (abs(lm[4].y - lm[8].y) < 0.12)

        # POSITIVE: Thumb has a natural curve.
        score += 0.25 * (0.15 * scale < self._distance(lm[4], lm[3]) < 0.45 * scale)

        # PENALTY: 3+ fingers fully extended = B (straight fingers), not C (curved).
        score -= 0.35 * (self._count_extended(fs) >= 3)

        # PENALTY: Large horizontal travel on index or middle = G or H territory.
        idx_sideways = abs(lm[8].x  - lm[5].x) > 0.12 * scale
        mid_sideways = abs(lm[12].x - lm[9].x) > 0.12 * scale
        score -= 0.25 * (idx_sideways or mid_sideways)

        # PENALTY: Straight fingers are characteristic of B/W or G/H.
        straight_count = 0
        for tip_idx, pip_idx, mcp_idx in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]:
            ang = self._angle_3pts(lm[tip_idx], lm[pip_idx], lm[mcp_idx])
            if ang > 165:
                straight_count += 1
        score -= 0.30 * (straight_count / 4)

        if fs.index_side_ext and self._angle_3pts(lm[8], lm[6], lm[5]) > 165:
            score -= 0.20
        if fs.middle_side_ext and self._angle_3pts(lm[12], lm[10], lm[9]) > 165:
            score -= 0.20

        return max(0.0, min(1.0, score))

    def _score_d(self, lm, scale, fs):
        """
        D - Index finger pointing straight up.
            Other fingers curl into a loop touching the thumb.

        Hand shape:
          - Index finger extends straight upward
          - Middle, ring, pinky curl down
          - Thumb tip meets (or nearly touches) the middle finger,
            forming the rounded back of the letter D

        Key differences:
          vs G : G points sideways; D points upward
          vs L : L has thumb extended sideways; D has thumb looping
          vs 1 : the "1" gesture has no thumb-middle contact
        """
        score = 0.0

        # POSITIVE: Index finger is extended.
        score += 0.25 * fs.index_ext

        # POSITIVE: Index tip is clearly above its MCP.
        score += 0.15 * (lm[8].y < lm[5].y - 0.05)

        # POSITIVE: Middle, ring, and pinky are all curled down.
        score += 0.25 * (not fs.middle_ext and not fs.ring_ext and not fs.pinky_ext)

        # POSITIVE: Thumb tip is close to the middle finger, forming the loop of the D.
        score += 0.25 * (self._distance(lm[4], lm[12]) < 0.28 * scale)

        # POSITIVE: Index tip is clearly higher than the middle fingertip.
        score += 0.10 * (lm[8].y < lm[12].y - 0.05)

        # PENALTY: Thumb extended sideways = L sign, not D.
        score -= 0.30 * fs.thumb_side

        # PENALTY: More than one finger extended.
        score -= 0.20 * (self._count_extended(fs) >= 2)

        return max(0.0, min(1.0, score))

    def _score_e(self, lm, scale, fs):
        """
        E - All fingers bent, tips curled down toward the palm.
            Thumb tucked LOW under the bent fingertips.

        Hand shape:
          - All 4 fingertips curl downward (claw / rake shape)
          - Thumb tip is LOW - below the index MCP level
          - Fingertips and thumb are close together in a flat claw
          - NOT a tight round fist - there is a slight gap

        Key differences:
          vs A : A has thumb HIGH beside the fist
          vs S : S thumb crosses OVER the front of the fist
          vs M/N: M and N have thumb BENEATH the fist (below thumb MCP);
                  E has thumb reaching FORWARD toward the fingertips
          vs O : O has a round open shape; E is flat and raked inward
        """
        score = 0.0

        # POSITIVE: All 4 fingers are curled.
        all_curled = not any([fs.index_ext, fs.middle_ext, fs.ring_ext, fs.pinky_ext])
        score += 0.25 * all_curled

        # POSITIVE: Each fingertip should be close to the thumb tip (claw geometry).
        thumb = lm[4]
        tips_near_thumb = sum(
            1 for t in [lm[8], lm[12], lm[16], lm[20]]
            if self._distance(t, thumb) < 0.30 * scale
        )
        score += 0.40 * (tips_near_thumb / 4)

        # POSITIVE: All fingertips at a similar height (flat claw alignment).
        tip_ys = [lm[8].y, lm[12].y, lm[16].y, lm[20].y]
        score += 0.20 * ((max(tip_ys) - min(tip_ys)) < 0.10)

        # POSITIVE: Thumb tip is LOW (at or below middle MCP level).
        score += 0.15 * (lm[4].y < lm[9].y + 0.05)

        # PENALTY: Thumb tip has curled BELOW its own MCP -> thumb is fully tucked
        # under the fist.  This is M/N/S behaviour, NOT E (claw where the thumb
        # reaches forward toward the fingertips from the side).
        score -= 0.60 * (lm[4].y > lm[2].y + 0.03)

        # PENALTY: Thumb beside fist (high and close to index PIP) -> A.
        thumb_beside_fist = (lm[4].y > lm[8].y + 0.02 and
                             self._distance(lm[4], lm[6]) < 0.20 * scale)
        score -= 0.25 * thumb_beside_fist

        return max(0.0, min(1.0, score))

    def _score_f(self, lm, scale, fs):
        """
        F - Thumb and index form a circle (touching tips).
            Middle, ring, pinky extend upward.

        Hand shape:
          - Thumb tip and index tip touch or nearly touch
          - Index curls down to meet the thumb (not extended up)
          - Middle, ring, pinky all point straight upward

        Key differences:
          vs D : D has thumb touching MIDDLE finger; F touches INDEX
          vs O : O has ALL fingers in a circle; F has 3 fingers up
          vs I : I has only pinky up; F has 3 fingers up
        """
        score = 0.0

        d = self._distance(lm[4], lm[8])
        three_up = fs.middle_ext and fs.ring_ext and fs.pinky_ext

        # POSITIVE: Thumb and index tips are very close.
        score += 0.45 * (d < 0.20 * scale)

        # POSITIVE: Middle, ring, pinky are all extended upward.
        score += 0.35 * three_up

        # POSITIVE: Index is NOT extended upward.
        score += 0.20 * (not fs.index_ext)

        # PENALTY: Index is extended.
        score -= 0.40 * fs.index_ext

        # PENALTY: Thumb and index are far apart.
        score -= 0.20 * (d > 0.30 * scale)

        # PENALTY: The three upper fingers are not raised.
        score -= 0.35 * (not three_up)

        return max(0.0, min(1.0, score))

    def _score_g(self, lm, scale, fs):
        """
        G - Index finger pointing SIDEWAYS and HORIZONTAL.
            Back of hand faces the camera.

        Hand shape:
          - ONLY the index finger extends, pointing to the side
          - Index tip is at roughly the same HEIGHT as its MCP
            (truly horizontal, not diagonal up)
          - Middle, ring, pinky are all curled into the palm
          - Thumb stays below or beside (does not extend sideways)

        Detection strategy for a LEVEL sideways finger
        -----------------------------------------------
        A truly horizontal index has:
          - Significant x-travel from MCP to tip (caught by index_side_ext,
            now triggered at 0.07 * scale instead of 0.12 * scale)
          - Very little y-travel between tip and MCP  (tip nearly level with MCP)
          - The tip is NOT above the MCP (not pointing up)
        We accept index_ext as a fallback for when the hand is slightly tilted,
        but the primary path for a perfectly level G is index_side_ext.

        Key differences:
          vs D : D points UPWARD; G points SIDEWAYS
          vs H : H has TWO fingers sideways; G has one
          vs C : C fingers are CURVED; G index is cleanly straight
          vs L : L has index UP and thumb sideways; G has index sideways
        """
        # HARD GATE: Middle, ring, and pinky must ALL be curled.
        # If any of them is up this is H (two fingers) or B/W (more fingers).
        if fs.middle_ext or fs.ring_ext or fs.pinky_ext:
            return 0.0

        # HARD GATE: Index must be doing SOMETHING extended —
        # either sideways (primary level-G path) or upward-ish (tilted G).
        if not fs.index_side_ext and not fs.index_ext:
            return 0.0

        score = 0.0

        x_travel = abs(lm[8].x - lm[5].x)   # tip to MCP horizontal distance
        y_travel = abs(lm[8].y - lm[5].y)   # tip to MCP vertical distance

        # POSITIVE: Index side-extension flag is the primary G signal.
        # This fires when x_travel > 0.07 * scale — the main path for level G.
        score += 0.45 * fs.index_side_ext

        # POSITIVE: The finger is roughly level — tip close in height to MCP.
        # A threshold of 0.18 allows natural hand tilt without excluding real G.
        score += 0.30 * (y_travel < 0.18)

        # POSITIVE: x-travel genuinely dominates y-travel (sideways, not diagonal-up).
        score += 0.15 * (x_travel > y_travel)

        # POSITIVE: Index finger is straight (not C-curved).
        # Vertex at PIP (lm[6]): straight = ~180°, curled = <160°.
        score += 0.10 * (self._angle_3pts(lm[8], lm[6], lm[5]) > 155)

        # PENALTY: Tip is clearly ABOVE its MCP → pointing upward (D or L), not G.
        # Threshold of 0.10 gives tolerance for very slight upward tilts.
        score -= 0.60 * (lm[8].y < lm[5].y - 0.10)

        # PENALTY: Thumb extended sideways → L not G.
        score -= 0.40 * fs.thumb_side

        # PENALTY: C-like hook — tip has dropped below its PIP joint.
        score -= 0.30 * (lm[8].y > lm[6].y + 0.02)

        # PENALTY: Negligible x-travel — finger hasn't actually moved sideways.
        # Only fires below 0.05 * scale (tighter than the 0.07 gate) so it
        # won't conflict with the relaxed side_ext threshold.
        score -= 0.40 * (x_travel < 0.05 * scale)

        return max(0.0, min(1.0, score))

    def _score_h(self, lm, scale, fs):
        """
        H - Index AND middle fingers both pointing SIDEWAYS and HORIZONTAL.
            Back of hand faces the camera.

        Hand shape:
          - Index AND middle both extend horizontally to the side
          - Both tips are at roughly the same HEIGHT as their MCPs
          - The two fingers are held TOGETHER (not spread apart)
          - Ring and pinky are curled

        Detection strategy for a LEVEL sideways H
        ------------------------------------------
        Same as G but both fingers must qualify.  The lowered side_ext
        threshold (0.07 * scale) means a truly horizontal H now triggers
        both index_side_ext and middle_side_ext even without any upward tilt.

        Key differences:
          vs G : G has ONE sideways finger; H has TWO
          vs U : U points UPWARD; H points SIDEWAYS
          vs V : V spreads fingers apart; H keeps them together
        """
        # HARD GATE: Both index AND middle must be extended (sideways or upward).
        if not fs.index_side_ext and not fs.index_ext:
            return 0.0
        if not fs.middle_side_ext and not fs.middle_ext:
            return 0.0

        # HARD GATE: Ring and pinky must be curled.
        if fs.ring_ext or fs.pinky_ext:
            return 0.0

        score = 0.0

        ix = abs(lm[8].x  - lm[5].x)   # index  x-travel (tip to MCP)
        mx = abs(lm[12].x - lm[9].x)   # middle x-travel (tip to MCP)
        iy = abs(lm[8].y  - lm[5].y)   # index  y-travel
        my = abs(lm[12].y - lm[9].y)   # middle y-travel

        # POSITIVE: Index side-extension flag — primary level-H signal.
        score += 0.25 * fs.index_side_ext

        # POSITIVE: Middle side-extension flag.
        score += 0.25 * fs.middle_side_ext

        # POSITIVE: Both tips at nearly the same HEIGHT as their MCPs — truly level.
        # Threshold 0.18 allows natural hand tilt without excluding real H.
        both_level = (iy < 0.18 and my < 0.18)
        score += 0.20 * both_level

        # POSITIVE: x-travel dominates y-travel on both fingers (sideways, not diagonal).
        score += 0.10 * (ix > iy and mx > my)

        # POSITIVE: Both fingers straight (not C-curved).
        # Vertex at PIP — straight = ~180°, curled = <155°.
        score += 0.10 * (self._angle_3pts(lm[8],  lm[6],  lm[5]) > 155)
        score += 0.10 * (self._angle_3pts(lm[12], lm[10], lm[9]) > 155)

        # POSITIVE: Ring and pinky are curled.
        score += 0.10 * (not fs.ring_ext and not fs.pinky_ext)

        # PENALTY: Both fingers pointing clearly UPWARD → U or V, not H.
        both_up = (lm[8].y < lm[5].y - 0.10 and lm[12].y < lm[9].y - 0.10)
        score -= 0.60 * both_up

        # PENALTY: C-shaped hooks on either finger.
        score -= 0.25 * (self._angle_3pts(lm[8],  lm[6],  lm[5]) < 150)
        score -= 0.25 * (self._angle_3pts(lm[12], lm[10], lm[9]) < 150)

        # PENALTY: Fingers spread far apart → V (peace sign), not tight H.
        score -= 0.30 * (abs(lm[8].x - lm[12].x) > 0.10)

        # PENALTY: Negligible x-travel on either finger → not actually sideways.
        score -= 0.40 * (ix < 0.05 * scale)
        score -= 0.40 * (mx < 0.05 * scale)

        return max(0.0, min(1.0, score))

    def _score_i(self, lm, scale, fs):
        """
        I - Pinky finger extended upward, all others curled.
            Thumb rests on or near the curled fingers.

        Hand shape:
          - ONLY the pinky extends upward
          - Index, middle, ring are all curled toward palm
          - Thumb rests close to the curled fingers / palm center
          - Thumb does NOT extend sideways

        Key differences:
          vs Y : Y has BOTH thumb AND pinky extended outward
          vs B : B has ALL 4 fingers extended; I has only pinky
          vs F : F has 3 fingers up; I has only 1
        """
        if not fs.pinky_ext:
            return 0.0

        score = 0.0

        score += 0.40

        score += 0.20 * (not (fs.index_ext or fs.middle_ext or fs.ring_ext))

        score += 0.15 * (lm[20].y < lm[17].y - 0.04)

        score += 0.10 * (lm[20].y < min(lm[8].y, lm[12].y, lm[16].y) - 0.05)

        cx, cy = self._palm_center(lm)
        thumb_to_palm = math.hypot(lm[4].x - cx, lm[4].y - cy)
        score += 0.05 * (thumb_to_palm < 0.18 * scale)

        score -= 0.25 * fs.thumb_side
        score -= 0.30 * (self._count_extended(fs) >= 2)

        return max(0.0, min(1.0, score))

    def _score_k(self, lm, scale, fs):
        """
        K - Index and middle fingers extended UP (index may angle forward),
            thumb tip sits BETWEEN the two raised fingers.
            Ring and pinky curled.

        vs V/U : spread fingers, NO thumb between them
        vs D   : only one finger up
        vs F   : F has middle/ring/pinky up, thumb touches index tip
        vs B   : all four fingers up
        """
        if not fs.index_ext or not fs.middle_ext:
            return 0.0
        if fs.ring_ext or fs.pinky_ext:
            return 0.0

        score = 0.0

        score += 0.20 * (lm[8].y < lm[5].y - 0.04 * scale)
        score += 0.20 * (lm[12].y < lm[9].y - 0.04 * scale)

        idx_x = lm[5].x
        mid_x = lm[9].x
        x_lo  = min(idx_x, mid_x) - 0.02
        x_hi  = max(idx_x, mid_x) + 0.02
        thumb_between_x = x_lo < lm[4].x < x_hi
        score += 0.25 * thumb_between_x

        thumb_height_ok = (lm[9].y + 0.02 > lm[4].y > lm[8].y - 0.05 * scale)
        score += 0.15 * thumb_height_ok

        score += 0.10 * (fs.ring_deep_curl and fs.pinky_deep_curl)
        score += 0.10 * (abs(lm[8].x - lm[12].x) < 0.12)

        score -= 0.50 * (not thumb_between_x)
        score -= 0.30 * (abs(lm[8].x - lm[12].x) > 0.15)
        score -= 0.40 * fs.thumb_side
        score -= 0.40 * (self._count_extended(fs) < 2)

        return max(0.0, min(1.0, score))

    def _score_l(self, lm, scale, fs):
        """
        L - Index pointing straight UP, thumb extended SIDEWAYS.
            Middle/ring/pinky all curled. L shape in profile.

        vs D   : D has thumb looping to middle finger, NOT sideways
        vs K   : K has two fingers up + thumb between them
        vs G   : G index points SIDEWAYS; L index points UPWARD
        """
        if not fs.index_ext:
            return 0.0
        if not fs.thumb_side:
            return 0.0

        score = 0.0

        score += 0.30 * (lm[8].y < lm[5].y - 0.05 * scale)
        score += 0.25 * (lm[4].x < lm[2].x - 0.05)

        three_curled = not fs.middle_ext and not fs.ring_ext and not fs.pinky_ext
        score += 0.25 * three_curled

        angle_l = self._angle_3pts(lm[8], lm[2], lm[4])
        score += 0.20 * (70 < angle_l < 120)

        score -= 0.50 * fs.middle_ext
        score -= 0.50 * (not fs.thumb_side)
        score -= 0.40 * (abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y))

        return max(0.0, min(1.0, score))

    def _score_m(self, lm, scale, fs):
        """
        M - Three fingers (index, middle, ring) folded OVER the thumb.
            The thumb is tucked beneath those three fingers, with its tip
            peeking out near the PINKY side of the fist.

        Physical geometry
        -----------------
        - Index, middle, AND ring are deeply curled over the palm.
        - Pinky is also curled (all four fingers down).
        - The thumb sneaks under the first THREE fingers, so its tip
          appears close to the RING-PINKY gap on the pinky side.
        - Because the thumb is beneath three fingers, thumb tip y is
          BELOW the thumb MCP (lm[2].y) — it is fully tucked under.
        - Thumb tip x is nearest to the midpoint of middle-MCP and
          pinky-MCP (the far / pinky side), NOT the middle-ring midpoint
          (which is the N slot).

        Key differences
        ---------------
          vs N : N thumb only tucks under TWO fingers → tip appears nearer
                 the middle-ring gap (index side of fist).
          vs A : A thumb is HIGH and visible BESIDE the fist, not under it.
          vs E : E is a CLAW — fingertips spread forward toward the thumb,
                 which itself is NOT tucked under its own MCP.
          vs S : S thumb crosses OVER the FRONT (nail side) of the fist.
        """
        # HARD GATE: closed fist — no fingers pointing upward.
        if fs.index_ext or fs.middle_ext or fs.ring_ext or fs.pinky_ext:
            return 0.0

        # HARD GATE: Thumb must be tucked UNDER its own MCP — this is the
        # single most reliable separator from A (thumb beside) and E (claw).
        if lm[4].y <= lm[2].y + 0.01:
            return 0.0

        score = 0.0

        # ── POSITIVE: tight fist ─────────────────────────────────────────────
        # At least 3 of the 4 fingers must be fully curled below their MCPs.
        deep_curled = self._count_deep_curled(fs)
        score += 0.20 * (deep_curled >= 3)

        # Compact fist: index tip to pinky tip span is small.
        score += 0.10 * (self._distance(lm[8], lm[20]) < 0.40 * scale)

        # ── POSITIVE: thumb tucked under (vertical confirmation) ─────────────
        # Thumb tip is clearly BELOW the thumb MCP — fully hidden beneath fingers.
        score += 0.25 * (lm[4].y > lm[2].y + 0.04)

        # Thumb tip is also below the index MCP (palm baseline) — deep tuck.
        score += 0.10 * (lm[4].y > lm[5].y + 0.02)

        # ── POSITIVE: thumb x-slot (M = pinky side) ─────────────────────────
        # We use a nearest-center approach: compare distance from thumb tip x
        # to the M-gap midpoint (middle MCP ↔ pinky MCP) vs the N-gap midpoint
        # (middle MCP ↔ ring MCP).  Whichever is smaller wins.
        # This avoids the compressed-MCP problem with fixed range checks.
        m_center = (lm[9].x + lm[17].x) / 2   # middle ↔ pinky midpoint
        n_center = (lm[9].x + lm[13].x) / 2   # middle ↔ ring midpoint
        dist_to_m = abs(lm[4].x - m_center)
        dist_to_n = abs(lm[4].x - n_center)
        thumb_nearest_m = dist_to_m < dist_to_n

        score += 0.30 * thumb_nearest_m

        # Soft bonus: thumb is genuinely on the pinky side (past ring MCP in x).
        # lm[13] = ring MCP; on a mirrored frame the pinky side has LARGER x.
        score += 0.10 * (lm[4].x > lm[13].x - 0.01)

        # ── PENALTIES ────────────────────────────────────────────────────────
        # Wrong x-slot: thumb is nearer to N gap → this is N, not M.
        score -= 0.55 * (not thumb_nearest_m)

        # Thumb NOT tucked under → A (beside) or E (claw).
        score -= 0.60 * (lm[4].y <= lm[5].y + 0.01)

        # Thumb extended sideways → not a fist at all.
        score -= 0.40 * fs.thumb_side

        # Claw shape: fingertips clustering near thumb tip → E, not M.
        tips_near_thumb = sum(
            1 for t in [lm[8], lm[12], lm[16], lm[20]]
            if self._distance(t, lm[4]) < 0.22 * scale
        )
        score -= 0.30 * (tips_near_thumb >= 3)

        # Thumb NOT below its own MCP: belt-and-suspenders guard vs A/E.
        score -= 0.50 * (lm[4].y <= lm[2].y + 0.01)

        return max(0.0, min(1.0, score))

    def _score_n(self, lm, scale, fs):
        """
        N - Two fingers (index, middle) folded OVER the thumb.
            The thumb is tucked beneath those TWO fingers, with its tip
            peeking out near the RING side of the fist (one slot toward
            the index compared with M).

        Physical geometry
        -----------------
        - Index and middle are deeply curled over the palm.
        - Ring and pinky also curl down (all four fingers down).
        - The thumb sneaks under only the first TWO fingers, so its tip
          appears closest to the MIDDLE-RING gap.
        - Thumb tip y is BELOW thumb MCP — tucked under the fist.
        - Thumb tip x is nearest to the midpoint of middle-MCP and
          ring-MCP (the N slot), not the middle-pinky midpoint (M slot).

        Key differences
        ---------------
          vs M : M thumb tucks under THREE fingers → tip on the pinky side.
          vs A : A thumb is HIGH beside the fist, not tucked under it.
          vs E : E is a claw — thumb NOT under its own MCP.
          vs S : S thumb is over the FRONT of the fist.
        """
        if fs.index_ext or fs.middle_ext or fs.ring_ext or fs.pinky_ext:
            return 0.0

        # HARD GATE: Thumb must be tucked UNDER its own MCP.
        if lm[4].y <= lm[2].y + 0.01:
            return 0.0

        score = 0.0

        # ── POSITIVE: tight fist ─────────────────────────────────────────────
        deep_curled = self._count_deep_curled(fs)
        score += 0.20 * (deep_curled >= 3)

        score += 0.10 * (self._distance(lm[8], lm[20]) < 0.40 * scale)

        # ── POSITIVE: thumb tucked under ─────────────────────────────────────
        score += 0.25 * (lm[4].y > lm[2].y + 0.04)

        score += 0.10 * (lm[4].y > lm[5].y + 0.02)

        # ── POSITIVE: thumb x-slot (N = ring side) ───────────────────────────
        m_center = (lm[9].x + lm[17].x) / 2
        n_center = (lm[9].x + lm[13].x) / 2
        dist_to_m = abs(lm[4].x - m_center)
        dist_to_n = abs(lm[4].x - n_center)
        thumb_nearest_n = dist_to_n < dist_to_m

        score += 0.30 * thumb_nearest_n

        # Soft bonus: thumb x is on the index/middle side (before ring MCP).
        score += 0.10 * (lm[4].x < lm[13].x + 0.01)

        # ── PENALTIES ────────────────────────────────────────────────────────
        # Wrong x-slot → M.
        score -= 0.55 * (not thumb_nearest_n)

        # Thumb NOT tucked under → A or E.
        score -= 0.60 * (lm[4].y <= lm[5].y + 0.01)

        score -= 0.40 * fs.thumb_side

        tips_near_thumb = sum(
            1 for t in [lm[8], lm[12], lm[16], lm[20]]
            if self._distance(t, lm[4]) < 0.22 * scale
        )
        score -= 0.30 * (tips_near_thumb >= 3)

        # Thumb NOT below its own MCP.
        score -= 0.50 * (lm[4].y <= lm[2].y + 0.01)

        return max(0.0, min(1.0, score))

    def _score_o(self, lm, scale, fs):
        """
        O - All fingers curve inward to meet the thumb, forming a round O shape.
            There is a visible hollow space inside the circle.

        vs F : F has 3 fingers straight up; O has all curved in
        vs C : C has a larger OPEN gap between thumb and index; O closes the gap
        vs A/E/S : tight fists; O has an open circle with space inside
        vs M/N : M/N are closed fists with thumb tucked under; O keeps thumb visible
        """
        # HARD GATE: no fully extended fingers — O has no straight spikes.
        if self._count_extended(fs) >= 2:
            return 0.0

        score = 0.0

        # POSITIVE: All fingertips close to thumb tip — this is the defining O circle.
        # Use tighter threshold (0.20 instead of 0.24) to distinguish from C/A (open gap/closed fist).
        thumb = lm[4]
        tips_near_thumb = sum(
            1 for t in [lm[8], lm[12], lm[16], lm[20]]
            if self._distance(t, thumb) < 0.20 * scale
        )
        score += 0.60 * (tips_near_thumb / 4)

        # POSITIVE: Thumb and index tips are close — top of the O is closed.
        score += 0.35 * (self._distance(lm[4], lm[8]) < 0.20 * scale)

        # POSITIVE: All fingertips at similar height — circular symmetry.
        tip_ys = [lm[8].y, lm[12].y, lm[16].y, lm[20].y]
        score += 0.25 * ((max(tip_ys) - min(tip_ys)) < 0.12)

        # POSITIVE: Some space inside — not a squashed flat fist.
        # Thumb IP to palm center distance confirms the thumb is arched outward.
        cx, cy = self._palm_center(lm)
        thumb_arch = math.hypot(lm[3].x - cx, lm[3].y - cy)
        score += 0.15 * (thumb_arch > 0.15 * scale)

        # POSITIVE: Thumb NOT tucked under its own MCP — O keeps the thumb arched.
        score += 0.10 * (lm[4].y <= lm[2].y + 0.03)

        # PENALTY: Any finger fully extended -> F or B, not O.
        score -= 0.60 * (self._count_extended(fs) >= 1)

        # PENALTY: Tight fist -> A/S/E.
        score -= 0.50 * (self._count_deep_curled(fs) >= 3)

        # PENALTY: Large thumb-index gap -> C not O.
        score -= 0.50 * (self._distance(lm[4], lm[8]) > 0.28 * scale)

        # PENALTY: Thumb tucked under its own MCP -> M/N, not O.
        score -= 0.50 * (lm[4].y > lm[2].y + 0.03)

        # PENALTY: Fingertips NOT near thumb -> C (open gap) or another sign.
        score -= 0.50 * (tips_near_thumb <= 1)

        return max(0.0, min(1.0, score))

    def _score_p(self, lm, scale, fs):
        """
        P - Like K rotated downward: index and middle point DOWN and FORWARD,
            thumb between them, wrist above the knuckles.

        For P the hand is oriented with fingers pointing downward (away from
        the signer) or diagonally down-forward.  Because of this orientation:
          - Index and middle TIPS are BELOW (higher y) than their MCPs
          - OR the tips have moved toward the camera (negative z relative to MCP)
          - Wrist is above the finger MCPs (lower y)
          - Thumb sits between the two lowered fingers

        vs K   : K tips are ABOVE MCPs; P tips are BELOW or camera-pointing
        vs G/H : G/H point sideways; P points downward or at camera
        """
        score = 0.0

        # Compute tip-relative positions for index and middle.
        idx_tip_below_mcp = lm[8].y  > lm[5].y   # index tip lower than its MCP
        mid_tip_below_mcp = lm[12].y > lm[9].y   # middle tip lower than its MCP

        # z-depth: tip closer to camera than MCP means pointing downward/forward.
        idx_toward_cam = self._finger_points_at_camera(lm[8],  lm[5])
        mid_toward_cam = self._finger_points_at_camera(lm[12], lm[9])

        # Each finger qualifies as "P-directed" if it points down OR at camera.
        idx_p = idx_tip_below_mcp or idx_toward_cam
        mid_p = mid_tip_below_mcp or mid_toward_cam

        # HARD GATE: At least BOTH fingers must be directed downward or at camera.
        # (P requires clear downward orientation, not just K-like upward)
        if not (idx_p and mid_p):
            return 0.0

        # HARD GATE: Wrist must be above (lower y than) the finger MCPs.
        # This is the "pointing down" orientation baseline.
        if lm[0].y > lm[5].y + 0.08 or lm[0].y > lm[9].y + 0.08:
            return 0.0

        # POSITIVE: Index tip clearly below its MCP (downward direction).
        score += 0.25 * (lm[8].y > lm[5].y + 0.04 * scale)

        # POSITIVE: Middle tip clearly below its MCP.
        score += 0.25 * (lm[12].y > lm[9].y + 0.04 * scale)

        # POSITIVE: z-depth signals — fingers pointing at/toward camera.
        score += 0.15 * idx_toward_cam
        score += 0.15 * mid_toward_cam

        # POSITIVE: Thumb between index and middle MCPs in x (same slot as K).
        idx_x = lm[5].x
        mid_x = lm[9].x
        x_lo  = min(idx_x, mid_x) - 0.03
        x_hi  = max(idx_x, mid_x) + 0.03
        thumb_between_x = x_lo < lm[4].x < x_hi
        score += 0.25 * thumb_between_x

        # POSITIVE: Ring and pinky curled.
        score += 0.20 * (not fs.ring_ext and not fs.pinky_ext)

        # POSITIVE: Both tips clearly below MCPs (strong indicator of downward point).
        # This is the most important distinguisher from K.
        both_below = (lm[8].y > lm[5].y + 0.03) and (lm[12].y > lm[9].y + 0.03)
        score += 0.35 * both_below

        # PENALTY: Thumb not between fingers.
        score -= 0.60 * (not thumb_between_x)

        # PENALTY: Both tips pointing UPWARD → K not P. This is critical distinction.
        score -= 0.90 * (lm[8].y < lm[5].y - 0.03 and lm[12].y < lm[9].y - 0.03)

        # PENALTY: Extra fingers extended (ring or pinky up).
        score -= 0.50 * (fs.ring_ext or fs.pinky_ext)

        # PENALTY: If index is not below its MCP, not pointing down.
        score -= 0.40 * (lm[8].y <= lm[5].y)

        # PENALTY: If middle is not below its MCP, not pointing down.
        score -= 0.40 * (lm[12].y <= lm[9].y)

        return max(0.0, min(1.0, score))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def classify(self, landmarks, handedness: Optional[str] = None,
                 target_letter: Optional[str] = None) -> Optional[Dict]:
        """
        Classify one frame of hand landmarks as an ASL letter.

        Parameters
        ----------
        landmarks     : list of 21 MediaPipe NormalizedLandmark objects
        handedness    : "Left" | "Right" | None
                        Right-hand landmarks are mirrored before scoring so
                        all scorers can assume the thumb is on the left side.
        target_letter : str | None
                        When provided (play mode), ONLY the scorer for that
                        letter runs.  This prevents other letters from
                        accidentally out-scoring the target, which would happen
                        in full competition mode when the hand is mid-transition.

        Returns
        -------
        dict  {"letter": str, "confidence": float, "scores": dict}
          or  None  if no letter clears min_confidence.

        "scores" always reflects what was actually scored:
          - In targeted mode it contains only the one target letter.
          - In full mode it contains all letter scores.
        """
        if landmarks is None or len(landmarks) < 21:
            return None

        # Mirror right-hand landmarks so all rules work for both hands.
        if handedness == "Right":
            landmarks = self._mirror_landmarks_x(landmarks)

        # Compute the normalisation scale.  Bail if the hand is too small
        # (e.g. partially off-screen) to get reliable measurements.
        scale = self._hand_scale(landmarks)
        if scale < 0.01:
            return None

        # Pre-compute all finger states once.
        fs = self._get_finger_states(landmarks, scale)

        # Map of letter -> scorer function.
        scorers = {
            "A": self._score_a,
            "B": self._score_b,
            "C": self._score_c,
            "D": self._score_d,
            "E": self._score_e,
            "F": self._score_f,
            "G": self._score_g,
            "H": self._score_h,
            "I": self._score_i,
            "K": self._score_k,
            "L": self._score_l,
            "M": self._score_m,
            "N": self._score_n,
            "O": self._score_o,
            "P": self._score_p,
        }

        # TARGETED MODE (play mode)
        if target_letter and target_letter in scorers:
            s = scorers[target_letter](landmarks, scale, fs)
            if s < self.min_confidence:
                return None
            return {
                "letter":     target_letter,
                "confidence": round(s, 3),
                "scores":     {target_letter: s},
            }

        # FULL COMPETITION MODE (debug mode)
        scores = {k: fn(landmarks, scale, fs) for k, fn in scorers.items()}
        best   = max(scores, key=scores.get)
        if scores[best] < self.min_confidence:
            return None

        # Uncomment to print all scores every frame for debugging:
        # print(f"{best} ({scores[best]:.2f}) | " +
        #       " ".join(f"{k}:{v:.2f}" for k, v in sorted(scores.items())))

        return {
            "letter":     best,
            "confidence": round(scores[best], 3),
            "scores":     scores,
        }