"""
asl_classifier_letters.py
=========================
Scoring-based static ASL letter classifier.

Active letters:  A  B  C  D  E  F  G  H  I
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
    # True = finger tip is ABOVE (lower y than) its PIP -> finger is extended/up
    index_ext:        bool
    middle_ext:       bool
    ring_ext:         bool
    pinky_ext:        bool

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
        return FingerState(
            # Extended = tip above PIP (tip.y < pip.y)
            index_ext        = self._is_extended(lm[8],  lm[6]),   # index tip vs index PIP
            middle_ext       = self._is_extended(lm[12], lm[10]),  # middle tip vs middle PIP
            ring_ext         = self._is_extended(lm[16], lm[14]),  # ring tip vs ring PIP
            pinky_ext        = self._is_extended(lm[20], lm[18]),  # pinky tip vs pinky PIP

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
          vs E : Thumb in A is clearly separate from all fingertips.
        """
        score = 0.0

        # HARD GATES
        # If index or middle finger is sticking up, this is not a fist at all.
        if fs.index_ext or fs.middle_ext:
            return 0.0

        # POSITIVE: All 4 fingers are curled (none are extended upward).
        # This is the base fist shape required for A.
        all_curled = not any([fs.index_ext, fs.middle_ext, fs.ring_ext, fs.pinky_ext])
        score += 0.30 * all_curled

        # POSITIVE: Compact fist - index tip (lm[8]) to pinky tip (lm[20]) distance is small.
        # A wide spread means fingers are not actually folded into a fist.
        score += 0.20 * (self._distance(lm[8], lm[20]) < 0.45 * scale)

        # POSITIVE: Thumb is HIGH - at or slightly above the index MCP (lm[5]).
        # This is THE defining A geometry: thumb stands upright beside the fist.
        # (E has the thumb low/tucked; A has it elevated.)
        score += 0.15 * (lm[4].y <= lm[5].y + 0.02)

        # POSITIVE: Thumb is beside the fist horizontally - close in x to the index MCP.
        # Prevents scoring if the thumb is pointing away from the hand sideways.
        score += 0.25 * (abs(lm[4].x - lm[5].x) < 0.12)

        # POSITIVE: Thumb tip is NOT close to any fingertip.
        # E has tips near the thumb (claw shape); A keeps the thumb clear.
        thumb_clear = all(self._distance(lm[4], lm[t]) > 0.18 * scale
                          for t in [8, 12, 16, 20])
        score += 0.30 * thumb_clear

        # PENALTY: Thumb is tucked LOW (well below the index MCP) -> this is E, not A.
        score -= 0.50 * (lm[4].y > lm[5].y + 0.06)

        # PENALTY: Thumb tip is ABOVE the index fingertip -> thumb crosses over the top
        # of the fist, which is the S sign, not A.
        score -= 0.30 * (lm[4].y < lm[8].y - 0.01)

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

        # POSITIVE: All 4 fingers must be extended upward. Primary B feature.
        score += 0.40 * all([fs.index_ext, fs.middle_ext, fs.ring_ext, fs.pinky_ext])

        # POSITIVE: Finger tips are all above their respective MCPs - confirms upward direction.
        # Using 3 fingers (index, middle, pinky) as a representative sample.
        score += 0.20 * (lm[8].y  < lm[5].y and   # index tip above index MCP
                         lm[12].y < lm[9].y and    # middle tip above middle MCP
                         lm[20].y < lm[17].y)       # pinky tip above pinky MCP

        # POSITIVE: Fingers are tightly grouped - small horizontal spread from index to pinky.
        # A spread > 0.10 normalised units suggests a W or open-palm pose.
        spread = abs(lm[8].x - lm[20].x)
        score += 0.20 * (spread < 0.10)

        # POSITIVE: Thumb is tucked - thumb tip (lm[4]) is below the index PIP (lm[6]).
        # This means the thumb has been folded across the palm, not extended.
        score += 0.20 * (lm[4].y > lm[6].y)

        # PENALTY: Thumb sticking out sideways = open hand or Y sign, not B.
        score -= 0.30 * fs.thumb_side

        # PENALTY: Fingers are spread wide -> looks like W, not tight B.
        score -= 0.20 * (spread > 0.14)

        # PENALTY: Thumb is far from the palm center -> it hasn't been tucked properly.
        # For a true B, the thumb should lie flat over the palm.
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
          vs O : O has thumb and index touching; C has a clear gap
          vs A : A is a closed fist; C is open with visible space
          vs G : G has a single finger pointing sideways cleanly;
                 C has ALL fingers bent in a curve
        """
        score = 0.0

        # POSITIVE: Count how many fingers are in the "partial curve zone".
        # A finger is partially curved when its tip is at roughly the same height
        # as its PIP joint (tip.y > pip.y - 0.02), meaning it has some bend
        # without being straight or fully folded.
        partial_curve_count = sum(
            1 for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]
            if lm[tip_idx].y > lm[pip_idx].y - 0.02
        )
        score += 0.45 * (partial_curve_count / 4)  # fraction of fingers curving

        # POSITIVE: The gap between thumb tip and index tip should be medium-sized -
        # wide enough to be clearly open (not O), narrow enough to be a C (not open hand).
        # Range: 28%-70% of hand scale.
        dist_thumb_index = self._distance(lm[4], lm[8])
        score += 0.35 * (0.28 * scale < dist_thumb_index < 0.70 * scale)

        # POSITIVE: Thumb and index tip are at a similar height - the C opens sideways,
        # not diagonally. A height difference > 0.12 would suggest a different pose.
        score += 0.20 * (abs(lm[4].y - lm[8].y) < 0.12)

        # POSITIVE: Thumb has a natural curve: the distance from thumb tip (lm[4]) to thumb
        # IP joint (lm[3]) is within an expected "curved but not flat" range.
        score += 0.25 * (0.15 * scale < self._distance(lm[4], lm[3]) < 0.45 * scale)

        # PENALTY: 3+ fingers fully extended = this is B (straight fingers), not C (curved).
        score -= 0.35 * (self._count_extended(fs) >= 3)

        # PENALTY: 3+ fingers deeply curled into fist = this is A or S, not the open C shape.
        score -= 0.45 * (self._count_deep_curled(fs) >= 3)

        # PENALTY: Large horizontal (x-axis) travel from MCP to tip on index or middle
        # means a finger is pointing sideways, which is G or H territory, not C.
        idx_sideways = abs(lm[8].x  - lm[5].x) > 0.12 * scale
        mid_sideways = abs(lm[12].x - lm[9].x) > 0.12 * scale
        score -= 0.25 * (idx_sideways or mid_sideways)

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

        # POSITIVE: Index finger is extended - the upward spike of the D.
        score += 0.25 * fs.index_ext

        # POSITIVE: Index tip is clearly above its MCP - confirming it points upward,
        # not just slightly extended. Threshold: 5% of hand scale above MCP.
        score += 0.15 * (lm[8].y < lm[5].y - 0.05)

        # POSITIVE: Middle, ring, and pinky are all curled down (not extended).
        # These form the curved body of the D.
        score += 0.25 * (not fs.middle_ext and not fs.ring_ext and not fs.pinky_ext)

        # POSITIVE: Thumb tip is close to the middle finger, forming the loop of the D.
        # Distance threshold: 28% of hand scale.
        score += 0.25 * (self._distance(lm[4], lm[12]) < 0.28 * scale)

        # POSITIVE: Index tip is clearly higher than the middle fingertip.
        # Confirms the index is the only thing pointing up.
        score += 0.10 * (lm[8].y < lm[12].y - 0.05)

        # PENALTY: Thumb extended sideways = L sign (index up, thumb sideways), not D.
        score -= 0.30 * fs.thumb_side

        # PENALTY: More than one finger extended = not the single-spike D shape.
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
          vs A : A has thumb HIGH beside the fist;
                 E has thumb LOW tucked under the curled tips
          vs S : S thumb crosses OVER the front of the fist
          vs C : C is open; E is closed with tips near thumb
        """
        score = 0.0

        # POSITIVE: All 4 fingers are curled (none point upward).
        # This is shared with A and S, so we need the thumb position to separate them.
        all_curled = not any([fs.index_ext, fs.middle_ext, fs.ring_ext, fs.pinky_ext])
        score += 0.25 * all_curled

        # POSITIVE: Each fingertip should be close to the thumb tip.
        # In E, the fingers claw inward toward the thumb - this is the defining
        # claw geometry. Threshold: 30% of hand scale.
        thumb = lm[4]
        tips_near_thumb = sum(
            1 for t in [lm[8], lm[12], lm[16], lm[20]]
            if self._distance(t, thumb) < 0.30 * scale
        )
        score += 0.40 * (tips_near_thumb / 4)  # fraction of tips near thumb

        # POSITIVE: All fingertips are at a similar height - the flat claw alignment.
        # A round fist (A/S) has more height variation between fingertips.
        tip_ys = [lm[8].y, lm[12].y, lm[16].y, lm[20].y]
        score += 0.20 * ((max(tip_ys) - min(tip_ys)) < 0.10)

        # POSITIVE: Thumb tip is LOW - at or below the middle MCP (lm[9]) level.
        # This is THE differentiator from A (where thumb is high).
        score += 0.15 * (lm[4].y < lm[9].y + 0.05)

        # PENALTY: Thumb is clearly beside the fist (high and close to index PIP) -> that's A.
        # Combination: thumb is above index tip AND close to index PIP side.
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

        # Distance between thumb tip (lm[4]) and index tip (lm[8]).
        d = self._distance(lm[4], lm[8])

        # Whether middle, ring, and pinky are all pointing up.
        three_up = fs.middle_ext and fs.ring_ext and fs.pinky_ext

        # POSITIVE: Thumb and index tips are very close - forming the circle.
        # Threshold: 20% of hand scale (tight contact range).
        score += 0.45 * (d < 0.20 * scale)

        # POSITIVE: Middle, ring, pinky are all extended upward - the three raised fingers.
        score += 0.35 * three_up

        # POSITIVE: Index is NOT extended upward - it has curled down to meet the thumb.
        score += 0.20 * (not fs.index_ext)

        # PENALTY: Index is extended = circle cannot be formed, looks like K or B.
        score -= 0.40 * fs.index_ext

        # PENALTY: Thumb and index are far apart = not touching = not F.
        score -= 0.20 * (d > 0.30 * scale)

        # PENALTY: The three upper fingers are not raised = could be I or another sign.
        score -= 0.35 * (not three_up)

        return max(0.0, min(1.0, score))

    def _score_g(self, lm, scale, fs):
        """
        G - Index finger pointing SIDEWAYS and HORIZONTAL.
            Back of hand faces the camera.

        Hand shape:
          - ONLY the index finger extends, pointing to the side
          - Index tip is at roughly the same HEIGHT as its MCP
            (truly horizontal, not diagonal)
          - Middle, ring, pinky are all curled into the palm
          - Thumb stays below or beside (does not extend sideways)

        Key differences:
          vs D : D points UPWARD; G points SIDEWAYS
          vs H : H has TWO fingers pointing sideways; G has one
          vs C : C fingers are CURVED; G index is cleanly extended
        """
        # HARD GATE: Index must be extended - without this G is impossible.
        if not fs.index_ext:
            return 0.0

        # HARD GATE: Middle, ring, and pinky must ALL be curled.
        # If any of them is up, this is H (two fingers) or B/W (more fingers).
        if fs.middle_ext or fs.ring_ext or fs.pinky_ext:
            return 0.0

        score = 0.0

        # x-travel: how far the index tip (lm[8]) has moved horizontally
        # from the index MCP (lm[5]). Large x-travel = pointing sideways.
        x_travel = abs(lm[8].x - lm[5].x)

        # POSITIVE: Index tip has clear horizontal travel from its MCP.
        # Threshold: 12% of hand scale.  Below this the finger is just bent
        # slightly, not pointing sideways.
        score += 0.45 * (x_travel > 0.12 * scale)

        # POSITIVE: Index tip is at nearly the SAME HEIGHT as its MCP - truly horizontal.
        # If tip is much higher than MCP the finger points upward (= D), not sideways.
        # Threshold: +/-12% of frame height.
        score += 0.35 * (abs(lm[8].y - lm[5].y) < 0.12)

        # POSITIVE: x-travel is larger than y-travel - the finger goes more sideways than vertical.
        # Factor of 1.5: x must be 50% more dominant than y.
        score += 0.20 * (x_travel > abs(lm[8].y - lm[5].y) * 1.5)

        # PENALTY: Index tip is well ABOVE its MCP -> finger is pointing upward, which is D.
        score -= 0.50 * (lm[8].y < lm[5].y - 0.10)

        # PENALTY: Very little x-travel -> finger is not pointing sideways at all.
        score -= 0.40 * (x_travel < 0.08 * scale)

        return max(0.0, min(1.0, score))

    def _score_h(self, lm, scale, fs):
        """
        H - Index AND middle fingers both pointing SIDEWAYS.
            Back of hand faces the camera.

        Hand shape:
          - Index AND middle both extend horizontally to the side
          - Both tips are at roughly the same height as their MCPs
          - The two fingers are held TOGETHER (not spread)
          - Ring and pinky are curled

        Key differences:
          vs G : G has ONE sideways finger; H has TWO
          vs U : U points UPWARD; H points SIDEWAYS
          vs V : V spreads apart; H keeps fingers together
        """
        # HARD GATE: Both index AND middle must be extended.
        # If only index is extended it is G; if neither are, it is A/E/S etc.
        if not fs.index_ext or not fs.middle_ext:
            return 0.0

        score = 0.0

        # x-travel for each of the two fingers (tip distance from their MCP in x).
        ix = abs(lm[8].x  - lm[5].x)   # index x-travel
        mx = abs(lm[12].x - lm[9].x)   # middle x-travel

        # POSITIVE: Index tip has clear sideways travel from its MCP.
        score += 0.25 * (ix > 0.12 * scale)

        # POSITIVE: Middle tip also has clear sideways travel from its MCP.
        score += 0.25 * (mx > 0.12 * scale)

        # POSITIVE: Both tips are at nearly the SAME HEIGHT as their own MCPs - truly horizontal.
        # Without this check a diagonal "up-and-sideways" hand would score high.
        both_level = (abs(lm[8].y  - lm[5].y) < 0.12 and
                      abs(lm[12].y - lm[9].y)  < 0.12)
        score += 0.25 * both_level

        # POSITIVE: x-travel dominates over y-travel for BOTH fingers simultaneously.
        # Confirms both are going sideways, not diagonally.
        idx_horiz = ix > abs(lm[8].y  - lm[5].y) * 1.5
        mid_horiz = mx > abs(lm[12].y - lm[9].y)  * 1.5
        score += 0.10 * (idx_horiz and mid_horiz)

        # POSITIVE: Ring and pinky are curled - only index and middle should be out.
        score += 0.15 * (not fs.ring_ext and not fs.pinky_ext)

        # PENALTY: Both fingers pointing straight UP = U or V sign, not H.
        both_pointing_up = (lm[8].y  < lm[5].y  - 0.08 and
                            lm[12].y < lm[9].y   - 0.08)
        score -= 0.50 * both_pointing_up

        # PENALTY: Little or no x-travel on either finger -> not pointing sideways.
        score -= 0.40 * (ix < 0.08 * scale or mx < 0.08 * scale)

        # PENALTY: Fingers spread far apart in x -> V sign (peace sign), not tight H.
        score -= 0.25 * (abs(lm[8].x - lm[12].x) > 0.10)

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
        # HARD GATE: Pinky must be extended - no pinky means no I.
        if not fs.pinky_ext:
            return 0.0

        score = 0.0

        # POSITIVE: Pinky is extended (already guaranteed by the gate above, but we
        # add weight here so a strong pinky lifts the score meaningfully).
        score += 0.40  # unconditional - we know pinky is up from the gate

        # POSITIVE: Index, middle, and ring are all curled (not extended).
        # This separates I from B (all 4 up) and Y (thumb + pinky out).
        score += 0.20 * (not (fs.index_ext or fs.middle_ext or fs.ring_ext))

        # POSITIVE: Pinky tip is noticeably above its MCP - confirming a clearly raised pinky,
        # not just slightly lifted. Threshold: 4% of frame height.
        score += 0.15 * (lm[20].y < lm[17].y - 0.04)

        # POSITIVE: Pinky tip is clearly above ALL other fingertips.
        # Ensures the pinky is the dominant raised element and not just one of
        # several roughly equal heights.
        score += 0.10 * (lm[20].y < min(lm[8].y, lm[12].y, lm[16].y) - 0.05)

        # POSITIVE: Thumb is resting close to the palm center (within 18% of hand scale).
        # A thumb held near the palm is a natural resting position for I.
        cx, cy = self._palm_center(lm)
        thumb_to_palm = math.hypot(lm[4].x - cx, lm[4].y - cy)
        score += 0.05 * (thumb_to_palm < 0.18 * scale)

        # PENALTY: Thumb extended sideways -> Y sign (pinky + thumb out), not I.
        score -= 0.25 * fs.thumb_side

        # PENALTY: More than one finger extended -> not the single-pinky I shape.
        score -= 0.30 * (self._count_extended(fs) >= 2)

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
          - In full mode it contains all A-I scores.
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
        # Add a new entry here when a new static letter is implemented.
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
        }

        # TARGETED MODE (play mode)
        # Run only the one scorer for the sign the player needs to show.
        # This avoids the situation where transitioning into a new hand shape
        # briefly matches a different letter and resets the hold timer.
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
        # Score every letter and return whichever wins.
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