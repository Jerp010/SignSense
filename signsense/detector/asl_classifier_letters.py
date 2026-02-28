from typing import List, Optional, Dict
import math
from dataclasses import dataclass


# ---------------------------------------------------
# ASL Scoring-Based Static Alphabet Classifier
# ---------------------------------------------------
#
# ACTIVE LETTERS: A, B, C, D, E, F, G, H
# All other letters return 0.0 until re-enabled.
#
# v4 Rebuild:
# - Scoped to A-H for stability before expanding
# - Unified threshold approach: ALL distances use scale-normalized values
# - ALL Y comparisons use raw normalized coords (consistent — MediaPipe y is 0..1)
# - Replaced unreliable _finger_curl_ratio with simple tip-vs-PIP and tip-vs-MCP checks
# - Added _is_extended(tip, pip) and _is_curled(tip, pip) helpers (single source of truth)
# - Reduced inter-letter score overlap by adding PENALTY logic:
#     each letter checks that competing conditions are NOT true
# - Dominant letter audit: B and H had too-easy paths; added spread/orientation gates
# ---------------------------------------------------
#
# v3 features preserved:
# - FingerState precomputation
# - _angle_3pts() for hook/bend detection
# - _hand_scale() normalization
# - Right-hand mirroring
# - _palm_center()
# ---------------------------------------------------
#
# DEBUGGING GUIDE (see bottom of file)
# ---------------------------------------------------

@dataclass
class FingerState:
    """Precomputed per-finger extended/curled states + thumb orientation."""
    # True = finger tip is above its PIP (extended upward)
    index_ext: bool
    middle_ext: bool
    ring_ext: bool
    pinky_ext: bool
    # True = finger tip is below its MCP (deeply curled into fist)
    index_deep_curl: bool
    middle_deep_curl: bool
    ring_deep_curl: bool
    pinky_deep_curl: bool
    # True = thumb is extended sideways (tip left of thumb base lm[2])
    thumb_side: bool
    # Pre-computed hand scale for use inside scoring
    scale: float


class ASLClassifierLetters:

    def __init__(self) -> None:
        self.min_confidence = 0.6  # minimum score required to return letter

    # ---------------------------------------------------
    # Utility Functions
    # ---------------------------------------------------

    def _distance(self, p1, p2) -> float:
        """Euclidean 2D distance."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _hand_scale(self, lm) -> float:
        """
        Dynamic scaling reference: Wrist (0) to Middle MCP (9).
        Used to normalize all distance thresholds.
        """
        return self._distance(lm[0], lm[9])

    def _mirror_landmarks_x(self, landmarks: List) -> List:
        """
        Mirrors X coordinates so left-hand logic works for right hand too.
        Call when handedness == 'Right'.
        """
        mirrored = []
        for lm in landmarks:
            new_lm = type(lm)()
            new_lm.x = 1.0 - lm.x
            new_lm.y = lm.y
            new_lm.z = getattr(lm, "z", 0)
            mirrored.append(new_lm)
        return mirrored

    def _is_extended(self, tip, pip) -> bool:
        """
        Finger is extended: tip is ABOVE (lower y) than PIP joint.
        In MediaPipe, y=0 is top of frame, y=1 is bottom.
        """
        return tip.y < pip.y

    def _is_curled(self, tip, pip) -> bool:
        """
        Finger is curled: tip is BELOW (higher y) than PIP joint.
        """
        return tip.y > pip.y

    def _is_deeply_curled(self, tip, mcp) -> bool:
        """
        Finger tip is below its MCP — deeply folded into fist.
        Stronger signal than PIP comparison for fist detection.
        """
        return tip.y > mcp.y

    def _get_finger_states(self, lm, scale: float) -> FingerState:
        """
        Precompute all per-finger states once per frame for efficiency.
        """
        return FingerState(
            index_ext=self._is_extended(lm[8], lm[6]),
            middle_ext=self._is_extended(lm[12], lm[10]),
            ring_ext=self._is_extended(lm[16], lm[14]),
            pinky_ext=self._is_extended(lm[20], lm[18]),
            index_deep_curl=self._is_deeply_curled(lm[8], lm[5]),
            middle_deep_curl=self._is_deeply_curled(lm[12], lm[9]),
            ring_deep_curl=self._is_deeply_curled(lm[16], lm[13]),
            pinky_deep_curl=self._is_deeply_curled(lm[20], lm[17]),
            thumb_side=(lm[4].x < lm[2].x - 0.03),
            scale=scale,
        )

    def _angle_3pts(self, a, b, c) -> float:
        """
        Angle at point B in degrees, formed by A-B-C.
        Useful for joint angles: 180=straight, <120=strongly bent.
        """
        ab = (a.x - b.x, a.y - b.y)
        cb = (c.x - b.x, c.y - b.y)
        dot = ab[0] * cb[0] + ab[1] * cb[1]
        mag_ab = math.hypot(ab[0], ab[1])
        mag_cb = math.hypot(cb[0], cb[1])
        if mag_ab * mag_cb == 0:
            return 180.0
        cos_angle = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
        return math.degrees(math.acos(cos_angle))

    def _palm_center(self, lm):
        """
        Approximate palm center using average of MCPs (5, 9, 13, 17).
        """
        cx = (lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 4
        cy = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4
        return cx, cy

    def _count_extended(self, fs: FingerState) -> int:
        """Count how many of the 4 fingers (not thumb) are extended."""
        return sum([fs.index_ext, fs.middle_ext, fs.ring_ext, fs.pinky_ext])

    def _count_deep_curled(self, fs: FingerState) -> int:
        """Count how many of the 4 fingers are deeply curled into fist."""
        return sum([fs.index_deep_curl, fs.middle_deep_curl, fs.ring_deep_curl, fs.pinky_deep_curl])

    # ---------------------------------------------------
    # LETTER SCORING FUNCTIONS (A-H active)
    # Each returns 0.0–1.0
    # Scoring philosophy:
    #   - Positive evidence adds score
    #   - Conflicting evidence (things that should NOT be true) subtracts
    #   - Weights sum to 1.0 per letter
    # ---------------------------------------------------

    def _score_a(self, lm, scale: float, fs: FingerState) -> float:
        """
        A: Closed fist. Thumb rises alongside the index finger, pointing upward.

        Physical description:
          - All 4 fingers curled into a compact fist (tips below PIPs)
          - Thumb tip is BESIDE the index finger pointing UP (not tucked under like E)
          - Thumb tip sits at roughly the same height as index MCP or above it
          - The thumb does NOT touch the fingertips - it stands apart beside the fist

        Key differentiators:
          - vs E: E has thumb tucked LOW under curled tips (thumb.y > index MCP.y);
                  A has thumb HIGH beside the fist (thumb.y <= index MCP.y)
          - vs C: C is open/curved with a gap; A is a closed compact fist
          - vs S: S wraps thumb OVER fingertips from the front; A keeps thumb beside
        """
        score = 0.0

        # HARD GATE: Any extended finger = not a fist
        if fs.index_ext or fs.middle_ext:
            return 0.0

        # POSITIVE: All 4 fingers curled at PIP
        all_pip_curled = not any([fs.index_ext, fs.middle_ext, fs.ring_ext, fs.pinky_ext])
        score += 0.30 * all_pip_curled

        # POSITIVE: Compact fist - index tip to pinky tip distance is small
        tip_spread = self._distance(lm[8], lm[20])
        compact_fist = tip_spread < 0.45 * scale
        score += 0.20 * compact_fist

        # POSITIVE: Thumb tip is HIGH - at or above the index MCP (lm[5]).
        thumb_high = lm[4].y <= lm[5].y + 0.02
        score += 0.15 * thumb_high

        # POSITIVE: Thumb tip close in x to index MCP - beside the fist
        thumb_beside_x = abs(lm[4].x - lm[5].x) < 0.12
        score += 0.25 * thumb_beside_x

        # POSITIVE: Thumb tip NOT close to any fingertip - thumb stands apart
        thumb_clear_of_tips = all(
            self._distance(lm[4], lm[tip]) > 0.18 * scale
            for tip in [8, 12, 16, 20]
        )
        score += 0.30 * thumb_clear_of_tips

        # PENALTY: Thumb LOW below index MCP = E-style tuck, not A
        thumb_tucked_low = lm[4].y > lm[5].y + 0.06
        score -= 0.50 * thumb_tucked_low

        # PENALTY: Thumb covering fingertips from above = S, not A
        thumb_covering = lm[4].y < lm[8].y - 0.01
        score -= 0.30 * thumb_covering

        return max(0.0, min(1.0, score))

    def _score_b(self, lm, scale: float, fs: FingerState) -> float:
        """
        B: Four fingers fully extended upward, tight together. Thumb tucked across palm.

        Physical description:
          - Index, middle, ring, pinky all pointing straight up
          - Fingers are close together (not spread like W)
          - Thumb is bent across the palm (not sticking out)

        Key differentiators:
          - vs W: all 4 fingers up (W only has 3), and fingers tight together
          - vs open hand: thumb must be tucked (not extended sideways)
          - vs H: B has all 4 fingers, H has only 2
        """
        score = 0.0

        # POSITIVE: All 4 fingers extended
        all_four_ext = all([fs.index_ext, fs.middle_ext, fs.ring_ext, fs.pinky_ext])
        score += 0.4 * all_four_ext

        # POSITIVE: Fingers pointing UP (tips above MCPs, not sideways)
        tips_above_mcps = (
                lm[8].y < lm[5].y and
                lm[12].y < lm[9].y and
                lm[20].y < lm[17].y
        )
        score += 0.2 * tips_above_mcps

        # POSITIVE: Fingers tightly grouped (index-to-pinky x-spread small)
        # Using raw coordinates — MediaPipe x is 0..1 normalized by frame width
        tip_spread = abs(lm[8].x - lm[20].x)
        tight = tip_spread < 0.10
        score += 0.2 * tight

        # POSITIVE: Thumb tucked (tip below index PIP, not sticking out sideways)
        thumb_tucked = lm[4].y > lm[6].y
        score += 0.2 * thumb_tucked

        # PENALTY: Thumb extended sideways = not B
        score -= 0.3 * fs.thumb_side

        # PENALTY: Spread fingers suggest W, not B
        spread_penalty = tip_spread > 0.14
        score -= 0.2 * spread_penalty

        # PENALTY: Thumb not lying on the palm. For a true B the thumb
        # should be tucked over the palm area; an open hand often leaves it
        # off to the side or elevated. We measure thumb distance to the
        # approximate palm center and penalize if it's too far away.
        cx, cy = self._palm_center(lm)
        thumb_to_palm = math.hypot(lm[4].x - cx, lm[4].y - cy)
        score -= 0.3 * (thumb_to_palm > 0.20 * scale)

        return max(0.0, min(1.0, score))

    def _score_c(self, lm, scale: float, fs: FingerState) -> float:
        """
        C: Curved open hand, fingers and thumb forming a C shape.

        Physical description:
          - All fingers partially curved (not straight, not fisted)
          - Thumb and index finger tips face each other with a gap
          - The opening of the C is roughly horizontal
          - Think of holding a can/cup shape

        Key differentiators:
          - vs O: C has a larger gap between thumb and index (O they touch)
          - vs B: C fingers are curved (not straight)
          - vs A: C is open (A is fully closed fist)
        """
        score = 0.0

        # POSITIVE: Fingers partially curved — tips between fully extended and PIP level
        # Each fingertip should be between its MCP and PIP (partial curl zone)
        partial_curve_count = 0
        for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            # Extended = tip above PIP; fully curled = tip below MCP
            # Partial = tip between PIP and MCP in y
            tip_y = lm[tip_idx].y
            pip_y = lm[pip_idx].y
            # tip is below PIP (some curl) but still relatively high
            if tip_y > pip_y - 0.02:  # slightly bent or flat
                partial_curve_count += 1
        score += 0.45 * (partial_curve_count / 4)

        # POSITIVE: Gap between thumb tip and index tip is C-shaped (medium distance)
        dist_th_idx = self._distance(lm[4], lm[8])
        c_gap = 0.28 * scale < dist_th_idx < 0.70 * scale
        score += 0.35 * c_gap

        # POSITIVE: Thumb and index tips at similar y-height (horizontal C opening)
        similar_height = abs(lm[4].y - lm[8].y) < 0.12
        score += 0.20 * similar_height

        # POSITIVE: Thumb has curve (tip not fully extended sideways, not folded)
        thumb_curved = 0.15 * scale < self._distance(lm[4], lm[3]) < 0.45 * scale
        score += 0.25 * thumb_curved

        # PENALTY: Fingers fully extended straight = B, not C
        score -= 0.35 * (self._count_extended(fs) >= 3)

        # PENALTY: Closed fist = A or S, not C
        score -= 0.45 * (self._count_deep_curled(fs) >= 3)

        # PENALTY: Strong sideways (horizontal) travel on index or middle is
        # indicative of G/H rather than a rounded C shape.
        idx_x_travel = abs(lm[8].x - lm[5].x)
        mid_x_travel = abs(lm[12].x - lm[9].x)
        sideways = idx_x_travel > 0.12 * scale or mid_x_travel > 0.12 * scale
        score -= 0.25 * sideways

        return max(0.0, min(1.0, score))

    def _score_d(self, lm, scale: float, fs: FingerState) -> float:
        """
        D: Index finger pointing straight up, other fingers curl into a loop touching the thumb.

        Physical description:
          - Index finger is straight and points up
          - Middle, ring, pinky are curled
          - Thumb tip meets/touches middle finger (forms the loop of D)
          - Index tip is significantly higher than all other fingertips

        Key differentiators:
          - vs G: D points UP, G points sideways
          - vs L: D has thumb meeting middle (loop), L has thumb out sideways
          - vs 1-finger signs: D has specific thumb-to-middle contact
        """
        score = 0.0

        # POSITIVE: Index extended upward
        score += 0.25 * fs.index_ext

        # POSITIVE: Index tip is ABOVE frame midpoint / above its own MCP significantly
        index_pointing_up = lm[8].y < lm[5].y - 0.05
        score += 0.15 * index_pointing_up

        # POSITIVE: Middle, ring, pinky are curled
        others_curled = not fs.middle_ext and not fs.ring_ext and not fs.pinky_ext
        score += 0.25 * others_curled

        # POSITIVE: Thumb touches or is very close to middle finger (the D loop)
        thumb_mid_touch = self._distance(lm[4], lm[12]) < 0.28 * scale
        score += 0.25 * thumb_mid_touch

        # POSITIVE: Index tip is clearly above other curled tips
        idx_above_mid = lm[8].y < lm[12].y - 0.05
        score += 0.1 * idx_above_mid

        # PENALTY: Thumb extended sideways = L, not D
        score -= 0.3 * fs.thumb_side

        # PENALTY: Multiple fingers extended = not D
        score -= 0.2 * (self._count_extended(fs) >= 2)

        return max(0.0, min(1.0, score))

    def _score_e(self, lm, scale: float, fs: FingerState) -> float:
        """
        E: All fingers bent with tips curled toward palm. Thumb tucked under finger pads.

        Physical description:
          - All 4 fingertips bend toward the palm (like clawing downward)
          - Thumb is tucked under — its tip is near/under the bent fingers
          - From the side it looks like a flattened claw
          - Important: NOT a tight fist (A/S) — there's a slight gap, fingers aren't fully folded

        Key differentiators:
          - vs A: E fingertips are near thumb; A is a closed fist with thumb beside
          - vs S: E has all tips closer to thumb; S thumb covers tips from outside
          - vs O: E all fingers involved; O just thumb meets index in a circle
        """
        score = 0.0

        # POSITIVE: All 4 fingers curled (tips below PIP)
        all_curled = not any([fs.index_ext, fs.middle_ext, fs.ring_ext, fs.pinky_ext])
        score += 0.25 * all_curled

        # POSITIVE: Each fingertip is close to thumb tip (claw toward thumb)
        thumb = lm[4]
        tip_close_count = sum(
            self._distance(tip, thumb) < 0.30 * scale
            for tip in [lm[8], lm[12], lm[16], lm[20]]
        )
        score += 0.40 * (tip_close_count / 4)

        # POSITIVE: Fingertips are at roughly similar heights (flat claw, not rounded fist)
        tip_ys = [lm[8].y, lm[12].y, lm[16].y, lm[20].y]
        flat_alignment = (max(tip_ys) - min(tip_ys)) < 0.10
        score += 0.20 * flat_alignment

        # POSITIVE: Thumb tip is tucked under (at or above middle MCP level)
        thumb_under = lm[4].y < lm[9].y + 0.05
        score += 0.15 * thumb_under

        # PENALTY: Thumb clearly beside fist = A, not E
        thumb_beside = lm[4].y > lm[8].y + 0.02 and self._distance(lm[4], lm[6]) < 0.20 * scale
        score -= 0.25 * thumb_beside

        return max(0.0, min(1.0, score))

    def _score_f(self, lm, scale: float, fs: FingerState) -> float:
        """
        F: Index and thumb form a circle (touching), middle/ring/pinky extended upward.

        Physical description:
          - Thumb tip and index fingertip are touching (or very close)
          - Index is curled down to meet the thumb
          - Middle, ring, pinky are extended straight up
          - The three extended fingers are spread slightly

        Key differentiators:
          - vs O: F has 3 fingers extended; O has all fingers curved
          - vs D: F has thumb-to-INDEX contact; D has thumb-to-MIDDLE
          - vs K: F has index curled down; K has index extended up
        """
        score = 0.0

        # POSITIVE: Thumb tip touching index tip (the circle)
        th_idx_dist = self._distance(lm[4], lm[8])
        touching = th_idx_dist < 0.20 * scale
        score += 0.45 * touching

        # POSITIVE: Middle, ring, pinky extended
        three_up = fs.middle_ext and fs.ring_ext and fs.pinky_ext
        score += 0.35 * three_up

        # POSITIVE: Index NOT extended (curled down to meet thumb)
        index_down = not fs.index_ext
        score += 0.20 * index_down

        # PENALTY: Index extended = not the F circle
        score -= 0.40 * fs.index_ext

        # PENALTY: Thumb-index too far apart = might be C or open hand
        score -= 0.20 * (th_idx_dist > 0.30 * scale)

        # PENALTY: F requires the three fingers (middle, ring, pinky) to be
        # extended. If they are not, it's likely an `I` (pinky-only) or another
        # curled configuration — penalize strongly to avoid F/I confusion.
        score -= 0.35 * (not three_up)

        return max(0.0, min(1.0, score))

    def _score_g(self, lm, scale: float, fs: FingerState) -> float:
        """
        G: Index finger pointing SIDEWAYS and HORIZONTAL. Back of hand faces camera.
        Only index extended. Middle/ring/pinky curled. Thumb below/beside.

        Physical description:
          - Index extends horizontally to the side at a level angle
          - The tip must be close to the same vertical height as its own MCP
          - Middle, ring, pinky all curled

        Key differentiators:
          - vs D: D has large y-travel (pointing up); G has large x-travel (pointing sideways)
          - vs H: G has ONE finger extended; H has TWO
          - vs C: C has curved/bent fingers; G has index cleanly extended with x-travel
        """
        score = 0.0

        # HARD GATE 1: Index must be extended
        if not fs.index_ext:
            return 0.0

        # HARD GATE 2: Middle, ring, AND pinky must all be curled
        if fs.middle_ext or fs.ring_ext or fs.pinky_ext:
            return 0.0

        # POSITIVE: Index tip has large horizontal (x) travel from its MCP
        idx_x_travel = abs(lm[8].x - lm[5].x)
        strong_sideways = idx_x_travel > 0.12 * scale
        score += 0.45 * strong_sideways

        # POSITIVE: Index tip is at NEARLY THE SAME y as its MCP (truly horizontal)
        idx_y_level = abs(lm[8].y - lm[5].y) < 0.12
        score += 0.35 * idx_y_level

        # POSITIVE: x-travel is larger than y-travel (more horizontal than vertical)
        more_horizontal = idx_x_travel > abs(lm[8].y - lm[5].y) * 1.5
        score += 0.20 * more_horizontal

        # PENALTY: Index pointing straight UP = D, not G
        pointing_up = lm[8].y < lm[5].y - 0.10
        score -= 0.50 * pointing_up

        # PENALTY: Little or no x-travel = not pointing sideways
        no_x_travel = idx_x_travel < 0.08 * scale
        score -= 0.40 * no_x_travel

        return max(0.0, min(1.0, score))

    def _score_h(self, lm, scale: float, fs: FingerState) -> float:
        """
        H: Index AND middle fingers BOTH pointing SIDEWAYS and HORIZONTAL.
        Back of hand faces camera. Ring and pinky curled.

        Physical description:
          - Both index and middle extend horizontally to the side at a level angle
          - Both tips must be close to the same y as their own MCPs (truly horizontal)
          - Ring and pinky curl into palm

        Key differentiators:
          - vs G: G has ONE finger extended; H has TWO
          - vs U: U has large y-travel (pointing up); H has large x-travel (sideways)
          - vs V: V points up and spreads; H points sideways with fingers together
          - vs C: C has curved bent fingers; H has two fingers cleanly extended with x-travel
        """
        score = 0.0

        # HARD GATE: Both index and middle must be extended
        if not fs.index_ext or not fs.middle_ext:
            return 0.0

        # POSITIVE: Index tip has large x-travel from its MCP
        idx_x_travel = abs(lm[8].x - lm[5].x)
        idx_sideways = idx_x_travel > 0.12 * scale
        score += 0.25 * idx_sideways

        # POSITIVE: Middle tip has large x-travel from its MCP
        mid_x_travel = abs(lm[12].x - lm[9].x)
        mid_sideways = mid_x_travel > 0.12 * scale
        score += 0.25 * mid_sideways

        # POSITIVE: Both tips at nearly the same y as their MCPs (truly horizontal)
        idx_y_level = abs(lm[8].y - lm[5].y) < 0.12
        mid_y_level = abs(lm[12].y - lm[9].y) < 0.12
        both_level = idx_y_level and mid_y_level
        score += 0.25 * both_level

        # POSITIVE: x-travel dominates over y-travel for both fingers
        idx_more_horiz = idx_x_travel > abs(lm[8].y - lm[5].y) * 1.5
        mid_more_horiz = mid_x_travel > abs(lm[12].y - lm[9].y) * 1.5
        score += 0.10 * (idx_more_horiz and mid_more_horiz)

        # POSITIVE: Ring and pinky curled
        ring_pinky_curled = not fs.ring_ext and not fs.pinky_ext
        score += 0.15 * ring_pinky_curled

        # PENALTY: Fingers pointing straight UP = U/V/B, not H
        both_up = lm[8].y < lm[5].y - 0.08 and lm[12].y < lm[9].y - 0.08
        score -= 0.50 * both_up

        # PENALTY: Little or no x-travel on either finger = not sideways
        no_x_travel = idx_x_travel < 0.08 * scale or mid_x_travel < 0.08 * scale
        score -= 0.40 * no_x_travel

        # PENALTY: Fingers spread wide apart = V, not H
        spread = abs(lm[8].x - lm[12].x) > 0.10
        score -= 0.25 * spread

        return max(0.0, min(1.0, score))

    # ---------------------------------------------------
    # TODO: LETTERS K-Z (currently inactive [I-Z])
    # ---------------------------------------------------

    def _score_i(self, lm, scale: float, fs: FingerState) -> float:
                """I: Pinky up, others curled, thumb rests on top of curled fingers.

                Physical description:
                    - Pinky (lm[20]) is extended upward
                    - Index/middle/ring are curled toward the palm
                    - Thumb rests near the curled fingers/palm (not sticking out)

                Key differentiators:
                    - vs Y: Y has thumb + pinky; I has thumb resting on curled fingers
                    - vs open palm/B: other fingers are curled, not straight
                """
                score = 0.0

                # HARD GATE: Pinky must be extended to be an I
                if not fs.pinky_ext:
                        return 0.0

                # POSITIVE: Pinky extended
                score += 0.40 * fs.pinky_ext

                # POSITIVE: Other fingers should be curled (index/middle/ring)
                others_curled = not (fs.index_ext or fs.middle_ext or fs.ring_ext)
                score += 0.20 * others_curled

                # POSITIVE: Pinky tip noticeably above its MCP (clearly raised)
                pinky_high = lm[20].y < lm[17].y - 0.04
                score += 0.15 * pinky_high

                # POSITIVE: Pinky higher than the other fingertips (distinct)
                pinky_above_others = lm[20].y < min(lm[8].y, lm[12].y, lm[16].y) - 0.05
                score += 0.10 * pinky_above_others

                # POSITIVE: Thumb resting near the palm/curled fingers (not far away)
                cx, cy = self._palm_center(lm)
                thumb_to_palm = math.hypot(lm[4].x - cx, lm[4].y - cy)
                thumb_resting = thumb_to_palm < 0.18 * scale
                score += 0.05 * thumb_resting

                # PENALTY: Thumb extended sideways (thumb sticking out) is not I
                score -= 0.25 * fs.thumb_side

                # PENALTY: If multiple fingers are extended it's unlikely to be I
                score -= 0.30 * (self._count_extended(fs) >= 2)

                return max(0.0, min(1.0, score))

    class JDetector:
        """
        Detects the ASL letter J by tracking the J-hook motion of the pinky tip.

        J = I position (pinky up) + hook motion: pinky traces down, curves, hooks up.

        The hook is detected by tracking three motion phases:
        Phase 0 (READY):    Hand is in I position, waiting for motion to start.
        Phase 1 (DOWN):     Pinky tip moves downward (y increases).
        Phase 2 (HOOK):     Pinky tip moves upward (y decreases) after going down.
        COMPLETE:           Hook confirmed — emit "J" once then reset.

        The detector resets if:
        - The hand leaves the I position during tracking
        - Too many frames pass without phase progression (timeout)
        - J is successfully detected (fires once then resets)
        """

    # Minimum pinky y-travel (in scale units) to count as "moved down"
    DOWN_THRESHOLD = 0.08
    # Minimum pinky y-travel back up (in scale units) to confirm the hook
    UP_THRESHOLD = 0.05
    # Frames allowed per phase before timeout/reset
    PHASE_TIMEOUT = 45
    # Frames the I position must be held before motion tracking begins
    I_HOLD_FRAMES = 4

    def __init__(self):
        self.reset()

    def reset(self):
        self._phase = 0             # 0=ready, 1=going down, 2=hooking up
        self._i_hold_count = 0      # frames held in I position
        self._phase_frames = 0      # frames spent in current phase
        self._down_start_y = None   # pinky y when downward motion began
        self._down_peak_y = None    # lowest pinky y reached during down phase
        self._fired = False         # True if J was just detected this frame

    def update(self, landmarks, handedness: Optional[str], classifier_result: Optional[Dict]) -> Optional[str]:
        """
        Call every frame with the current landmarks and classifier result.

        Returns "J" on the frame the J motion is completed, None otherwise.
        """
        self._fired = False

        if landmarks is None or len(landmarks) < 21:
            self.reset()
            return None

        # Mirror landmarks for right hand (same as classifier)
        lm = landmarks
        if handedness == "Right":
            mirrored = []
            for pt in landmarks:
                new_pt = type(pt)()
                new_pt.x = 1.0 - pt.x
                new_pt.y = pt.y
                new_pt.z = getattr(pt, "z", 0)
                mirrored.append(new_pt)
            lm = mirrored

        scale = math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y)
        if scale < 0.01:
            self.reset()
            return None

        pinky_y = lm[20].y
        in_i_position = (
            classifier_result is not None and
            classifier_result.get("letter") == "I"
        )

        # ── Phase 0: Wait for a stable I hold ─────────────────────────────────
        if self._phase == 0:
            if in_i_position:
                self._i_hold_count += 1
            else:
                self._i_hold_count = 0

            if self._i_hold_count >= self.I_HOLD_FRAMES:
                # I held long enough — start tracking motion
                self._phase = 1
                self._down_start_y = pinky_y
                self._down_peak_y = pinky_y
                self._phase_frames = 0
            return None

        # ── Phase 1: Pinky must move DOWN ──────────────────────────────────────
        if self._phase == 1:
            self._phase_frames += 1

            # Track the lowest point the pinky reaches
            if pinky_y > self._down_peak_y:
                self._down_peak_y = pinky_y

            down_travel = self._down_peak_y - self._down_start_y

            # Enough downward travel — transition to hook-up phase
            if down_travel > self.DOWN_THRESHOLD * scale:
                self._phase = 2
                self._phase_frames = 0
                return None

            # Timeout or hand left I position — reset
            if self._phase_frames > self.PHASE_TIMEOUT or not in_i_position:
                self.reset()
            return None

        # ── Phase 2: Pinky must hook back UP ──────────────────────────────────
        if self._phase == 2:
            self._phase_frames += 1

            up_travel = self._down_peak_y - pinky_y  # positive = moving up

            if up_travel > self.UP_THRESHOLD * scale:
                # J hook complete!
                self.reset()
                return "J"

            # Timeout — reset
            if self._phase_frames > self.PHASE_TIMEOUT:
                self.reset()
            return None

        return None

    # def _score_J(self, lm, scale: float, fs: FingerState) -> float:
    #     """J (static component):

    #     This score only reflects the *initial* handshape used at the
    #     beginning of the J gesture (essentially the "I" configuration with a
    #     slight sideways orientation).  The dynamic arc motion is handled by
    #     the state machine in ``signs/dynamic_signs.py`` and is **not** part of
    #     this scoring function.

    #     Positive evidence increases the score; conflicting features subtract
    #     from it.  We intentionally bias the function toward being conservative
    #     so that ``J`` won't dominate over other similar one‑finger signs.
    #     """
    #     score = 0.0

    #     # --- HARD GATES --------------------------------------------------
    #     # if the basic "I" shape isn't present, give up early
    #     if not fs.pinky_ext:
    #         return 0.0
    #     if fs.index_ext or fs.middle_ext or fs.ring_ext:
    #         return 0.0

    #     # --- POSITIVE FEATURES ------------------------------------------
    #     # pinky extended clearly above its MCP
    #     pinky_high = lm[20].y < lm[17].y - 0.04
    #     score += 0.40 * pinky_high

    #     # other fingers curled down
    #     others_curled = not (fs.index_ext or fs.middle_ext or fs.ring_ext)
    #     score += 0.20 * others_curled

    #     # thumb resting near palm (not flared out)
    #     cx, cy = self._palm_center(lm)
    #     thumb_to_palm = math.hypot(lm[4].x - cx, lm[4].y - cy)
    #     thumb_resting = thumb_to_palm < 0.18 * scale
    #     score += 0.10 * thumb_resting

    #     # slight sideways orientation: width of palm (index MCP to pinky MCP)
    #     palm_width = abs(lm[5].x - lm[17].x)
    #     oriented_sideways = palm_width > 0.20 * scale
    #     score += 0.10 * oriented_sideways

    #     # a little extra credit if pinky is noticeably above the other tips
    #     pinky_above_others = lm[20].y < min(lm[8].y, lm[12].y, lm[16].y) - 0.05
    #     score += 0.05 * pinky_above_others

    #     # --- PENALTIES ---------------------------------------------------
    #     # thumb sticking out sideways isn't part of the I shape
    #     score -= 0.25 * fs.thumb_side

    #     # if more than one non‑pinky finger sneaks up, it's probably a different
    #     # letter (B, C, etc.)
    #     score -= 0.30 * (self._count_extended(fs) >= 2)

    #     return max(0.0, min(1.0, score))

    # def _score_k(self, lm, scale: float, fs: FingerState) -> float:
    #     """K: Index+middle up, thumb between."""
    #     return 0.0

    # def _score_l(self, lm, scale: float, fs: FingerState) -> float:
    #     """L: Index up, thumb sideways L."""
    #     return 0.0

    # def _score_m(self, lm, scale: float, fs: FingerState) -> float:
    #     """M: 3 fingers over tucked thumb."""
    #     return 0.0

    # def _score_n(self, lm, scale: float, fs: FingerState) -> float:
    #     """N: 2 fingers over tucked thumb."""
    #     return 0.0

    # def _score_o(self, lm, scale: float, fs: FingerState) -> float:
    #     """O: All fingers circle to thumb."""
    #     return 0.0

    # def _score_s(self, lm, scale: float, fs: FingerState) -> float:
    #     """S: Fist with thumb over tips."""
    #     return 0.0

    # def _score_t(self, lm, scale: float, fs: FingerState) -> float:
    #     """T: Thumb between index+middle."""
    #     return 0.0

    # def _score_u(self, lm, scale: float, fs: FingerState) -> float:
    #     """U: Index+middle up together."""
    #     return 0.0

    # def _score_v(self, lm, scale: float, fs: FingerState) -> float:
    #     """V: Index+middle up spread."""
    #     return 0.0

    # def _score_w(self, lm, scale: float, fs: FingerState) -> float:
    #     """W: 3 fingers up spread."""
    #     return 0.0

    # def _score_x(self, lm, scale: float, fs: FingerState) -> float:
    #     """X: Index hooked."""
    #     return 0.0

    # def _score_y(self, lm, scale: float, fs: FingerState) -> float:
    #     """Y: Thumb sideways + pinky up."""
    #     return 0.0

    # ---------------------------------------------------
    # CLASSIFIER
    # ---------------------------------------------------

    def classify(self, landmarks, handedness: Optional[str] = None) -> Optional[Dict]:
        """
        Classify hand landmarks as an ASL letter (A-H only).

        Returns a dict: {"letter": str, "confidence": float, "scores": dict}
        or None if below min_confidence threshold.
        """
        if landmarks is None or len(landmarks) < 21:
            return None

        if handedness == "Right":
            landmarks = self._mirror_landmarks_x(landmarks)

        scale = self._hand_scale(landmarks)
        if scale < 0.01:
            return None

        fs = self._get_finger_states(landmarks, scale)

        Jvar = self.j_detector.update(landmarks, handedness, None)
        if Jvar == "J":
            return {
                "letter": "J",
                "confidence": 0.95,
                "scores": {"J": 0.95},
            }

        # Only A-H active — disabled letters return 0.0
        scores = {
            "A": self._score_a(landmarks, scale, fs),
            "B": self._score_b(landmarks, scale, fs),
            "C": self._score_c(landmarks, scale, fs),
            "D": self._score_d(landmarks, scale, fs),
            "E": self._score_e(landmarks, scale, fs),
            "F": self._score_f(landmarks, scale, fs),
            "G": self._score_g(landmarks, scale, fs),
            "H": self._score_h(landmarks, scale, fs),
            "I": self._score_i(landmarks, scale, fs),
            "J": Jvar == "J",
            # "J": self._score_J(landmarks, scale, fs),
            # "K": self._score_k(landmarks, scale, fs),
            # "L": self._score_l(landmarks, scale, fs),
            # "M": self._score_m(landmarks, scale, fs),
            # "N": self._score_n(landmarks, scale, fs),
            # "O": self._score_o(landmarks, scale, fs),
            # "P": self._score_p(landmarks, scale, fs),
            # "Q": self._score_q(landmarks, scale, fs),
            # "R": self._score_r(landmarks, scale, fs),
            # "S": self._score_s(landmarks, scale, fs),
            # "T": self._score_t(landmarks, scale, fs),
            # "U": self._score_u(landmarks, scale, fs),
            # "V": self._score_v(landmarks, scale, fs),
            # "W": self._score_w(landmarks, scale, fs),
            # "X": self._score_x(landmarks, scale, fs),
            # "Y": self._score_y(landmarks, scale, fs),
            # "Z": self._score_z(landmarks, scale, fs),
        }

        best_letter = max(scores, key=scores.get)
        best_score = scores[best_letter]

        # Debugging line (optional) — uncomment to print all scores each frame
        # print(f"{best_letter} ({best_score:.2f}) | " + " ".join(f"{k}:{v:.2f}" for k,v in sorted(scores.items())))

        if best_score < self.min_confidence:
            return None

        return {
            "letter": best_letter,
            "confidence": round(best_score, 3),
            "scores": scores,
        }

# ---------------------------------------------------
# DEBUGGING GUIDE
# ---------------------------------------------------
#
# HOW TO SEND DEBUG INFO:
#   1. Uncomment the print() line inside classify()
#   2. Run your app and hold the problematic sign
#   3. Copy 5-10 lines of printed output
#   4. Tell me:
#      - Which letter you INTENDED (e.g. "I was signing A")
#      - What score it got (e.g. "A: 0.48")
#      - Which letter DOMINATED (e.g. "E dominated with 0.71")
#   That's all I need to fix any threshold.
#
# COMMON PROBLEMS & HOW TO READ THEM:
#
#   "Nothing detected" (returns None):
#     - Lower min_confidence to 0.3 temporarily
#     - The right letter probably scores 0.45-0.58 and just needs threshold tuning
#
#   "Wrong letter always wins":
#     - The dominant letter has conditions that are too easy to satisfy
#     - Look at its POSITIVE checks vs your actual sign - one is probably true by accident
#     - Increase the PENALTY for the dominant letter, or add a PENALTY to it for the feature
#       your intended letter has (e.g. if B dominates when you sign C, add a PENALTY to B
#       for partial curl)
#
#   "Letter flickers between two letters":
#     - The two letters have overlapping scores (both ~0.65)
#     - Find the ONE geometric feature that distinguishes them
#     - Add it as a high-weight POSITIVE to the intended letter and PENALTY to the other
#
# MediaPipe Landmark Index Reference:
#   0=Wrist
#   1=Thumb CMC, 2=Thumb MCP, 3=Thumb IP, 4=Thumb tip
#   5=Index MCP, 6=Index PIP, 7=Index DIP, 8=Index tip
#   9=Middle MCP, 10=Middle PIP, 11=Middle DIP, 12=Middle tip
#   13=Ring MCP, 14=Ring PIP, 15=Ring DIP, 16=Ring tip
#   17=Pinky MCP, 18=Pinky PIP, 19=Pinky DIP, 20=Pinky tip
#
# Coordinate system:
#   x: 0=left edge, 1=right edge (of frame)
#   y: 0=TOP of frame, 1=BOTTOM (inverted from screen coordinates!)
#   So: tip.y < pip.y means tip is HIGHER on screen = finger pointing UP
# ---------------------------------------------------
