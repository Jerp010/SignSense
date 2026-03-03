import math
from signsense.detector.asl_classifier_letters import ASLClassifierLetters


class _LM:
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def make_landmarks(overrides=None):
    """
    Create 21 landmark objects.  Wrist (lm[0]) and middle MCP (lm[9]) are
    placed far apart so scale ≈ 0.70, giving realistic normalised thresholds.
    """
    lm = [_LM() for _ in range(21)]
    lm[0].x, lm[0].y = 0.50, 0.90   # wrist  (bottom)
    lm[9].x, lm[9].y = 0.50, 0.20   # middle MCP  (scale ≈ 0.70)

    # Sensible defaults for every joint — individual tests override what matters
    defaults = {
        1:  (0.48, 0.75),  # thumb CMC
        2:  (0.46, 0.65),  # thumb MCP
        3:  (0.43, 0.58),  # thumb IP
        4:  (0.41, 0.52),  # thumb tip  (slightly left = thumb_side borderline)
        5:  (0.44, 0.55),  # index MCP
        6:  (0.44, 0.62),  # index PIP
        7:  (0.44, 0.68),  # index DIP
        8:  (0.44, 0.72),  # index tip  (curled by default)
        10: (0.50, 0.60),  # middle PIP
        11: (0.50, 0.66),  # middle DIP
        12: (0.50, 0.70),  # middle tip (curled by default)
        13: (0.54, 0.56),  # ring MCP
        14: (0.54, 0.63),  # ring PIP
        15: (0.54, 0.69),  # ring DIP
        16: (0.54, 0.73),  # ring tip   (curled by default)
        17: (0.58, 0.57),  # pinky MCP
        18: (0.58, 0.64),  # pinky PIP
        19: (0.58, 0.70),  # pinky DIP
        20: (0.58, 0.74),  # pinky tip  (curled by default)
    }
    for i, (x, y) in defaults.items():
        lm[i].x, lm[i].y = x, y

    for idx, (x, y) in (overrides or {}).items():
        lm[idx].x = x
        lm[idx].y = y

    return lm


def run_targeted(letter, lm, confidence=0.40):
    clf = ASLClassifierLetters()
    clf.min_confidence = confidence
    return clf.classify(lm, handedness=None, target_letter=letter)


# ---------------------------------------------------------------------------
# K
# ---------------------------------------------------------------------------

def test_k_recognised():
    """
    K: index + middle both extended upward, thumb tip between them in x.
    Index may be slightly angled (not perfectly vertical).
    """
    overrides = {
        # Index pointing up (slightly angled forward — tip left of MCP)
        5:  (0.44, 0.55),   # index MCP
        6:  (0.43, 0.48),   # index PIP
        8:  (0.42, 0.35),   # index tip — above PIP and MCP, slightly forward
        # Middle pointing up
        9:  (0.50, 0.55),   # middle MCP
        10: (0.50, 0.48),   # middle PIP
        12: (0.50, 0.34),   # middle tip — above MCP
        # Thumb between index MCP (x=0.44) and middle MCP (x=0.50) → x=0.47
        2:  (0.52, 0.60),   # thumb MCP (after mirror: thumb on left)
        4:  (0.47, 0.44),   # thumb tip — x between 0.44 and 0.50
        # Ring and pinky curled (tips below their MCPs)
        13: (0.55, 0.55),  16: (0.55, 0.72),
        17: (0.59, 0.57),  20: (0.59, 0.75),
    }
    lm = make_landmarks(overrides)
    res = run_targeted("K", lm)
    assert res is not None and res["letter"] == "K", \
        f"K not recognised, got {res}"


def test_k_not_confused_with_v():
    """V has no thumb between the fingers — K scorer should beat V-like poses."""
    overrides = {
        5:  (0.44, 0.55),  8:  (0.44, 0.34),   # index up
        9:  (0.56, 0.55),  12: (0.56, 0.34),   # middle up, spread wide
        # Thumb FAR outside the finger range (to the left) — not between them
        2:  (0.52, 0.60),  4:  (0.35, 0.52),
        13: (0.58, 0.55),  16: (0.58, 0.72),
        17: (0.62, 0.57),  20: (0.62, 0.75),
    }
    lm = make_landmarks(overrides)
    res = run_targeted("K", lm, confidence=0.60)
    assert res is None, "K scorer should not fire when thumb is not between fingers"


# ---------------------------------------------------------------------------
# L
# ---------------------------------------------------------------------------

def test_l_recognised():
    """
    L: index pointing straight up, thumb extended sideways (leftward after mirror).
    Middle/ring/pinky all curled.
    """
    overrides = {
        # Index pointing up
        5:  (0.45, 0.55),  6:  (0.45, 0.48),  8:  (0.45, 0.28),
        # Thumb extended sideways: tip well left of thumb MCP
        2:  (0.52, 0.62),  4:  (0.30, 0.60),  # lm[4].x << lm[2].x -> thumb_side=True
        # Middle/ring/pinky curled
        9:  (0.50, 0.55),  12: (0.50, 0.72),
        13: (0.54, 0.56),  16: (0.54, 0.73),
        17: (0.58, 0.57),  20: (0.58, 0.75),
    }
    lm = make_landmarks(overrides)
    res = run_targeted("L", lm)
    assert res is not None and res["letter"] == "L", \
        f"L not recognised, got {res}"


def test_l_not_confused_with_d():
    """D has thumb looping to middle, not extending sideways — should not score as L."""
    overrides = {
        5:  (0.45, 0.55),  8:  (0.45, 0.28),   # index up
        # Thumb close to middle tip (D loop), NOT sideways
        2:  (0.52, 0.62),  4:  (0.50, 0.52),   # thumb tip near centre
        9:  (0.50, 0.55),  12: (0.50, 0.52),   # middle tip near thumb
        13: (0.54, 0.56),  16: (0.54, 0.73),
        17: (0.58, 0.57),  20: (0.58, 0.75),
    }
    lm = make_landmarks(overrides)
    # thumb_side will be False here, so L gate fires immediately
    res = run_targeted("L", lm, confidence=0.60)
    assert res is None, "L scorer should not fire when thumb is not sideways"


# ---------------------------------------------------------------------------
# M
# ---------------------------------------------------------------------------

def test_m_recognised():
    """
    M: closed fist, all fingers curled over thumb.
    Thumb tip between MIDDLE MCP (lm[9], x≈0.50) and PINKY MCP (lm[17], x≈0.58).
    Thumb is LOW (below index MCP).
    """
    overrides = {
        # All fingers deeply curled (tips well below their MCPs)
        5:  (0.44, 0.55),  6:  (0.44, 0.62),  8:  (0.44, 0.72),   # index
        9:  (0.50, 0.55),  10: (0.50, 0.62),  12: (0.50, 0.72),   # middle
        13: (0.54, 0.56),  14: (0.54, 0.63),  16: (0.54, 0.74),   # ring
        17: (0.58, 0.57),  18: (0.58, 0.64),  20: (0.58, 0.75),   # pinky
        # Thumb tip in M slot: between mid_x=0.50 and pky_x=0.58 → x=0.54
        # and LOW: y > index MCP y (0.55)
        2:  (0.52, 0.58),  4:  (0.54, 0.65),
    }
    lm = make_landmarks(overrides)
    res = run_targeted("M", lm)
    assert res is not None and res["letter"] == "M", \
        f"M not recognised, got {res}"


def test_m_not_confused_with_n():
    """When thumb is in the N slot (middle–ring gap), M scorer should not fire."""
    overrides = {
        5:  (0.44, 0.55),  8:  (0.44, 0.72),
        9:  (0.50, 0.55),  12: (0.50, 0.72),
        13: (0.54, 0.56),  16: (0.54, 0.74),
        17: (0.58, 0.57),  20: (0.58, 0.75),
        # Thumb in N slot: between mid_x=0.50 and ring_x=0.54 → x=0.52
        2:  (0.52, 0.58),  4:  (0.52, 0.65),
    }
    lm = make_landmarks(overrides)
    res = run_targeted("M", lm, confidence=0.60)
    assert res is None, "M scorer should not fire when thumb is in the N slot"


# ---------------------------------------------------------------------------
# N
# ---------------------------------------------------------------------------

def test_n_recognised():
    """
    N: closed fist, thumb tip between MIDDLE MCP (x≈0.50) and RING MCP (x≈0.54).
    All fingers curled over thumb. Thumb is LOW.
    """
    overrides = {
        5:  (0.44, 0.55),  8:  (0.44, 0.72),
        9:  (0.50, 0.55),  12: (0.50, 0.72),
        13: (0.54, 0.56),  16: (0.54, 0.74),
        17: (0.58, 0.57),  20: (0.58, 0.75),
        # Thumb in N slot: between mid_x=0.50 and ring_x=0.54 → x=0.52
        # and LOW: y=0.66 > index MCP y=0.55
        2:  (0.52, 0.58),  4:  (0.52, 0.66),
    }
    lm = make_landmarks(overrides)
    res = run_targeted("N", lm)
    assert res is not None and res["letter"] == "N", \
        f"N not recognised, got {res}"


def test_n_not_confused_with_m():
    """When thumb is in the M slot (middle–pinky gap), N scorer should not fire."""
    overrides = {
        5:  (0.44, 0.55),  8:  (0.44, 0.72),
        9:  (0.50, 0.55),  12: (0.50, 0.72),
        13: (0.54, 0.56),  16: (0.54, 0.74),
        17: (0.58, 0.57),  20: (0.58, 0.75),
        # Thumb in M slot: between mid_x=0.50 and pky_x=0.58 → x=0.54
        2:  (0.52, 0.58),  4:  (0.54, 0.65),
    }
    lm = make_landmarks(overrides)
    res = run_targeted("N", lm, confidence=0.60)
    assert res is None, "N scorer should not fire when thumb is in the M slot"


# ---------------------------------------------------------------------------
# O
# ---------------------------------------------------------------------------

def test_o_recognised():
    """
    O: all fingertips curve inward to meet the thumb, forming a round O.
    No finger is straight. Tips all near the thumb tip.
    """
    overrides = {
        # All fingertips converge near (0.50, 0.62) — thumb tip
        4:  (0.50, 0.62),   # thumb tip (centre of the O)
        8:  (0.48, 0.60),   # index tip near thumb
        12: (0.50, 0.58),   # middle tip near thumb
        16: (0.52, 0.60),   # ring tip near thumb
        20: (0.51, 0.63),   # pinky tip near thumb
        # PIPs sit above the tips so _is_extended is False (tips not above PIPs)
        6:  (0.46, 0.50),   # index PIP above index tip
        10: (0.50, 0.48),
        14: (0.53, 0.50),
        18: (0.56, 0.52),
        # MCPs at natural positions
        5:  (0.44, 0.42),
        9:  (0.50, 0.40),
        13: (0.54, 0.43),
        17: (0.58, 0.45),
    }
    lm = make_landmarks(overrides)
    res = run_targeted("O", lm)
    assert res is not None and res["letter"] == "O", \
        f"O not recognised, got {res}"


# ---------------------------------------------------------------------------
# P
# ---------------------------------------------------------------------------

def test_p_recognised():
    """
    P: like K but entire hand points downward.
    Index and middle tips are BELOW their MCPs. Thumb between the two MCPs.
    Wrist is ABOVE the finger MCPs.
    """
    overrides = {
        # Wrist high up (small y = top of frame)
        0:  (0.50, 0.10),
        # Index MCP higher than tip (hand pointing down)
        5:  (0.44, 0.30),   # index MCP (high)
        6:  (0.44, 0.45),   # index PIP
        8:  (0.44, 0.58),   # index tip — BELOW MCP (y=0.58 > y=0.30)
        # Middle MCP higher than tip
        9:  (0.50, 0.30),   # middle MCP (high)
        10: (0.50, 0.45),
        12: (0.50, 0.58),   # middle tip — BELOW MCP
        # Thumb between index MCP (x=0.44) and middle MCP (x=0.50) → x=0.47
        2:  (0.52, 0.22),
        4:  (0.47, 0.35),
        # Ring and pinky curled (above their tips)
        13: (0.55, 0.32),  16: (0.55, 0.52),
        17: (0.59, 0.33),  20: (0.59, 0.53),
    }
    lm = make_landmarks(overrides)
    res = run_targeted("P", lm)
    assert res is not None and res["letter"] == "P", \
        f"P not recognised, got {res}"


def test_p_not_confused_with_k():
    """K has tips above MCPs — should not score as P."""
    overrides = {
        0:  (0.50, 0.90),   # wrist at bottom (normal orientation)
        5:  (0.44, 0.55),  8:  (0.44, 0.34),   # index tip ABOVE MCP
        9:  (0.50, 0.55),  12: (0.50, 0.33),   # middle tip ABOVE MCP
        2:  (0.52, 0.60),  4:  (0.47, 0.44),   # thumb between
        13: (0.55, 0.56),  16: (0.55, 0.73),
        17: (0.59, 0.57),  20: (0.59, 0.75),
    }
    lm = make_landmarks(overrides)
    # P hard gate requires tips BELOW MCPs — should return None
    res = run_targeted("P", lm, confidence=0.60)
    assert res is None, "P scorer should not fire when tips are above MCPs (that's K)"


# ---------------------------------------------------------------------------
# Cross-contamination: M and N must not beat each other's canonical poses
# ---------------------------------------------------------------------------

def test_m_not_confused_with_a():
    """A is a closed fist; an M pose should NOT look like A."""
    overrides = {
        5:  (0.44, 0.55),  8:  (0.44, 0.72),
        9:  (0.50, 0.55),  12: (0.50, 0.72),
        13: (0.54, 0.56),  16: (0.54, 0.74),
        17: (0.58, 0.57),  20: (0.58, 0.75),
        2:  (0.52, 0.58),  4:  (0.54, 0.65),   # M slot
    }
    lm = make_landmarks(overrides)
    clf = ASLClassifierLetters()
    clf.min_confidence = 0.30
    res = clf.classify(lm, handedness=None)
    assert res is not None and res["letter"] == "M", \
        f"M should outrank A on M pose, got {res}"

def test_n_not_confused_with_e():
    """An N pose should not be mis‑scored as E (claw)."""
    overrides = {
        5:  (0.44, 0.55),  8:  (0.44, 0.72),
        9:  (0.50, 0.55),  12: (0.50, 0.72),
        13: (0.54, 0.56),  16: (0.54, 0.74),
        17: (0.58, 0.57),  20: (0.58, 0.75),
        2:  (0.52, 0.58),  4:  (0.52, 0.66),   # N slot
    }
    lm = make_landmarks(overrides)
    clf = ASLClassifierLetters()
    clf.min_confidence = 0.30
    res = clf.classify(lm, handedness=None)
    assert res is not None and res["letter"] == "N", \
        f"N should outrank E on N pose, got {res}"


def test_m_beats_n_on_m_pose():
    """On the canonical M pose, M should outscore N in full competition mode."""
    overrides = {
        5:  (0.44, 0.55),  8:  (0.44, 0.72),
        9:  (0.50, 0.55),  12: (0.50, 0.72),
        13: (0.54, 0.56),  16: (0.54, 0.74),
        17: (0.58, 0.57),  20: (0.58, 0.75),
        2:  (0.52, 0.58),  4:  (0.54, 0.65),   # M slot
    }
    lm = make_landmarks(overrides)
    clf = ASLClassifierLetters()
    clf.min_confidence = 0.30
    res = clf.classify(lm, handedness=None)
    assert res is not None and res["letter"] == "M", \
        f"M should win on M pose, got {res}"


def test_n_beats_m_on_n_pose():
    """On the canonical N pose, N should outscore M in full competition mode."""
    overrides = {
        5:  (0.44, 0.55),  8:  (0.44, 0.72),
        9:  (0.50, 0.55),  12: (0.50, 0.72),
        13: (0.54, 0.56),  16: (0.54, 0.74),
        17: (0.58, 0.57),  20: (0.58, 0.75),
        2:  (0.52, 0.58),  4:  (0.52, 0.66),   # N slot
    }
    lm = make_landmarks(overrides)
    clf = ASLClassifierLetters()
    clf.min_confidence = 0.30
    res = clf.classify(lm, handedness=None)
    assert res is not None and res["letter"] == "N", \
        f"N should win on N pose, got {res}"