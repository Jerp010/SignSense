"""
sign_registry.py
================
Single source of truth for every sign SignSense knows about.

Adding a new sign
-----------------
1. Add an entry to SIGN_REGISTRY with the correct SignType.
2. If it is DYNAMIC, implement a detector in dynamic_signs.py and add it to
   DYNAMIC_DETECTORS at the bottom of this file.
3. Drop preview images into  assets/signs/<LETTER>/view_1.png  (etc.).
   Until images exist the UI shows a styled placeholder automatically.

Registry fields
---------------
letter      : str          — display key ("A", "B", …)
sign_type   : SignType     — STATIC (single frame) or DYNAMIC (motion)
description : str          — one-line hint shown to the user
preview_count : int        — how many preview images exist (pages in the box)
preview_dir : str          — path relative to project root for preview assets
enabled     : bool         — True = not yet implemented, skipped in play mode
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List


class SignType(Enum):
    STATIC  = auto()   # single-frame pose
    DYNAMIC = auto()   # requires motion/trajectory detection


@dataclass
class SignEntry:
    letter:        str
    sign_type:     SignType
    description:   str
    preview_count: int  = 3       # number of view images available
    preview_dir:   str  = ""      # e.g. "assets/signs/A"
    enabled:       bool = True


# ---------------------------------------------------------------------------
# REGISTRY  — add new signs here, in order
# ---------------------------------------------------------------------------
SIGN_REGISTRY: List[SignEntry] = [
    SignEntry("A", SignType.STATIC,  "Closed fist, thumb beside index"),
    SignEntry("B", SignType.STATIC,  "Four fingers up, thumb tucked"),
    SignEntry("C", SignType.STATIC,  "Curved hand forming a C"),
    SignEntry("D", SignType.STATIC,  "Index up, thumb-middle loop"),
    SignEntry("E", SignType.STATIC,  "Fingers clawed, thumb tucked under"),
    SignEntry("F", SignType.STATIC,  "Thumb-index circle, three fingers up"),
    SignEntry("G", SignType.STATIC,  "Index pointing sideways"),
    SignEntry("H", SignType.STATIC,  "Index + middle pointing sideways"),
    SignEntry("I", SignType.STATIC,  "Pinky up, others curled"),
    SignEntry("J", SignType.DYNAMIC, "I shape + pinky traces a J hook"),
    SignEntry("K", SignType.STATIC,  "Index + middle up, thumb between",  enabled=True),
    SignEntry("L", SignType.STATIC,  "Index up, thumb sideways L shape",  enabled=True),
    SignEntry("M", SignType.STATIC,  "Three fingers over tucked thumb",   enabled=True),
    SignEntry("N", SignType.STATIC,  "Two fingers over tucked thumb",     enabled=True),
    SignEntry("O", SignType.STATIC,  "All fingers circle to thumb",       enabled=True),
    SignEntry("P", SignType.STATIC,  "Like K but pointing down",          enabled=True),
    SignEntry("Q", SignType.STATIC,  "Like G but pointing down",          enabled=True),
    SignEntry("R", SignType.STATIC,  "Index + middle crossed",            enabled=True),
    SignEntry("S", SignType.STATIC,  "Fist with thumb over fingers",      enabled=True),
    SignEntry("T", SignType.STATIC,  "Thumb between index + middle",      enabled=True),
    SignEntry("U", SignType.STATIC,  "Index + middle up together",        enabled=True),
    SignEntry("V", SignType.STATIC,  "Index + middle up spread (peace)",  enabled=True),
    SignEntry("W", SignType.STATIC,  "Three fingers up spread",           enabled=True),
    SignEntry("X", SignType.STATIC,  "Index finger hooked",               enabled=True),
    SignEntry("Y", SignType.STATIC,  "Thumb + pinky out",                 enabled=True),
            # --- not yet implemented — will be skipped in play mode ---
    SignEntry("Z", SignType.DYNAMIC, "Index traces a Z in the air",       enabled=False),
]

# Convenience: only the enabled signs, in order
ACTIVE_SIGNS: List[SignEntry] = [s for s in SIGN_REGISTRY if s.enabled]

# Lookup by letter
SIGN_BY_LETTER = {s.letter: s for s in SIGN_REGISTRY}


def get_active_letters() -> List[str]:
    """Return letters for active signs in order."""
    return [s.letter for s in ACTIVE_SIGNS]


def get_sign(letter: str) -> Optional[SignEntry]:
    return SIGN_BY_LETTER.get(letter.upper())