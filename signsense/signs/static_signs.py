"""Static ASL sign definitions for single-hand position recognition."""

from typing import Dict, List, Any


# TODO: Implement geometric matching algorithms for static sign recognition
# TODO: Add conditions based on:
#   - Finger extended/bent states
#   - Relative positions of landmarks
#   - Angles between joints
#   - Distances between key points

# ---------------------------------------------------
# Condition Format Reference:
# Each condition is a dict describing a single geometric check:
# {
#   "check": str,          # what to measure (see below)
#   "landmarks": [...],    # landmark indices involved
#   "relation": str,       # "lt", "gt", "close", "apart", "between"
#   "threshold": float,    # value or scale fraction for comparisons
#   "weight": float,       # contribution to overall score (0.0–1.0, all weights sum to 1.0)
#   "note": str            # human-readable explanation of what this detects
# }
#
# check types:
#   "y_compare"   → compare y positions of two landmarks (tip vs pip = curled/extended)
#   "x_compare"   → compare x positions
#   "distance"    → euclidean distance between two landmarks (scaled)
#   "x_between"   → x of landmark is between x of two other landmarks
#   "x_spread"    → horizontal spread between two tips (U vs V style)
#   "tip_height"  → tip is significantly above MCP (pointing up vs sideways)
# ---------------------------------------------------

STATIC_SIGNS: Dict[str, Dict[str, Any]] = {
    "A": {
        "description": "Closed fist with thumb resting on the side of the index finger (not over fingertips like S)",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",  # tip.y > pip.y → curled
                "weight": 0.15,
                "note": "Index finger is curled"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.1,
                "note": "Middle finger is curled"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.1,
                "note": "Ring finger is curled"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "gt",
                "weight": 0.1,
                "note": "Pinky is curled"
            },
            {
                "check": "y_compare",
                "landmarks": [4, 8],
                "relation": "gt",  # thumb tip y > index tip y → thumb below fingertips (beside fist)
                "weight": 0.3,
                "note": "Thumb beside fist, not over fingertips (distinguishes A from S)"
            },
            {
                "check": "x_compare",
                "landmarks": [4, 5],
                "relation": "close",
                "threshold": 0.12,
                "weight": 0.25,
                "note": "Thumb tip x close to index MCP x (beside index knuckle)"
            }
        ]
    },
    "B": {
        "description": "All four fingers extended upward, thumb tucked across palm",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "lt",  # tip above pip = extended
                "weight": 0.15,
                "note": "Index finger extended"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "lt",
                "weight": 0.15,
                "note": "Middle finger extended"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "lt",
                "weight": 0.15,
                "note": "Ring finger extended"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "lt",
                "weight": 0.15,
                "note": "Pinky extended"
            },
            {
                "check": "y_compare",
                "landmarks": [4, 6],
                "relation": "gt",  # thumb tip lower than index PIP = tucked
                "weight": 0.2,
                "note": "Thumb tucked across palm"
            },
            {
                "check": "x_spread",
                "landmarks": [8, 20],
                "relation": "lt",
                "threshold": 0.12,
                "weight": 0.2,
                "note": "Fingers are tightly grouped (not spread like W or V)"
            }
        ]
    },
    "C": {
        "description": "Curved open hand, fingers and thumb forming a C shape",
        "conditions": [
            {
                "check": "distance",
                "landmarks": [4, 8],
                "relation": "between",
                "threshold": [0.30, 0.65],  # fraction of hand scale
                "weight": 0.4,
                "note": "Thumb-to-index gap is mid-range (not closed like O, not open like B)"
            },
            {
                "check": "y_compare",
                "landmarks": [4, 8],
                "relation": "close",
                "threshold": 0.15,
                "weight": 0.3,
                "note": "Thumb and index tip at similar heights (C opening is horizontal)"
            },
            {
                "check": "partial_curl",
                "landmarks": [8, 12, 16, 20],
                "relation": "partial",
                "weight": 0.3,
                "note": "Fingers not fully extended and not fully curled"
            }
        ]
    },
    "D": {
        "description": "Index finger extended upward, other fingers curl into a circle with thumb",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "lt",
                "weight": 0.3,
                "note": "Index finger extended upward"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.1,
                "note": "Middle finger curled"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.1,
                "note": "Ring finger curled"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "gt",
                "weight": 0.1,
                "note": "Pinky curled"
            },
            {
                "check": "distance",
                "landmarks": [4, 12],
                "relation": "lt",
                "threshold": 0.25,
                "weight": 0.25,
                "note": "Thumb touching middle finger (forms the circle in D)"
            },
            {
                "check": "tip_height",
                "landmarks": [8, 12],
                "relation": "lt",
                "threshold": 0.05,
                "weight": 0.15,
                "note": "Index tip significantly above middle tip"
            }
        ]
    },
    "E": {
        "description": "All fingers bent at tips toward palm, thumb tucked under (flat curl)",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",
                "weight": 0.1,
                "note": "Index bent"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.1,
                "note": "Middle bent"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.1,
                "note": "Ring bent"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "gt",
                "weight": 0.1,
                "note": "Pinky bent"
            },
            {
                "check": "distance",
                "landmarks": [4, 8],
                "relation": "lt",
                "threshold": 0.28,
                "weight": 0.15,
                "note": "Index tip near thumb"
            },
            {
                "check": "distance",
                "landmarks": [4, 12],
                "relation": "lt",
                "threshold": 0.28,
                "weight": 0.15,
                "note": "Middle tip near thumb"
            },
            {
                "check": "distance",
                "landmarks": [4, 16],
                "relation": "lt",
                "threshold": 0.28,
                "weight": 0.1,
                "note": "Ring tip near thumb"
            },
            {
                "check": "distance",
                "landmarks": [4, 20],
                "relation": "lt",
                "threshold": 0.28,
                "weight": 0.1,
                "note": "Pinky tip near thumb"
            },
            {
                "check": "tip_alignment",
                "landmarks": [8, 12, 16, 20],
                "relation": "lt",
                "threshold": 0.08,
                "weight": 0.1,
                "note": "All fingertips at similar height (flat curl, not rounded fist)"
            }
        ]
    },
    "F": {
        "description": "Index and thumb form a circle (touching), middle/ring/pinky extended",
        "conditions": [
            {
                "check": "distance",
                "landmarks": [4, 8],
                "relation": "lt",
                "threshold": 0.22,
                "weight": 0.5,
                "note": "Thumb and index tip touching (the circle)"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "lt",
                "weight": 0.15,
                "note": "Middle finger extended"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "lt",
                "weight": 0.15,
                "note": "Ring finger extended"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "lt",
                "weight": 0.1,
                "note": "Pinky extended"
            },
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",
                "weight": 0.1,
                "note": "Index finger is curled/down (not extended like in K)"
            }
        ]
    },
    "G": {
        "description": "Index finger extended horizontally, thumb extended parallel, hand rotated sideways",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "lt",
                "weight": 0.3,
                "note": "Index finger extended"
            },
            {
                "check": "y_compare",
                "landmarks": [4, 8],
                "relation": "close",
                "threshold": 0.06,
                "weight": 0.4,
                "note": "Thumb and index at same height (both pointing horizontally)"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.3,
                "note": "Middle finger curled"
            }
        ]
    },
    "H": {
        "description": "Index and middle fingers extended together horizontally, ring and pinky curled",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "lt",
                "weight": 0.25,
                "note": "Index extended"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "lt",
                "weight": 0.25,
                "note": "Middle extended"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.15,
                "note": "Ring curled"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "gt",
                "weight": 0.15,
                "note": "Pinky curled"
            },
            {
                "check": "tip_alignment",
                "landmarks": [8, 12],
                "relation": "lt",
                "threshold": 0.05,
                "weight": 0.2,
                "note": "Index and middle tips at same height (parallel, pointing sideways)"
            }
        ]
    },
    "I": {
        "description": "Pinky extended upward, all other fingers curled into fist",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "lt",
                "weight": 0.5,
                "note": "Pinky extended upward"
            },
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",
                "weight": 0.2,
                "note": "Index curled"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.15,
                "note": "Middle curled"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.15,
                "note": "Ring curled"
            }
        ]
    },
    "K": {
        "description": "Index and middle extended upward, thumb pointing up between them",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "lt",
                "weight": 0.2,
                "note": "Index extended"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "lt",
                "weight": 0.2,
                "note": "Middle extended"
            },
            {
                "check": "y_compare",
                "landmarks": [4, 6],
                "relation": "lt",  # thumb tip above index PIP = pointing up
                "weight": 0.3,
                "note": "Thumb pointing upward between the two fingers"
            },
            {
                "check": "distance",
                "landmarks": [4, 12],
                "relation": "lt",
                "threshold": 0.30,
                "weight": 0.3,
                "note": "Thumb close to middle finger (touching in K)"
            }
        ]
    },
    "L": {
        "description": "Index pointing up and thumb pointing sideways — forming an L shape",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "lt",
                "weight": 0.4,
                "note": "Index finger extended upward"
            },
            {
                "check": "x_compare",
                "landmarks": [4, 2],
                "relation": "lt",
                "threshold": 0.04,
                "weight": 0.4,
                "note": "Thumb extended sideways (left of thumb base)"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.1,
                "note": "Middle finger curled"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.1,
                "note": "Ring finger curled"
            }
        ]
    },
    "M": {
        "description": "Three fingers (index, middle, ring) folded over thumb tucked inside",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",
                "weight": 0.1,
                "note": "Index curled"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.1,
                "note": "Middle curled"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.1,
                "note": "Ring curled"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "gt",
                "weight": 0.1,
                "note": "Pinky curled"
            },
            {
                "check": "x_compare",
                "landmarks": [4, 13],
                "relation": "gt",  # thumb past ring MCP
                "weight": 0.4,
                "note": "Thumb tucked deep under 3 fingers (past ring MCP)"
            },
            {
                "check": "y_compare",
                "landmarks": [4, 8],
                "relation": "gt",
                "weight": 0.2,
                "note": "Thumb tip below fingertips (inside fist)"
            }
        ]
    },
    "N": {
        "description": "Two fingers (index, middle) folded over thumb tucked inside",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",
                "weight": 0.1,
                "note": "Index curled"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.1,
                "note": "Middle curled"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.1,
                "note": "Ring curled"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "gt",
                "weight": 0.1,
                "note": "Pinky curled"
            },
            {
                "check": "x_between",
                "landmarks": [4, 9, 13],  # thumb x between middle MCP (9) and ring MCP (13)
                "relation": "between",
                "weight": 0.4,
                "note": "Thumb tucked under 2 fingers only (between middle and ring MCPs)"
            },
            {
                "check": "y_compare",
                "landmarks": [4, 8],
                "relation": "gt",
                "weight": 0.2,
                "note": "Thumb tip below index tip (inside fist)"
            }
        ]
    },
    "O": {
        "description": "All fingers curved to meet thumb, forming an O circle",
        "conditions": [
            {
                "check": "distance",
                "landmarks": [4, 8],
                "relation": "lt",
                "threshold": 0.25,
                "weight": 0.5,
                "note": "Thumb and index tip very close (circle contact)"
            },
            {
                "check": "distance",
                "landmarks": [4, 12],
                "relation": "lt",
                "threshold": 0.35,
                "weight": 0.15,
                "note": "Middle tip near thumb"
            },
            {
                "check": "distance",
                "landmarks": [4, 16],
                "relation": "lt",
                "threshold": 0.35,
                "weight": 0.1,
                "note": "Ring tip near thumb"
            },
            {
                "check": "distance",
                "landmarks": [4, 20],
                "relation": "lt",
                "threshold": 0.35,
                "weight": 0.05,
                "note": "Pinky tip near thumb"
            },
            {
                "check": "extended_count",
                "landmarks": [8, 12, 16, 20],
                "relation": "lte",
                "threshold": 1,
                "weight": 0.2,
                "note": "Most fingers partially curled (not extended like F)"
            }
        ]
    },
    "P": {
        "description": "Hand pointing downward; index points forward/down, middle extends down, thumb out",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",  # index tip BELOW pip = pointing down
                "weight": 0.5,
                "note": "Index pointing downward"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "lt",
                "weight": 0.5,
                "note": "Middle extended (downward in P)"
            }
        ]
    },
    "Q": {
        "description": "Index and thumb pointing downward, similar to G rotated downward",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",
                "weight": 1.0,
                "note": "Index tip below PIP (pointing down)"
            }
        ]
    },
    "R": {
        "description": "Index and middle fingers extended and crossed (index over middle)",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "lt",
                "weight": 0.2,
                "note": "Index extended"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "lt",
                "weight": 0.2,
                "note": "Middle extended"
            },
            {
                "check": "x_compare",
                "landmarks": [8, 12],
                "relation": "gt",  # index tip x > middle tip x = crossed
                "weight": 0.4,
                "note": "Index crosses over middle (the defining R feature)"
            },
            {
                "check": "x_spread",
                "landmarks": [8, 12],
                "relation": "lt",
                "threshold": 0.06,
                "weight": 0.2,
                "note": "Fingers are close together (crossed, not spread like V)"
            }
        ]
    },
    "S": {
        "description": "Closed fist with thumb wrapping over the curled fingertips",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",
                "weight": 0.1,
                "note": "Index curled"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.1,
                "note": "Middle curled"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.1,
                "note": "Ring curled"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "gt",
                "weight": 0.1,
                "note": "Pinky curled"
            },
            {
                "check": "y_compare",
                "landmarks": [4, 8],
                "relation": "lt",  # thumb ABOVE index tip = covering it
                "weight": 0.3,
                "note": "Thumb wraps OVER fingertips (distinguishes S from A)"
            },
            {
                "check": "x_compare",
                "landmarks": [4, 8],
                "relation": "gt",
                "weight": 0.3,
                "note": "Thumb x in front of fist (covering position)"
            }
        ]
    },
    "T": {
        "description": "Thumb inserted between index and middle fingers, all fingers curled",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",
                "weight": 0.15,
                "note": "Index curled"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.15,
                "note": "Middle curled"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.1,
                "note": "Ring curled"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "gt",
                "weight": 0.1,
                "note": "Pinky curled"
            },
            {
                "check": "x_between",
                "landmarks": [4, 5, 9],  # thumb x between index MCP (5) and middle MCP (9)
                "relation": "between",
                "weight": 0.3,
                "note": "Thumb inserted between index and middle MCPs"
            },
            {
                "check": "y_compare",
                "landmarks": [4, 8],
                "relation": "lt",  # thumb above index tip = at knuckle level
                "weight": 0.2,
                "note": "Thumb at or above knuckle level (inserted position)"
            }
        ]
    },
    "U": {
        "description": "Index and middle fingers extended upward together (not spread)",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "lt",
                "weight": 0.2,
                "note": "Index extended upward"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "lt",
                "weight": 0.2,
                "note": "Middle extended upward"
            },
            {
                "check": "x_spread",
                "landmarks": [8, 12],
                "relation": "lt",
                "threshold": 0.05,
                "weight": 0.4,
                "note": "Fingers together (small gap — distinguishes U from V)"
            },
            {
                "check": "tip_height",
                "landmarks": [8, 5],
                "relation": "lt",
                "threshold": 0.0,
                "weight": 0.2,
                "note": "Tips significantly above MCPs (pointing up, not sideways like H)"
            }
        ]
    },
    "V": {
        "description": "Index and middle fingers extended in a V shape (spread apart)",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "lt",
                "weight": 0.2,
                "note": "Index extended"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "lt",
                "weight": 0.2,
                "note": "Middle extended"
            },
            {
                "check": "x_spread",
                "landmarks": [8, 12],
                "relation": "gt",
                "threshold": 0.07,
                "weight": 0.4,
                "note": "Fingers spread apart (the V shape — distinguishes from U)"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.1,
                "note": "Ring curled"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "gt",
                "weight": 0.1,
                "note": "Pinky curled"
            }
        ]
    },
    "W": {
        "description": "Index, middle, and ring fingers extended and spread, pinky curled",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "lt",
                "weight": 0.2,
                "note": "Index extended"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "lt",
                "weight": 0.2,
                "note": "Middle extended"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "lt",
                "weight": 0.2,
                "note": "Ring extended"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "gt",
                "weight": 0.2,
                "note": "Pinky curled (distinguishes W from B)"
            },
            {
                "check": "x_spread",
                "landmarks": [8, 16],
                "relation": "gt",
                "threshold": 0.08,
                "weight": 0.2,
                "note": "Index to ring spread (three fingers fanned out)"
            }
        ]
    },
    "X": {
        "description": "Index finger bent/hooked downward, all other fingers curled",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",  # tip BELOW pip = bent hook
                "weight": 0.4,
                "note": "Index finger bent/hooked (tip below PIP)"
            },
            {
                "check": "distance",
                "landmarks": [8, 5],
                "relation": "lt",
                "threshold": 0.40,
                "weight": 0.3,
                "note": "Index tip close to MCP (hooked toward palm)"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.1,
                "note": "Middle curled"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.1,
                "note": "Ring curled"
            },
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "gt",
                "weight": 0.1,
                "note": "Pinky curled"
            }
        ]
    },
    "Y": {
        "description": "Thumb and pinky extended, index/middle/ring fingers closed",
        "conditions": [
            {
                "check": "y_compare",
                "landmarks": [20, 18],
                "relation": "lt",
                "weight": 0.4,
                "note": "Pinky extended upward"
            },
            {
                "check": "x_compare",
                "landmarks": [4, 2],
                "relation": "lt",
                "threshold": 0.04,
                "weight": 0.4,
                "note": "Thumb extended sideways (distinguishes Y from I)"
            },
            {
                "check": "y_compare",
                "landmarks": [8, 6],
                "relation": "gt",
                "weight": 0.07,
                "note": "Index curled"
            },
            {
                "check": "y_compare",
                "landmarks": [12, 10],
                "relation": "gt",
                "weight": 0.07,
                "note": "Middle curled"
            },
            {
                "check": "y_compare",
                "landmarks": [16, 14],
                "relation": "gt",
                "weight": 0.06,
                "note": "Ring curled"
            }
        ]
    },
    "Z": {
        "description": "Index finger draws a Z shape in the air (dynamic sign)",
        "conditions": []  # Z is a dynamic/motion sign — cannot be captured by static landmark snapshot
    }
}