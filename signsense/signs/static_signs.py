"""Static ASL sign definitions for single-hand position recognition."""

from typing import Dict, List, Any


# TODO: Implement geometric matching algorithms for static sign recognition
# TODO: Add conditions based on:
#   - Finger extended/bent states
#   - Relative positions of landmarks
#   - Angles between joints
#   - Distances between key points

STATIC_SIGNS: Dict[str, Dict[str, Any]] = {
    "A": {
        "description": "Closed fist with thumb outside",
        "conditions": []
    },
    "B": {
        "description": "All fingers extended, thumb folded",
        "conditions": []
    },
    "C": {
        "description": "Curved hand shape like letter C",
        "conditions": []
    },
    "D": {
        "description": "Index finger extended upward, other fingers closed",
        "conditions": []
    },
    "E": {
        "description": "All fingers bent, thumb across palm",
        "conditions": []
    },
    "F": {
        "description": "Index and thumb touching, other fingers extended",
        "conditions": []
    },
    "G": {
        "description": "Index finger extended horizontally, thumb extended",
        "conditions": []
    },
    "H": {
        "description": "Index and middle fingers extended together",
        "conditions": []
    },
    "I": {
        "description": "Pinky extended upward, other fingers closed",
        "conditions": []
    },
    "K": {
        "description": "Index and middle fingers extended, thumb up",
        "conditions": []
    },
    "L": {
        "description": "Index finger and thumb extended, forming L shape",
        "conditions": []
    },
    "M": {
        "description": "Three fingers (index, middle, ring) folded, thumb inside",
        "conditions": []
    },
    "N": {
        "description": "Two fingers (index, middle) folded, thumb inside",
        "conditions": []
    },
    "O": {
        "description": "Fingers curved to form O shape",
        "conditions": []
    },
    "P": {
        "description": "Index finger pointing down, middle finger extended",
        "conditions": []
    },
    "Q": {
        "description": "Index finger and thumb extended downward",
        "conditions": []
    },
    "R": {
        "description": "Index and middle fingers crossed",
        "conditions": []
    },
    "S": {
        "description": "Closed fist",
        "conditions": []
    },
    "T": {
        "description": "Thumb between index and middle fingers",
        "conditions": []
    },
    "U": {
        "description": "Index and middle fingers extended upward together",
        "conditions": []
    },
    "V": {
        "description": "Index and middle fingers extended in V shape",
        "conditions": []
    },
    "W": {
        "description": "Three fingers (index, middle, ring) extended",
        "conditions": []
    },
    "X": {
        "description": "Index finger bent",
        "conditions": []
    },
    "Y": {
        "description": "Thumb and pinky extended, other fingers closed",
        "conditions": []
    },
    "Z": {
        "description": "Index finger drawing Z shape",
        "conditions": []
    }
}
