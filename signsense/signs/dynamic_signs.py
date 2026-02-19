"""Dynamic ASL sign definitions for multi-state gesture recognition."""

from typing import Dict, List, Any


# TODO: Implement sequential state validation using GestureStateMachine
# TODO: Add temporal constraints and transition conditions
# TODO: Support face-hand spatial validation for context-aware recognition

DYNAMIC_SIGNS: Dict[str, Dict[str, Any]] = {
    "HELLO": {
        "description": "Wave greeting gesture",
        "states": [
            {
                "description": "Hand near forehead",
                "conditions": []
            },
            {
                "description": "Hand away from forehead",
                "conditions": []
            }
        ]
    },
    "THANK_YOU": {
        "description": "Hand moving from chin forward",
        "states": [
            {
                "description": "Fingertips touching chin",
                "conditions": []
            },
            {
                "description": "Hand extended forward",
                "conditions": []
            }
        ]
    },
    "PLEASE": {
        "description": "Circular motion with flat hand",
        "states": [
            {
                "description": "Flat hand at chest",
                "conditions": []
            },
            {
                "description": "Circular motion",
                "conditions": []
            }
        ]
    },
    "YES": {
        "description": "Fist moving up and down",
        "states": [
            {
                "description": "Fist at lower position",
                "conditions": []
            },
            {
                "description": "Fist at upper position",
                "conditions": []
            }
        ]
    },
    "NO": {
        "description": "Index and middle fingers snapping together",
        "states": [
            {
                "description": "Fingers extended apart",
                "conditions": []
            },
            {
                "description": "Fingers snapped together",
                "conditions": []
            }
        ]
    },
    "GOODBYE": {
        "description": "Open hand waving side to side",
        "states": [
            {
                "description": "Hand at center",
                "conditions": []
            },
            {
                "description": "Hand moved to side",
                "conditions": []
            }
        ]
    }
}
