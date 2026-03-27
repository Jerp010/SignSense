"""Data-driven gesture definitions for ASL signs.

This module defines anatomical requirements for 9 ASL gesture signs:
HELLO, THANK YOU, NAME, GOOD, HELP, WATER, YES, NO, BAD

Each sign is defined with its hand requirements, facial expressions,
hand shapes, and movement patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class HandShape(Enum):
    """Standard ASL hand shape classifications."""
    FLAT_HAND_5 = "5"  # Flat open hand, all fingers extended
    N_SHAPE = "N"      # Index and middle fingers up (N-shape)
    G_SHAPE = "G"      # Index up, thumb touching middle finger
    OPEN_HAND_5 = "5"  # Same as FLAT_HAND_5
    W_SHAPE = "W"      # Three fingers extended (W-shape)
    FIST_THUMB_UP = "Fist"  # Thumb-up fist
    INDEX_FINGER_1 = "1"    # Index finger extended only
    Y_SHAPE = "Y"      # Thumb and pinky extended (Y-shape)


class MovementPattern(Enum):
    """Movement patterns for ASL signs."""
    TOUCH_FOREHEAD_MOVE_OUT = "touch_forehead_move_out"  # Hello sign
    CHIN_TO_FORWARD = "chin_to_forward"  # Thank you, Water
    CHEEK_SLIGHT_MOVE = "cheek_slight_move"  # Name sign
    CHIN_TO_DOWN_OUT = "chin_to_down_out"  # Good, Bad
    BODY_TO_UP_OUT = "body_to_up_out"  # Help sign
    HEAD_NOD = "head_nod"  # Yes sign
    HEAD_SIDE_TO_SIDE = "head_side_to_side"  # No sign


class FacialExpression(Enum):
    """Facial expressions for ASL signs."""
    SMILE = "smile"
    NONE = "none"


@dataclass
class GestureClass:
    """Defines the anatomical requirements for an ASL sign gesture.
    
    Attributes:
        name: Sign name (e.g., "HELLO", "THANK YOU")
        hands_required: Number of hands required (1 or 2)
        uses_face: Whether facial expression is required
        uses_lips: Whether lip reading is required for this sign
        hand_shape: The required hand shape for this sign
        movement_pattern: The movement pattern for this sign
        confidence_threshold: Minimum confidence for detection (default 0.7)
        dominant_hand: Whether this sign uses the dominant hand specifically
        target_region: Body region where the sign is performed
    """
    name: str
    hands_required: int
    uses_face: bool
    uses_lips: bool
    hand_shape: HandShape
    movement_pattern: MovementPattern
    confidence_threshold: float = 0.7
    dominant_hand: bool = True  # Most signs use dominant hand
    target_region: str = ""  # e.g., "forehead", "chin", "cheek"


# Gesture registry mapping sign names to GestureClass instances
GESTURE_REGISTRY: Dict[str, GestureClass] = {}


def _register_gesture(gesture: GestureClass) -> None:
    """Register a gesture in the global registry."""
    GESTURE_REGISTRY[gesture.name] = gesture


# Define all 9 ASL gesture signs

# HELLO: 1 hand (dominant), smile, flat hand (5), touch forehead then move out
_register_gesture(GestureClass(
    name="HELLO",
    hands_required=1,
    uses_face=True,
    uses_lips=False,
    hand_shape=HandShape.FLAT_HAND_5,
    movement_pattern=MovementPattern.TOUCH_FOREHEAD_MOVE_OUT,
    confidence_threshold=0.7,
    dominant_hand=True,
    target_region="forehead"
))

# THANK YOU: 1 hand, nod, flat hand (5), chin to forward
_register_gesture(GestureClass(
    name="THANK YOU",
    hands_required=1,
    uses_face=True,
    uses_lips=False,
    hand_shape=HandShape.FLAT_HAND_5,
    movement_pattern=MovementPattern.CHIN_TO_FORWARD,
    confidence_threshold=0.7,
    dominant_hand=True,
    target_region="chin"
))

# NAME: 1 hand, none, N-shape (index+middle up), near cheek, slight move
_register_gesture(GestureClass(
    name="NAME",
    hands_required=1,
    uses_face=False,
    uses_lips=False,
    hand_shape=HandShape.N_SHAPE,
    movement_pattern=MovementPattern.CHEEK_SLIGHT_MOVE,
    confidence_threshold=0.7,
    dominant_hand=True,
    target_region="cheek"
))

# YES: 1 or head, head nod, thumb-up fist, up-down motion
_register_gesture(GestureClass(
    name="YES",
    hands_required=1,  # Can be hand or just head
    uses_face=True,
    uses_lips=False,
    hand_shape=HandShape.FIST_THUMB_UP,
    movement_pattern=MovementPattern.HEAD_NOD,
    confidence_threshold=0.7,
    dominant_hand=True,
    target_region="head"
))

# NO: 1 or head, head shake, index finger (1), side-to-side
_register_gesture(GestureClass(
    name="NO",
    hands_required=1,  # Can be hand or just head
    uses_face=True,
    uses_lips=False,
    hand_shape=HandShape.INDEX_FINGER_1,
    movement_pattern=MovementPattern.HEAD_SIDE_TO_SIDE,
    confidence_threshold=0.7,
    dominant_hand=True,
    target_region="head"
))


def get_gesture(name: str) -> Optional[GestureClass]:
    """Get a gesture definition by name.
    
    Args:
        name: The sign name (e.g., "HELLO", "THANK YOU")
    
    Returns:
        GestureClass instance or None if not found
    """
    return GESTURE_REGISTRY.get(name.upper())


def get_all_gestures() -> Dict[str, GestureClass]:
    """Get all registered gestures.
    
    Returns:
        Dictionary mapping gesture names to GestureClass instances
    """
    return GESTURE_REGISTRY.copy()


def get_gestures_by_hand_count(hands: int) -> Dict[str, GestureClass]:
    """Get all gestures requiring a specific number of hands.
    
    Args:
        hands: Number of hands (1 or 2)
    
    Returns:
        Dictionary of matching gestures
    """
    return {
        name: gesture 
        for name, gesture in GESTURE_REGISTRY.items()
        if gesture.hands_required == hands
    }


def get_gestures_requiring_face() -> Dict[str, GestureClass]:
    """Get all gestures that require facial expression.
    
    Returns:
        Dictionary of gestures requiring facial expression
    """
    return {
        name: gesture 
        for name, gesture in GESTURE_REGISTRY.items()
        if gesture.uses_face
    }