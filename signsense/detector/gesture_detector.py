"""Gesture detector for ASL signs.

This module implements detection for 5 ASL gesture signs:
HELLO, THANK YOU, NAME, YES, NO

Note: Hand gesture recognition should be trained via the dynamic recorder
instead of using hardcoded logic. This module retains only essential
detector functions for face tracking and non-hand-specific input methods.
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

from signsense.signs.gesture_definitions import (
    GestureClass,
    GESTURE_REGISTRY,
    HandShape,
    MovementPattern,
    get_gesture
)


# Standard confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# Hand landmark indices for finger detection
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# Wrist position
WRIST = 0


@dataclass
class DetectionResult:
    """Result of gesture detection.
    
    Attributes:
        sign_name: The detected sign name
        confidence: Detection confidence (0.0-1.0)
        hand_shape: Detected hand shape
        movement: Detected movement pattern
        is_gesture: Whether a gesture was detected
    """
    sign_name: Optional[str] = None
    confidence: float = 0.0
    hand_shape: Optional[str] = None
    movement: Optional[str] = None
    is_gesture: bool = False


class GestureDetector:
    """Modular detection pipeline for ASL gesture signs.
    
    Provides detection for one-handed signs, two-handed signs,
    facial expressions, and lip movements.
    """
    
    def __init__(
        self,
        hand_tracker: Any = None,
        face_tracker: Any = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> None:
        """Initialize the gesture detector.
        
        Args:
            hand_tracker: HandTracker instance for hand detection
            face_tracker: FaceTracker instance for face detection
            confidence_threshold: Minimum confidence for detection
        """
        self.hand_tracker = hand_tracker
        self.face_tracker = face_tracker
        self.confidence_threshold = confidence_threshold
        
        # History for movement detection
        self._hand_position_history: List[Dict[str, Any]] = []
        self._face_position_history: List[Dict[str, Any]] = []
        self._max_history_length = 30  # ~1 second at 30fps
        
        # Current detection state
        self._current_hand_data: Optional[Dict[str, Any]] = None
        self._current_face_data: Optional[Dict[str, Any]] = None
    
    def update(self, frame: np.ndarray) -> None:
        """Update detector with new frame data.
        
        Args:
            frame: Input frame in RGB format
        """
        # Process hand tracking
        if self.hand_tracker:
            hand_result = self.hand_tracker.process_frame(frame)
            self._current_hand_data = hand_result
            
            if hand_result:
                self._hand_position_history.append(hand_result)
                if len(self._hand_position_history) > self._max_history_length:
                    self._hand_position_history.pop(0)
        
        # Process face tracking
        if self.face_tracker:
            face_result = self.face_tracker.process_frame(frame)
            self._current_face_data = face_result
            
            if face_result and face_result.get("detected"):
                self._face_position_history.append(face_result)
                if len(self._face_position_history) > self._max_history_length:
                    self._face_position_history.pop(0)
    
    def detect(self, target_gesture: str = None) -> DetectionResult:
        """Detect any gesture from current frame data.
        
        Args:
            target_gesture: Optional target gesture for focused detection.
        
        Returns:
            DetectionResult with detected gesture or empty result
        """
        # Try one-handed detection first
        result = self.single_hand_detect(target_gesture)
        if result.is_gesture:
            return result
        
        # Try two-handed detection
        result = self.two_hand_detect()
        if result.is_gesture:
            return result
        
        # Try facial detection
        result = self.facial_detect()
        if result.is_gesture:
            return result
        
        return DetectionResult()
    
    def single_hand_detect(self, target_gesture: str = None) -> DetectionResult:
        """Detect one-handed signs using dominant hand tracker.
        
        Note: This method is deprecated. Hand gesture recognition should be
        trained via the dynamic recorder instead of using hardcoded logic.
        
        Args:
            target_gesture: Optional target gesture name for focused detection.
                           If provided, only checks for that gesture (play mode optimization).
        
        Signs: HELLO, THANK YOU, NAME, YES, NO
        
        Returns:
            DetectionResult for one-handed signs
        """
        if not self._current_hand_data:
            return DetectionResult()
        
        hand_data = self._current_hand_data
        landmarks = hand_data.get("landmarks")
        if not landmarks:
            return DetectionResult()
        
        # Get hand shape
        hand_shape = self._detect_hand_shape(landmarks)
        
        # Get dominant hand (right hand by default)
        handedness = hand_data.get("handedness", "Right")
        
        # Get hand position for movement detection
        wrist = landmarks[WRIST]
        hand_position = (wrist.x, wrist.y, wrist.z)
        
        # Detect movement pattern
        movement = self._detect_movement_pattern(hand_position)
        
        # Check each one-handed gesture
        gestures_to_check = [(target_gesture, GESTURE_REGISTRY[target_gesture])] if target_gesture and target_gesture in GESTURE_REGISTRY else GESTURE_REGISTRY.items()
        
        for sign_name, gesture in gestures_to_check:
            if gesture.hands_required != 1:
                continue
            
            # Check hand shape match
            if not self._match_hand_shape(hand_shape, gesture.hand_shape):
                continue
            
            # Check movement pattern match
            if not self._match_movement(movement, gesture.movement_pattern):
                continue
            
            # NEW: Check target region (hand position matches expected body region)
            if not self._validate_hand_region(hand_position, gesture):
                continue
            
            # Calculate confidence based on matches
            confidence = self._calculate_confidence(hand_shape, movement, gesture)
            
            if confidence >= self.confidence_threshold:
                return DetectionResult(
                    sign_name=sign_name,
                    confidence=confidence,
                    hand_shape=hand_shape,
                    movement=movement,
                    is_gesture=True
                )
        
        return DetectionResult(
            hand_shape=hand_shape,
            movement=movement
        )
    
    def two_hand_detect(self) -> DetectionResult:
        """Detect two-handed signs tracking both hands independently.
        
        Currently no two-handed signs in the 9-sign set.
        
        Returns:
            DetectionResult for two-handed signs
        """
        # Note: The current 9-sign set doesn't include two-handed signs
        # This method is provided for extensibility
        return DetectionResult()
    
    def facial_detect(self) -> DetectionResult:
        """Detect signs requiring facial recognition.
        
        Note: This method is deprecated. Hand gesture recognition should be
        trained via the dynamic recorder instead of using hardcoded logic.
        
        Signs: YES (head nod), NO (head shake)
        
        Returns:
            DetectionResult for facial-based signs
        """
        if not self._current_face_data:
            return DetectionResult()
        
        face_data = self._current_face_data
        if not face_data.get("detected"):
            return DetectionResult()
        
        # Get facial landmarks
        landmarks = face_data.get("landmarks")
        if not landmarks:
            return DetectionResult()
        
        # Detect head movement
        head_movement = self._detect_head_movement()
        
        # Check for YES (head nod)
        if head_movement == "head_nod":
            # Check if hand is in thumb-up position for YES
            if self._current_hand_data:
                hand_shape = self._detect_hand_shape(
                    self._current_hand_data.get("landmarks")
                )
                if hand_shape == "fist_thumb_up":
                    return DetectionResult(
                        sign_name="YES",
                        confidence=self.confidence_threshold,
                        hand_shape=hand_shape,
                        movement="head_nod",
                        is_gesture=True
                    )
            # Or just head nod alone counts as YES
            return DetectionResult(
                sign_name="YES",
                confidence=self.confidence_threshold,
                movement="head_nod",
                is_gesture=True
            )
        
        # Check for NO (head shake)
        if head_movement == "head_side_to_side":
            return DetectionResult(
                sign_name="NO",
                confidence=self.confidence_threshold,
                movement="head_side_to_side",
                is_gesture=True
            )
        
        return DetectionResult()
    
    def lip_read_detect(self) -> DetectionResult:
        """Detect mouth movement for lip reading.
        
        Note: This method is deprecated. Hand gesture recognition should be
        trained via the dynamic recorder instead of using hardcoded logic.
        
        Note: Current 5-sign set doesn't require lip reading.
        This method is provided for extensibility.
        
        Returns:
            DetectionResult for lip-based signs
        """
        # Note: The current 9-sign set doesn't include lip-reading signs
        # This method is provided for extensibility
        return DetectionResult()
    
    def _detect_hand_shape(self, landmarks: List[Any]) -> str:
        """Detect hand shape from landmarks.
        
        Note: This method is deprecated. Hand gesture recognition should be
        trained via the dynamic recorder instead of using hardcoded logic.
        
        Args:
            landmarks: List of 21 hand landmarks
        
        Returns:
            Detected hand shape string
        """
        # Deprecated: Use TrainedDynamicDetector for gesture recognition
        return "unknown"
    
    def _match_hand_shape(self, detected: str, expected: HandShape) -> bool:
        """Check if detected hand shape matches expected.
        
        Note: This method is deprecated. Hand gesture recognition should be
        trained via the dynamic recorder instead of using hardcoded logic.
        
        Args:
            detected: Detected hand shape
            expected: Expected hand shape from GestureClass
        
        Returns:
            True if shapes match
        """
        # Deprecated: Use TrainedDynamicDetector for gesture recognition
        return False
    
    def _validate_hand_region(self, hand_position: Tuple[float, float, float], gesture: GestureClass) -> bool:
        """Validate that hand is in the expected body region for the gesture.
        
        Note: This method is deprecated. Hand gesture recognition should be
        trained via the dynamic recorder instead of using hardcoded logic.
        
        Args:
            hand_position: (x, y, z) position of wrist
            gesture: GestureClass to validate against
        
        Returns:
            True if hand is in expected region
        """
        # Deprecated: Use TrainedDynamicDetector for gesture recognition
        return True

    def _detect_movement_pattern(self, current_position: Tuple[float, float, float]) -> str:
        """Detect movement pattern from hand position history.
        
        Note: This method is deprecated. Hand gesture recognition should be
        trained via the dynamic recorder instead of using hardcoded logic.
        
        Args:
            current_position: Current hand position (x, y, z)
        
        Returns:
            Detected movement pattern
        """
        # Deprecated: Use TrainedDynamicDetector for gesture recognition
        return "static"
    
    def _match_movement(self, detected: str, expected: MovementPattern) -> bool:
        """Check if detected movement matches expected.
        
        Note: This method is deprecated. Hand gesture recognition should be
        trained via the dynamic recorder instead of using hardcoded logic.
        
        Args:
            detected: Detected movement pattern
            expected: Expected movement pattern from GestureClass
        
        Returns:
            True if movements match
        """
        # Deprecated: Use TrainedDynamicDetector for gesture recognition
        return False
    
    def _detect_head_movement(self) -> str:
        """Detect head movement from face position history.
        
        Returns:
            Detected head movement pattern
        """
        if len(self._face_position_history) < 3:
            return "static"
        
        # Get chin positions
        chin_positions = [
            f.get("chin")
            for f in self._face_position_history[-10:]
            if f.get("chin")
        ]
        
        if len(chin_positions) < 3:
            return "static"
        
        # Calculate movement
        first_pos = chin_positions[0]
        last_pos = chin_positions[-1]
        
        dy = last_pos[1] - first_pos[1]  # Vertical (nod)
        dx = last_pos[0] - first_pos[0]  # Horizontal (shake)
        
        # Detect nod (vertical movement)
        if abs(dy) > 0.02:
            return "head_nod"
        
        # Detect shake (horizontal movement)
        if abs(dx) > 0.02:
            return "head_side_to_side"
        
        return "static"
    
    def _calculate_confidence(
        self,
        hand_shape: str,
        movement: str,
        gesture: GestureClass
    ) -> float:
        """Calculate detection confidence.
        
        Note: This method is deprecated. Hand gesture recognition should be
        trained via the dynamic recorder instead of using hardcoded logic.
        
        Args:
            hand_shape: Detected hand shape
            movement: Detected movement pattern
            gesture: GestureClass to match against
        
        Returns:
            Confidence score (0.0-1.0)
        """
        # Deprecated: Use TrainedDynamicDetector for gesture recognition
        return 0.0
    
    def reset(self) -> None:
        """Reset detector state."""
        self._hand_position_history.clear()
        self._face_position_history.clear()
        self._current_hand_data = None
        self._current_face_data = None


def create_detector(
    hand_tracker: Any = None,
    face_tracker: Any = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> GestureDetector:
    """Create a GestureDetector instance.
    
    Args:
        hand_tracker: HandTracker instance
        face_tracker: FaceTracker instance
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        Configured GestureDetector instance
    """
    return GestureDetector(
        hand_tracker=hand_tracker,
        face_tracker=face_tracker,
        confidence_threshold=confidence_threshold
    )
