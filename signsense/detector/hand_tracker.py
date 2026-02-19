"""Hand tracking module using MediaPipe Tasks Hand Landmarker."""

import sys
from pathlib import Path

import numpy as np
from typing import Optional, Dict, Any

from mediapipe.tasks.python import vision, BaseOptions
from mediapipe import Image, ImageFormat


# Hand landmark connections for drawing (21 points)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _get_model_path() -> str:
    """Return path to hand_landmarker.task, downloading if needed."""
    base = Path(__file__).resolve().parent.parent
    model_dir = base / "assets" / "models"
    model_path = model_dir / "hand_landmarker.task"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        try:
            import urllib.request
            print("Downloading hand_landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, str(model_path))
        except Exception as e:
            print(f"Failed to download model: {e}", file=sys.stderr)
            sys.exit(1)

    return str(model_path)


class HandTracker:
    """
    MediaPipe Tasks Hand Landmarker wrapper for real-time hand tracking.
    Optimized for reliability during fast movement.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        max_num_hands: int = 1,
    ) -> None:
        """
        Initialize the HandTracker with tuned parameters.

        Args:
            min_detection_confidence: Minimum confidence for detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for tracking (0.0-1.0)
            max_num_hands: Maximum number of hands (1 for performance)
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands
        self._frame_timestamp_ms = 0
        self._timestamp_increment_ms = 33  # ~30 fps default

        model_path = _get_model_path()
        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def update_timestamp_increment(self, fps: float) -> None:
        """Update timestamp increment based on actual FPS for continuous tracking."""
        if fps > 0:
            self._timestamp_increment_ms = int(1000 / fps)

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a single frame for hand detection.

        Args:
            frame: Input frame in RGB format (numpy array)

        Returns:
            Dictionary with landmarks, handedness, confidence or None
        """
        if frame is None:
            return None

        mp_image = Image(image_format=ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(
            mp_image, self._frame_timestamp_ms
        )
        self._frame_timestamp_ms += self._timestamp_increment_ms

        if not result.hand_landmarks:
            return None

        hand_landmarks = result.hand_landmarks[0]
        handedness = None
        if hasattr(result, "handedness") and result.handedness and result.handedness[0]:
            h = result.handedness[0]
            if hasattr(h, "categories") and h.categories:
                handedness = h.categories[0].category_name
            elif hasattr(h, "display_name"):
                handedness = h.display_name

        return {
            "landmarks": hand_landmarks,
            "handedness": handedness,
            "confidence": self.min_detection_confidence,
        }

    def __del__(self) -> None:
        """Cleanup MediaPipe resources."""
        if hasattr(self, "_landmarker"):
            try:
                self._landmarker.close()
            except Exception:
                pass
