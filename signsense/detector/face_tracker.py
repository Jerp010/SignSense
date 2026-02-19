"""Face tracking module using MediaPipe Tasks Face Landmarker."""

import sys
from pathlib import Path

import numpy as np
from typing import Optional, Dict, Any, Tuple

from mediapipe.tasks.python import vision, BaseOptions
from mediapipe import Image, ImageFormat


# Key face landmark indices (468 total)
NOSE_TIP_INDEX = 1
CHIN_INDEX = 175

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _get_model_path() -> str:
    """Return path to face_landmarker.task, downloading if needed."""
    base = Path(__file__).resolve().parent.parent
    model_dir = base / "assets" / "models"
    model_path = model_dir / "face_landmarker.task"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        try:
            import urllib.request
            print("Downloading face_landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, str(model_path))
        except Exception as e:
            print(f"Failed to download model: {e}", file=sys.stderr)
            sys.exit(1)

    return str(model_path)


class FaceTracker:
    """
    MediaPipe Tasks Face Landmarker wrapper.
    Extracts nose tip and chin for spatial reference.
    """

    def __init__(self) -> None:
        """Initialize FaceTracker with minimal configuration for performance."""
        self._frame_timestamp_ms = 0
        self._timestamp_increment_ms = 33

        model_path = _get_model_path()
        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def update_timestamp_increment(self, fps: float) -> None:
        """Update timestamp increment based on actual FPS."""
        if fps > 0:
            self._timestamp_increment_ms = int(1000 / fps)

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a single frame for face detection.

        Args:
            frame: Input frame in RGB format (numpy array)

        Returns:
            Dictionary with landmarks, nose_tip, chin, detected or None
        """
        if frame is None:
            return None

        mp_image = Image(image_format=ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(
            mp_image, self._frame_timestamp_ms
        )
        self._frame_timestamp_ms += self._timestamp_increment_ms

        if not result.face_landmarks:
            return {"detected": False}

        landmarks = result.face_landmarks[0]
        height, width = frame.shape[:2]

        nose_tip = None
        chin = None
        if len(landmarks) > max(NOSE_TIP_INDEX, CHIN_INDEX):
            nose_lm = landmarks[NOSE_TIP_INDEX]
            chin_lm = landmarks[CHIN_INDEX]
            nose_tip = (
                int(max(0, min(1, nose_lm.x)) * width),
                int(max(0, min(1, nose_lm.y)) * height),
            )
            chin = (
                int(max(0, min(1, chin_lm.x)) * width),
                int(max(0, min(1, chin_lm.y)) * height),
            )

        return {
            "landmarks": landmarks,
            "nose_tip": nose_tip,
            "chin": chin,
            "detected": True,
        }

    def __del__(self) -> None:
        """Cleanup MediaPipe resources."""
        if hasattr(self, "_landmarker"):
            try:
                self._landmarker.close()
            except Exception:
                pass
