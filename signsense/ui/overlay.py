"""UI overlay module for rendering hand landmarks and application status."""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple

# Hand landmark connections (MediaPipe 21-point topology)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class Overlay:
    """
    Overlay rendering class for SignSense application.
    
    Handles drawing of hand landmarks, status information, FPS,
    and fingertip coordinates on the video frame.
    """
    
    def __init__(self, window_title: str = "SignSense - Prototype") -> None:
        """
        Initialize the Overlay.
        
        Args:
            window_title: Title for the application window
        """
        self.window_title = window_title

        # Color definitions
        self.color_green = (0, 255, 0)
        self.color_red = (0, 0, 255)
        self.color_white = (255, 255, 255)
        self.color_blue = (255, 0, 0)
        self.color_yellow = (0, 255, 255)
        self.color_cyan = (255, 255, 0)

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.line_type = cv2.LINE_AA
    
    def draw(
        self,
        frame: np.ndarray,
        hand_data: Optional[Dict[str, Any]],
        fps: float,
        face_data: Optional[Dict[str, Any]] = None,
        detected_letter: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw overlay on the frame.

        Args:
            frame: Input frame (BGR format)
            hand_data: Dictionary containing hand detection data or None
            fps: Current FPS value
            face_data: Face detection data or None
            detected_letter: Stable ASL letter ('A'-'F') or None

        Returns:
            Frame with overlay drawn
        """
        output_frame = frame.copy()

        self._draw_header_bar(output_frame)

        hand_detected = hand_data is not None
        self._draw_status(output_frame, hand_detected)

        face_detected = face_data is not None and face_data.get("detected", False)
        self._draw_face_status(output_frame, face_detected)

        self._draw_fps(output_frame, fps)
        self._draw_detected_letter(output_frame, detected_letter)

        if hand_detected and hand_data:
            landmarks = hand_data.get("landmarks")
            if landmarks:
                self._draw_landmarks(output_frame, landmarks)
                height, width = frame.shape[:2]
                index_fingertip = landmarks[8]
                x = int(index_fingertip.x * width)
                y = int(index_fingertip.y * height)
                self._draw_coordinates(output_frame, x, y)

        return output_frame
    
    def _draw_header_bar(self, frame: np.ndarray) -> None:
        """
        Draw header bar at the top of the frame.
        
        Args:
            frame: Frame to draw on
        """
        height, width = frame.shape[:2]
        bar_height = 50
        
        # Draw semi-transparent bar
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (width, bar_height),
            (30, 30, 30),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw title
        title_text = "SignSense - Prototype"
        text_size = cv2.getTextSize(
            title_text,
            self.font,
            self.font_scale + 0.2,
            self.font_thickness + 1
        )[0]
        text_x = (width - text_size[0]) // 2
        text_y = bar_height // 2 + text_size[1] // 2
        cv2.putText(
            frame,
            title_text,
            (text_x, text_y),
            self.font,
            self.font_scale + 0.2,
            self.color_white,
            self.font_thickness + 1,
            self.line_type
        )
    
    def _draw_face_status(self, frame: np.ndarray, face_detected: bool) -> None:
        """Draw face detection status."""
        status_text = "Face: Yes" if face_detected else "Face: No"
        status_color = self.color_green if face_detected else self.color_red
        text_x = 10
        text_y = 95
        text_size = cv2.getTextSize(status_text, self.font, self.font_scale, self.font_thickness)[0]
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0), -1,
        )
        cv2.putText(frame, status_text, (text_x, text_y), self.font, self.font_scale, status_color, self.font_thickness, self.line_type)

    def _draw_detected_letter(self, frame: np.ndarray, letter: Optional[str]) -> None:
        """Draw detected ASL letter (large, prominent)."""
        display = letter if letter else "?"
        text_x, text_y = 20, 155
        scale = 2.0
        thickness = 4
        text_size = cv2.getTextSize(display, self.font, scale, thickness)[0]
        cv2.rectangle(
            frame,
            (text_x - 10, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 10, text_y + 10),
            (0, 0, 0), -1,
        )
        cv2.putText(frame, display, (text_x, text_y), self.font, scale, self.color_cyan, thickness, self.line_type)

    def _draw_status(self, frame: np.ndarray, hand_detected: bool) -> None:
        """
        Draw hand detection status.
        
        Args:
            frame: Frame to draw on
            hand_detected: Whether a hand is currently detected
        """
        status_text = "Hand Detected" if hand_detected else "No Hand Detected"
        status_color = self.color_green if hand_detected else self.color_red
        
        # Position in top-left corner, below header
        text_x = 10
        text_y = 70
        
        # Draw background rectangle for better visibility
        text_size = cv2.getTextSize(
            status_text,
            self.font,
            self.font_scale,
            self.font_thickness
        )[0]
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            frame,
            status_text,
            (text_x, text_y),
            self.font,
            self.font_scale,
            status_color,
            self.font_thickness,
            self.line_type
        )
    
    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """
        Draw FPS counter.
        
        Args:
            frame: Frame to draw on
            fps: Current FPS value
        """
        fps_text = f"FPS: {fps:.1f}"
        height, width = frame.shape[:2]
        
        # Position in top-right corner
        text_size = cv2.getTextSize(
            fps_text,
            self.font,
            self.font_scale,
            self.font_thickness
        )[0]
        text_x = width - text_size[0] - 10
        text_y = 70
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            frame,
            fps_text,
            (text_x, text_y),
            self.font,
            self.font_scale,
            self.color_yellow,
            self.font_thickness,
            self.line_type
        )
    
    def _draw_coordinates(self, frame: np.ndarray, x: int, y: int) -> None:
        """
        Draw index fingertip coordinates.
        
        Args:
            frame: Frame to draw on
            x: X coordinate of index fingertip
            y: Y coordinate of index fingertip
        """
        coord_text = f"Index Tip: ({x}, {y})"
        
        # Position below letter display
        text_x = 10
        text_y = 200
        
        # Draw background rectangle
        text_size = cv2.getTextSize(
            coord_text,
            self.font,
            self.font_scale,
            self.font_thickness
        )[0]
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            frame,
            coord_text,
            (text_x, text_y),
            self.font,
            self.font_scale,
            self.color_white,
            self.font_thickness,
            self.line_type
        )
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks) -> None:
        """
        Draw hand landmarks using OpenCV (Tasks API format: list of landmarks).

        Args:
            frame: Frame to draw on
            landmarks: List of NormalizedLandmark objects with x, y, z
        """
        height, width = frame.shape[:2]
        pts = [
            (
                int(max(0, min(1, lm.x)) * width),
                int(max(0, min(1, lm.y)) * height),
            )
            for lm in landmarks
        ]

        # Draw connections
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(
                    frame, pts[a], pts[b],
                    (0, 255, 0), 2, cv2.LINE_AA
                )

        # Draw landmark points
        for (x, y) in pts:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Highlight index fingertip (landmark 8) with a circle
        if len(pts) > 8:
            tip_x, tip_y = pts[8]
            cv2.circle(frame, (tip_x, tip_y), 10, self.color_yellow, 3)
            cv2.circle(frame, (tip_x, tip_y), 5, self.color_yellow, -1)
