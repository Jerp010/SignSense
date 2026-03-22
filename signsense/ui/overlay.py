"""UI overlay module for rendering hand landmarks and application status."""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
import time

# Import HAND_CONNECTIONS from detector to avoid duplication
from detector.hand_tracker import HAND_CONNECTIONS

# Active letters in A-H scope (used for score bar display)
# 'I' and 'J' are included for debugging/expansion
ACTIVE_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]


class SignHoldTimer:
    """
    Tracks how long a specific ASL sign has been held consistently.
    A sign is "confirmed" when the same letter is held above confidence threshold
    for HOLD_DURATION seconds without switching.

    Usage:
        timer = SignHoldTimer(hold_duration=1.5)
        result = timer.update(classifier_result)  # returns confirmed letter or None
    """

    def __init__(self, hold_duration: float = 1.5) -> None:
        """
        Args:
            hold_duration: Seconds the same letter must be held to confirm (default 1.5s)
        """
        self.hold_duration = hold_duration
        self._current_letter: Optional[str] = None
        self._hold_start: Optional[float] = None
        self._confirmed_letter: Optional[str] = None
        self._last_confirmed_letter: Optional[str] = None
        self._confirmation_time: Optional[float] = None

    @property
    def current_letter(self) -> Optional[str]:
        """The letter currently being tracked (held but not yet confirmed)."""
        return self._current_letter

    @property
    def hold_progress(self) -> float:
        """Progress toward confirmation (0.0 to 1.0)."""
        if self._current_letter is None or self._hold_start is None:
            return 0.0
        elapsed = time.time() - self._hold_start
        return min(1.0, elapsed / self.hold_duration)

    @property
    def confirmed_letter(self) -> Optional[str]:
        """The last fully confirmed letter."""
        return self._confirmed_letter

    @property
    def is_confirmed(self) -> bool:
        """True for a brief window after a letter is just confirmed."""
        if self._confirmation_time is None:
            return False
        return (time.time() - self._confirmation_time) < 1.0  # show for 1s

    def update(self, classifier_result: Optional[Dict]) -> Optional[str]:
        """
        Feed in the latest classifier result each frame.
        Returns the confirmed letter if this frame completed confirmation, else None.
        """
        letter = classifier_result["letter"] if classifier_result else None

        if letter != self._current_letter:
            # Sign changed - restart timer
            self._current_letter = letter
            self._hold_start = time.time() if letter else None
            return None

        if letter is None:
            return None

        # Same letter still held - check if duration reached
        elapsed = time.time() - self._hold_start
        if elapsed >= self.hold_duration:
            # Avoid re-confirming the same letter repeatedly
            if letter != self._last_confirmed_letter:
                self._confirmed_letter = letter
                self._last_confirmed_letter = letter
                self._confirmation_time = time.time()
                # Reset so user can sign again after lifting hand
                self._hold_start = time.time()  # prevents immediate re-trigger
                return letter

        return None

    def reset(self) -> None:
        """Reset all state (e.g. when hand leaves frame)."""
        self._current_letter = None
        self._hold_start = None
        self._last_confirmed_letter = None


class Overlay:
    """
    Overlay rendering class for SignSense application.

    Handles drawing of hand landmarks, status information, FPS,
    sign hold timer, confidence bar, and per-letter score display.
    """

    def __init__(self, window_title: str = "SignSense - Prototype") -> None:
        """
        Initialize the Overlay.

        Args:
            window_title: Title for the application window (unused, kept for API compatibility)
        """

        # Color definitions (BGR format for OpenCV)
        self.color_green = (0, 255, 0)
        self.color_red = (0, 0, 255)
        self.color_white = (255, 255, 255)
        self.color_blue = (255, 0, 0)
        self.color_yellow = (0, 255, 255)
        self.color_cyan = (255, 255, 0)
        self.color_orange = (0, 165, 255)
        self.color_gray = (160, 160, 160)
        self.color_dark = (30, 30, 30)
        self.color_confirmed = (0, 220, 100)  # bright green for confirmed state

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
        classifier_result: Optional[Dict] = None,
        hold_timer: Optional[SignHoldTimer] = None,
        dynamic_info: Optional[Dict[str, Any]] = None,  # name/stage data for multi-stage signs
    ) -> np.ndarray:
        """
        Draw overlay on the frame.

        Args:
            frame: Input frame (BGR format)
            hand_data: Dictionary containing hand detection data or None
            fps: Current FPS value
            face_data: Face detection data or None
            classifier_result: Dict from ASLClassifier.classify() containing letter, confidence, scores
            hold_timer: SignHoldTimer instance for progress display

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

        # Draw classifier result + hold timer panel (right side)
        self._draw_sign_panel(output_frame, classifier_result, hold_timer, dynamic_info)

        # Draw score bars for all active letters (bottom panel)
        if classifier_result is not None:
            self._draw_score_bars(output_frame, classifier_result.get("scores", {}))

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
        """Draw header bar at the top of the frame."""
        height, width = frame.shape[:2]
        bar_height = 50

        # Draw semi-transparent bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw title
        title_text = "SignSense - Prototype"
        text_size = cv2.getTextSize(title_text, self.font, self.font_scale + 0.2, self.font_thickness + 1)[0]
        text_x = (width - text_size[0]) // 2
        text_y = bar_height // 2 + text_size[1] // 2
        cv2.putText(frame, title_text, (text_x, text_y), self.font,
                    self.font_scale + 0.2, self.color_white, self.font_thickness + 1, self.line_type)

    def _draw_face_status(self, frame: np.ndarray, face_detected: bool) -> None:
        """Draw face detection status."""
        status_text = "Face: Yes" if face_detected else "Face: No"
        status_color = self.color_green if face_detected else self.color_red
        text_x, text_y = 10, 95
        text_size = cv2.getTextSize(status_text, self.font, self.font_scale, self.font_thickness)[0]
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (text_x, text_y), self.font,
                    self.font_scale, status_color, self.font_thickness, self.line_type)

    def _draw_sign_panel(
        self,
        frame: np.ndarray,
        classifier_result: Optional[Dict],
        hold_timer: Optional["SignHoldTimer"],
        dynamic_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Draw the main sign recognition panel on the right side.
        Shows: detected letter (large), confidence percentage, hold progress arc.
        """
        height, width = frame.shape[:2]
        panel_w = 200
        panel_x = width - panel_w - 10
        panel_y = 60
        panel_h = 220

        # Semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 10, panel_y), (panel_x + panel_w, panel_y + panel_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (panel_x - 10, panel_y), (panel_x + panel_w, panel_y + panel_h),
                      (80, 80, 80), 1)

        # --- Letter display ---
        letter = None
        confidence = 0.0
        if classifier_result:
            letter = classifier_result.get("letter")
            confidence = classifier_result.get("confidence", 0.0)

        # Check if just confirmed
        is_confirmed = hold_timer.is_confirmed if hold_timer else False
        confirmed_letter = hold_timer.confirmed_letter if hold_timer else None

        # Large letter
        display_letter = letter if letter else "?"
        letter_color = self.color_confirmed if is_confirmed else (self.color_cyan if letter else self.color_gray)
        letter_scale = 3.0
        letter_thickness = 5
        letter_size = cv2.getTextSize(display_letter, self.font, letter_scale, letter_thickness)[0]
        lx = panel_x + (panel_w - letter_size[0]) // 2 - 5
        ly = panel_y + 90
        cv2.putText(frame, display_letter, (lx, ly), self.font,
                    letter_scale, letter_color, letter_thickness, self.line_type)

        # Confirmed flash label
        if is_confirmed:
            cv2.putText(frame, "CONFIRMED!", (panel_x - 5, panel_y + 115), self.font,
                        0.55, self.color_confirmed, 2, self.line_type)

        # --- Confidence bar ---
        bar_x = panel_x
        bar_y = panel_y + 130
        bar_w = panel_w - 10
        bar_h = 14
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        # Fill
        if confidence > 0:
            fill_w = int(bar_w * confidence)
            conf_color = self._confidence_color(confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), conf_color, -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
        # Label
        conf_text = f"Conf: {confidence * 100:.0f}%"
        cv2.putText(frame, conf_text, (bar_x, bar_y - 4), self.font,
                    0.45, self.color_white, 1, self.line_type)

        # --- Hold progress bar ---
        hold_y = bar_y + 30
        hold_progress = hold_timer.hold_progress if hold_timer else 0.0
        hold_letter = hold_timer.current_letter if hold_timer else None
        # Background
        cv2.rectangle(frame, (bar_x, hold_y), (bar_x + bar_w, hold_y + bar_h), (40, 40, 40), -1)
        # Fill (orange → green as it fills)
        if hold_progress > 0:
            fill_w = int(bar_w * hold_progress)
            hold_color = self.color_confirmed if hold_progress >= 1.0 else self.color_orange
            cv2.rectangle(frame, (bar_x, hold_y), (bar_x + fill_w, hold_y + bar_h), hold_color, -1)
        cv2.rectangle(frame, (bar_x, hold_y), (bar_x + bar_w, hold_y + bar_h), (100, 100, 100), 1)
        # Label
        hold_label = f"Hold: {hold_progress * 100:.0f}%"
        cv2.putText(frame, hold_label, (bar_x, hold_y - 4), self.font,
                    0.45, self.color_white, 1, self.line_type)

        # Instruction text
        instr_y = hold_y + 30
        instr = "Hold sign to confirm" if (letter and hold_progress < 1.0) else ("" if not letter else "")
        if instr:
            cv2.putText(frame, instr, (bar_x, instr_y), self.font,
                        0.38, self.color_gray, 1, self.line_type)

        # --- Dynamic stage indicator ---
        # only display once the tracker has advanced past the initial state
        if dynamic_info and dynamic_info.get("stage", 0) > 0:
            stage = dynamic_info.get("stage", 0)
            total = dynamic_info.get("total", None)
            desc = dynamic_info.get("description") or ""
            stage_text = f"Stage: {stage}"
            if total:
                stage_text += f"/{total}"
            if desc:
                stage_text += f" ({desc})"
            dy = instr_y + 20
            cv2.putText(frame, stage_text, (bar_x, dy), self.font,
                        0.38, self.color_yellow, 1, self.line_type)

    def _draw_score_bars(self, frame: np.ndarray, scores: Dict[str, float]) -> None:
        """
        Draw a horizontal bar for each active letter showing its current score.
        Displayed at the bottom of the frame as a mini diagnostic panel.
        """
        if not scores:
            return

        height, width = frame.shape[:2]
        bar_area_h = 100
        bar_area_y = height - bar_area_h - 5
        bar_area_x = 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_area_x - 5, bar_area_y - 5),
                      (bar_area_x + 300, bar_area_y + bar_area_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Label
        cv2.putText(frame, "Scores:", (bar_area_x, bar_area_y + 12), self.font,
                    0.42, self.color_gray, 1, self.line_type)

        # Each letter as a mini horizontal bar
        bar_max_w = 200
        bar_h = 9
        col_w = 70
        row_h = 13
        letters_per_row = 4

        for i, letter in enumerate(ACTIVE_LETTERS):
            score = scores.get(letter, 0.0)
            row = i // letters_per_row
            col = i % letters_per_row

            x = bar_area_x + col * col_w
            y = bar_area_y + 22 + row * (bar_h + row_h)

            # Letter label
            is_best = score == max(scores.values()) and score > 0
            label_color = self.color_cyan if is_best else self.color_white
            cv2.putText(frame, f"{letter}:", (x, y + bar_h), self.font,
                        0.40, label_color, 1, self.line_type)

            # Bar background
            bx = x + 18
            cv2.rectangle(frame, (bx, y), (bx + bar_max_w // 4, y + bar_h), (50, 50, 50), -1)

            # Bar fill
            if score > 0:
                fill = int((bar_max_w // 4) * score)
                bar_col = self._confidence_color(score)
                cv2.rectangle(frame, (bx, y), (bx + fill, y + bar_h), bar_col, -1)

            # Score text
            cv2.putText(frame, f"{score:.2f}", (bx + (bar_max_w // 4) + 2, y + bar_h),
                        self.font, 0.35, self.color_gray, 1, self.line_type)

    def _confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Return BGR color on a red→yellow→green gradient based on confidence value."""
        if confidence < 0.5:
            # Red to yellow (0.0 → 0.5)
            ratio = confidence / 0.5
            return (0, int(255 * ratio), 255)
        else:
            # Yellow to green (0.5 → 1.0)
            ratio = (confidence - 0.5) / 0.5
            return (0, 255, int(255 * (1 - ratio)))

    def _draw_status(self, frame: np.ndarray, hand_detected: bool) -> None:
        """Draw hand detection status."""
        status_text = "Hand Detected" if hand_detected else "No Hand Detected"
        status_color = self.color_green if hand_detected else self.color_red

        # Position in top-left corner, below header
        text_x, text_y = 10, 70

        # Draw background rectangle for better visibility
        text_size = cv2.getTextSize(status_text, self.font, self.font_scale, self.font_thickness)[0]
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)

        cv2.putText(frame, status_text, (text_x, text_y), self.font,
                    self.font_scale, status_color, self.font_thickness, self.line_type)

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS counter."""
        fps_text = f"FPS: {fps:.1f}"
        height, width = frame.shape[:2]

        # Position in top-right area (left of sign panel)
        text_size = cv2.getTextSize(fps_text, self.font, self.font_scale, self.font_thickness)[0]
        text_x = width - text_size[0] - 220  # offset left of the sign panel
        text_y = 70

        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)

        cv2.putText(frame, fps_text, (text_x, text_y), self.font,
                    self.font_scale, self.color_yellow, self.font_thickness, self.line_type)

    def _draw_coordinates(self, frame: np.ndarray, x: int, y: int) -> None:
        """Draw index fingertip coordinates."""
        coord_text = f"Index Tip: ({x}, {y})"
        text_x, text_y = 10, 120

        text_size = cv2.getTextSize(coord_text, self.font, self.font_scale, self.font_thickness)[0]
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)

        cv2.putText(frame, coord_text, (text_x, text_y), self.font,
                    self.font_scale, self.color_white, self.font_thickness, self.line_type)

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
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)

        # Draw landmark points
        for (x, y) in pts:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Highlight index fingertip (landmark 8) with a circle
        if len(pts) > 8:
            tip_x, tip_y = pts[8]
            cv2.circle(frame, (tip_x, tip_y), 10, self.color_yellow, 3)
            cv2.circle(frame, (tip_x, tip_y), 5, self.color_yellow, -1)