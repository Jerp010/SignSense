"""SignSense - Main entry point for the application."""

import sys

if sys.version_info >= (3, 14):
    print(
        "SignSense requires Python 3.9 through 3.12. "
        f"Current: Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        "Create a venv with Python 3.11 or 3.12: "
        "py -3.12 -m venv .venv && .venv\\Scripts\\activate",
    )
    sys.exit(1)

import cv2
import time

from detector.hand_tracker import HandTracker
from detector.face_tracker import FaceTracker
from detector.asl_classifier import ASLClassifier
from utils.smoothing import PredictionSmoother
from ui.overlay import Overlay


def main() -> None:
    """
    Main application loop.
    Initializes camera, detectors, classifier, smoother, and overlay.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Part 1: Camera performance optimization
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Part 2: Hand tracker with tuned params
    hand_tracker = HandTracker(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        max_num_hands=1,
    )
    face_tracker = FaceTracker()
    asl_classifier = ASLClassifier()
    smoother = PredictionSmoother(buffer_size=5, min_confidence=3)
    overlay = Overlay(window_title="SignSense - Prototype")

    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 30.0

    print("SignSense started. Press ESC to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_data = hand_tracker.process_frame(rgb_frame)
            face_data = face_tracker.process_frame(rgb_frame)

            raw_letter = None
            if hand_data and hand_data.get("landmarks"):
                raw_letter = asl_classifier.classify(hand_data["landmarks"])
            stable_letter = smoother.add_prediction(raw_letter)

            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()
                hand_tracker.update_timestamp_increment(fps)
                face_tracker.update_timestamp_increment(fps)

            output_frame = overlay.draw(
                frame,
                hand_data,
                fps,
                face_data=face_data,
                detected_letter=stable_letter,
            )
            cv2.imshow(overlay.window_title, output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("SignSense closed.")


if __name__ == "__main__":
    main()
