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
from detector.asl_classifier_letters import ASLClassifierLetters
from utils.smoothing import PredictionSmoother
from ui.overlay import Overlay, SignHoldTimer  # SignHoldTimer added
from signs.dynamic_signs import create_j_tracker


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
    asl_classifier_letters = ASLClassifierLetters()
    smoother = PredictionSmoother(buffer_size=5, min_confidence=3)
    overlay = Overlay(window_title="SignSense - Prototype")
    hold_timer = SignHoldTimer(hold_duration=1.5)  # seconds to hold before confirming
    # dynamic sign tracker for letter J
    j_tracker = create_j_tracker()

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

            # classifier.classify() now returns a dict {"letter", "confidence", "scores"}
            # or None if no sign detected above threshold
            classifier_result = None
            if hand_data and hand_data.get("landmarks"):
                handedness = hand_data.get("handedness")  # pass if your tracker provides it
                classifier_result = asl_classifier_letters.classify(
                    hand_data["landmarks"],
                    handedness=handedness,
                )

            # Smoother still works on the letter string for stability
            raw_letter = classifier_result["letter"] if classifier_result else None
            stable_letter = smoother.add_prediction(raw_letter)

            # Rebuild result with smoothed letter so UI shows stabilized output
            if stable_letter and classifier_result:
                classifier_result["letter"] = stable_letter
            elif not stable_letter:
                classifier_result = None

            # Update hold timer — returns confirmed letter when hold is complete
            confirmed_letter = hold_timer.update(classifier_result)
            if confirmed_letter:
                print(f"Confirmed: {confirmed_letter}")

            # dynamic J tracker update - only expose UI once stage1 has been
            # satisfied.  stage_idx is 0‑based; 0 = not started, 1 = handshape seen.
            dynamic_info = None
            if hand_data and hand_data.get("landmarks"):
                stage_idx = j_tracker.update(hand_data.get("landmarks"), classifier_result)
                if stage_idx > 0:
                    dynamic_info = {
                        "name": "J",
                        "stage": stage_idx,
                        "total": 2,
                        "description": j_tracker.stage_description(),
                    }
            else:
                dynamic_info = None
                j_tracker.reset()

            # Reset hold timer when hand leaves frame
            if not hand_data:
                hold_timer.reset()


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
                classifier_result=classifier_result,  # replaces detected_letter
                hold_timer=hold_timer,
                dynamic_info=dynamic_info,
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