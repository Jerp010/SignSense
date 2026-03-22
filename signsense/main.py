"""
main.py
=======
SignSense — entry point.

Application state machine
--------------------------
  MAIN_MENU      Camera OFF.  Show main menu.
  LEVEL_SELECT   Camera OFF.  Show level selection.
  PLAY           Camera ON.   Stage-by-stage sign learning mode.
  DEBUG          Camera ON.   Raw classifier output with full score bars
                              (the original prototype view).

Transitions
-----------
  MAIN_MENU  → [Play]        → LEVEL_SELECT
  MAIN_MENU  → [Debug]       → DEBUG
  MAIN_MENU  → [Quit] / ESC  → exit

  LEVEL_SELECT → [letters]   → PLAY  (level_id="letters")
  LEVEL_SELECT → [Back] / ESC → MAIN_MENU

  PLAY    ESC                → MAIN_MENU  (camera released)
  DEBUG   ESC                → MAIN_MENU  (camera released)
"""

import sys
import concurrent.futures
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ensure logging subsystem is initialised immediately
# bring in the logger *instance* plus helper functions
from utils.logger import (
    logger,
    log_init,
    log_success,
    log_error,
    log_warning,
    log_session_start,
    log_session_end,
    TimingContext,
    perf_tracker,
)


if sys.version_info >= (3, 14):
    print(
        "SignSense requires Python 3.9–3.12. "
        f"Current: Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        "Create a venv with Python 3.11: py -3.12 -m venv .venv"
    )
    sys.exit(1)

import cv2
import time

from detector.hand_tracker           import HandTracker
from detector.face_tracker           import FaceTracker
from detector.asl_classifier_letters import ASLClassifierLetters
from utils.smoothing                 import PredictionSmoother
from ui.overlay                      import Overlay, SignHoldTimer
from ui.menu                         import MainMenu, LevelSelect, DebugMenu, RecordMenu
from ui.play_mode                    import PlayModeRenderer, StageTracker
from signs.sign_registry             import ACTIVE_SIGNS, SignType

# Global detector cache for model reuse
_detector_cache = {
    'hand_tracker': None,
    'face_tracker': None,
    'classifier': None
}

def get_cached_hand_tracker():
    if _detector_cache['hand_tracker'] is None:
        _detector_cache['hand_tracker'] = HandTracker(0.6, 0.6, 1)
    return _detector_cache['hand_tracker']

def get_cached_face_tracker():
    if _detector_cache['face_tracker'] is None:
        _detector_cache['face_tracker'] = FaceTracker()
    return _detector_cache['face_tracker']

def get_cached_classifier():
    if _detector_cache['classifier'] is None:
        _detector_cache['classifier'] = ASLClassifierLetters()
    return _detector_cache['classifier']

def initialize_detectors():
    """Initialize all detectors in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        hand_future = executor.submit(get_cached_hand_tracker)
        face_future = executor.submit(get_cached_face_tracker)
        classifier_future = executor.submit(get_cached_classifier)
        
        hand_tracker = hand_future.result()
        face_tracker = face_future.result()
        classifier = classifier_future.result()
        
    return hand_tracker, face_tracker, classifier

def open_camera(W=640, H=480):
    """Open camera with specified resolution and optimized buffer settings."""
    logger.debug(f"Attempting to open camera with resolution {W}x{H}")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Camera not accessible")
        return None
        
    # Configure camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    logger.debug("Camera opened successfully")
    return cap


def release_camera(cap):
    if cap and cap.isOpened():
        cap.release()


def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.flip(frame, 1)


# ---------------------------------------------------------------------------
# Window size helper
# ---------------------------------------------------------------------------

def get_window_size(win_name: str, default_w=640, default_h=480):
    """
    Return the current (w, h) of an OpenCV window.
    Falls back to defaults if the window isn't ready yet.
    """
    try:
        rect = cv2.getWindowImageRect(win_name)
        w, h = rect[2], rect[3]
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return default_w, default_h


# ---------------------------------------------------------------------------
# Mouse callback shim
# ---------------------------------------------------------------------------

_mouse_event = None   # (event_type, (x, y))

def _mouse_cb(event, x, y, flags, param):
    global _mouse_event
    if event == cv2.EVENT_MOUSEMOVE:
        _mouse_event = ("mouse_move",  (x, y))
    elif event == cv2.EVENT_LBUTTONDOWN:
        _mouse_event = ("mouse_click", (x, y))


# ---------------------------------------------------------------------------
# State: MAIN MENU  (no camera)
# ---------------------------------------------------------------------------

def run_main_menu(W=640, H=480) -> str:
    """Blocks until user picks an action. Returns 'play'|'debug'|'quit'."""
    WIN  = "SignSense"
    cv2.setMouseCallback(WIN, _mouse_cb)
    menu      = MainMenu(W, H)
    last_size = (W, H)

    global _mouse_event
    while True:
        cur_size = get_window_size(WIN, W, H)
        if cur_size != last_size:
            menu      = MainMenu(*cur_size)
            last_size = cur_size

        frame = menu.render()
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(16) & 0xFF
        result = menu.handle_event("key", key)
        if result:
            return result

        if _mouse_event:
            et, data = _mouse_event
            _mouse_event = None
            result = menu.handle_event(et, data)
            if result:
                return result

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"


# ---------------------------------------------------------------------------
# State: LEVEL SELECT  (no camera)
# ---------------------------------------------------------------------------

def run_level_select(W=640, H=480) -> str:
    """Returns level_id or 'back'."""
    WIN       = "SignSense"
    cv2.setMouseCallback(WIN, _mouse_cb)
    ls        = LevelSelect(W, H)
    last_size = (W, H)

    global _mouse_event
    while True:
        cur_size = get_window_size(WIN, W, H)
        if cur_size != last_size:
            ls        = LevelSelect(*cur_size)
            last_size = cur_size

        frame = ls.render()
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(16) & 0xFF
        result = ls.handle_event("key", key)
        if result:
            return result

        if _mouse_event:
            et, data = _mouse_event
            _mouse_event = None
            result = ls.handle_event(et, data)
            if result:
                return result

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            return "back"


# ---------------------------------------------------------------------------
# State: DEBUG MENU  (no camera)
# ---------------------------------------------------------------------------

def run_debug_menu(W=640, H=480) -> str:
    """Returns 'record'|'train'|'back'."""
    WIN       = "SignSense"
    cv2.setMouseCallback(WIN, _mouse_cb)
    menu      = DebugMenu(W, H)
    last_size = (W, H)

    global _mouse_event
    while True:
        cur_size = get_window_size(WIN, W, H)
        if cur_size != last_size:
            menu      = DebugMenu(*cur_size)
            last_size = cur_size

        frame = menu.render()
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(16) & 0xFF
        result = menu.handle_event("key", key)
        if result:
            return result

        if _mouse_event:
            et, data = _mouse_event
            _mouse_event = None
            result = menu.handle_event(et, data)
            if result:
                return result

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            return "back"


# ---------------------------------------------------------------------------
# State: RECORD MENU  (no camera)
# ---------------------------------------------------------------------------

def run_record_menu(W=640, H=480) -> str:
    """Returns 'record_static'|'record_dynamic'|'back'."""
    WIN       = "SignSense"
    cv2.setMouseCallback(WIN, _mouse_cb)
    menu      = RecordMenu(W, H)
    last_size = (W, H)

    global _mouse_event
    while True:
        cur_size = get_window_size(WIN, W, H)
        if cur_size != last_size:
            menu      = RecordMenu(*cur_size)
            last_size = cur_size

        frame = menu.render()
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(16) & 0xFF
        result = menu.handle_event("key", key)
        if result:
            return result

        if _mouse_event:
            et, data = _mouse_event
            _mouse_event = None
            result = menu.handle_event(et, data)
            if result:
                return result

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            return "back"


# ---------------------------------------------------------------------------
# State: PLAY MODE
# ---------------------------------------------------------------------------

def run_play_mode(level_id: str, W=640, H=480) -> str:
    """Returns 'menu' when done or ESC pressed."""
    # log entry time to track pre‑camera delay
    mode_entry = time.perf_counter()
    log_init("Play mode", f"level={level_id}")

    # Initialize detectors first (parallel)
    with TimingContext("Initialize play mode detectors and components"):
        hand_tracker, face_tracker, classifier = initialize_detectors()
        smoother = PredictionSmoother(buffer_size=5, min_confidence=3)
        signs_for_level = ACTIVE_SIGNS   # currently only one level
        stage_tracker = StageTracker(signs_for_level)
        renderer = PlayModeRenderer(W, H)
    
    # Open camera synchronously (more reliable)
    cam_start_time = time.perf_counter()
    cap = open_camera(W, H)
    cam_delay = time.perf_counter() - cam_start_time
    
    if cap is None or not cap.isOpened():
        log_error("Play mode", RuntimeError("camera open failed"))
        return "menu"
        
    total_startup = time.perf_counter() - mode_entry
    log_success(
        f"Open camera {W}x{H} | delay={cam_delay:.4f}s | startup={total_startup:.4f}s"
    )
    log_success(f"Play mode initialized with {len(signs_for_level)} signs")

    WIN = "SignSense"
    cv2.setMouseCallback(WIN, _mouse_cb)

    fps_start = time.time()
    fps_count = 0
    fps       = 30.0
    frame_num = 0  # running count for debug logging

    global _mouse_event
    # track overall camera run duration
    cam_start = time.time()
    try:
        while True:
            frame_start = time.perf_counter()
            frame = read_frame(cap)
            frame_time = time.perf_counter() - frame_start
            perf_tracker.record_timing("camera_read", frame_time)
            if frame is None:
                logger.warning("Camera frame read returned None, exiting play mode loop")
                break
            frame_num += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_data = hand_tracker.process_frame(rgb)
            landmarks  = hand_data.get("landmarks") if hand_data else None
            handedness = hand_data.get("handedness") if hand_data else None

            # --- Classify -----------------------------------------------
            current_sign    = stage_tracker.current_sign
            classifier_result = None

            if landmarks and current_sign:
                if current_sign.sign_type == SignType.STATIC:
                    # Targeted mode: only score the sign we're looking for
                    classifier_result = classifier.classify(
                        landmarks, handedness,
                        target_letter=current_sign.letter)
                else:
                    # Dynamic sign — classifier still runs for UI feedback
                    # but the stage tracker uses its own detector
                    classifier_result = classifier.classify(landmarks, handedness)

            # Smooth static results
            if current_sign and current_sign.sign_type == SignType.STATIC:
                raw = classifier_result["letter"] if classifier_result else None
                stable = smoother.add_prediction(raw)
                if stable and classifier_result:
                    classifier_result["letter"] = stable
                elif not stable:
                    classifier_result = None

            # --- Stage update -------------------------------------------
            confirmed = stage_tracker.update(
                classifier_result,
                landmarks=landmarks,
                handedness=handedness,
            )
            if confirmed:
                cur = stage_tracker.current_sign
                letter = cur.letter if cur else "?"
                logger.info(f"Confirmed letter {letter} at frame {frame_num}")
                perf_tracker.record_timing("letter_confirm", 0.0)  # marker, zero duration

            # --- FPS --------------------------------------------------------
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps       = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()
                hand_tracker.update_timestamp_increment(fps)
                face_tracker.update_timestamp_increment(fps)
                logger.debug(f"Frame {frame_num} | FPS: {fps:.1f}")

            # --- Render ---------------------------------------------------
            # Scale frame to current window size so layout fills the window
            win_w, win_h = get_window_size(WIN, W, H)
            if (win_w, win_h) != (frame.shape[1], frame.shape[0]):
                display_frame = cv2.resize(frame, (win_w, win_h))
            else:
                display_frame = frame

            output = renderer.render(
                display_frame, stage_tracker, classifier_result,
                fps, hand_data is not None)
            cv2.imshow(WIN, output)

            # --- Events ---------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                logger.info("ESC pressed, exiting play mode")
                return "menu"
            if key != 255:
                renderer.handle_event("key", key, stage_tracker)

            if _mouse_event:
                et, data = _mouse_event
                _mouse_event = None
                renderer.handle_event(et, data, stage_tracker)

            if stage_tracker.is_complete:
                # Show completion screen until ESC
                comp = renderer.render(frame, stage_tracker,
                                       None, fps, False)
                cv2.imshow(WIN, comp)
                while True:
                    k = cv2.waitKey(16) & 0xFF
                    if k == 27:
                        return "menu"
                    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                        return "menu"

            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                return "menu"

    except KeyboardInterrupt:
        pass
    finally:
        release_camera(cap)
        elapsed = time.time() - cam_start
        log_success(f"Camera ran for {elapsed:.2f}s")
        log_success(f"Frames processed: {frame_num}")

    return "menu"


# ---------------------------------------------------------------------------
# State: DEBUG MODE  (original prototype view)
# ---------------------------------------------------------------------------

def run_debug_mode(W=640, H=480) -> str:
    """Returns 'menu' when ESC pressed."""
    mode_entry = time.perf_counter()
    log_init("Debug mode")
    
    # Initialize detectors first (parallel)
    with TimingContext("Initialize debug mode components"):
        hand_tracker, face_tracker, classifier = initialize_detectors()
        smoother = PredictionSmoother(buffer_size=5, min_confidence=3)
        overlay = Overlay(window_title="SignSense - Debug")
        hold_timer = SignHoldTimer(hold_duration=1.5)
    
    # Open camera synchronously (more reliable)
    cam_start_time = time.perf_counter()
    cap = open_camera(W, H)
    cam_delay = time.perf_counter() - cam_start_time
    
    if cap is None or not cap.isOpened():
        log_error("Debug mode", RuntimeError("camera open failed"))
        return "menu"
        
    total_startup = time.perf_counter() - mode_entry
    log_success(
        f"Open camera {W}x{H} | delay={cam_delay:.4f}s | startup={total_startup:.4f}s"
    )
    log_success("Debug mode components initialized")

    WIN = "SignSense"
    cv2.setMouseCallback(WIN, _mouse_cb)

    fps_start = time.time()
    fps_count = 0
    fps       = 30.0
    frame_num = 0
    cam_start = time.time()  # track duration of camera usage in this mode for logging purposes

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                logger.warning("Camera frame read returned None, exiting debug mode loop")
                break
            frame_num += 1

            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_data = hand_tracker.process_frame(rgb)
            face_data = face_tracker.process_frame(rgb)

            classifier_result = None
            if hand_data and hand_data.get("landmarks"):
                classifier_result = classifier.classify(
                    hand_data["landmarks"],
                    handedness=hand_data.get("handedness"))

            raw    = classifier_result["letter"] if classifier_result else None
            stable = smoother.add_prediction(raw)
            if stable and classifier_result:
                classifier_result["letter"] = stable
            elif not stable:
                classifier_result = None

            confirmed = hold_timer.update(classifier_result)
            if confirmed:
                let = classifier_result["letter"] if classifier_result else "?"
                logger.info(f"Debug confirmed sign '{let}' (frame {frame_num})")
                perf_tracker.record_timing("letter_confirm", 0.0)

            if not hand_data:
                hold_timer.reset()

            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps       = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()
                hand_tracker.update_timestamp_increment(fps)
                face_tracker.update_timestamp_increment(fps)
                logger.debug(f"Debug frame {frame_num} | FPS: {fps:.1f}")

            win_w, win_h = get_window_size(WIN, W, H)
            if (win_w, win_h) != (frame.shape[1], frame.shape[0]):
                display_frame = cv2.resize(frame, (win_w, win_h))
            else:
                display_frame = frame

            output = overlay.draw(
                display_frame, hand_data, fps,
                face_data=face_data,
                classifier_result=classifier_result,
                hold_timer=hold_timer,
            )
            cv2.imshow(WIN, output)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                logger.info("ESC pressed, exiting debug mode")
                return "menu"

            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                return "menu"

    except KeyboardInterrupt:
        pass
    finally:
        release_camera(cap)
        elapsed = time.time() - cam_start
        log_success(f"Camera ran for {elapsed:.2f}s")
        log_success(f"Frames processed: {frame_num}")

    return "menu"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # log session boundaries
    log_session_start()

    DEFAULT_W, DEFAULT_H = 640, 480
    WIN = "SignSense"

    # WINDOW_NORMAL lets the user resize freely by dragging the window edge.
    # Set a minimum comfortable size via resizeWindow — the user can go larger.
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DEFAULT_W, DEFAULT_H)
    cv2.setMouseCallback(WIN, _mouse_cb)

    state     = "MAIN_MENU"
    _level_id = "letters"

    print("SignSense started.  Drag the window corner to resize.")
    try:
        while state != "QUIT":
            # Always pass the current window size so menus and layouts adapt.
            W, H = get_window_size(WIN, DEFAULT_W, DEFAULT_H)

            if state == "MAIN_MENU":
                action = run_main_menu(W, H)
                logger.info(f"Main menu action: {action}")
                if action == "play":
                    state = "LEVEL_SELECT"
                elif action == "debug":
                    state = "DEBUG_MENU"
                else:
                    state = "QUIT"
                logger.info(f"STATE: {state}")

            elif state == "DEBUG_MENU":
                action = run_debug_menu(W, H)
                logger.info(f"Debug menu action: {action}")
                if action == "record":
                    state = "RECORD_MENU"
                elif action in ("train", "back"):
                    state = "MAIN_MENU"
                else:
                    state = "MAIN_MENU"
                logger.info(f"STATE: {state}")

            elif state == "RECORD_MENU":
                action = run_record_menu(W, H)
                logger.info(f"Record menu action: {action}")
                if action == "record_static":
                    # Launch static landmark recorder
                    from ml.record_landmarks import LandmarkRecorder
                    recorder = LandmarkRecorder()
                    recorder.run()
                    state = "RECORD_MENU"  # Return to record menu after
                elif action == "record_dynamic":
                    # Launch dynamic gesture recorder
                    from ml.dynamic_recorder import DynamicSignRecorder
                    recorder = DynamicSignRecorder()
                    recorder.run()
                    state = "RECORD_MENU"  # Return to record menu after
                elif action == "back":
                    state = "DEBUG_MENU"
                else:
                    state = "DEBUG_MENU"
                logger.info(f"STATE: {state}")

            elif state == "LEVEL_SELECT":
                action = run_level_select(W, H)
                logger.info(f"Level select result: {action}")
                if action == "back":
                    state = "MAIN_MENU"
                elif action == "quit":
                    state = "QUIT"
                else:
                    state     = "PLAY"
                    _level_id = action
                logger.info(f"STATE: {state}")

            elif state == "PLAY":
                run_play_mode(_level_id, W, H)
                state = "MAIN_MENU"

            elif state == "DEBUG":
                run_debug_mode(W, H)
                state = "MAIN_MENU"

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        cv2.destroyAllWindows()
        print("SignSense closed.")
        log_session_end()


if __name__ == "__main__":
    main()