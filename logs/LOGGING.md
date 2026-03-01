# SignSense Logging System

## Overview

SignSense includes a comprehensive logging system that helps you debug issues, track failures, and identify performance bottlenecks. All logs are saved to disk with full detail, while the console shows a simplified view.

## Log Files

Logs are saved in the `logs/` directory at the workspace root:
```
logs/
  signsense_20260301_120530.log
  signsense_20260301_124015.log
  ...
```

Each session creates a new timestamped log file (format: `YYYYMMDD_HHMMSS`).

## What Gets Logged

### Initialization & Component Loading
- Application startup information (Python version, platform)
- Camera initialization and configuration
- Detector loading (HandTracker, FaceTracker, ASLClassifierLetters)
- Display components (renderers, overlays)
- Timing for each initialization step

### Runtime Events
  Each time the camera is opened we log two values: the time spent inside `open_camera()` itself (`delay`) and the total startup duration from mode entry until the camera became ready (`startup`).
- **Detections**: Hand/face tracking updates (logged at DEBUG level).
- **Stage tracker events**: Dynamic sign completion, advancement to next sign, and other state changes.  When a letter is confirmed the log records the letter and frame number (also marked in performance stats).- **User actions**: Main menu and level‑select button clicks, ESC presses, window events

### Errors & Failures
- Camera failures (camera unavailable, frame read errors)
- Detector errors (HandTracker, FaceTracker, ASLClassifierLetters failures)
- Rendering errors (Overlay, PlayModeRenderer failures)
- Component initialization failures
- Each error logs the frame number and context for easier debugging

### Performance Metrics
- Component initialization times (with warnings for >1s, >5s)
- FPS measurements during gameplay
- Frame processing statistics
- Performance summary at session end

## Log Levels

### Console Output (INFO and above)
- **INFO** (INFO symbols): Normal operation flow
- **WARNING**: Issues that need attention (slow loads, errors)
- **ERROR**: Component failures

### File Output (DEBUG and above)
- **DEBUG** (detailed): Frame numbers, FPS, resize events
- **INFO**: Same as console
- **WARNING**: Same as console
- **ERROR**: Same as console
- **CRITICAL**: System-level failures

## Reading a Log File

### Example Structure
```
2026-03-01 12:05:30 | INFO     | __main__                | main                 | ======================================================================
2026-03-01 12:05:30 | INFO     | __main__                | main                 | SIGNSENSE SESSION STARTED
2026-03-01 12:05:30 | INFO     | __main__                | main                 | ======================================================================
2026-03-01 12:05:30 | INFO     | __main__                | main                 | Log file: D:/CODE FOLDER JEFF/SignSense/logs/signsense_20260301_120530.log
2026-03-01 12:05:30 | INFO     | __main__                | main                 | Python: 3.11.0 (main, Oct 24 2022, 18:26:48) [MSC v.1933 64 bit (AMD64)]
2026-03-01 12:05:30 | INFO     | __main__                | main                 | Platform: win32
2026-03-01 12:05:30 | INFO     | __main__                | main                 | ======================================================================
2026-03-01 12:05:31 | INFO     | __main__                | main                 | SignSense window created: 640x480
2026-03-01 12:05:31 | INFO     | __main__                | main                 | Drag the window corner to resize.
2026-03-01 12:05:31 | INFO     | __main__                | main                 | STATE: MAIN_MENU
2026-03-01 12:05:31 | INFO     | __main__                | main                 | -> Rendering MAIN MENU
2026-03-01 12:05:35 | INFO     | __main__                | main                 | Main menu action: play
2026-03-01 12:05:35 | INFO     | __main__                | main                 | STATE: LEVEL_SELECT
2026-03-01 12:05:35 | INFO     | __main__                | main                 | -> Rendering LEVEL SELECT
2026-03-01 12:05:38 | INFO     | __main__                | main                 | Level select result: letters
2026-03-01 12:05:38 | INFO     | __main__                | main                 | STATE: PLAY
2026-03-01 12:05:38 | INFO     | __main__                | main                 | Entering PLAY MODE | level=letters
2026-03-01 12:05:38 | INFO     | utils.logger            | log_init             | INIT: Initializing Camera - resolution 640x480
2026-03-01 12:05:38 | INFO     | utils.logger            | log_success          | SUCCESS: Open camera 640x480 | Time: 0.0021s
2026-03-01 12:05:38 | INFO     | utils.logger            | log_success          | SUCCESS: Camera opened and configured
2026-03-01 12:05:38 | INFO     | utils.logger            | log_success          | SUCCESS: Initialize play mode detectors and components | Time: 5.2341s
2026-03-01 12:05:43 | WARNING  | utils.logger            | <lambda>             | SLOW DONE: Initialize play mode detectors and components | Time: 5.2341s
2026-03-01 12:05:43 | INFO     | utils.logger            | log_success          | SUCCESS: Play mode initialized with 10 signs
2026-03-01 12:05:43 | DEBUG    | utils.logger            | <lambda>             | Frame 15 | FPS: 30.0
2026-03-01 12:05:44 | DEBUG    | utils.logger            | <lambda>             | Frame 45 | FPS: 30.1
```

### Key Columns
- **Timestamp**: When the event occurred
- **Level**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Module**: Which file the log came from
- **Function**: Which function created the log
- **Message**: The actual log message with emoji indicators

## Common Issues to Look For

### Slow Component Loading
Look for WARNING entries with times > 1 second:
```
SLOW DONE: Initialize play mode detectors and components | Time: 5.2341s
```
This indicates model loading might be slow. Check hand_landmarker.task and face_landmarker.task file sizes.

### Camera Issues
```
FAILED: Open camera 640x480 | FileNotFoundError: Camera not available
```
This means your camera isn't accessible. Check:
- Is your camera in use by another application?
- Do you have the right permissions?
- Is your camera properly connected?

### Frame Processing Errors
```
Error in Classifier: RuntimeError: Model inference failed [frame 42]
```
This indicates a model or data issue. Frame 42 likely has unusual hand poses that the model can't handle.

### Performance Bottlenecks
Look at consistent DEBUG messages:
```
Frame 1500 | FPS: 25.3
Frame 1501 | FPS: 24.8
Frame 1502 | FPS: 22.1
```
If FPS drops significantly, you're hitting performance limits. Check what's running in the background.

## Debugging Tips

1. **Check the log file first** - Always review the timestamped log file, not just console output
   
2. **Look for the first ERROR** - Scroll to the first ERROR line, this usually indicates the root cause

3. **Use timestamps to correlate events** - If something crashed at 12:06:15, look at that time in the log

4. **Check initialization times** - If HandTracker takes 3+ seconds to load, models may be corrupt or network is slow

5. **Look for repeating errors** - Errors that occur every 10 frames suggest a specific pattern (e.g., certain hand poses)

6. **Performance summary at end** - The final PERFORMANCE REPORT shows statistics:
   ```
   ======================================================================
   PERFORMANCE REPORT
   ======================================================================
   
   Timing Statistics:
     Open camera 640x480              | Count:   1 | Avg:  0.0021s | Min:  0.0021s | Max:  0.0021s
     Initialize play mode detectors   | Count:   1 | Avg:  5.2341s | Min:  5.2341s | Max:  5.2341s
   ```

## Session Files

After each session closes, a summary report is generated showing:
- Total files processed (frame count)
- Errors encountered
- Average operation times
- Slowest operations
The report now also includes timings for play/debug mode initialization and any errors recorded by the performance tracker.
## In Code: Using the Logger

### Basic Logging
```python
from utils.logger import logger, log_init, log_error, log_success, TimingContext

# Log normal operations
log_init("HandTracker", "loading model...")
log_success("HandTracker ready")

# Log errors
try:
    result = classifier.classify(landmarks, handedness)
except Exception as e:
    log_error("Classifier", e, f"frame {frame_number}")

# Time operations
with TimingContext("Process frame"):
    # ... your code ...
    pass

# Decorator for timing functions
from utils.logger import time_it

@time_it("Load model")
def load_my_model():
    # ... code ...
    pass
```

### Performance Tracking
```python
from utils.logger import perf_tracker

# Record timing (manual)
start = time.perf_counter()
# ... do work ...
perf_tracker.record_timing("operation_name", time.perf_counter() - start)

# Record error
perf_tracker.record_error("operation_name", "error message")

# Get stats
stats = perf_tracker.get_stats("operation_name")
# {'count': 10, 'min': 0.01, 'max': 0.05, 'avg': 0.025, 'total': 0.25}
```

## Log Retention

Log files are saved indefinitely but the logs/ directory is in .gitignore to avoid committing them. You can:
- Delete logs manually when they pile up
- Archive old logs for analysis
- Set up a cleanup script if needed

## Customizing Logging

The logger is configured in `utils/logger.py`. To adjust:

- **Change log level**: Modify `file_handler.setLevel()` or `console_handler.setLevel()`
- **Change log format**: Edit the `Formatter` strings
- **Add colors**: Modify the `COLORS` dictionary in `ColoredFormatter`
- **Change log directory**: Modify `LOG_DIR` path

## Next Steps

Run the application and check the log file to verify everything is working:

```powershell
python main.py
```

Then check the latest log in `logs/`:
```powershell
Get-Content logs/ -Tail 50
```

Or open the log file in your editor for full analysis.
