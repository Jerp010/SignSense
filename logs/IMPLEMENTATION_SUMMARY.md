# SignSense Logging Implementation Summary

## What Was Added

### 1. Core Logging Module
**File**: `signsense/utils/logger.py` (570+ lines)

Complete logging framework with:
- **File logging**: All events saved to timestamped log files in `logs/` directory
- **Console logging**: User-friendly colored output (green=info, yellow=warning, red=error)
- **Timing context manager**: `TimingContext()` tracks how long operations take
- **Decorator**: `@time_it()` to time any function automatically
- **Performance tracker**: Collects statistics on all operations
- **Helper functions**:
  - `log_init()` - Log component initialization
  - `log_error()` - Log errors with context
  - `log_success()` - Log successful operations
  - `log_warning()` - Log warnings
  - `log_session_start()` / `log_session_end()` - Session bookends

### 2. Main Application Integration
**File**: `signsense/main.py` (updated)

Added logging at every level:
- **Session level**: `log_session_start()` / `log_session_end()` + performance report
- **State transitions**: Each state (MAIN_MENU -> LEVEL_SELECT -> PLAY/DEBUG) is logged
- **Component initialization**: All detectors, classifiers, and renderers timed
- **Runtime events**: FPS measurements, frame counts, window events
- **Error handling**: Try-catch blocks around processors with detailed error logging
- **User actions**: Button clicks, ESC presses, menu selections

### 3. Documentation
**Files created**:
- `LOGGING.md` - Comprehensive logging guide (600+ lines)
- `DEBUG_QUICK_START.txt` - Quick reference for finding issues
- `.gitignore` - Updated to ignore `logs/` directory

## How It Works

### Log File Creation
When you run `main.py`, a new log file is created:
```
logs/
  signsense_20260301_120530.log  <- Session timestamp
```

### Logging Hierarchy
```
DEBUG       File only - detailed frame-by-frame info
   ↓
INFO        Console + File - normal operations ([OK]/->/<-)
   ↓
WARNING     Console + File - slow operations ([WARN])
   ↓
ERROR       Console + File - component failures ([FAIL])
   ↓
CRITICAL    Console + File - system failures
```

### When Logs Are Created

| Event | Log Entry |
|-------|-----------|
| App starts | `SIGNSENSE SESSION STARTED` + Python version, platform |
| Camera opens | `INIT: Initializing Camera` -> `SUCCESS: Camera opened and configured` |
| Detector loads | `DONE: Initialize play mode detectors ... | Time: 5.23s` |
| Level selected | `<- Level select result: letters` |
| Play mode starts | `-> Entering PLAY MODE \| level=letters` |
| Frame processed | `DEBUG: Frame 150 \| FPS: 30.0` |
| Error occurs | `FAILED: Classifier ... RuntimeError: ...` |
| Component loads slow | `SLOW DONE: Initialize ... | Time: 8.34s` |
| App closes | `SIGNSENSE SESSION ENDED` + performance report |

## Key Features

### 1. **Timing Tracking**
Everything that takes >100ms is timed:
```
DONE: Open camera 640x480 | Time: 0.0021s
DONE: Initialize play mode detectors | Time: 5.2341s  [Warning >1s]
```

### 2. **Error Capture**
All errors logged with context:
```
Error in HandTracker: RuntimeError: Model inference failed [frame 42]
Error in Overlay: TypeError: ...missing required positional argument... [frame 125]
```

### 3. **Performance Summary**
After each session, automatic report:
```
======================================================================
PERFORMANCE REPORT
======================================================================

Timing Statistics:
  Open camera 640x480              | Count:   1 | Avg:  0.0021s | Min:  0.0021s | Max:  0.0021s
  Initialize play mode detectors   | Count:   1 | Avg:  5.2341s | Min:  5.2341s | Max:  5.2341s
  Process frame                    | Count: 150 | Avg:  0.0331s | Min:  0.0021s | Max:  0.1523s

Errors Encountered:
  Classifier                       | Count: 3
  FaceTracker                      | Count: 1
```

### 4. **Frame-Level Diagnostics**
Debug mode logs every Nth frame:
```
DEBUG: Frame 15 | FPS: 30.0
DEBUG: Frame 45 | FPS: 30.1
DEBUG: Frame 123 | FPS: 29.8
```

## Reading Logs

### Quick View (Windows PowerShell)
```powershell
# Show last 50 lines of most recent log
Get-Content (Get-Item logs/ | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName | Select-Object -Last 50

# Show all errors
Get-Content logs/signsense_*.log | Select-String "Error|ERROR|FAILED"

# Show all warnings
Get-Content logs/signsense_*.log | Select-String "[WARN]|WARNING"
```

### Log File Location
```
D:\CODE FOLDER JEFF\SignSense\logs\signsense_YYYYMMDD_HHMMSS.log
```

## Common Debug Scenarios

### Issue: Slow Load Times
Look for warnings with high times:
```
SLOW DONE: Initialize play mode detectors | Time: 8.2341s
```
Solutions:
- Check if hand_landmarker.task/face_landmarker.task exist
- Consider smaller models or different model formats
- Check CPU/GPU availability

### Issue: Camera Not Working
Look for errors:
```
FAILED: Open camera 640x480 | FileNotFoundError: Camera not available
```
Solutions:
- Close other apps using camera (Zoom, Teams, etc.)
- Check camera connection
- Restart application

### Issue: FPS Dropping
Watch for decreasing FPS:
```
DEBUG: Frame 150 | FPS: 30.0
DEBUG: Frame 160 | FPS: 25.3
DEBUG: Frame 170 | FPS: 18.2
```
Solutions:
- Close background apps
- Lower window resolution
- Check if processing is hanging (look for missing frame debug entries)

### Issue: Random Crashes
Look for the first ERROR:
```
Error in Classifier: RuntimeError: ... [frame 42]
```
Then look for patterns - if same frame count each session, likely frame data issue.

## Integration with Your Workflow

1. **During development**: Check console output for INFO/WARNING levels
2. **When debugging**: Open the log file to see full DEBUG detail
3. **For performance**: Review PERFORMANCE REPORT at session end
4. **For failures**: Search for "ERROR" or "FAILED" in log file

## Customization

### To adjust log detail:
Edit `signsense/utils/logger.py`:
```python
# More verbose (show DEBUG in console)
console_handler.setLevel(logging.DEBUG)

# Less verbose (show only WARNING+)
console_handler.setLevel(logging.WARNING)
```

### To change log location:
```python
LOG_DIR = Path(__file__).parent.parent.parent / "your_log_dir"
```

### To add custom logging in detector modules:
```python
from utils.logger import logger, log_error, TimingContext

# Log errors
try:
    result = model.process(frame)
except Exception as e:
    log_error("MyDetector", e, f"frame {frame_number}")

# Time operations
with TimingContext("Process frame"):
    # ... your code ...
    pass
```

## Files Modified/Created

| Path | Status | Purpose |
|------|--------|---------|
| `signsense/utils/logger.py` | **CREATED** | Core logging framework |
| `signsense/main.py` | UPDATED | Integrated logging throughout |
| `LOGGING.md` | **CREATED** | Comprehensive documentation |
| `DEBUG_QUICK_START.txt` | **CREATED** | Quick reference guide |
| `.gitignore` | UPDATED | Added `logs/` directory |

## Next Steps

1. **Run the application**:
   ```powershell
   cd d:\CODE FOLDER JEFF\SignSense\signsense
   python main.py
   ```

2. **Check the log file**:
   ```powershell
   Get-Content ../logs/signsense*.log | Select-Object -Last 50
   ```

3. **Look for any issues** in the log (search for ERROR, [FAIL], or [WARN])

4. **Share logs when reporting bugs** - they contain all diagnostic info

## Summary

You now have:
- [OK] Automatic logging to timestamped files
- [OK] Colored console output for quick feedback
- [OK] Timing information for all operations
- [OK] Performance statistics at session end
- [OK] Error capture with context (frame numbers, component names)
- [OK] State transition tracking for debugging flow
- [OK] Comprehensive documentation for log analysis

All logs go to `logs/signsense_*.log` and are never committed to git.
