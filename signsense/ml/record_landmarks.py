"""
ml/record_landmarks.py
=====================
Interactive CSV recorder for ASL hand landmark training data.

Supports:
- Custom gesture labels via text input
- Adding more samples to existing labels
- Continuous recording across sessions

Usage:
  python -m ml.record_landmarks

Controls:
  - T: Enter text mode to type custom label
  - A-Z: Quick select letter label
  - SPACE: Start/stop recording for current label
  - [: save and exit
  - ESC: exit without saving
  - Close window: exit without saving

Output:
  - Appends to ml/data/landmarks.csv across sessions
  - Backs up to ml/data/backups/landmarks_<timestamp>.csv on each save
  - Format: label, x0,y0,z0, x1,y1,z1, ..., x20,y20,z20 (64 columns total)
"""

import cv2
import csv
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# Add parent directory to path so we can import signsense modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signsense.detector.hand_tracker import HandTracker
from signsense.ml.model import LandmarkNormaliser


class LandmarkRecorder:
    """Record hand landmarks to CSV file, organised by ASL letter label."""

    def __init__(self, output_dir: str = None):
        # Determine the correct output directory relative to the script location
        if output_dir is None:
            script_dir = Path(__file__).parent
            # The correct data directory is at the root level (SignSense/ml/data), not inside signsense
            self.output_dir = script_dir.parent.parent / "ml" / "data"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "landmarks.csv"
        
        self.normaliser = LandmarkNormaliser()
        self.hand_tracker = HandTracker()
        
        # Recording state
        self.active_label: Optional[str] = None
        self.is_recording: bool = False
        self.frame_count_for_label: int = 0
        self.total_rows_recorded: int = 0
        self.recorded_data: list = []  # Store data in memory until saved
        
        # Input mode
        self.text_input_mode: bool = False
        self.current_text: str = ""
        
        # Load existing data stats
        self.label_counts: Dict[str, int] = {}
        self._load_existing_counts()
        
        # Create CSV header if doesn't exist
        self._ensure_csv_header()

    def _load_existing_counts(self):
        """Load existing sample counts per label."""
        if not self.csv_path.exists():
            return
        
        try:
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row.get("label", "")
                    if label:
                        self.label_counts[label] = self.label_counts.get(label, 0) + 1
        except Exception:
            pass
        
        self.total_rows_recorded = sum(self.label_counts.values())
        print(f"Loaded existing data: {self.total_rows_recorded} samples")
        if self.label_counts:
            print(f"Labels: {list(self.label_counts.keys())}")

    def _ensure_csv_header(self) -> None:
        """Create CSV with header if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["label"] + [f"f{i}" for i in range(63)]
                writer.writerow(header)

    def _append_row(self, label: str, normalized_features) -> None:
        """Append one landmark row to memory buffer."""
        row = [label] + normalized_features.tolist()
        self.recorded_data.append(row)
        self.total_rows_recorded += 1
        self.frame_count_for_label += 1
        
        # Update label counts
        self.label_counts[label] = self.label_counts.get(label, 0) + 1

    def set_label(self, label: str) -> None:
        """Set the active label."""
        self.active_label = label.upper().replace(" ", "_")
        self.frame_count_for_label = 0
        self.is_recording = True
        
        existing_count = self.label_counts.get(self.active_label, 0)
        print(f"\nLabel set to: {self.active_label}")
        print(f"  Existing samples: {existing_count}")
        print(f"  Recording: ON")

    def handle_text_input(self, key: int) -> bool:
        """
        Handle text input mode.
        
        Returns:
            True if text input is active, False otherwise
        """
        if not self.text_input_mode:
            return False
        
        # Enter key - confirm label
        if key == 13:  # Enter
            if self.current_text.strip():
                self.set_label(self.current_text.strip())
            self.text_input_mode = False
            self.current_text = ""
            return False
        
        # Escape - cancel text input
        elif key == 27:  # Escape
            self.text_input_mode = False
            self.current_text = ""
            return False
        
        # Backspace
        elif key == 8:  # Backspace
            self.current_text = self.current_text[:-1]
        
        # Regular character input
        elif 32 <= key <= 126:  # Printable characters
            char = chr(key)
            # Limit length
            if len(self.current_text) < 20:
                self.current_text += char
        
        return True

    def _draw_text(self, img, text: str, pos, fontScale=0.6, color=(255, 255, 255)) -> None:
        """Draw text on image with black outline."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        
        # Black outline
        cv2.putText(img, text, pos, font, fontScale, (0, 0, 0), thickness + 2)
        # White text
        cv2.putText(img, text, pos, font, fontScale, color, thickness)

    def _draw_landmarks(self, img, landmarks) -> None:
        """Draw hand landmarks and skeleton on image."""
        h, w = img.shape[:2]
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]
        
        for start, end in connections:
            if start < len(landmarks) and end < len(landmarks):
                p1 = landmarks[start]
                p2 = landmarks[end]
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw points
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

    def run(self) -> None:
        """Main recording loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        target_frames_per_label = 450  # ~15 sec at 30 fps
        
        print("\n" + "=" * 60)
        print("LANDMARK RECORDER")
        print("=" * 60)
        print("Controls:")
        print("  T         - Type custom label (e.g., 'hello', 'goodbye')")
        print("  A-Z       - Quick select letter label")
        print("  SPACE     - Toggle recording for current label")
        print("  [         - Save and exit")
        print("  ESC       - Exit without saving")
        print("=" * 60 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # Mirror for selfie view
                h, w = frame.shape[:2]
                
                # Detect hand
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_data = self.hand_tracker.process_frame(rgb)
                landmarks = hand_data["landmarks"] if hand_data else None
                
                # If recording and hand detected, save data
                if self.is_recording and landmarks and len(landmarks) == 21:
                    normalized_features = self.normaliser.normalise(landmarks)
                    self._append_row(self.active_label, normalized_features)

                # Draw landmarks if detected
                if landmarks:
                    self._draw_landmarks(frame, landmarks)
                
                # Draw UI overlay
                y_offset = 30
                
                # Title bar
                cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 40), -1)
                
                # Status
                if self.text_input_mode:
                    status = f"Type label: {self.current_text}_"
                    color = (255, 200, 100)
                else:
                    rec_status = "●REC" if self.is_recording else "PAUSED"
                    label_text = f"Label: {self.active_label or 'NOT SET'}"
                    color = (0, 255, 0) if self.is_recording else (255, 255, 0)
                    status = f"{label_text} | {rec_status}"
                
                self._draw_text(frame, status, (10, y_offset), fontScale=0.7, color=color)
                y_offset += 40
                
                # Progress bar (if label is set)
                if self.active_label and not self.text_input_mode:
                    existing = self.label_counts.get(self.active_label, 0)
                    total = existing + self.frame_count_for_label
                    
                    progress = min(total / target_frames_per_label, 1.0)
                    bar_width = int(progress * 200)
                    cv2.rectangle(frame, (10, y_offset), (10 + 200, y_offset + 20), (100, 100, 100), -1)
                    cv2.rectangle(frame, (10, y_offset), (10 + bar_width, y_offset + 20), (0, 255, 0), -1)
                    pct = int(progress * 100)
                    self._draw_text(frame, f"{pct}% ({total}/{target_frames_per_label})", (220, y_offset + 15), fontScale=0.5)
                    y_offset += 35
                
                # Stats
                self._draw_text(
                    frame,
                    f"Frames (this session): {self.frame_count_for_label} | Total: {self.total_rows_recorded}",
                    (10, y_offset),
                    fontScale=0.5,
                    color=(200, 200, 200)
                )
                y_offset += 25
                
                # Existing samples for current label
                if self.active_label and not self.text_input_mode:
                    existing = self.label_counts.get(self.active_label, 0)
                    self._draw_text(
                        frame,
                        f"Existing samples for '{self.active_label}': {existing}",
                        (10, y_offset),
                        fontScale=0.5,
                        color=(150, 150, 150)
                    )
                
                # Hand detection status
                hand_status = "Hand: DETECTED" if hand_data else "Hand: NOT DETECTED"
                color = (0, 255, 0) if hand_data else (0, 0, 255)
                self._draw_text(frame, hand_status, (w - 250, 30), fontScale=0.6, color=color)
                
                cv2.imshow("Landmark Recorder", frame)
                
                # Handle key input
                key = cv2.waitKey(1) & 0xFF
                
                # Handle text input mode
                if self.handle_text_input(key):
                    continue
                
                # T - Enter text input mode
                if key == ord('t') or key == ord('T'):
                    self.text_input_mode = True
                    self.current_text = ""
                    print("\nEnter label name (press Enter to confirm, ESC to cancel):")
                
                # ESC - Exit
                elif key == 27:  # ESC
                    print("\nExit without saving.")
                    break
                
                # [ - Save and exit
                elif key == ord('['):
                    self._save_and_exit()
                    break
                
                # A-Z (upper and lower) - Quick select
                elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
                    label = chr(key).upper()
                    self.set_label(label)
                
                # SPACE - Toggle recording
                elif key == ord(' '):
                    if self.active_label:
                        self.is_recording = not self.is_recording
                        status = "RECORDING" if self.is_recording else "PAUSED"
                        print(f"{self.active_label}: {status}")
                    else:
                        print("Error: Set a label first (T or A-Z)")
                
                # Check if window was closed
                if cv2.getWindowProperty("Landmark Recorder", cv2.WND_PROP_VISIBLE) < 1:
                    print("\nWindow closed - exiting without saving.")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _save_and_exit(self) -> None:
        """Save recorded data to CSV and exit gracefully."""
        # Write all recorded data to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for row in self.recorded_data:
                writer.writerow(row)
        
        # Create backup in separate directory
        backup_dir = self.output_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"landmarks_{timestamp}.csv"
        
        with open(self.csv_path, "r") as f_in:
            with open(backup_path, "w") as f_out:
                f_out.write(f_in.read())
        
        print(f"\nSaved {len(self.recorded_data)} new rows (total: {self.total_rows_recorded}) to {self.csv_path}")
        print(f"Backup: {backup_path}")


if __name__ == "__main__":
    recorder = LandmarkRecorder()
    recorder.run()
