"""
ml/dynamic_recorder.py
=====================
Interactive recorder for dynamic ASL sign sequences.

Supports:
- Custom gesture names (not just letters)
- Simple gestures (single-stage) and complex gestures (multi-stage)
- Adding more samples to existing gestures
- Auto-configuration for new gestures

Usage:
  python -m ml.dynamic_recorder

Controls:
  - T: Enter text mode to type custom gesture name
  - A-Z: Quick select predefined gesture (letters)
  - 0-9: Set current stage (for complex gestures)
  - S: Toggle simple/complex mode
  - SPACE: Start/stop recording
  - N: Next stage (complex mode) or new sequence (simple mode)
  - P: Previous stage (go back)
  - Q: Save and exit
  - ESC: Exit without saving

Output:
  - Saves to ml/data/dynamic/<GESTURE_NAME>/ directory
  - Creates separate .npy files for each recorded sequence
  - Metadata: ml/data/dynamic/<GESTURE_NAME>/metadata.json
"""

import cv2
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signsense.detector.hand_tracker import HandTracker
from signsense.ml.model import LandmarkNormaliser
from signsense.config.dynamic_config import (
    get_config, get_all_sign_names, SignConfig, 
    GestureType, create_default_config
)


class DynamicSignRecorder:
    """
    Record sequences of hand landmarks for dynamic signs with stage labels.
    
    Features:
    - Custom gesture names via text input
    - Simple (single-stage) and complex (multi-stage) gesture support
    - Add more samples to existing gestures
    - Auto-create config for new gestures
    """

    def __init__(self, output_base: str = None):
        # Determine the correct output directory
        if output_base is None:
            script_dir = Path(__file__).parent
            self.output_base = script_dir.parent.parent / "ml" / "data" / "dynamic"
        else:
            self.output_base = Path(output_base)
        
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        self.normaliser = LandmarkNormaliser()
        self.hand_tracker = HandTracker()
        
        # Recording state
        self.sign_name: Optional[str] = None
        self.sign_config: Optional[SignConfig] = None
        self.current_stage: int = 0
        self.is_recording: bool = False
        self.current_sequence: List[np.ndarray] = []
        self.stage_sequences: dict = {}  # stage_num -> list of sequences
        
        # Input mode
        self.text_input_mode: bool = False
        self.current_text: str = ""
        
        # Gesture mode (simple or complex)
        self.gesture_mode: str = "simple"  # "simple" or "complex"
        
        # Existing gestures info
        self._load_existing_gestures()
    
    def _load_existing_gestures(self):
        """Load information about existing gestures."""
        self.existing_gestures = {}
        
        if not self.output_base.exists():
            return
        
        for gesture_dir in self.output_base.iterdir():
            if not gesture_dir.is_dir():
                continue
            
            metadata_file = gesture_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.existing_gestures[gesture_dir.name.upper()] = metadata
                except Exception:
                    pass
    
    def _get_sign_dir(self) -> Path:
        """Get output directory for current sign."""
        if not self.sign_name:
            raise ValueError("Sign name not set. Use T to type a name or A-Z to select.")
        return self.output_base / self.sign_name
    
    def _ensure_sign_dir(self) -> None:
        """Create sign directory if needed."""
        self._get_sign_dir().mkdir(parents=True, exist_ok=True)
    
    def _load_existing_data(self):
        """Load existing sequences for the current gesture."""
        # Initialize existing_sequences before early return
        self.existing_sequences = {}
        
        sign_dir = self._get_sign_dir()
        
        if not sign_dir.exists():
            return
        
        # Load metadata
        metadata_file = sign_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.sign_config_data = json.load(f)
            except Exception:
                self.sign_config_data = {}
        else:
            self.sign_config_data = {}
        
        # Count existing sequences per stage
        self.existing_sequences = {}
        for npy_file in sign_dir.glob("*.npy"):
            # Parse filename: GESTURE_s0_0.npy
            try:
                parts = npy_file.name.split("_s")
                if len(parts) >= 2:
                    stage_num = int(parts[1].split("_")[0])
                    if stage_num not in self.existing_sequences:
                        self.existing_sequences[stage_num] = 0
                    self.existing_sequences[stage_num] += 1
            except Exception:
                pass
    
    def _save_sequence(self, stage_num: int, sequence: List[np.ndarray]) -> None:
        """Save a completed sequence to disk."""
        if not sequence:
            return
        
        self._ensure_sign_dir()
        sign_dir = self._get_sign_dir()
        
        # Find next sequence number for this stage
        # Account for existing sequences
        base_idx = self.existing_sequences.get(stage_num, 0)
        
        # Find next available index
        existing = list(sign_dir.glob(f"{self.sign_name.upper()}_s{stage_num}_*.npy"))
        next_idx = base_idx + len(existing)
        
        filename = sign_dir / f"{self.sign_name.upper()}_s{stage_num}_{next_idx}.npy"
        
        # Stack all frames in sequence (shape: num_frames x 63)
        sequence_array = np.vstack(sequence)
        np.save(filename, sequence_array)
        
        print(f"  ✓ Saved: {filename.name} ({len(sequence)} frames)")
        
        # Update existing sequences count
        if stage_num not in self.existing_sequences:
            self.existing_sequences[stage_num] = 0
        self.existing_sequences[stage_num] += 1
    
    def _save_all_sequences(self) -> None:
        """Save all recorded sequences and update metadata."""
        if not self.sign_name:
            print("No sign recorded.")
            return
        
        self._ensure_sign_dir()
        
        # Build metadata
        sign_dir = self._get_sign_dir()
        
        # Save each sequence to disk FIRST
        for stage_num, sequences in self.stage_sequences.items():
            for seq in sequences:
                self._save_sequence(stage_num, seq)
        
        # Count total sequences saved
        total_sequences = sum(len(seqs) for seqs in self.stage_sequences.values())
        
        metadata = {
            "sign_name": self.sign_name,
            "gesture_type": self.gesture_mode,
            "num_stages": len(self.stage_sequences),
            "sequences_per_stage": {
                str(s): len(seqs) for s, seqs in self.stage_sequences.items()
            },
            "total_sequences": total_sequences,
            "recorded_at": datetime.now().isoformat(),
            "description": self.sign_config.description if self.sign_config else "",
        }
        
        metadata_path = sign_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved metadata to {metadata_path.name}")
        print(f"  Gesture: {self.sign_name}")
        print(f"  Type: {self.gesture_mode}")
        print(f"  Total sequences: {total_sequences}")
        for stage, count in metadata['sequences_per_stage'].items():
            print(f"    Stage {stage}: {count} sequences")
    
    def set_sign_name(self, name: str) -> None:
        """Set the sign name and load/create config."""
        self.sign_name = name.upper().replace(" ", "_")
        
        # Try to get existing config
        self.sign_config = get_config(self.sign_name)
        
        if self.sign_config:
            # Use existing config
            self.gesture_mode = "simple" if self.sign_config.is_simple else "complex"
            print(f"Loaded existing config for: {self.sign_name}")
            print(f"  Type: {self.gesture_mode}")
            if self.sign_config.description:
                print(f"  Description: {self.sign_config.description}")
        else:
            # Create default config for new gesture
            description = f"Custom gesture: {self.sign_name}"
            self.sign_config = create_default_config(
                self.sign_name,
                GestureType.SIMPLE if self.gesture_mode == "simple" else GestureType.COMPLEX,
                description
            )
            print(f"Created new config for: {self.sign_name}")
        
        # Reset recording state
        self.stage_sequences = {}
        self.current_stage = 0
        self.is_recording = False
        self.current_sequence = []
        
        # Load existing data if any
        self._load_existing_data()
        
        print(f"Gesture: {self.sign_name}")
        print(f"Mode: {self.gesture_mode}")
        if self.existing_sequences:
            print(f"Existing samples: {sum(self.existing_sequences.values())}")
        print(f"Current stage: {self.current_stage}")
    
    def toggle_gesture_mode(self) -> None:
        """Toggle between simple and complex gesture mode."""
        if self.gesture_mode == "simple":
            self.gesture_mode = "complex"
        else:
            self.gesture_mode = "simple"
        print(f"Gesture mode: {self.gesture_mode}")
    
    def set_stage(self, stage_num: int) -> None:
        """Set current stage number."""
        if self.current_sequence and self.is_recording:
            # Save current sequence before switching
            if self.current_stage not in self.stage_sequences:
                self.stage_sequences[self.current_stage] = []
            self.stage_sequences[self.current_stage].append(self.current_sequence)
            print(f"Auto-saved stage {self.current_stage} sequence ({len(self.current_sequence)} frames)")
            self.current_sequence = []
        
        self.current_stage = stage_num
        self.is_recording = False
        
        stage_name = ""
        if self.sign_config and self.sign_config.stages:
            stage_name = f" - {self.sign_config.stages.get(stage_num, '')}"
        
        print(f"Stage set to: {self.current_stage}{stage_name}")
    
    def toggle_recording(self) -> None:
        """Start/stop recording."""
        if not self.sign_name:
            print("Error: Set gesture name first (T to type, or A-Z for letter)")
            return
        
        self.is_recording = not self.is_recording
        
        if self.is_recording:
            self.current_sequence = []
            print(f"Recording {'stage ' + str(self.current_stage) + '...' if self.gesture_mode == 'complex' else 'gesture...'}")
        else:
            if self.current_sequence:
                if self.current_stage not in self.stage_sequences:
                    self.stage_sequences[self.current_stage] = []
                self.stage_sequences[self.current_stage].append(self.current_sequence)
                print(f"✓ Saved {'stage ' + str(self.current_stage) + ' ' if self.gesture_mode == 'complex' else ''}sequence ({len(self.current_sequence)} frames)")
            self.current_sequence = []
    
    def next_stage(self) -> None:
        """Advance to next stage."""
        # Save current sequence if recording
        if self.current_sequence and self.is_recording:
            self.toggle_recording()  # Stop recording to save
        
        # Move to next stage
        self.current_stage += 1
        print(f"Moved to stage {self.current_stage}")
        
        stage_name = ""
        if self.sign_config and self.sign_config.stages:
            stage_name = f" - {self.sign_config.stages.get(self.current_stage, '')}"
        
        print(f"Stage: {self.current_stage}{stage_name}")
    
    def prev_stage(self) -> None:
        """Go back to previous stage."""
        if self.current_stage > 0:
            # Save current sequence if recording
            if self.current_sequence and self.is_recording:
                self.toggle_recording()  # Stop recording to save
            
            # Move to previous stage
            self.current_stage -= 1
            print(f"Moved back to stage {self.current_stage}")
            
            stage_name = ""
            if self.sign_config and self.sign_config.stages:
                stage_name = f" - {self.sign_config.stages.get(self.current_stage, '')}"
            
            print(f"Stage: {self.current_stage}{stage_name}")
        else:
            print("Already at stage 0 - cannot go back")
    
    def handle_text_input(self, key: int) -> bool:
        """
        Handle text input mode.
        
        Returns:
            True if text input is active, False otherwise
        """
        if not self.text_input_mode:
            return False
        
        # Enter key - confirm gesture name
        if key == 13:  # Enter
            if self.current_text.strip():
                self.set_sign_name(self.current_text.strip())
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
            if len(self.current_text) < 30:
                self.current_text += char
        
        return True
    
    def run(self) -> None:
        """Main interactive loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        print("""
============================================================
DYNAMIC GESTURE RECORDER
============================================================
Controls:
  T         - Type custom gesture name (e.g., "hello", "thank_you")
  A-Z       - Quick select letter gesture (J, Z, etc.)
  S         - Toggle simple/complex mode
  0-9       - Set stage (complex mode only)
  SPACE     - Start/stop recording
  N         - Next stage (complex) / new sequence (simple)
  Q         - Save and exit
  ESC       - Exit without saving
============================================================
        """)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror for selfie view
            
            # Detect hand landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_data = self.hand_tracker.process_frame(rgb)
            hand_landmarks = hand_data["landmarks"] if hand_data else None
            
            # Record if active
            if self.is_recording and hand_landmarks:
                normalized = self.normaliser.normalise(hand_landmarks)
                self.current_sequence.append(normalized)
            
            # Draw UI
            h, w = frame.shape[:2]
            
            # Title bar
            cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 40), -1)
            
            # Status text
            if self.text_input_mode:
                status = f"Type name: {self.current_text}_"
                color = (255, 200, 100)
            else:
                mode_indicator = "[SIMPLE]" if self.gesture_mode == "simple" else "[COMPLEX]"
                rec_indicator = "●REC" if self.is_recording else ""
                status = f"Gesture: {self.sign_name or 'NOT SET'} | {mode_indicator} | Stage: {self.current_stage} | {rec_indicator}"
                color = (0, 255, 0) if self.is_recording else (255, 255, 255)
            
            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw landmarks
            if hand_landmarks:
                self._draw_landmarks(frame, hand_landmarks)
            
            # Info panel
            info_y = 80
            
            # Existing samples info
            if self.sign_name and self.existing_sequences:
                total_existing = sum(self.existing_sequences.values())
                cv2.putText(
                    frame,
                    f"Existing samples: {total_existing}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (150, 150, 150),
                    1
                )
                info_y += 25
            
            # Current recording info
            if self.is_recording and self.current_sequence:
                cv2.putText(
                    frame,
                    f"Recording: {len(self.current_sequence)} frames",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                info_y += 30
            
            # Stage info for complex mode
            if self.gesture_mode == "complex" and self.sign_name:
                stage_count = len(self.stage_sequences.get(self.current_stage, []))
                cv2.putText(
                    frame,
                    f"Stage {self.current_stage}: {stage_count} sequences",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (100, 200, 100),
                    1
                )
                info_y += 25
            
            # Instructions
            cv2.putText(
                frame,
                "T: Type name | SPACE: Record | N: Next | Q: Save",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 100, 100),
                1
            )
            
            cv2.imshow("Dynamic Gesture Recorder", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle text input mode
            if self.handle_text_input(key):
                continue
            
            # T - Enter text input mode
            if key == ord('t') or key == ord('T'):
                self.text_input_mode = True
                self.current_text = ""
                print("\nEnter gesture name (press Enter to confirm, ESC to cancel):")
            
            # A-Z - Quick select letter
            elif 65 <= key <= 90:  # uppercase A-Z
                self.set_sign_name(chr(key))
            
            # S - Toggle gesture mode
            elif key == ord('s') or key == ord('S'):
                self.toggle_gesture_mode()
            
            # 0-9 - Set stage (complex mode)
            elif ord('0') <= key <= ord('9'):
                stage_num = int(chr(key))
                if self.gesture_mode == "complex":
                    self.set_stage(stage_num)
            
            # SPACE - Toggle recording
            elif key == ord(' '):
                self.toggle_recording()
            
            # N - Next stage / new sequence
            elif key in (ord('n'), ord('N')):
                self.next_stage()
            
            # P - Previous stage
            elif key in (ord('p'), ord('P')):
                self.prev_stage()
            
            # Q - Save and exit
            elif key in (ord('q'), ord('Q')):
                self._save_all_sequences()
                break
            
            # ESC - Exit without saving
            elif key == 27:  # ESC
                print("\nExiting without saving.")
                break
            
            # Check if window was closed
            if cv2.getWindowProperty("Dynamic Gesture Recorder", cv2.WND_PROP_VISIBLE) < 1:
                print("\nWindow closed - exiting without saving.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _draw_landmarks(self, img, landmarks) -> None:
        """Draw hand landmarks on image."""
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


if __name__ == "__main__":
    recorder = DynamicSignRecorder()
    recorder.run()
