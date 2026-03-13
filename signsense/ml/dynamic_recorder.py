"""
ml/dynamic_recorder.py
======================
Interactive recorder for dynamic ASL sign sequences.

A dynamic sign has multiple STAGES that unfold over time (e.g., J: Hold I → Hook down → Palm away).
This tool records sequences where each frame is labeled with its current stage number.

Usage:
  python -m ml.dynamic_recorder

Controls:
  - Press A-Z to set the sign name instantly (e.g., press J for sign "J")
  - Press 0-9 to set the current STAGE (e.g., 0=Hold I, 1=Hook down, 2=Palm away)
  - While recording, every detected hand frame is saved with its stage label
  - SPACE: start/stop recording sequence for current stage
  - N: advance to next stage and start a new sequence
  - Q: save all sequences and exit
  - ESC: exit without saving

Output:
  - Saves to ml/data/dynamic/<SIGN_NAME>/ directory
  - Creates separate .npy files for each recorded sequence
  - Metadata: ml/data/dynamic/<SIGN_NAME>/metadata.json
  
Example:
  For J sign:
    Stage 0: [frame1, frame2, frame3, ...]       (Hold I position)
    Stage 1: [frame4, frame5, frame6, ...]       (Pinky moves down)
    Stage 2: [frame7, frame8, frame9, ...]       (Palm away + hold I)
    
  Each stage is a separate sequence file: J_s0_0.npy, J_s0_1.npy, ... (multiple examples)
"""

import cv2
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signsense.detector.hand_tracker import HandTracker
from signsense.ml.model import LandmarkNormaliser
from signsense.config.dynamic_config import get_config, get_all_sign_names, SignConfig


class DynamicSignRecorder:
    """Record sequences of hand landmarks for dynamic signs with stage labels."""

    def __init__(self, output_base: str = "ml/data/dynamic"):
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        self.normaliser = LandmarkNormaliser()
        self.hand_tracker = HandTracker()
        
        self.sign_name: Optional[str] = None
        self.sign_config: Optional[SignConfig] = None
        self.current_stage: int = 0
        self.is_recording = False
        self.current_sequence: List[np.ndarray] = []
        self.stage_sequences: dict = {}  # stage_num -> list of sequences
        
    def _get_sign_dir(self) -> Path:
        """Get output directory for current sign."""
        if not self.sign_name:
            raise ValueError("Sign name not set. Press A-Z to set sign name.")
        return self.output_base / self.sign_name
    
    def _ensure_sign_dir(self) -> None:
        """Create sign directory if needed."""
        self._get_sign_dir().mkdir(parents=True, exist_ok=True)
    
    def _save_sequence(self, stage_num: int, sequence: List[np.ndarray]) -> None:
        """Save a completed sequence to disk."""
        if not sequence:
            return
        
        self._ensure_sign_dir()
        sign_dir = self._get_sign_dir()
        
        # Find next sequence number for this stage
        existing = list(sign_dir.glob(f"{self.sign_name}_s{stage_num}_*.npy"))
        next_idx = len(existing)
        
        filename = sign_dir / f"{self.sign_name}_s{stage_num}_{next_idx}.npy"
        
        # Stack all frames in sequence (shape: num_frames x 63)
        sequence_array = np.vstack(sequence)
        np.save(filename, sequence_array)
        
        print(f"  ✓ Saved: {filename.name} ({len(sequence)} frames)")
    
    def _save_all_sequences(self) -> None:
        """Save all recorded sequences."""
        if not self.sign_name:
            print("No sign recorded.")
            return
        
        self._ensure_sign_dir()
        
        # Save metadata
        sign_dir = self._get_sign_dir()
        metadata = {
            "sign_name": self.sign_name,
            "num_stages": len(self.stage_sequences),
            "sequences_per_stage": {
                str(s): len(seqs) for s, seqs in self.stage_sequences.items()
            },
            "recorded_at": datetime.now().isoformat(),
        }
        
        metadata_path = sign_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved metadata to {metadata_path.name}")
        print(f"  Sign: {self.sign_name}")
        print(f"  Stages: {metadata['num_stages']}")
        for stage, count in metadata['sequences_per_stage'].items():
            print(f"    Stage {stage}: {count} sequences")
    
    def set_sign_name(self, name: str) -> None:
        """Set the sign name."""
        self.sign_name = name.upper()
        self.sign_config = get_config(self.sign_name)
        self.stage_sequences = {}
        self.current_stage = 0
        
        print(f"Sign set to: {self.sign_name}")
        if self.sign_config:
            print(f"Stages defined: {list(self.sign_config.stages.values())}")
        else:
            print(f"Warning: No configuration found for sign '{self.sign_name}'")
        print(f"Current stage: {self.current_stage}")
    
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
        
        if self.sign_config and self.current_stage in self.sign_config.stages:
            print(f"Stage set to: {self.current_stage} - {self.sign_config.stages[self.current_stage]}")
        else:
            print(f"Stage set to: {self.current_stage}")
    
    def toggle_recording(self) -> None:
        """Start/stop recording current stage."""
        if not self.sign_name:
            print("Error: Set sign name first (press A-Z)")
            return
        
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.current_sequence = []
            print(f"Recording stage {self.current_stage}...")
        else:
            if self.current_sequence:
                if self.current_stage not in self.stage_sequences:
                    self.stage_sequences[self.current_stage] = []
                self.stage_sequences[self.current_stage].append(self.current_sequence)
                print(f"✓ Saved stage {self.current_stage} sequence ({len(self.current_sequence)} frames)")
            self.current_sequence = []
    
    def run(self) -> None:
        """Main interactive loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        print("""
============================================================
DYNAMIC SIGN RECORDER
============================================================
A-Z: set sign name to that letter
0-9: set current stage
SPACE: start/stop recording current stage
N: move to next stage
Q: save and exit
ESC: exit without saving
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
            handedness = hand_data["handedness"] if hand_data else None
            
            # Record if active
            if self.is_recording and hand_landmarks:
                normalized = self.normaliser.normalise(hand_landmarks)
                self.current_sequence.append(normalized)
            
            # Draw info
            h, w = frame.shape[:2]
            info_color = (0, 255, 0)
            
            # Status box
            status = f"Sign: {self.sign_name or 'NOT SET'} | Stage: {self.current_stage} | Recording: {'YES' if self.is_recording else 'no'}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
            
            if self.sign_name and self.current_stage in self.stage_sequences:
                count = len(self.stage_sequences[self.current_stage])
                cv2.putText(
                    frame,
                    f"Stage {self.current_stage}: {count} sequences",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 200, 100),
                    2
                )
            
            # Frame count in current sequence
            if self.current_sequence:
                cv2.putText(
                    frame,
                    f"Current sequence: {len(self.current_sequence)} frames",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 100),
                    2
                )
            
            cv2.imshow("Dynamic Sign Recorder", frame)
            
            key = cv2.waitKey(1) & 0xFF

            # A-Z (upper and lower) → set sign name instantly, no blocking input()
            if 65 <= key <= 90:         # uppercase A-Z
                self.set_sign_name(chr(key))
            elif 97 <= key <= 122:      # lowercase a-z
                self.set_sign_name(chr(key))

            elif ord('0') <= key <= ord('9'):
                stage_num = int(chr(key))
                self.set_stage(stage_num)
            
            elif key == ord(' '):
                self.toggle_recording()
            
            elif key in (ord('n'), ord('N')):
                self.set_stage(self.current_stage + 1)
            
            elif key in (ord('q'), ord('Q')):
                self._save_all_sequences()
                break
            
            elif key == 27:  # ESC
                print("Exiting without saving.")
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recorder = DynamicSignRecorder()
    recorder.run()