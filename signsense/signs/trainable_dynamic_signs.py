"""
signs/trainable_dynamic_signs.py
================================
Learned detector for dynamic ASL signs using LSTM models.

Replaces hardcoded state machines (JDetector, ZDetector) with trainable models.
Maintains same interface as dynamic_signs.py for compatibility.

Usage:
  detector = TrainedDynamicDetector("J")
  detector.update(landmarks, handedness)
  if detector.phase_complete:
      print(f"Sign complete at stage {detector.current_stage}")
"""

from typing import Optional
from pathlib import Path
import torch
import numpy as np

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from signsense.ml.dynamic_model import load_dynamic_model
from signsense.config.dynamic_config import get_config, SignConfig


class TrainedDynamicDetector:
    """
    Learned detector for dynamic signs using trained LSTM.
    
    Tracks hand landmarks through a sequence, predicts current stage,
    and detects when a sign is complete.
    Uses configuration from config/dynamic_signs.yaml.
    """
    
    def __init__(self, sign_name: str, model_dir: str = "ml/models"):
        """
        Args:
            sign_name: Name of sign (e.g., "J", "Z")
            model_dir: Directory containing trained models
        """
        self.sign_name = sign_name.upper()
        self.model_dir = Path(model_dir)
        
        # Load configuration
        self._config = get_config(self.sign_name)
        if self._config is None:
            raise ValueError(f"Configuration for sign '{self.sign_name}' not found in dynamic_signs.yaml")
        
        self._model = None
        self._normaliser = None
        self._num_stages = 0
        self._stage_names = self._config.stages
        self._device = "cpu"
        
        self._phase_complete = False
        self._current_stage = 0
        self._frame_buffer = []  # Store recent frames for sequence processing
        self._stage_frames = 0
        self._stage_confidence = 0.0
        self._final_stage_hold = 0
        self._consecutive_stage_preds = 0  # Track consecutive predictions for validation
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model checkpoint."""
        model_path = self.model_dir / f"dynamic_{self.sign_name}.pt"
        
        if not model_path.exists():
            print(
                f"[TrainedDynamicDetector] Warning: Model not found at {model_path}\n"
                f"  Run: python -m ml.dynamic_recorder  (to record sequences)\n"
                f"  Then: python -m ml.dynamic_train {self.sign_name}  (to train the model)\n"
                f"  Until then, {self.sign_name} detection will not work."
            )
            return
        
        try:
            model, normaliser, num_stages = load_dynamic_model(
                str(model_path),
                self.sign_name,
                device=self._device
            )
            self._model = model
            self._normaliser = normaliser
            self._num_stages = num_stages
            print(f"[TrainedDynamicDetector] Loaded {self.sign_name} model ({num_stages} stages)")
        except Exception as e:
            print(f"[TrainedDynamicDetector] Error loading model: {e}")
    
    @property
    def stage_label(self) -> str:
        """Human-readable label for current stage."""
        if not self._model:
            return ""
        return f"Stage {self._current_stage}/{self._num_stages - 1} (conf: {self._stage_confidence:.2f})"
    
    @property
    def phase_complete(self) -> bool:
        """True for exactly one frame when sign completes."""
        return self._phase_complete
    
    @property
    def current_stage(self) -> int:
        """Current stage index (0 to num_stages-1)."""
        return self._current_stage
    
    def reset(self) -> None:
        """Reset detector state."""
        self._phase_complete = False
        self._current_stage = 0
        self._frame_buffer = []
        self._stage_frames = 0
        self._stage_confidence = 0.0
        self._final_stage_hold = 0
        self._final_stage_hold = 0
    
    @property
    def stage_info(self) -> dict:
        """Get current stage info for UI display."""
        return {
            'current_stage': self._current_stage,
            'num_stages': self._num_stages,
            'confidence': self._stage_confidence,
            'phase_complete': self._phase_complete,
            'stage_progress': self._stage_frames,
            'final_stage_hold': getattr(self, '_final_stage_hold', 0)
        }
    
    def update(self, landmarks, handedness: Optional[str]) -> bool:
        """
        Update detector with new frame.
        
        Args:
            landmarks: 21 MediaPipe hand landmarks
            handedness: "Left" or "Right"
        
        Returns:
            True if sign completed this frame, False otherwise
        """
        self._phase_complete = False
        
        if not self._model or landmarks is None or len(landmarks) < 21:
            self.reset()
            return False
        
        try:
            # Normalise landmarks and add to buffer
            landmarks_list = [landmarks]
            normalized = self._normaliser.normalise_sequence(landmarks_list)
            self._frame_buffer.append(normalized[0])  # Add single frame
            
            # Keep buffer to recent frames (don't need entire history)
            max_buffer = 50
            if len(self._frame_buffer) > max_buffer:
                self._frame_buffer = self._frame_buffer[-max_buffer:]
            
            # Predict only if we have enough frames
            if len(self._frame_buffer) < 3:
                return False
            
            # Stack buffer into sequence
            frame_sequence = np.vstack(self._frame_buffer)  # (seq_len, 63)
            
            # Send to model
            with torch.no_grad():
                seq_tensor = torch.from_numpy(frame_sequence[np.newaxis, :, :]).float()
                seq_tensor = seq_tensor.to(self._device)
                
                stage_logits, transition_probs = self._model(seq_tensor)
                # stage_logits: [1, seq_len, num_stages]
                # Get prediction for LAST frame in sequence
                last_frame_logits = stage_logits[0, -1, :]
                
                # Get current stage
                predicted_stage = last_frame_logits.argmax().item()
                stage_confidence = torch.softmax(last_frame_logits, dim=0)[predicted_stage].item()
                transition_confidence = transition_probs[0, -1, 0].item()
        
        except Exception as e:
            print(f"[TrainedDynamicDetector] Error in update: {e}")
            self.reset()
            return False
        
        # Update stage tracking
        self._stage_confidence = stage_confidence
        
        # ENFORCE SEQUENTIAL STAGE PROGRESSION
        # Only allow progression to the next sequential stage (current + 1)
        # Reject any jumps to non-sequential stages
        # Also require minimum confidence and consistency for valid transitions
        min_confidence = 0.4  # Default threshold for stage recognition
        
        if predicted_stage != self._current_stage:
            # Track consecutive predictions for the same stage
            if predicted_stage == getattr(self, '_last_predicted_stage', None):
                self._consecutive_stage_preds += 1
            else:
                self._consecutive_stage_preds = 1
            self._last_predicted_stage = predicted_stage
            
            # Require confidence threshold (lowered threshold for easier detection)
            if stage_confidence >= min_confidence:
                # Calculate the expected next stage (must be current + 1)
                expected_next_stage = self._current_stage + 1
                
                # Only allow progression to the immediate next stage
                if predicted_stage == expected_next_stage:
                    # Valid sequential progression
                    print(f"[{self.sign_name}] Stage {self._current_stage} → {predicted_stage} (sequential, conf={stage_confidence:.2f})")
                    self._current_stage = predicted_stage
                    self._stage_frames = 0  # Reset timeout counter
                    self._consecutive_stage_preds = 0  # Reset consistency counter
                elif predicted_stage > self._current_stage:
                    # Invalid jump (e.g., 0→2) - reject and wait for intermediate stage
                    print(f"[{self.sign_name}] Stage {self._current_stage} → {predicted_stage} (REJECTED - must go through {expected_next_stage} first)")
                    # Don't update stage, but also don't reset the timeout counter
                else:
                    # Regression (e.g., 2→1) - allow it but reset hold counter
                    print(f"[{self.sign_name}] Stage {self._current_stage} → {predicted_stage} (regression)")
                    self._current_stage = predicted_stage
                    self._stage_frames = 0  # Reset timeout counter
                    self._consecutive_stage_preds = 0  # Reset consistency counter
            else:
                # Not confident enough yet - keep current stage
                if stage_confidence < min_confidence:
                    print(f"[{self.sign_name}] Stage {predicted_stage} (low confidence {stage_confidence:.2f}, ignoring)")
                else:
                    print(f"[{self.sign_name}] Stage {predicted_stage} (need {self._consecutive_stage_preds}/2 consistent predictions)")
        else:
            # Same stage - increment timeout counter
            self._stage_frames += 1
            self._consecutive_stage_preds = 0  # Reset since we're staying in same stage
        
        # Reset hold counter when reaching final stage
        if self._current_stage >= self._num_stages - 1:
            self._final_stage_hold = getattr(self, '_final_stage_hold', 0) + 1
            # Need to hold final stage for minimum frames before completing
            min_hold_frames = 10  # ~0.3 seconds at 30fps
            if self._final_stage_hold >= min_hold_frames:
                self._phase_complete = True
                print(f"[{self.sign_name}] Sign complete! ✓ (held {self._final_stage_hold} frames)")
                # Reset detector after completion to prevent repeated confirmations
                self.reset()
                return True
        
        # Timeout if stuck in stage too long
        if self._stage_frames > self._config.detector.phase_timeout:
            print(f"[{self.sign_name}] Stage timeout - resetting")
            self.reset()
            return False
        
        return False


if __name__ == "__main__":
    # Simple test
    print("TrainedDynamicDetector test script")
    
    # Try to load J model (may not exist yet)
    detector = TrainedDynamicDetector("J")
    print(f"Loaded: {detector.sign_name}")
    print(f"Current stage: {detector.current_stage}")
    print(f"Stage label: {detector.stage_label}")
