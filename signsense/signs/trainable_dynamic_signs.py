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
import math

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from signsense.ml.dynamic_model_enhanced import load_enhanced_dynamic_model as load_dynamic_model
from signsense.config.dynamic_config import get_config, SignConfig
import warnings


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
        self._visited_stages = set()  # Track which stages have been visited
        self._last_landmarks = None  # Store previous frame landmarks for movement detection
        self._movement_history = []  # Track movement between frames
        self._static_pose_frames = 0  # Count frames with minimal movement
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model checkpoint."""
        # Try enhanced model first (preferred), then fall back to basic
        model_path = self.model_dir / f"dynamic_{self.sign_name}_enhanced.pt"
        
        if not model_path.exists():
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
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
        self._visited_stages = set()  # Track which stages have been visited
        self._last_landmarks = None  # Store previous frame landmarks for movement detection
        self._movement_history = []  # Track movement between frames
        self._static_pose_frames = 0  # Count frames with minimal movement
        self._frames_since_completion = 0  # Cooldown counter
    
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
    
    def _calculate_movement(self, current_landmarks, previous_landmarks) -> float:
        """
        Calculate the movement/distance between two sets of landmarks.
        
        Args:
            current_landmarks: Current frame landmarks (21 points)
            previous_landmarks: Previous frame landmarks (21 points)
        
        Returns:
            Average Euclidean distance between corresponding landmarks
        """
        if previous_landmarks is None or current_landmarks is None:
            return 0.0
        
        if len(current_landmarks) < 21 or len(previous_landmarks) < 21:
            return 0.0
        
        total_distance = 0.0
        for i in range(21):
            # Calculate Euclidean distance for each landmark
            dx = current_landmarks[i].x - previous_landmarks[i].x
            dy = current_landmarks[i].y - previous_landmarks[i].y
            dz = getattr(current_landmarks[i], 'z', 0.0) - getattr(previous_landmarks[i], 'z', 0.0)
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            total_distance += distance
        
        # Return average distance across all landmarks
        return total_distance / 21.0

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
                
                model_output = self._model(seq_tensor)
                
                # Handle both basic model (2 outputs) and enhanced model (3 outputs)
                if isinstance(model_output, tuple):
                    if len(model_output) == 3:
                        stage_logits, transition_probs, confidence_scores = model_output
                    else:
                        stage_logits, transition_probs = model_output
                else:
                    stage_logits = model_output
                    transition_probs = None
                
                # stage_logits: [1, seq_len, num_stages]
                # Get prediction for LAST frame in sequence
                last_frame_logits = stage_logits[0, -1, :]
                
                # Get current stage
                predicted_stage = last_frame_logits.argmax().item()
                stage_confidence = torch.softmax(last_frame_logits, dim=0)[predicted_stage].item()
                transition_confidence = transition_probs[0, -1, 0].item() if transition_probs is not None else 0.5
        
        except Exception as e:
            print(f"[TrainedDynamicDetector] Error in update: {e}")
            self.reset()
            return False
        
        # Update stage tracking
        self._stage_confidence = stage_confidence
        
        # Calculate movement between frames for validation
        current_movement = self._calculate_movement(landmarks, self._last_landmarks)
        self._movement_history.append(current_movement)
        
        # Keep only recent movement history (last 30 frames)
        if len(self._movement_history) > 30:
            self._movement_history = self._movement_history[-30:]
        
        # Detect static pose (minimal movement)
        movement_threshold = 0.01  # Threshold for detecting movement
        if current_movement < movement_threshold:
            self._static_pose_frames += 1
        else:
            self._static_pose_frames = 0
        
        # Update last landmarks for next frame's movement calculation
        self._last_landmarks = landmarks
        
        # ENFORCE SEQUENTIAL STAGE PROGRESSION
        # Only allow progression to the next sequential stage (current + 1)
        # Reject any jumps to non-sequential stages
        # Also require minimum confidence and consistency for valid transitions
        min_confidence = 0.4  # Default threshold for stage recognition
        
        # SPECIAL CASE: Simple/single-stage models
        # For models configured as "simple" type, skip sequential progression enforcement
        # and allow any stage prediction to complete the sign
        # This handles cases where a model was trained with multiple stages but is
        # configured as a simple (single-stage) motion gesture
        is_simple_type = hasattr(self._config, 'is_simple') and self._config.is_simple
        
        if is_simple_type or self._num_stages == 1:
            # Simple/single-stage model: accept any stage prediction with sufficient confidence
            if stage_confidence >= min_confidence:
                # For simple models, jump directly to final stage
                if self._current_stage < self._num_stages - 1:
                    print(f"[{self.sign_name}] Simple model: advancing to final stage (conf={stage_confidence:.2f})")
                    # Track that we've visited the current stage before advancing
                    self._visited_stages.add(self._current_stage)
                    self._current_stage = self._num_stages - 1
                    self._stage_frames = 0
                    self._consecutive_stage_preds = 0
            else:
                # Low confidence - stay in current stage
                self._stage_frames += 1
        elif predicted_stage != self._current_stage:
            # Multi-stage model: enforce sequential progression
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
                    # Track that we've visited the current stage before advancing
                    self._visited_stages.add(self._current_stage)
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
                    # Track that we've visited the current stage before regressing
                    self._visited_stages.add(self._current_stage)
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
                # VALIDATE MOVEMENT: Check that user actually moved through stages
                # For simple-type models, we only require visiting the final stage
                # For multi-stage models, we require visiting all stages
                is_simple_type = hasattr(self._config, 'is_simple') and self._config.is_simple
                
                if is_simple_type:
                    # Simple model: only require being at final stage (already there)
                    all_stages_visited = True
                else:
                    # Multi-stage model: require visiting all stages
                    all_stages_visited = len(self._visited_stages) >= self._num_stages - 1
                
                # 2. Check for movement between stages (not just static pose)
                # Calculate average movement over recent frames
                avg_movement = sum(self._movement_history[-10:]) / len(self._movement_history[-10:]) if self._movement_history else 0.0
                has_movement = avg_movement > 0.015  # Increased minimum movement threshold - requires more deliberate motion
                
                # 3. Check that user is not static in final position
                not_static_in_final = self._static_pose_frames < 15  # Less than 15 frames of static pose
                
                # Only allow completion if all validation criteria are met
                if all_stages_visited and has_movement and not_static_in_final:
                    # Check cooldown to prevent consecutive confirmations
                    frames_since_last_completion = getattr(self, '_frames_since_completion', 0)
                    cooldown_frames = 30  # ~1 second cooldown at 30fps
                    
                    if frames_since_last_completion >= cooldown_frames:
                        self._phase_complete = True
                        print(f"[{self.sign_name}] Sign complete! ✓ (held {self._final_stage_hold} frames, visited {len(self._visited_stages)}/{self._num_stages} stages, movement={avg_movement:.4f})")
                        # Reset detector after completion to prevent repeated confirmations
                        self.reset()
                        # Set cooldown counter
                        self._frames_since_completion = 0
                        return True
                    else:
                        # Still in cooldown period - don't confirm yet
                        self._frames_since_completion = frames_since_last_completion + 1
                else:
                    # Validation failed - don't complete yet
                    if not all_stages_visited:
                        print(f"[{self.sign_name}] Completion blocked: not all stages visited ({len(self._visited_stages)}/{self._num_stages})")
                    if not has_movement:
                        print(f"[{self.sign_name}] Completion blocked: insufficient movement (avg={avg_movement:.4f})")
                    if not not_static_in_final:
                        print(f"[{self.sign_name}] Completion blocked: static in final position ({self._static_pose_frames} frames)")
            else:
                # Increment cooldown counter if we're not at final stage
                if hasattr(self, '_frames_since_completion'):
                    self._frames_since_completion += 1
        else:
            # Increment cooldown counter if we're not at final stage
            if hasattr(self, '_frames_since_completion'):
                self._frames_since_completion += 1
        
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
