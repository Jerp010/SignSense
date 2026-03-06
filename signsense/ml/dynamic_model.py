"""
ml/dynamic_model.py
===================
LSTM architecture for learning dynamic ASL signs.

A dynamic sign consists of multiple sequential stages. This model:
  1. Takes variable-length sequences of normalized hand landmarks
  2. Predicts current stage at each frame
  3. Detects stage transitions
  4. Judges sign completion based on stage progression

Architecture:
  Sequence of frames (variable length) [shape: seq_len x 63]
    ↓
  LSTM (input_size=63, hidden_size=128, num_layers=2, dropout=0.3)
    ↓
  Output (num_stages)
    ↓
  Softmax to get stage probabilities + stage transition confidence
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path


class DynamicSignLSTM(nn.Module):
    """
    LSTM-based classifier for dynamic ASL sign sequences.
    
    Learns to recognize multi-stage sign sequences and detect transitions.
    """
    
    def __init__(self, num_stages: int, input_size: int = 63, hidden_size: int = 128):
        """
        Args:
            num_stages: Number of sequential stages in this sign (e.g., 3 for J)
            input_size: Landmark feature size (63 = 21 landmarks * 3 coords)
            hidden_size: LSTM hidden dimension
        """
        super().__init__()
        self.num_stages = num_stages
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM: process sequence of landmarks
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )
        
        # Head 1: Stage classifier (what stage is the hand in?)
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_stages)
        )
        
        # Head 2: Transition detector (is this frame a stage transition?)
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output: 0-1 probability
        )
    
    def forward(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequences: Batch of sequences [batch_size, seq_len, 63]
                      (variable length sequences padded to same size)
        
        Returns:
            stage_logits: [batch_size, seq_len, num_stages] - stage probabilities per frame
            transition_probs: [batch_size, seq_len, 1] - transition confidence per frame
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(sequences)
        # lstm_out shape: [batch_size, seq_len, hidden_size]
        
        # Stage prediction for each frame
        stage_logits = self.stage_head(lstm_out)
        # [batch_size, seq_len, num_stages]
        
        # Transition detection for each frame
        transition_probs = self.transition_head(lstm_out)
        # [batch_size, seq_len, 1]
        
        return stage_logits, transition_probs


class DynamicSignNormaliser:
    """Normalise landmark sequences (same as static normaliser but for sequences)."""
    
    def normalise_sequence(self, landmarks_sequence: List) -> np.ndarray:
        """
        Normalise a sequence of hand landmarks.
        
        Args:
            landmarks_sequence: List of landmark lists (each is 21 landmarks)
        
        Returns:
            np.ndarray of shape (len(landmarks_sequence), 63)
        """
        import math
        
        normalized_frames = []
        for frame_landmarks in landmarks_sequence:
            # Same logic as static normaliser
            wrist = frame_landmarks[0]
            middle_mcp = frame_landmarks[9]
            scale = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)
            
            if scale < 1e-6:
                scale = 1e-6
            
            wrist_z = getattr(frame_landmarks[0], "z", 0.0)
            features = []
            
            for lm in frame_landmarks:
                dx = (lm.x - wrist.x) / scale
                dy = (lm.y - wrist.y) / scale
                dz = (getattr(lm, "z", 0.0) - wrist_z) / scale
                features.extend([dx, dy, dz])
            
            normalized_frames.append(features)
        
        return np.array(normalized_frames, dtype=np.float32)


def load_dynamic_model(
    checkpoint_path: str,
    sign_name: str,
    device: str = "cpu"
) -> Tuple[DynamicSignLSTM, DynamicSignNormaliser, int]:
    """
    Load a trained dynamic sign model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        sign_name: Name of sign (e.g., "J")
        device: "cpu" or "cuda"
    
    Returns:
        (model, normaliser, num_stages)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    num_stages = checkpoint["num_stages"]
    model = DynamicSignLSTM(num_stages=num_stages)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    normaliser = DynamicSignNormaliser()
    
    return model, normaliser, num_stages
