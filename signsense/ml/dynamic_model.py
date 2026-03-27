"""
ml/dynamic_model.py
==================
Basic LSTM model for dynamic ASL sign recognition.

This is the basic model architecture - for enhanced model with 
bidirectional LSTM and attention, see dynamic_model_enhanced.py.

Note: This is a placeholder/basic implementation. The enhanced model
(dynamic_model_enhanced.py) is the primary implementation.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import numpy as np


class DynamicSignLSTM(nn.Module):
    """
    Basic unidirectional LSTM for dynamic sign stage recognition.
    
    Architecture:
        Input (63 features) → LSTM (2 layers) → Stage Head → Stage Prediction
    
    For enhanced model with bidirectional LSTM and attention, 
    use EnhancedDynamicSignLSTM from dynamic_model_enhanced.py.
    """
    
    def __init__(self, num_stages: int = 4, hidden_size: int = 128):
        """
        Args:
            num_stages: Number of stages in the sign sequence
            hidden_size: LSTM hidden dimension
        """
        super().__init__()
        self.num_stages = num_stages
        self.hidden_size = hidden_size
        
        # Feature encoder: 63 → hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(63, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Unidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Stage prediction head
        self.stage_head = nn.Linear(hidden_size, num_stages)
        
        # Transition detection head
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, 63)
        
        Returns:
            (stage_logits, transition_probs)
            - stage_logits: (batch_size, seq_len, num_stages)
            - transition_probs: (batch_size, seq_len, 1)
        """
        # Encode features
        encoded = self.encoder(x)  # (batch, seq_len, hidden_size)
        
        # LSTM
        lstm_out, _ = self.lstm(encoded)  # (batch, seq_len, hidden_size)
        
        # Stage predictions
        stage_logits = self.stage_head(lstm_out)
        
        # Transition predictions
        transition_probs = self.transition_head(lstm_out)
        
        return stage_logits, transition_probs


class DynamicSignNormaliser:
    """Normaliser for dynamic sign landmark sequences."""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, sequences):
        """Fit normaliser to training sequences."""
        all_frames = np.vstack([seq for seq in sequences])
        self.mean = np.mean(all_frames, axis=0)
        self.std = np.std(all_frames, axis=0) + 1e-8
    
    def transform(self, sequence):
        """Normalize a single sequence."""
        if self.mean is None:
            return sequence
        return (sequence - self.mean) / self.std
    
    def fit_transform(self, sequences):
        """Fit and transform in one step."""
        self.fit(sequences)
        return [self.transform(seq) for seq in sequences]


def load_dynamic_model(
    checkpoint_path: str,
    sign_name: str,
    device: str = "cpu"
) -> Tuple[DynamicSignLSTM, DynamicSignNormaliser, int]:
    """
    Load a trained dynamic sign model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        sign_name: Name of sign (e.g., "J", "Z")
        device: "cpu" or "cuda"
    
    Returns:
        (model, normaliser, num_stages)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    num_stages = checkpoint.get("num_stages", 4)
    hidden_size = checkpoint.get("hidden_size", 128)
    
    model = DynamicSignLSTM(num_stages=num_stages, hidden_size=hidden_size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    normaliser = DynamicSignNormaliser()
    if "normaliser_mean" in checkpoint:
        normaliser.mean = checkpoint["normaliser_mean"]
        normaliser.std = checkpoint["normaliser_std"]
    
    return model, normaliser, num_stages


def save_model(
    model: DynamicSignLSTM,
    normaliser: DynamicSignNormaliser,
    num_stages: int,
    path: str,
    sign_name: str = "",
    hidden_size: int = 128
):
    """Save model checkpoint."""
    checkpoint = {
        "model_state": model.state_dict(),
        "num_stages": num_stages,
        "hidden_size": hidden_size,
        "sign_name": sign_name,
    }
    if normaliser.mean is not None:
        checkpoint["normaliser_mean"] = normaliser.mean
        checkpoint["normaliser_std"] = normaliser.std
    
    torch.save(checkpoint, path)