"""
ml/dynamic_model_enhanced.py
=============================
Enhanced LSTM architecture for learning dynamic ASL signs.

Improvements over basic model:
  1. Bidirectional LSTM (captures past AND future context)
  2. Multi-head attention (focuses on important frames)
  3. Residual connections (prevents gradient vanishing)
  4. Stage-specific feature extractors
  5. Confidence calibration layer

Architecture:
  Sequence of frames (variable length) [shape: seq_len x 63]
    ↓
  Local Feature Encoder (63 → 64)
    ↓
  Bidirectional LSTM (64 → 128*2 = 256)
    ↓
  Multi-Head Attention (4 heads)
    ↓
  Residual Connection + Layer Norm
    ↓
  Stage-Specific Feature Extractors
    ↓
  Output Heads (Stage + Transition + Confidence)
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional, Dict, List
from pathlib import Path


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism for temporal focus.
    
    Allows the model to focus on the most important frames
    in the sequence for stage prediction.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            embed_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional mask [batch_size, seq_len] (1=valid, 0=padded)
        
        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)  # [batch, seq, embed_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # Shape: [batch, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # Shape: [batch, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class EnhancedDynamicSignLSTM(nn.Module):
    """
    Enhanced LSTM with bidirectional processing, attention, and residual connections.
    
    Key improvements over basic model:
    - Bidirectional LSTM: Captures context from both past and future frames
    - Multi-head attention: Focuses on the most important frames
    - Residual connections: Helps training deeper networks
    - Stage-specific features: Better discrimination between stages
    - Confidence calibration: Knows when it's uncertain
    """
    
    def __init__(
        self,
        num_stages: int,
        input_size: int = 63,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Args:
            num_stages: Number of sequential stages in this sign (e.g., 3 for J)
            input_size: Landmark feature size (63 = 21 landmarks * 3 coords)
            hidden_size: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_stages = num_stages
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Local feature encoder
        # Compresses 63-dim landmarks to 64-dim features
        self.local_encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # Lighter dropout here
        )
        
        # Bidirectional LSTM
        # Processes sequence in both directions for full context
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # KEY IMPROVEMENT: looks at past AND future
        )
        
        # Multi-head attention
        # Focuses on the most important frames for stage prediction
        self.attention = MultiHeadAttention(
            embed_dim=hidden_size * 2,  # *2 because bidirectional
            num_heads=num_attention_heads,
            dropout=dropout * 0.5
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Residual connection projection
        # Projects input to match LSTM output dimensions
        self.residual_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU()
        )
        
        # Stage-specific feature extractors
        # Each stage gets its own feature processing
        self.stage_features = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ) for _ in range(num_stages)
        ])
        
        # Stage classification head
        # Predicts which stage the hand is in at each frame
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_size * 2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_stages)
        )
        
        # Transition detection head
        # Detects when the hand moves from one stage to another
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output: 0-1 probability of transition
        )
        
        # Confidence calibration head
        # Estimates how confident the model is in its prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output: 0-1 confidence score
        )
    
    def forward(
        self,
        sequences: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sequences: Batch of sequences [batch_size, seq_len, 63]
                      (variable length sequences padded to same size)
            mask: Optional mask [batch_size, seq_len] (1=valid, 0=padded)
        
        Returns:
            stage_logits: [batch_size, seq_len, num_stages] - stage probabilities per frame
            transition_probs: [batch_size, seq_len, 1] - transition confidence per frame
            confidence_scores: [batch_size, seq_len, 1] - prediction confidence per frame
        """
        batch_size, seq_len, _ = sequences.shape
        
        # Step 1: Local feature encoding
        # Compress landmarks to lower-dimensional features
        local_features = self.local_encoder(sequences)
        # Shape: [batch, seq_len, 64]
        
        # Step 2: Bidirectional LSTM
        # Process sequence in both directions
        lstm_out, (h_n, c_n) = self.lstm(local_features)
        # lstm_out shape: [batch, seq_len, hidden_size*2]
        
        # Step 3: Multi-head attention
        # Focus on important frames
        attended = self.attention(lstm_out, mask)
        # Shape: [batch, seq_len, hidden_size*2]
        
        # Step 4: Residual connection with layer normalization
        # Add skip connection from input (helps training)
        residual = self.residual_proj(sequences)
        combined = self.layer_norm(attended + residual)
        # Shape: [batch, seq_len, hidden_size*2]
        
        # Step 5: Stage-specific feature extraction
        # Get features specialized for each stage
        stage_features_list = []
        for stage_idx in range(self.num_stages):
            stage_feat = self.stage_features[stage_idx](combined)
            stage_features_list.append(stage_feat)
        
        # Concatenate all stage features
        # Shape: [batch, seq_len, 64*num_stages]
        all_stage_features = torch.cat(stage_features_list, dim=-1)
        
        # Step 6: Stage prediction
        # Combine LSTM output with stage-specific features
        stage_input = torch.cat([combined, all_stage_features[:, :, :64]], dim=-1)
        stage_logits = self.stage_head(stage_input)
        # Shape: [batch, seq_len, num_stages]
        
        # Step 7: Transition detection
        transition_probs = self.transition_head(combined)
        # Shape: [batch, seq_len, 1]
        
        # Step 8: Confidence calibration
        confidence_scores = self.confidence_head(combined)
        # Shape: [batch, seq_len, 1]
        
        return stage_logits, transition_probs, confidence_scores
    
    def get_attention_weights(
        self,
        sequences: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get attention weights for visualization/debugging.
        
        Args:
            sequences: Batch of sequences [batch_size, seq_len, 63]
            mask: Optional mask [batch_size, seq_len]
        
        Returns:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # This is a simplified version - in practice you'd need to
        # modify the attention module to return weights
        batch_size, seq_len, _ = sequences.shape
        
        local_features = self.local_encoder(sequences)
        lstm_out, _ = self.lstm(local_features)
        
        # For now, return dummy weights
        # In full implementation, modify MultiHeadAttention to return weights
        return torch.ones(batch_size, 4, seq_len, seq_len) / seq_len


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


def load_enhanced_dynamic_model(
    checkpoint_path: str,
    sign_name: str,
    device: str = "cpu"
) -> Tuple[EnhancedDynamicSignLSTM, DynamicSignNormaliser, int]:
    """
    Load a trained enhanced dynamic sign model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        sign_name: Name of sign (e.g., "J")
        device: "cpu" or "cuda"
    
    Returns:
        (model, normaliser, num_stages)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    num_stages = checkpoint["num_stages"]
    model = EnhancedDynamicSignLSTM(num_stages=num_stages)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    normaliser = DynamicSignNormaliser()
    
    return model, normaliser, num_stages


def compare_models():
    """
    Compare basic vs enhanced model architectures.
    
    Returns:
        Dictionary with comparison metrics
    """
    num_stages = 3
    batch_size = 8
    seq_len = 120
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, 63)
    
    # Basic model (from dynamic_model.py)
    from signsense.ml.dynamic_model import DynamicSignLSTM
    basic_model = DynamicSignLSTM(num_stages=num_stages)
    
    # Enhanced model
    enhanced_model = EnhancedDynamicSignLSTM(num_stages=num_stages)
    
    # Count parameters
    basic_params = sum(p.numel() for p in basic_model.parameters())
    enhanced_params = sum(p.numel() for p in enhanced_model.parameters())
    
    # Forward pass
    basic_out = basic_model(x)
    enhanced_out = enhanced_model(x)
    
    return {
        "basic": {
            "parameters": basic_params,
            "output_shapes": [o.shape for o in basic_out]
        },
        "enhanced": {
            "parameters": enhanced_params,
            "output_shapes": [o.shape for o in enhanced_out],
            "improvements": [
                "Bidirectional LSTM (captures future context)",
                "Multi-head attention (focuses on important frames)",
                "Residual connections (better gradient flow)",
                "Stage-specific features (better discrimination)",
                "Confidence calibration (uncertainty estimation)"
            ]
        }
    }


if __name__ == "__main__":
    # Test the enhanced model
    print("Testing Enhanced Dynamic Sign LSTM Model")
    print("=" * 50)
    
    comparison = compare_models()
    
    print(f"\nBasic Model:")
    print(f"  Parameters: {comparison['basic']['parameters']:,}")
    print(f"  Output shapes: {comparison['basic']['output_shapes']}")
    
    print(f"\nEnhanced Model:")
    print(f"  Parameters: {comparison['enhanced']['parameters']:,}")
    print(f"  Output shapes: {comparison['enhanced']['output_shapes']}")
    print(f"\n  Key Improvements:")
    for improvement in comparison['enhanced']['improvements']:
        print(f"    - {improvement}")
    
    param_increase = (
        (comparison['enhanced']['parameters'] - comparison['basic']['parameters'])
        / comparison['basic']['parameters'] * 100
    )
    print(f"\nParameter increase: {param_increase:.1f}%")
