"""
ml/model.py
===========
MLP architecture, landmark normalisation, and model loading for ASL classification.

Architecture:
  Input (63 floats: 21 landmarks × 3 coords)
  → Linear(63, 256) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(256, 128) → BatchNorm1d → ReLU → Dropout(0.2)
  → Linear(128, 64) → BatchNorm1d → ReLU
  → Linear(64, num_classes)
  → output logits (softmax at inference)

Normalisation:
  1. Translate all 21 landmarks so wrist (lm[0]) is at origin
  2. Scale by wrist→middle-MCP (lm[0]→lm[9]) distance
  3. Flatten to 63 floats: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
  → result: np.ndarray shape (63,), dtype float32
"""

import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Dict, Optional


class LandmarkNormaliser:
    """
    Normalise MediaPipe 21 hand landmarks to camera-distance-invariant features.
    
    Steps:
      1. Compute scale: distance from wrist (lm[0]) to middle MCP (lm[9])
      2. Translate all landmarks relative to wrist
      3. Scale all coordinates by this distance
      4. Flatten to [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
    """

    def normalise(self, landmarks) -> np.ndarray:
        """
        Args:
            landmarks: list of 21 MediaPipe NormalizedLandmark objects
                      (each with .x, .y, .z attributes in [0, 1])

        Returns:
            np.ndarray of shape (63,), dtype float32
        """
        # Compute scale: wrist (lm[0]) to middle MCP (lm[9])
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        scale = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)
        
        # Avoid division by zero
        if scale < 1e-6:
            scale = 1e-6

        # Translate & scale each landmark relative to wrist
        wrist_z = getattr(landmarks[0], "z", 0.0)
        features = []
        for lm in landmarks:
            # Translate to wrist origin
            dx = (lm.x - wrist.x) / scale
            dy = (lm.y - wrist.y) / scale
            dz = (getattr(lm, "z", 0.0) - wrist_z) / scale
            
            features.append(dx)
            features.append(dy)
            features.append(dz)

        return np.array(features, dtype=np.float32)


class SignMLP(nn.Module):
    """
    Multi-layer perceptron for ASL letter classification.
    
    Takes normalized landmark vector (63 dims) and outputs logits over letter classes.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Args:
            num_classes: number of ASL letters to classify (e.g., 15 for A-P)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (batch, 63)

        Returns:
            tensor of shape (batch, num_classes) — logits
        """
        return self.net(x)


def load_model(
    model_path: str,
    label_map: Optional[Dict[str, int]] = None,
    device: Optional[str] = None,
) -> Tuple[SignMLP, LandmarkNormaliser, Dict[str, int], Dict[int, str]]:
    """
    Load a trained MLP model from disk.

    Args:
        model_path: path to .pt checkpoint (e.g., "ml/models/sign_mlp.pt")
        label_map: optional label map (dict[str, int]). If provided, will be
                   validated against the checkpoint. If None, loaded from checkpoint.
        device: torch device string (default: "cpu")

    Returns:
        Tuple of:
          - model: SignMLP instance (on device, eval mode)
          - normaliser: LandmarkNormaliser instance
          - label_map: dict {letter: int_class_index}
          - inv_label_map: dict {int_class_index: letter}

    Raises:
        FileNotFoundError: if model_path does not exist
        KeyError: if checkpoint is missing required keys
    """
    device = device or "cpu"
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(path, map_location=device)

    # Extract metadata from checkpoint
    num_classes = checkpoint.get("num_classes")
    ckpt_label_map = checkpoint.get("label_map")
    model_state = checkpoint.get("model_state")

    if num_classes is None or ckpt_label_map is None or model_state is None:
        raise KeyError(
            "Checkpoint missing required keys: 'num_classes', 'label_map', 'model_state'"
        )

    # Validate against provided label_map if given
    if label_map is not None:
        if label_map != ckpt_label_map:
            raise ValueError(
                f"Provided label_map differs from checkpoint. "
                f"Checkpoint: {ckpt_label_map}, Provided: {label_map}"
            )
    else:
        label_map = ckpt_label_map

    # Rebuild model and load weights
    model = SignMLP(num_classes=num_classes).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Create inverse label map for inference
    inv_label_map = {v: k for k, v in label_map.items()}

    normaliser = LandmarkNormaliser()

    return model, normaliser, label_map, inv_label_map
