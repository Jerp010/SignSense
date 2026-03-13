"""
asl_classifier_letters.py
==========================
MLP-based ASL letter classifier — inference wrapper.

This replaces the hand-crafted scoring system with a PyTorch MLP trained on
recorded hand landmark data. The classifier maintains the same public API so
main.py requires no changes.

INTERFACE CONTRACT
------------------
class ASLClassifierLetters:
    def __init__(self) -> None: ...
    def classify(
        self,
        landmarks,                      # list of 21 MediaPipe NormalizedLandmark
        handedness: Optional[str] = None,   # "Left" | "Right" | None
        target_letter: Optional[str] = None # play-mode targeted scoring
    ) -> Optional[Dict]:
        # Returns: {"letter": str, "confidence": float, "scores": dict[str, float]}
        #          or None if no detection above threshold

    min_confidence: float = 0.6         # public attribute

ACTIVE LETTERS
--------------
The MLP classifies static single-frame poses only. Dynamics signs (J, Z, ...)
are handled by signs/dynamic_signs.py and main.py's targeted-mode logic.

MODEL LOADING
-------------
- Looks for ml/models/sign_mlp.pt (PyTorch checkpoint)
- If missing, logs a warning and runs in "no-model" mode (returns None every frame)
- Allows the app to start and run before any training has been done
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict
from types import SimpleNamespace

import torch
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.model import load_model


class ASLClassifierLetters:
    """
    MLP-based ASL letter classifier.
    
    Loads a trained PyTorch model and performs real-time inference on hand
    landmarks. Maintains the same public API as the previous scoring-based
    classifier for compatibility with main.py and the rest of the codebase.
    """

    MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "models" / "sign_mlp.pt"

    def __init__(self) -> None:
        """Initialize classifier. Loads model if available; runs gracefully without it."""
        self.min_confidence = 0.6
        self._model = None
        self._normaliser = None
        self._label_map = {}
        self._inv_label_map = {}
        self._device = "cpu"
        
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the trained MLP model from disk.
        
        If MODEL_PATH does not exist, logs a warning and sets _model = None.
        The classifier will return None for every frame, allowing the app to
        start before training has been done.
        """
        model_path = Path(self.MODEL_PATH)
        
        if not model_path.exists():
            print(
                f"[ASLClassifierLetters] Warning: Model not found at {self.MODEL_PATH}\n"
                f"  Run: python -m ml.record_landmarks  (to collect data)\n"
                f"  Then: python -m ml.train            (to train the model)\n"
                f"  Until then, the classifier will return None (no detection)."
            )
            return
        
        try:
            model, normaliser, label_map, inv_label_map = load_model(
                str(model_path),
                device=self._device
            )
            self._model = model
            self._normaliser = normaliser
            self._label_map = label_map
            self._inv_label_map = inv_label_map
            print(f"[ASLClassifierLetters] Loaded model from {self.MODEL_PATH}")
            print(f"  Letters: {sorted(label_map.keys())}")
        except Exception as e:
            print(f"[ASLClassifierLetters] Error loading model: {e}")
            self._model = None

    def classify(
        self,
        landmarks: List,
        handedness: Optional[str] = None,
        target_letter: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Classify a hand pose into an ASL letter.

        Args:
            landmarks: list of 21 MediaPipe NormalizedLandmark objects
            handedness: "Left" | "Right" | None  (affects mirroring)
            target_letter: if set (play mode), only evaluate this letter

        Returns:
            {"letter": str, "confidence": float, "scores": dict[str, float]}
            or None if no detection above min_confidence
        """
        # Bail if model not loaded
        if self._model is None:
            return None
        
        # Validate input
        if landmarks is None or len(landmarks) < 21:
            return None
        
        # Mirror right-hand landmarks so all trained patterns assume left-hand orientation
        if handedness == "Right":
            landmarks = self._mirror_landmarks_x(landmarks)
        
        # Normalise landmarks
        try:
            vec = self._normaliser.normalise(landmarks)  # shape: (63,)
        except Exception as e:
            print(f"[ASLClassifierLetters] Normalisation error: {e}")
            return None
        
        # Forward pass
        try:
            self._model.eval()
            with torch.no_grad():
                x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self._device)
                logits = self._model(x)  # shape: (1, num_classes)
                probs = torch.softmax(logits, dim=1).squeeze()  # shape: (num_classes,)
        except Exception as e:
            print(f"[ASLClassifierLetters] Inference error: {e}")
            return None
        
        # Build scores dict for all letters
        scores = {
            self._inv_label_map[i]: float(probs[i].cpu().numpy())
            for i in range(len(self._inv_label_map))
        }
        
        # Targeted mode (play mode): check only target_letter
        if target_letter:
            confidence = scores.get(target_letter, 0.0)
            if confidence < self.min_confidence:
                return None
            return {
                "letter": target_letter,
                "confidence": round(confidence, 3),
                "scores": scores,
            }
        
        # Debug mode: return best-scoring letter if above threshold
        best_letter = max(scores, key=scores.get)
        best_confidence = scores[best_letter]
        
        if best_confidence < self.min_confidence:
            return None
        
        return {
            "letter": best_letter,
            "confidence": round(best_confidence, 3),
            "scores": scores,
        }

    def _mirror_landmarks_x(self, landmarks: List) -> List:
        """
        Flip x-coordinates so a right hand looks like a left hand.

        Why? The MLP was trained assuming left-hand orientation (thumb on left).
        By mirroring right-hand landmarks before normalisation, we apply the
        same patterns learned for left hands.

        Args:
            landmarks: list of 21 MediaPipe NormalizedLandmark objects

        Returns:
            list of 21 mirrored landmarks (same type as input)
        """
        mirrored = []
        for lm in landmarks:
            n = SimpleNamespace(
                x=1.0 - lm.x,
                y=lm.y,
                z=getattr(lm, "z", 0.0)
            )
            mirrored.append(n)
        return mirrored
        