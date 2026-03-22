"""Prediction smoothing for stable ASL letter display."""

from collections import deque, Counter
from typing import Optional


class PredictionSmoother:
    """
    Smooths predictions using a sliding buffer.
    Confirms letter only when same result appears min_confidence times.
    """

    def __init__(self, buffer_size: int = 5, min_confidence: int = 3) -> None:
        """
        Initialize the smoother.

        Args:
            buffer_size: Max number of recent predictions to keep
            min_confidence: Same prediction must appear this many times to confirm
        """
        self.buffer: deque = deque(maxlen=buffer_size)
        self.min_confidence = min_confidence

    def add_prediction(self, prediction: Optional[str]) -> Optional[str]:
        """
        Add a prediction and return stable result if threshold met.

        Args:
            prediction: Current frame prediction ('A'-'F' or None)

        Returns:
            Confirmed letter if same prediction >= min_confidence times, else None
        """
        self.buffer.append(prediction)

        if len(self.buffer) < self.min_confidence:
            return None

        # Count most common non-None prediction
        non_none = [p for p in self.buffer if p is not None]
        if not non_none:
            return None

        most_common = Counter(non_none).most_common(1)[0]
        letter, count = most_common
        if count >= self.min_confidence:
            return letter
        return None

    def reset(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
