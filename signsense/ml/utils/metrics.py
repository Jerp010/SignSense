"""
ml/utils/metrics.py
==================
Metrics tracking and visualization for SignSense training.

Provides:
- Training metrics logging
- Metrics history management
- Training curve generation
- Performance reporting
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    additional_metrics: Dict[str, float] = field(default_factory=dict)


class TrainingMetrics:
    """
    Track and manage training metrics.
    
    Features:
    - Log metrics per epoch
    - Calculate running statistics
    - Generate training curves
    - Export metrics to various formats
    
    Usage:
        metrics = TrainingMetrics("ml/models/metrics")
        
        # Log epoch metrics
        metrics.log_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            val_acc=0.85,
            learning_rate=0.001
        )
        
        # Get history
        history = metrics.get_history()
        
        # Plot curves
        metrics.plot_curves("training_curves.png")
    """
    
    def __init__(
        self,
        metrics_dir: str = "ml/models/metrics",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize training metrics tracker.
        
        Args:
            metrics_dir: Directory to save metrics
            experiment_name: Optional experiment name
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Metrics storage
        self.epochs: List[EpochMetrics] = []
        
        # Running statistics
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Load existing metrics if available
        self._load_metrics()
    
    def _get_metrics_file(self) -> Path:
        """Get path to metrics file."""
        return self.metrics_dir / f"{self.experiment_name}_metrics.json"
    
    def _load_metrics(self):
        """Load existing metrics from disk."""
        metrics_file = self._get_metrics_file()
        
        if not metrics_file.exists():
            return
        
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                self.epochs = [EpochMetrics(**e) for e in data.get("epochs", [])]
                self.best_val_acc = data.get("best_val_acc", 0.0)
                self.best_val_loss = data.get("best_val_loss", float('inf'))
                self.best_epoch = data.get("best_epoch", 0)
        except Exception as e:
            print(f"Warning: Could not load metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to disk."""
        metrics_file = self._get_metrics_file()
        
        data = {
            "experiment_name": self.experiment_name,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "epochs": [asdict(e) for e in self.epochs]
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        learning_rate: float = 0.0,
        epoch_time: float = 0.0,
        **additional_metrics
    ):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            val_acc: Validation accuracy
            learning_rate: Learning rate
            epoch_time: Time taken for epoch
            **additional_metrics: Any additional metrics to track
        """
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=learning_rate,
            epoch_time=epoch_time,
            additional_metrics=additional_metrics
        )
        
        self.epochs.append(epoch_metrics)
        
        # Update best metrics
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
        
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
        # Save to disk
        self._save_metrics()
    
    def get_history(self) -> Dict[str, List[float]]:
        """
        Get metrics history as dictionary.
        
        Returns:
            Dictionary with lists of metric values
        """
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
            "epoch_time": []
        }
        
        for e in self.epochs:
            history["epoch"].append(e.epoch)
            history["train_loss"].append(e.train_loss)
            
            if e.val_loss is not None:
                history["val_loss"].append(e.val_loss)
            if e.val_acc is not None:
                history["val_acc"].append(e.val_acc)
            if e.learning_rate is not None:
                history["learning_rate"].append(e.learning_rate)
            if e.epoch_time is not None:
                history["epoch_time"].append(e.epoch_time)
        
        return history
    
    def get_epoch_count(self) -> int:
        """Get total number of epochs logged."""
        return len(self.epochs)
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get best metrics achieved.
        
        Returns:
            Dictionary with best metric values
        """
        return {
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch
        }
    
    def get_latest_metrics(self) -> Optional[EpochMetrics]:
        """Get the most recent epoch metrics."""
        if not self.epochs:
            return None
        return self.epochs[-1]
    
    def get_metrics_df(self):
        """
        Get metrics as pandas DataFrame.
        
        Returns:
            DataFrame with metrics
        """
        import pandas as pd
        
        data = []
        for e in self.epochs:
            row = {
                "epoch": e.epoch,
                "train_loss": e.train_loss,
                "val_loss": e.val_loss,
                "val_acc": e.val_acc,
                "learning_rate": e.learning_rate,
                "epoch_time": e.epoch_time
            }
            row.update(e.additional_metrics)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_curves(
        self,
        output_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot training curves.
        
        Args:
            output_path: Path to save plot
            show: Whether to display the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available, skipping plot")
            return
        
        history = self.get_history()
        
        if not history["epoch"]:
            print("No data to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Training Curves - {self.experiment_name}")
        
        # Plot training loss
        if history["train_loss"]:
            axes[0, 0].plot(history["epoch"], history["train_loss"], 'b-', label='Train Loss')
            if history["val_loss"]:
                axes[0, 0].plot(history["epoch"], history["val_loss"], 'r-', label='Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training & Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot validation accuracy
        if history["val_acc"]:
            axes[0, 1].plot(history["epoch"], history["val_acc"], 'g-', label='Val Accuracy')
            axes[0, 1].axhline(
                y=self.best_val_acc, 
                color='r', 
                linestyle='--', 
                label=f'Best ({self.best_val_acc:.3f})'
            )
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1])
        
        # Plot learning rate
        if history["learning_rate"]:
            axes[1, 0].plot(history["epoch"], history["learning_rate"], 'purple', label='Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # Plot epoch time
        if history["epoch_time"]:
            axes[1, 1].plot(history["epoch"], history["epoch_time"], 'orange', label='Epoch Time')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_title('Epoch Training Time')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {output_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def print_summary(self):
        """Print a summary of training metrics."""
        if not self.epochs:
            print("No training metrics available")
            return
        
        print("\n" + "=" * 60)
        print(f"Training Summary - {self.experiment_name}")
        print("=" * 60)
        print(f"Total Epochs: {len(self.epochs)}")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f} (epoch {self.best_epoch})")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        
        latest = self.get_latest_metrics()
        if latest:
            print(f"\nFinal Epoch ({latest.epoch}):")
            print(f"  Train Loss: {latest.train_loss:.4f}")
            if latest.val_loss:
                print(f"  Val Loss: {latest.val_loss:.4f}")
            if latest.val_acc:
                print(f"  Val Accuracy: {latest.val_acc:.4f}")
            print(f"  Learning Rate: {latest.learning_rate:.6f}")
            print(f"  Epoch Time: {latest.epoch_time:.2f}s")
        
        print("=" * 60 + "\n")
    
    def reset(self):
        """Reset all metrics."""
        self.epochs = []
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self._save_metrics()


# Helper functions for computing metrics

def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float(np.mean(predictions == targets))


def compute_confusion_matrix(
    predictions: np.ndarray, 
    targets: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """Compute confusion matrix."""
    return np.bincount(
        num_classes * targets + predictions,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def compute_per_class_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> Dict[int, float]:
    """Compute per-class accuracy."""
    per_class_acc = {}
    
    for class_idx in range(num_classes):
        mask = targets == class_idx
        if mask.sum() > 0:
            per_class_acc[class_idx] = float(
                np.mean(predictions[mask] == targets[mask])
            )
    
    return per_class_acc


# Import dataclass for asdict
from dataclasses import asdict


class MetricsLogger:
    """
    Simple metrics logger for real-time tracking.
    
    Provides a simpler interface for logging metrics during training.
    
    Usage:
        logger = MetricsLogger()
        
        # Log batch loss
        logger.log_batch_loss(0.5)
        
        # Log epoch metrics
        logger.log_epoch(train_loss=0.5, val_acc=0.9)
        
        # Get metrics
        metrics = logger.get_metrics()
    """
    
    def __init__(self):
        """Initialize the metrics logger."""
        self.batch_losses: List[float] = []
        self.epoch_metrics: List[Dict[str, float]] = []
        
        # Current epoch tracking
        self.current_epoch = 0
        self.current_train_loss = 0.0
        self.current_batch_count = 0
    
    def log_batch_loss(self, loss: float):
        """Log a batch loss value."""
        self.batch_losses.append(loss)
        self.current_train_loss += loss
        self.current_batch_count += 1
    
    def begin_epoch(self, epoch: int):
        """Mark the beginning of an epoch."""
        self.current_epoch = epoch
        self.current_train_loss = 0.0
        self.current_batch_count = 0
        self.batch_losses = []
    
    def log_epoch(self, **metrics):
        """Log metrics for the current epoch."""
        if self.current_batch_count > 0:
            avg_train_loss = self.current_train_loss / self.current_batch_count
            metrics["train_loss"] = avg_train_loss
        
        metrics["epoch"] = self.current_epoch
        self.epoch_metrics.append(metrics)
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get all logged metrics."""
        result = {"epoch": []}
        
        for m in self.epoch_metrics:
            result["epoch"].append(m.get("epoch", 0))
            for key, value in m.items():
                if key != "epoch" and isinstance(value, (int, float)):
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
        
        return result
    
    def get_latest(self) -> Optional[Dict[str, float]]:
        """Get the most recent epoch metrics."""
        if not self.epoch_metrics:
            return None
        return self.epoch_metrics[-1]
    
    def reset(self):
        """Reset all metrics."""
        self.batch_losses = []
        self.epoch_metrics = []
        self.current_epoch = 0
        self.current_train_loss = 0.0
        self.current_batch_count = 0
