"""
ml/utils/checkpoint.py
=====================
Checkpoint management for model training.

Handles saving/loading checkpoints with versioning, automatic cleanup,
and best model tracking.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import json


class CheckpointManager:
    """
    Manages model checkpoints with versioning and automatic cleanup.
    
    Features:
    - Versioned checkpoint saving
    - Best model tracking
    - Automatic cleanup of old checkpoints
    - Resume training from last checkpoint
    - Metadata saving with each checkpoint
    
    Usage:
        checkpoint_manager = CheckpointManager("ml/models/checkpoints", max_keep=5)
        
        # Save checkpoint
        checkpoint_manager.save({
            "epoch": 10,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": {"val_acc": 0.95}
        }, is_best=True)
        
        # Load latest checkpoint
        checkpoint = checkpoint_manager.load_latest()
        
        # Resume training
        if checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            start_epoch = checkpoint["epoch"] + 1
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "ml/models/checkpoints",
        max_keep: int = 5,
        save_frequency: int = 5
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_keep: Maximum number of best checkpoints to keep
            save_frequency: Save checkpoint every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_keep = max_keep
        self.save_frequency = save_frequency
        
        # Track best metrics
        self.best_metric = 0.0
        self.best_metric_name = "val_acc"
        self.best_epoch = 0
        
        # Checkpoint history
        self.checkpoints: List[Dict] = []
        self._load_checkpoint_history()
    
    def _load_checkpoint_history(self):
        """Load checkpoint history from disk."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoints = data.get("checkpoints", [])
                    self.best_metric = data.get("best_metric", 0.0)
                    self.best_epoch = data.get("best_epoch", 0)
            except Exception as e:
                print(f"Warning: Could not load checkpoint history: {e}")
                self.checkpoints = []
    
    def _save_checkpoint_history(self):
        """Save checkpoint history to disk."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        
        data = {
            "checkpoints": self.checkpoints,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch
        }
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save(
        self,
        state: Dict[str, Any],
        is_best: bool = False,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            state: Dictionary containing model state and other data
            is_best: Whether this is the best model so far
            epoch: Current epoch number
            metrics: Optional metrics dictionary
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add metadata to state
        state["checkpoint_metadata"] = {
            "timestamp": timestamp,
            "epoch": epoch,
            "metrics": metrics or {},
            "pytorch_version": torch.__version__
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}.pt"
        torch.save(state, checkpoint_path)
        
        # Track checkpoint
        checkpoint_info = {
            "path": str(checkpoint_path),
            "timestamp": timestamp,
            "epoch": epoch,
            "metrics": metrics or {},
            "is_best": is_best
        }
        self.checkpoints.append(checkpoint_info)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            
            # Update best metrics
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if value > self.best_metric:
                            self.best_metric = value
                            self.best_metric_name = key
                            self.best_epoch = epoch
            
            # Save best model info
            best_info = {
                "epoch": epoch,
                "metrics": metrics,
                "timestamp": timestamp
            }
            with open(self.checkpoint_dir / "best_model_info.json", 'w') as f:
                json.dump(best_info, f, indent=2)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save history
        self._save_checkpoint_history()
        
        return checkpoint_path
    
    def save_if_needed(
        self,
        state: Dict[str, Any],
        epoch: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> Optional[Path]:
        """
        Save checkpoint if it meets the save frequency requirement.
        
        Args:
            state: Checkpoint state dictionary
            epoch: Current epoch
            metrics: Optional metrics
        
        Returns:
            Path to saved checkpoint or None
        """
        if epoch % self.save_frequency == 0:
            is_best = False
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and value > self.best_metric:
                        is_best = True
                        break
            
            return self.save(state, is_best=is_best, epoch=epoch, metrics=metrics)
        
        # Always check for best model
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value > self.best_metric:
                    self.save(state, is_best=True, epoch=epoch, metrics=metrics)
                    break
        
        return None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the best N."""
        # Get all checkpoint files sorted by modification time
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Keep max_keep checkpoints (plus best_model.pt)
        files_to_delete = checkpoint_files[self.max_keep:]
        
        for checkpoint_file in files_to_delete:
            try:
                checkpoint_file.unlink()
                # Remove from history
                self.checkpoints = [
                    c for c in self.checkpoints 
                    if c["path"] != str(checkpoint_file)
                ]
            except Exception as e:
                print(f"Warning: Could not delete old checkpoint {checkpoint_file}: {e}")
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Checkpoint dictionary or None if no checkpoints exist
        """
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not checkpoint_files:
            return None
        
        try:
            return torch.load(checkpoint_files[0], map_location='cpu')
        except Exception as e:
            print(f"Error loading latest checkpoint: {e}")
            return None
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """
        Load the best model checkpoint.
        
        Returns:
            Best model checkpoint dictionary or None
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        
        if not best_path.exists():
            return None
        
        try:
            return torch.load(best_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading best checkpoint: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint dictionary or None
        """
        path = Path(checkpoint_path)
        
        if not path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            return torch.load(path, map_location='cpu')
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def get_best_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the best model.
        
        Returns:
            Dictionary with best model info or None
        """
        info_path = self.checkpoint_dir / "best_model_info.json"
        
        if not info_path.exists():
            return None
        
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading best model info: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        return sorted(
            self.checkpoints, 
            key=lambda x: x.get("epoch", 0), 
            reverse=True
        )
    
    def reset(self):
        """Reset the checkpoint manager (for testing)."""
        # Delete all checkpoints
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pt"):
            try:
                checkpoint_file.unlink()
            except Exception:
                pass
        
        # Delete best model
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            try:
                best_path.unlink()
            except Exception:
                pass
        
        # Reset state
        self.best_metric = 0.0
        self.best_epoch = 0
        self.checkpoints = []
        self._save_checkpoint_history()


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str = "ml/models/checkpoints",
    is_best: bool = False
) -> Path:
    """
    Convenience function to save a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Metrics dictionary
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model
    
    Returns:
        Path to saved checkpoint
    """
    manager = CheckpointManager(checkpoint_dir)
    
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics
    }
    
    return manager.save(state, is_best=is_best, epoch=epoch, metrics=metrics)


def load_model_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Convenience function to load a model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load model to
    
    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    
    # Load optimizer state
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    return checkpoint
