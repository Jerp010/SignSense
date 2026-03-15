"""
ml/trainers/base_trainer.py
===========================
Base trainer class for SignSense models.

Provides common training functionality including:
- Device management
- Checkpoint management  
- Metrics tracking
- Experiment logging
- Training loop with early stopping
"""

import sys
import time
import random
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import training utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signsense.utils.logger import logger, TimingContext
from signsense.ml.utils.checkpoint import CheckpointManager
from signsense.ml.utils.experiment_tracker import get_tracker
from signsense.ml.utils.metrics import TrainingMetrics
from signsense.ml.config.config_loader import (
    load_config, get_model_config, get_output_config, 
    get_defaults, set_random_seed, get_device
)


class BaseTrainer(ABC):
    """
    Abstract base trainer class for SignSense models.
    
    Provides common training infrastructure including:
    - Device setup
    - Checkpoint management
    - Metrics tracking
    - Experiment logging
    - Training loop with early stopping
    
    Subclasses must implement:
    - _create_model()
    - _create_dataloaders()
    - _train_step()
    - _validate()
    - _get_model_name()
    
    Usage:
        class MyTrainer(BaseTrainer):
            def _create_model(self): ...
            def _create_dataloaders(self): ...
            ...
        
        trainer = MyTrainer(config)
        trainer.train()
    """
    
    def __init__(
        self,
        model_config: Optional[Dict] = None,
        output_dir: str = "ml/models",
        experiment_name: Optional[str] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model_config: Model configuration dictionary
            output_dir: Directory for outputs
            experiment_name: Name for experiment tracking
            device: Device to use (cpu/cuda/auto)
            seed: Random seed for reproducibility
        """
        # Load configuration
        self.config = model_config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get defaults
        defaults = get_defaults()
        
        # Device setup
        if device:
            self.device_str = device
        else:
            self.device_str = defaults.get("device", "auto")
        
        if self.device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_str)
        
        # Random seed
        self.seed = seed if seed is not None else defaults.get("random_seed", 42)
        set_random_seed(self.seed)
        
        # Get training config
        training_config = self.config.get("training", {})
        self.epochs = training_config.get("epochs", 100)
        self.batch_size = training_config.get("batch_size", 64)
        self.learning_rate = training_config.get("learning_rate", 0.001)
        self.weight_decay = training_config.get("weight_decay", 0.0001)
        self.gradient_clip = training_config.get("gradient_clip", 1.0)
        
        # Early stopping config
        early_stopping = training_config.get("early_stopping", {})
        self.early_stopping_enabled = early_stopping.get("enabled", True)
        self.patience = early_stopping.get("patience", 15)
        self.min_delta = early_stopping.get("min_delta", 0.001)
        self.early_stopping_mode = early_stopping.get("mode", "max")
        
        # Output config
        output_config = get_output_config()
        self.checkpoint_dir = output_config.get("checkpoint_dir", "ml/models/checkpoints")
        self.metrics_dir = output_config.get("metrics_dir", "ml/models/metrics")
        self.experiment_dir = output_config.get("experiment_dir", "ml/experiments")
        
        # Experiment name
        self.experiment_name = experiment_name or self._get_model_name()
        
        # Initialize components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.checkpoint_dir,
            max_keep=output_config.get("max_checkpoints_keep", 5)
        )
        
        # Metrics tracker
        self.metrics = TrainingMetrics(
            metrics_dir=self.metrics_dir,
            experiment_name=self.experiment_name
        )
        
        # Experiment tracker
        self.experiment_tracker = get_tracker(self.experiment_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.is_training = False
        
        # Mixed precision
        self.use_amp = False
        self.scaler = None
    
    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create and return the model."""
        pass
    
    @abstractmethod
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create and return train and validation dataloaders."""
        pass
    
    @abstractmethod
    def _train_step(
        self, 
        batch: Any, 
        batch_idx: int
    ) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Batch of training data
            batch_idx: Batch index
        
        Returns:
            Dictionary of metrics
        """
        pass
    
    @abstractmethod
    def _validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    @abstractmethod
    def _get_model_name(self) -> str:
        """Get the model name for saving."""
        pass
    
    def _setup_optimizer(self):
        """Setup the optimizer."""
        training_config = self.config.get("training", {})
        optimizer_name = training_config.get("optimizer", "adamw").lower()
        
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == "sgd":
            momentum = training_config.get("momentum", 0.9)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=momentum,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        logger.info(f"Optimizer: {optimizer_name}, LR: {self.learning_rate}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        training_config = self.config.get("training", {})
        scheduler_name = training_config.get("scheduler", "cosine_annealing").lower()
        
        if scheduler_name == "cosine_annealing":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=training_config.get("min_lr", 1e-6)
            )
        elif scheduler_name == "step_lr":
            step_size = training_config.get("step_size", 10)
            gamma = training_config.get("gamma", 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_name == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.early_stopping_mode,
                factor=0.1,
                patience=5
            )
        else:
            self.scheduler = None
        
        if self.scheduler:
            logger.info(f"Scheduler: {scheduler_name}")
    
    def _setup_criterion(self):
        """Setup the loss criterion."""
        training_config = self.config.get("training", {})
        label_smoothing = training_config.get("label_smoothing", 0.0)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def setup(self):
        """Setup all training components."""
        logger.info("Setting up trainer...")
        
        # Create model
        with TimingContext("model_creation"):
            self.model = self._create_model()
            self.model.to(self.device)
        
        logger.info(f"Model: {self._get_model_name()}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create dataloaders
        with TimingContext("dataloader_creation"):
            self.train_loader, self.val_loader = self._create_dataloaders()
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        
        # Setup optimizer, scheduler, criterion
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_criterion()
        
        logger.info("Trainer setup complete")
    
    def train(self, resume: bool = False) -> Dict[str, Any]:
        """
        Run the training loop.
        
        Args:
            resume: Whether to resume from latest checkpoint
        
        Returns:
            Training results dictionary
        """
        # Start experiment
        exp = self.experiment_tracker.start_experiment(
            name=self.experiment_name,
            config=self.config
        )
        
        # Resume from checkpoint if requested
        start_epoch = 0
        if resume:
            checkpoint = self.checkpoint_manager.load_latest()
            if checkpoint:
                start_epoch = checkpoint.get("epoch", 0) + 1
                self._load_checkpoint_state(checkpoint)
                logger.info(f"Resuming from epoch {start_epoch}")
        
        self.is_training = True
        
        # Log training start
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Starting Training: {self.experiment_name}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 70)
        
        try:
            for epoch in range(start_epoch, self.epochs):
                self.current_epoch = epoch
                
                # Train one epoch
                with TimingContext(f"epoch_{epoch + 1}"):
                    epoch_metrics = self._train_one_epoch(epoch)
                
                # Validate
                val_metrics = self._validate()
                
                # Update scheduler
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get("val_loss", 0))
                    else:
                        self.scheduler.step()
                
                # Log metrics
                metrics = {**epoch_metrics, **val_metrics, "learning_rate": current_lr}
                
                self.metrics.log_epoch(
                    epoch=epoch + 1,
                    train_loss=epoch_metrics.get("train_loss", 0),
                    val_loss=val_metrics.get("val_loss"),
                    val_acc=val_metrics.get("val_acc"),
                    learning_rate=current_lr,
                    epoch_time=epoch_metrics.get("epoch_time", 0)
                )
                
                # Log to experiment tracker
                self.experiment_tracker.log_metrics(exp.id, metrics)
                
                # Print progress
                self._print_epoch_progress(epoch + 1, metrics)
                
                # Check for improvement
                metric_to_check = "val_acc" if self.early_stopping_mode == "max" else "val_loss"
                current_metric = val_metrics.get(metric_to_check, 0)
                
                is_best = False
                if self.early_stopping_mode == "max":
                    is_best = current_metric > self.best_metric + self.min_delta
                else:
                    is_best = current_metric < self.best_metric - self.min_delta
                
                if is_best:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                checkpoint_state = self._get_checkpoint_state(epoch)
                self.checkpoint_manager.save_if_needed(
                    checkpoint_state,
                    epoch=epoch + 1,
                    metrics=metrics
                )
                
                # Early stopping
                if self.early_stopping_enabled and self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Training complete
            self._on_training_complete()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.experiment_tracker.fail_experiment(exp.id, str(e))
            raise
        
        finally:
            self.is_training = False
        
        # Return results
        return {
            "best_metric": self.best_metric,
            "epochs_completed": self.current_epoch + 1,
            "experiment_id": exp.id
        }
    
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Training step
                step_metrics = self._train_step(batch, batch_idx)
                
                total_loss += step_metrics.get("loss", 0)
                num_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"GPU OOM at batch {batch_idx}, skipping")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            "train_loss": avg_loss,
            "num_batches": num_batches
        }
    
    def _get_checkpoint_state(self, epoch: int) -> Dict[str, Any]:
        """Get state dictionary for checkpoint."""
        return {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "config": self.config,
            "experiment_name": self.experiment_name
        }
    
    def _load_checkpoint_state(self, checkpoint: Dict[str, Any]):
        """Load state from checkpoint."""
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        
        if "optimizer_state" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if "scheduler_state" in checkpoint and self.scheduler and checkpoint["scheduler_state"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        if "best_metric" in checkpoint:
            self.best_metric = checkpoint["best_metric"]
    
    def _print_epoch_progress(self, epoch: int, metrics: Dict[str, float]):
        """Print training progress for an epoch."""
        # Build progress string
        parts = [
            f"Epoch {epoch:3d}/{self.epochs}"
        ]
        
        if "train_loss" in metrics:
            parts.append(f"train_loss: {metrics['train_loss']:.4f}")
        
        if "val_loss" in metrics:
            parts.append(f"val_loss: {metrics['val_loss']:.4f}")
        
        if "val_acc" in metrics:
            parts.append(f"val_acc: {metrics['val_acc']:.4f}")
        
        if "learning_rate" in metrics:
            parts.append(f"lr: {metrics['learning_rate']:.6f}")
        
        # Check if this is the best epoch
        metric_to_check = "val_acc" if self.early_stopping_mode == "max" else "val_loss"
        current_metric = metrics.get(metric_to_check, 0)
        
        if self.early_stopping_mode == "max":
            is_best = current_metric >= self.best_metric
        else:
            is_best = current_metric <= self.best_metric
        
        marker = " ✓" if is_best and self.patience_counter == 0 else ""
        
        logger.info(" | ".join(parts) + marker)
    
    def _on_training_complete(self):
        """Called when training completes."""
        # Log completion
        logger.info("")
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info(f"Best {self.early_stopping_mode}: {self.best_metric:.4f}")
        logger.info(f"Epochs: {self.current_epoch + 1}")
        logger.info("=" * 70)
        
        # Print metrics summary
        self.metrics.print_summary()
        
        # Complete experiment
        self.experiment_tracker.complete_experiment(
            self.experiment_tracker.get_experiment_by_name(self.experiment_name).id
        )
        
        # Plot training curves
        curves_path = self.output_dir / f"training_curves_{self.experiment_name}.png"
        self.metrics.plot_curves(str(curves_path))
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the validation set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call setup() first.")
        
        self.model.eval()
        return self._validate()
    
    def save_model(self, path: str):
        """Save the model to a file."""
        if not self.model:
            raise RuntimeError("Model not initialized.")
        
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "experiment_name": self.experiment_name
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        
        logger.info(f"Model loaded from {path}")
