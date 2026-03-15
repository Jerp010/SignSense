"""
ml/config/config_loader.py
=========================
Unified configuration loader with validation for SignSense training.

Provides type-safe access to training configuration with validation,
CLI override support, and backward compatibility with existing config.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from copy import deepcopy

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


# =============================================================================
# Dataclasses for Type-Safe Configuration
# =============================================================================

@dataclass
class DefaultsConfig:
    """Default settings applied to all models."""
    random_seed: int = 42
    device: str = "auto"  # "cpu", "cuda", "auto"
    precision: str = "fp32"  # "fp32", "fp16", "mixed"


@dataclass
class DataConfig:
    """Data configuration for static and dynamic signs."""
    path: str = "ml/data/landmarks.csv"
    train_split: float = 0.85
    validation_split: float = 0.15
    stratify: bool = True
    cache_enabled: bool = True


@dataclass
class DynamicDataConfig:
    """Data configuration for dynamic signs."""
    base_path: str = "ml/data/dynamic"
    train_split: float = 0.80
    validation_split: float = 0.20
    stratify: bool = False
    sequence_length: int = 120


@dataclass
class DataSettingsConfig:
    """Container for all data settings."""
    static: DataConfig = field(default_factory=DataConfig)
    dynamic: DynamicDataConfig = field(default_factory=DynamicDataConfig)


@dataclass
class ArchitectureConfig:
    """Model architecture configuration."""
    input_size: int = 63
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.0])
    activation: str = "relu"
    batch_norm: bool = True
    use_bias: bool = True


@dataclass
class LSTMArchitectureConfig:
    """LSTM architecture configuration."""
    input_size: int = 63
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    use_attention: bool = False


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    patience: int = 15
    min_delta: float = 0.001
    mode: str = "max"  # "max" for accuracy, "min" for loss


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    label_smoothing: float = 0.0
    gradient_clip: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-08
    
    # Scheduler
    scheduler: str = "cosine_annealing"
    warmup_epochs: int = 0
    min_lr: float = 1e-06
    
    # Early stopping
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class AugmentationConfig:
    """Data augmentation settings."""
    enabled: bool = True
    noise_std: float = 0.005
    x_flip_prob: float = 0.5
    scale_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    rotation_range: float = 0


@dataclass
class SequenceConfig:
    """Sequence-specific settings for LSTM."""
    max_length: int = 120
    padding_value: float = 0.0
    label_padding: int = -1
    use_mask: bool = True


@dataclass
class ModelConfig:
    """Complete model configuration."""
    name: str = "SignMLP"
    architecture: Union[ArchitectureConfig, LSTMArchitectureConfig] = field(
        default_factory=ArchitectureConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)


@dataclass
class OutputConfig:
    """Output and logging configuration."""
    checkpoint_dir: str = "ml/models/checkpoints"
    save_best_only: bool = True
    save_frequency: int = 5
    max_checkpoints_keep: int = 5
    
    log_dir: str = "logs/training"
    log_frequency: int = 1
    print_frequency: int = 1
    use_tensorboard: bool = False
    
    metrics_dir: str = "ml/models/metrics"
    save_training_curves: bool = True
    curves_format: str = "png"
    
    experiment_dir: str = "ml/experiments"
    track_experiments: bool = True


@dataclass
class SearchSpaceConfig:
    """Hyperparameter search space for AutoML."""
    learning_rate: List[float] = field(default_factory=lambda: [0.001])
    batch_size: List[int] = field(default_factory=lambda: [64])


# =============================================================================
# Main Configuration Loader
# =============================================================================

class TrainingConfigLoader:
    """
    Unified configuration loader for SignSense training.
    
    Supports:
    - YAML config files with validation
    - CLI argument overrides
    - Dynamic sign specific configs from dynamic_signs.yaml
    - Default values for missing configurations
    
    Usage:
        config = TrainingConfigLoader.load()
        config = TrainingConfigLoader.load(model_type="static_mlp")
        config = TrainingConfigLoader.load_from_args()
    """
    
    _instance = None
    _config: Optional[Dict] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the config loader."""
        self._config = None
        self._dynamic_config = None
    
    def _get_config_path(self) -> Path:
        """Get path to training.yaml config file."""
        config_path = Path(__file__).parent / "training.yaml"
        if not config_path.exists():
            # Fallback to project root
            config_path = PROJECT_ROOT / "signsense" / "ml" / "config" / "training.yaml"
        return config_path
    
    def _get_dynamic_config_path(self) -> Path:
        """Get path to dynamic_signs.yaml config file."""
        return Path(__file__).parent.parent.parent / "config" / "dynamic_signs.yaml"
    
    def load(self, config_path: Optional[Path] = None) -> Dict:
        """
        Load the training configuration from YAML.
        
        Args:
            config_path: Optional path to config file. If None, uses default.
        
        Returns:
            Dictionary containing all configuration
        """
        if self._config is not None:
            return self._config
        
        if config_path is None:
            config_path = self._get_config_path()
        
        if not config_path.exists():
            print(f"Warning: Config file not found at {config_path}, using defaults")
            self._config = self._get_default_config()
            return self._config
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # Load dynamic sign configs
        self._load_dynamic_signs()
        
        # Apply defaults
        self._apply_defaults()
        
        return self._config
    
    def _load_dynamic_signs(self):
        """Load dynamic sign configurations from dynamic_signs.yaml."""
        dynamic_path = self._get_dynamic_config_path()
        
        if dynamic_path.exists():
            with open(dynamic_path, 'r', encoding='utf-8') as f:
                dynamic_data = yaml.safe_load(f)
            
            if dynamic_data and "dynamic_signs" in dynamic_data:
                if self._config is None:
                    self._config = {}
                self._config["dynamic_signs"] = dynamic_data["dynamic_signs"]
    
    def _apply_defaults(self):
        """Apply default values for missing configurations."""
        defaults = {
            "defaults": DefaultsConfig().__dict__,
            "data": {
                "static": DataConfig().__dict__,
                "dynamic": DynamicDataConfig().__dict__
            },
            "output": OutputConfig().__dict__
        }
        
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
    
    def _get_default_config(self) -> Dict:
        """Get default configuration when no config file exists."""
        return {
            "defaults": DefaultsConfig().__dict__,
            "data": {
                "static": DataConfig().__dict__,
                "dynamic": DynamicDataConfig().__dict__
            },
            "models": {
                "static_mlp": {
                    "name": "SignMLP",
                    "architecture": ArchitectureConfig().__dict__,
                    "training": TrainingConfig().__dict__,
                    "augmentation": AugmentationConfig().__dict__
                },
                "dynamic_lstm": {
                    "name": "DynamicSignLSTM",
                    "architecture": LSTMArchitectureConfig().__dict__,
                    "training": TrainingConfig().__dict__,
                    "sequence": SequenceConfig().__dict__
                }
            },
            "output": OutputConfig().__dict__
        }
    
    def get_model_config(self, model_type: str) -> Dict:
        """
        Get configuration for a specific model type.
        
        Args:
            model_type: "static_mlp" or "dynamic_lstm"
        
        Returns:
            Dictionary containing model configuration
        
        Raises:
            ValueError: If model_type is not supported
        """
        config = self.load()
        
        if model_type not in config.get("models", {}):
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(config.get('models', {}).keys())}"
            )
        
        return config["models"][model_type]
    
    def get_dynamic_sign_config(self, sign_name: str) -> Optional[Dict]:
        """
        Get configuration for a specific dynamic sign.
        
        Args:
            sign_name: Name of the dynamic sign (e.g., "J", "Z")
        
        Returns:
            Dictionary containing sign configuration or None
        """
        config = self.load()
        
        dynamic_signs = config.get("dynamic_signs", {})
        return dynamic_signs.get(sign_name.upper())
    
    def get_all_dynamic_signs(self) -> List[str]:
        """Get list of all configured dynamic signs."""
        config = self.load()
        dynamic_signs = config.get("dynamic_signs", {})
        return list(dynamic_signs.keys())
    
    def get_output_config(self) -> Dict:
        """Get output configuration."""
        config = self.load()
        return config.get("output", OutputConfig().__dict__)
    
    def get_defaults(self) -> Dict:
        """Get default settings."""
        config = self.load()
        return config.get("defaults", DefaultsConfig().__dict__)
    
    def get_data_config(self, data_type: str = "static") -> Dict:
        """
        Get data configuration.
        
        Args:
            data_type: "static" or "dynamic"
        
        Returns:
            Data configuration dictionary
        """
        config = self.load()
        data = config.get("data", {})
        return data.get(data_type, DataConfig().__dict__)
    
    def override_from_args(self, args: argparse.Namespace) -> Dict:
        """
        Override configuration from CLI arguments.
        
        Args:
            args: Parsed command-line arguments
        
        Returns:
            Updated configuration dictionary
        """
        config = self.load()
        
        # Override with CLI args
        if hasattr(args, 'epochs') and args.epochs is not None:
            config["models"]["static_mlp"]["training"]["epochs"] = args.epochs
        
        if hasattr(args, 'lr') and args.lr is not None:
            config["models"]["static_mlp"]["training"]["learning_rate"] = args.lr
        
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            config["models"]["static_mlp"]["training"]["batch_size"] = args.batch_size
        
        if hasattr(args, 'device') and args.device is not None:
            config["defaults"]["device"] = args.device
        
        return config
    
    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        """
        Create CLI argument parser with common training arguments.
        
        Returns:
            Configured ArgumentParser
        """
        parser = argparse.ArgumentParser(
            description="SignSense Training Configuration"
        )
        
        # Model selection
        parser.add_argument(
            "--model", 
            type=str, 
            default="static_mlp",
            choices=["static_mlp", "dynamic_lstm"],
            help="Model type to train"
        )
        
        # Training hyperparameters
        parser.add_argument("--epochs", type=int, help="Number of training epochs")
        parser.add_argument("--lr", type=float, help="Learning rate")
        parser.add_argument("--batch-size", type=int, help="Batch size")
        parser.add_argument("--weight-decay", type=float, help="Weight decay")
        
        # Device
        parser.add_argument(
            "--device", 
            type=str, 
            choices=["cpu", "cuda", "auto"],
            help="Device to use for training"
        )
        
        # Config file
        parser.add_argument(
            "--config", 
            type=str, 
            help="Path to custom config file"
        )
        
        # Data paths
        parser.add_argument(
            "--data", 
            type=str, 
            help="Path to training data"
        )
        
        # Output
        parser.add_argument(
            "--output", 
            type=str, 
            help="Path for model output"
        )
        
        # Options
        parser.add_argument(
            "--no-augmentation", 
            action="store_true",
            help="Disable data augmentation"
        )
        
        parser.add_argument(
            "--seed", 
            type=int, 
            help="Random seed"
        )
        
        return parser
    
    def reset(self):
        """Reset the config cache (useful for testing)."""
        self._config = None
        self._dynamic_config = None


# =============================================================================
# Convenience Functions
# =============================================================================

# Global config instance
_config_loader = TrainingConfigLoader()


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load the training configuration."""
    return _config_loader.load(config_path)


def get_model_config(model_type: str) -> Dict:
    """Get configuration for a specific model type."""
    return _config_loader.get_model_config(model_type)


def get_dynamic_sign_config(sign_name: str) -> Optional[Dict]:
    """Get configuration for a specific dynamic sign."""
    return _config_loader.get_dynamic_sign_config(sign_name)


def get_all_dynamic_signs() -> List[str]:
    """Get list of all configured dynamic signs."""
    return _config_loader.get_all_dynamic_signs()


def get_output_config() -> Dict:
    """Get output configuration."""
    return _config_loader.get_output_config()


def get_defaults() -> Dict:
    """Get default settings."""
    return _config_loader.get_defaults()


def get_data_config(data_type: str = "static") -> Dict:
    """Get data configuration."""
    return _config_loader.get_data_config(data_type)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    return _config_loader.create_parser()


def load_from_args(args: Optional[argparse.Namespace] = None) -> Dict:
    """
    Load configuration from args or sys.argv.
    
    Args:
        args: Parsed arguments. If None, parses sys.argv
    
    Returns:
        Configuration dictionary with CLI overrides
    """
    if args is None:
        parser = create_parser()
        args = parser.parse_args()
    
    return _config_loader.override_from_args(args)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_config(config: Dict) -> List[str]:
    """
    Validate configuration and return list of warnings.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        List of validation warning messages
    """
    warnings = []
    
    # Validate model configs
    for model_name, model_config in config.get("models", {}).items():
        # Check learning rate
        lr = model_config.get("training", {}).get("learning_rate", 0)
        if lr > 0.1:
            warnings.append(f"{model_name}: Learning rate {lr} is very high")
        elif lr < 1e-06:
            warnings.append(f"{model_name}: Learning rate {lr} is very low")
        
        # Check batch size
        bs = model_config.get("training", {}).get("batch_size", 0)
        if bs > 256:
            warnings.append(f"{model_name}: Batch size {bs} is very large")
        elif bs < 1:
            warnings.append(f"{model_name}: Batch size must be positive")
        
        # Check epochs
        epochs = model_config.get("training", {}).get("epochs", 0)
        if epochs > 1000:
            warnings.append(f"{model_name}: Epochs {epochs} is very high")
    
    # Validate data splits
    for data_type, data_config in config.get("data", {}).items():
        train_split = data_config.get("train_split", 0)
        val_split = data_config.get("validation_split", 0)
        
        if train_split + val_split > 1.0:
            warnings.append(
                f"data.{data_type}: train_split + validation_split > 1.0"
            )
    
    return warnings


def get_device() -> str:
    """
    Get the best available device (cuda/cpu).
    
    Returns:
        Device string
    """
    import torch
    
    config = _config_loader.get_defaults()
    device = config.get("device", "auto")
    
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU")
        return "cpu"
    
    return device


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
