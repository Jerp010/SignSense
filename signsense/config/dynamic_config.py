"""
Configuration manager for dynamic ASL signs.

Loads and validates dynamic sign configurations from YAML files,
providing a type-safe API for accessing detector and training parameters.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


@dataclass
class DetectorConfig:
    """Configuration for dynamic sign detectors."""
    i_hold_frames: int = 6
    down_threshold: float = 0.06
    finish_hold_frames: int = 8
    phase_timeout: int = 70
    diagonal_threshold: float = 0.12
    horizontal_threshold: float = 0.08


@dataclass
class TrainingConfig:
    """Configuration for dynamic sign training."""
    sequence_length: int = 120
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 0.001
    hidden_size: int = 128


@dataclass
class SignConfig:
    """Complete configuration for a specific dynamic sign."""
    name: str
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stages: Dict[int, str] = field(default_factory=dict)

    @property
    def num_stages(self) -> int:
        return len(self.stages)


class DynamicSignConfig:
    """Main configuration manager for dynamic signs."""
    
    _instance = None
    _config: Dict[str, SignConfig] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from dynamic_signs.yaml."""
        config_path = Path(__file__).parent / "dynamic_signs.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Dynamic sign config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        dynamic_signs = yaml_data.get("dynamic_signs", {})
        
        for sign_name, config_data in dynamic_signs.items():
            # Create detector config
            detector_config = DetectorConfig(**config_data.get("detector", {}))
            
            # Create training config
            training_config = TrainingConfig(**config_data.get("training", {}))
            
            # Get stage descriptions
            stages = config_data.get("stages", {})
            
            # Convert string keys to integers
            stages_int = {int(k): v for k, v in stages.items()}
            
            # Create and store sign config
            self._config[sign_name.upper()] = SignConfig(
                name=sign_name.upper(),
                detector=detector_config,
                training=training_config,
                stages=stages_int
            )
    
    def get_config(self, sign_name: str) -> Optional[SignConfig]:
        """Get configuration for a specific sign."""
        return self._config.get(sign_name.upper())
    
    def has_config(self, sign_name: str) -> bool:
        """Check if configuration exists for a specific sign."""
        return sign_name.upper() in self._config
    
    def get_all_sign_names(self) -> List[str]:
        """Get list of all configured dynamic signs."""
        return list(self._config.keys())
    
    def get_sign_configs(self) -> Dict[str, SignConfig]:
        """Get all sign configurations as a dictionary."""
        return self._config.copy()


# Global config instance
_config = DynamicSignConfig()


def get_config(sign_name: str) -> Optional[SignConfig]:
    """Get configuration for a specific dynamic sign (convenience function)."""
    return _config.get_config(sign_name)


def has_config(sign_name: str) -> bool:
    """Check if configuration exists for a specific dynamic sign (convenience function)."""
    return _config.has_config(sign_name)


def get_all_sign_names() -> List[str]:
    """Get list of all configured dynamic signs (convenience function)."""
    return _config.get_all_sign_names()


def get_sign_configs() -> Dict[str, SignConfig]:
    """Get all sign configurations as a dictionary (convenience function)."""
    return _config.get_sign_configs()
