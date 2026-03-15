"""
Configuration manager for dynamic ASL signs.

Loads and validates dynamic sign configurations from YAML files,
providing a type-safe API for accessing detector and training parameters.
Supports both predefined complex gestures and dynamically created simple gestures.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class GestureType(Enum):
    """Type of dynamic gesture."""
    SIMPLE = "simple"  # Single-stage gesture (e.g., wave, hello)
    COMPLEX = "complex"  # Multi-stage gesture (e.g., letters J, Z)


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
    gesture_type: GestureType = GestureType.COMPLEX
    description: str = ""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stages: Dict[int, str] = field(default_factory=dict)
    min_samples: int = 10  # For simple gestures

    @property
    def num_stages(self) -> int:
        if self.gesture_type == GestureType.SIMPLE:
            return 1
        return len(self.stages)
    
    @property
    def is_simple(self) -> bool:
        return self.gesture_type == GestureType.SIMPLE
    
    @property
    def is_complex(self) -> bool:
        return self.gesture_type == GestureType.COMPLEX


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
            # Parse gesture type
            gesture_type_str = config_data.get("type", "complex")
            gesture_type = GestureType.COMPLEX
            if gesture_type_str.lower() == "simple":
                gesture_type = GestureType.SIMPLE
            
            # Create detector config
            detector_config = DetectorConfig(**config_data.get("detector", {}))
            
            # Create training config
            training_config = TrainingConfig(**config_data.get("training", {}))
            
            # Get stage descriptions
            stages = config_data.get("stages", {})
            
            # Convert string keys to integers
            stages_int = {int(k): v for k, v in stages.items()}
            
            # Get description
            description = config_data.get("description", f"Gesture {sign_name}")
            
            # Get min_samples for simple gestures
            min_samples = config_data.get("min_samples", 10)
            
            # Create and store sign config
            self._config[sign_name.upper()] = SignConfig(
                name=sign_name.upper(),
                gesture_type=gesture_type,
                description=description,
                detector=detector_config,
                training=training_config,
                stages=stages_int,
                min_samples=min_samples
            )
    
    def _save_config(self):
        """Save current configuration to YAML file."""
        config_path = Path(__file__).parent / "dynamic_signs.yaml"
        
        # Build YAML structure
        yaml_data = {"dynamic_signs": {}}
        
        for sign_name, config in self._config.items():
            sign_data = {
                "type": config.gesture_type.value,
                "description": config.description,
                "detector": {
                    "i_hold_frames": config.detector.i_hold_frames,
                    "down_threshold": config.detector.down_threshold,
                    "finish_hold_frames": config.detector.finish_hold_frames,
                    "phase_timeout": config.detector.phase_timeout,
                    "diagonal_threshold": config.detector.diagonal_threshold,
                    "horizontal_threshold": config.detector.horizontal_threshold,
                },
                "training": {
                    "sequence_length": config.training.sequence_length,
                    "batch_size": config.training.batch_size,
                    "epochs": config.training.epochs,
                    "learning_rate": config.training.learning_rate,
                    "hidden_size": config.training.hidden_size,
                },
                "stages": {str(k): v for k, v in config.stages.items()},
            }
            
            if config.gesture_type == GestureType.SIMPLE:
                sign_data["min_samples"] = config.min_samples
            
            yaml_data["dynamic_signs"][sign_name] = sign_data
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    
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
    
    def get_simple_gestures(self) -> List[SignConfig]:
        """Get all simple gesture configurations."""
        return [c for c in self._config.values() if c.gesture_type == GestureType.SIMPLE]
    
    def get_complex_gestures(self) -> List[SignConfig]:
        """Get all complex gesture configurations."""
        return [c for c in self._config.values() if c.gesture_type == GestureType.COMPLEX]
    
    def add_gesture(self, sign_name: str, gesture_type: GestureType = GestureType.SIMPLE,
                    description: str = "", stages: Dict[int, str] = None) -> SignConfig:
        """
        Add a new gesture configuration.
        
        Args:
            sign_name: Name of the gesture
            gesture_type: Type of gesture (SIMPLE or COMPLEX)
            description: Description of the gesture
            stages: Stage descriptions for complex gestures
        
        Returns:
            The created SignConfig
        """
        sign_name = sign_name.upper()
        
        if stages is None:
            stages = {0: "Single stage"} if gesture_type == GestureType.SIMPLE else {}
        
        config = SignConfig(
            name=sign_name,
            gesture_type=gesture_type,
            description=description or f"Custom gesture: {sign_name}",
            stages=stages
        )
        
        self._config[sign_name] = config
        self._save_config()
        
        return config
    
    def update_gesture(self, sign_name: str, **kwargs):
        """
        Update an existing gesture configuration.
        
        Args:
            sign_name: Name of the gesture to update
            **kwargs: Fields to update
        """
        sign_name = sign_name.upper()
        
        if sign_name not in self._config:
            raise ValueError(f"Gesture '{sign_name}' not found")
        
        config = self._config[sign_name]
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self._save_config()
    
    def remove_gesture(self, sign_name: str):
        """
        Remove a gesture configuration.
        
        Args:
            sign_name: Name of the gesture to remove
        """
        sign_name = sign_name.upper()
        
        if sign_name in self._config:
            del self._config[sign_name]
            self._save_config()
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance (useful for testing)."""
        if cls._instance is not None:
            cls._instance._config = {}
            cls._instance = None


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


def get_simple_gestures() -> List[SignConfig]:
    """Get all simple gesture configurations (convenience function)."""
    return _config.get_simple_gestures()


def get_complex_gestures() -> List[SignConfig]:
    """Get all complex gesture configurations (convenience function)."""
    return _config.get_complex_gestures()


def add_gesture(sign_name: str, gesture_type: GestureType = GestureType.SIMPLE,
                description: str = "", stages: Dict[int, str] = None) -> SignConfig:
    """Add a new gesture configuration (convenience function)."""
    return _config.add_gesture(sign_name, gesture_type, description, stages)


def create_default_config(gesture_name: str, gesture_type: GestureType = GestureType.SIMPLE,
                          description: str = "") -> SignConfig:
    """
    Create a default configuration for a new gesture.
    
    This function creates a default config if one doesn't exist,
    allowing dynamic recording of custom gestures.
    
    Args:
        gesture_name: Name of the gesture
        gesture_type: Type of gesture (SIMPLE or COMPLEX)
        description: Description of the gesture
    
    Returns:
        The SignConfig (existing or newly created)
    """
    gesture_name = gesture_name.upper()
    
    # Return existing config if it exists
    existing = _config.get_config(gesture_name)
    if existing:
        return existing
    
    # Create default config for new gesture
    if gesture_type == GestureType.SIMPLE:
        return _config.add_gesture(
            gesture_name,
            GestureType.SIMPLE,
            description or f"Custom simple gesture: {gesture_name}",
            {0: "Single stage gesture"}
        )
    else:
        return _config.add_gesture(
            gesture_name,
            GestureType.COMPLEX,
            description or f"Custom complex gesture: {gesture_name}",
            {0: "Stage 1", 1: "Stage 2", 2: "Stage 3"}
        )
