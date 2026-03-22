"""
Configuration manager for dynamic ASL signs.

Loads and validates dynamic sign configurations from YAML files,
providing a type-safe API for accessing detector and training parameters.
Supports both predefined complex gestures and dynamically created simple gestures.
"""

import yaml
import csv
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


@dataclass
class CSVRegistryEntry:
    """Entry from the dynamic signs CSV registry."""
    letter: str
    name: str
    gesture_type: GestureType
    description: str
    enabled: bool = True
    detector_type: str = "auto"  # auto, trained, hardcoded
    num_stages: int = 1
    min_samples: int = 10
    model_exists: bool = False
    data_path: str = ""


def load_csv_registry(csv_path: str = None) -> Dict[str, CSVRegistryEntry]:
    """
    Load dynamic signs registry from CSV file.
    
    Args:
        csv_path: Path to the CSV file. If None, uses default location.
    
    Returns:
        Dictionary mapping letter -> CSVRegistryEntry
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / "dynamic_signs.csv"
    else:
        csv_path = Path(csv_path)
    
    registry: Dict[str, CSVRegistryEntry] = {}
    
    if not csv_path.exists():
        return registry
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty rows or comment-like rows
            if not row.get('letter') or row.get('letter', '').startswith('#'):
                continue
            
            # Parse gesture type
            gesture_type = GestureType.COMPLEX
            if row.get('type', '').lower() == 'simple':
                gesture_type = GestureType.SIMPLE
            
            # Parse enabled
            enabled = row.get('enabled', 'true').lower() == 'true'
            
            # Parse model exists
            model_exists = row.get('model_exists', 'false').lower() == 'true'
            
            # Parse numeric fields
            num_stages = int(row.get('num_stages', 1))
            min_samples = int(row.get('min_samples', 10))
            
            entry = CSVRegistryEntry(
                letter=row['letter'].upper(),
                name=row.get('name', row['letter']),
                gesture_type=gesture_type,
                description=row.get('description', ''),
                enabled=enabled,
                detector_type=row.get('detector_type', 'auto'),
                num_stages=num_stages,
                min_samples=min_samples,
                model_exists=model_exists,
                data_path=row.get('data_path', '')
            )
            registry[entry.letter] = entry
    
    return registry


def save_csv_registry(registry: Dict[str, CSVRegistryEntry], csv_path: str = None):
    """
    Save dynamic signs registry to CSV file.
    
    Args:
        registry: Dictionary mapping letter -> CSVRegistryEntry
        csv_path: Path to save the CSV file. If None, uses default location.
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / "dynamic_signs.csv"
    else:
        csv_path = Path(csv_path)
    
    # Add header comment
    header_comment = """# SignSense Dynamic Signs Registry
# This CSV provides a quick overview of all dynamic signs and their status.
# Edit this file to enable/disable signs or modify basic metadata.
# For detailed configuration, see dynamic_signs.yaml
#
# Columns:
# - letter: Sign identifier (e.g., J, Z)
# - name: Display name
# - type: complex (multi-stage) or simple (single-stage)
# - description: Human-readable description
# - enabled: Whether the sign is active (true/false)
# - detector_type: auto (prefer trained, fallback to hardcoded), trained, or hardcoded
# - num_stages: Number of stages in the gesture sequence
# - min_samples: Minimum samples required for training
# - model_exists: Whether a trained model file exists (auto-updated)
# - data_path: Path to recorded training data directory
#"""
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(header_comment + "\n")
        f.write("letter,name,type,description,enabled,detector_type,num_stages,min_samples,model_exists,data_path\n")
        
        for letter, entry in sorted(registry.items()):
            f.write(f"{entry.letter},{entry.name},{entry.gesture_type.value},"
                   f"\"{entry.description}\",{str(entry.enabled).lower()},"
                   f"{entry.detector_type},{entry.num_stages},{entry.min_samples},"
                   f"{str(entry.model_exists).lower()},{entry.data_path}\n")


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


# Convenience functions for CSV registry
global_csv_registry: Dict[str, CSVRegistryEntry] = {}


def get_csv_registry() -> Dict[str, CSVRegistryEntry]:
    """Get the global CSV registry (lazy-loaded)."""
    global global_csv_registry
    if not global_csv_registry:
        global_csv_registry = load_csv_registry()
    return global_csv_registry


def reload_csv_registry() -> Dict[str, CSVRegistryEntry]:
    """Force reload the CSV registry from disk."""
    global global_csv_registry
    global_csv_registry = load_csv_registry()
    return global_csv_registry


def get_enabled_dynamic_signs() -> List[str]:
    """Get list of enabled dynamic signs from CSV registry."""
    registry = get_csv_registry()
    return [letter for letter, entry in registry.items() if entry.enabled]


def is_dynamic_sign_enabled(letter: str) -> bool:
    """Check if a dynamic sign is enabled in the CSV registry."""
    registry = get_csv_registry()
    entry = registry.get(letter.upper())
    return entry.enabled if entry else False


def get_csv_entry(letter: str) -> Optional[CSVRegistryEntry]:
    """Get CSV registry entry for a specific sign."""
    registry = get_csv_registry()
    return registry.get(letter.upper())
