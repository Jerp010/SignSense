"""
Dynamic Sign Factory - Compatibility Layer

Provides a unified interface for both hardcoded dynamic sign detectors
(from dynamic_signs.py) and trained LSTM-based detectors
(from trainable_dynamic_signs.py).

This factory allows the application to switch between different detector types
while maintaining compatibility with existing code.
"""

from typing import Optional
from pathlib import Path

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from signsense.config.dynamic_config import get_config, has_config
from signsense.signs.dynamic_signs import DynamicDetector, JDetector
from signsense.signs.trainable_dynamic_signs import TrainedDynamicDetector


class DynamicSignFactory:
    """Factory for creating dynamic sign detectors with fallback behavior."""
    
    @staticmethod
    def create_detector(sign_name: str, detector_type: str = "auto") -> Optional[DynamicDetector]:
        """
        Create a dynamic sign detector for the specified sign.
        
        Args:
            sign_name: The name of the sign to create a detector for (e.g., "J", "Z")
            detector_type: Type of detector to create:
                - "auto": Try trained detector first, fallback to hardcoded
                - "trained": Only try trained detector
                - "hardcoded": Only use hardcoded detector
                
        Returns:
            DynamicDetector instance or None if no suitable detector found
        """
        sign_name = sign_name.upper()
        
        # Validate configuration
        if not has_config(sign_name):
            return None
            
        config = get_config(sign_name)
        
        if detector_type == "auto" or detector_type == "trained":
            try:
                return TrainedDynamicDetector(sign_name)
            except Exception as e:
                print(f"Failed to load trained detector for '{sign_name}': {e}")
                if detector_type == "trained":
                    return None
                    
        if detector_type == "auto" or detector_type == "hardcoded":
            try:
                return DynamicSignFactory._create_hardcoded_detector(sign_name)
            except Exception as e:
                print(f"Failed to load hardcoded detector for '{sign_name}': {e}")
                return None
                
        return None
        
    @staticmethod
    def _create_hardcoded_detector(sign_name: str) -> Optional[DynamicDetector]:
        """
        Create a hardcoded dynamic sign detector.
        
        Args:
            sign_name: The name of the sign to create a detector for
            
        Returns:
            DynamicDetector instance or None
        """
        sign_name = sign_name.upper()
        
        if sign_name == "J":
            return JDetector()
            
        # Add other hardcoded detectors here as needed
        # elif sign_name == "Z":
        #     return ZDetector()
            
        return None
        
    @staticmethod
    def has_trained_model(sign_name: str) -> bool:
        """
        Check if a trained model exists for the specified sign.
        
        Args:
            sign_name: The name of the sign to check
            
        Returns:
            True if a trained model file exists, False otherwise
        """
        sign_name = sign_name.upper()
        model_path = Path("ml/models") / f"dynamic_{sign_name}.pt"
        return model_path.exists()
        
    @staticmethod
    def detector_type_available(sign_name: str, detector_type: str) -> bool:
        """
        Check if a specific detector type is available for the sign.
        
        Args:
            sign_name: The name of the sign to check
            detector_type: Type of detector to check
            
        Returns:
            True if the detector type is available
        """
        sign_name = sign_name.upper()
        
        if detector_type == "trained":
            return DynamicSignFactory.has_trained_model(sign_name)
            
        if detector_type == "hardcoded":
            return sign_name in ["J"]  # Add other hardcoded signs here
            
        if detector_type == "auto":
            return (DynamicSignFactory.detector_type_available(sign_name, "trained") or
                   DynamicSignFactory.detector_type_available(sign_name, "hardcoded"))
                   
        return False
        
    @staticmethod
    def get_available_detector_types(sign_name: str) -> list:
        """
        Get list of available detector types for a specific sign.
        
        Args:
            sign_name: The name of the sign to check
            
        Returns:
            List of available detector types
        """
        available = []
        
        if DynamicSignFactory.detector_type_available(sign_name, "trained"):
            available.append("trained")
            
        if DynamicSignFactory.detector_type_available(sign_name, "hardcoded"):
            available.append("hardcoded")
            
        return available


# Convenience function for backward compatibility
def get_detector(sign_name: str, detector_type: str = "auto") -> Optional[DynamicDetector]:
    """
    Create a dynamic sign detector (backward compatible function).
    
    Args:
        sign_name: The name of the sign to create a detector for
        detector_type: Type of detector to create (default: "auto")
        
    Returns:
        DynamicDetector instance or None
    """
    return DynamicSignFactory.create_detector(sign_name, detector_type)
