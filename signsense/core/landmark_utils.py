"""Utility functions for landmark geometry calculations."""

import numpy as np
from typing import Tuple, List, Union
import math


def get_landmark_coordinates(
    landmark,
    frame_shape: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Convert normalized landmark coordinates to pixel coordinates.
    
    Args:
        landmark: MediaPipe landmark object with x, y, z attributes
        frame_shape: Tuple of (height, width) of the frame
        
    Returns:
        Tuple of (x, y) pixel coordinates
    """
    height, width = frame_shape[:2]
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return (x, y)


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        point1: Tuple of (x, y) coordinates
        point2: Tuple of (x, y) coordinates
        
    Returns:
        Euclidean distance as float
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.sqrt(dx * dx + dy * dy)


def calculate_angle(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float]
) -> float:
    """
    Calculate the angle at point b formed by points a, b, c.
    
    Args:
        a: First point (x, y)
        b: Vertex point (x, y)
        c: Third point (x, y)
        
    Returns:
        Angle in degrees (0-180)
    """
    # Calculate vectors
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    
    # Calculate dot product and magnitudes
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0] * ba[0] + ba[1] * ba[1])
    magnitude_bc = math.sqrt(bc[0] * bc[0] + bc[1] * bc[1])
    
    # Avoid division by zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0
    
    # Calculate angle in radians, then convert to degrees
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def normalize_landmarks(landmarks) -> np.ndarray:
    """
    Normalize landmark coordinates to a standard range.
    
    Args:
        landmarks: MediaPipe hand landmarks object
        
    Returns:
        Numpy array of normalized landmark coordinates (x, y, z)
        
    TODO: This will be used for ASL sign matching algorithms
    """
    if landmarks is None:
        return np.array([])
    
    normalized = []
    for landmark in landmarks.landmark:
        normalized.append([landmark.x, landmark.y, landmark.z])
    
    return np.array(normalized)
