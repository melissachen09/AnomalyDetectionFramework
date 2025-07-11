"""
Detection detectors package.

This package contains all detector implementations for the anomaly detection framework.
Importing this package will automatically register all available detectors.
"""

# Import all detector implementations to trigger registration
from .base_detector import BaseDetector, DetectionResult, get_registered_detectors, create_detector
from .threshold_detector import ThresholdDetector
from .statistical_detector import StatisticalDetector

__all__ = [
    'BaseDetector',
    'DetectionResult', 
    'ThresholdDetector',
    'StatisticalDetector',
    'get_registered_detectors',
    'create_detector'
]