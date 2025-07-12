"""
Anomaly Detection Framework - Detection Module

This module provides the core detection interfaces and utilities for building
anomaly detection plugins.

Main Components:
    BaseDetector: Abstract base class for all detection plugins
    DetectionResult: Standardized result format for anomaly detection
    DetectorRegistry: Plugin registry and factory for detector management
    
Usage:
    from detection.base_detector import BaseDetector, DetectionResult, detector_registry
    
    @detector_registry.register("my_detector")
    class MyDetector(BaseDetector):
        def detect(self, start_date, end_date):
            # Implementation...
            pass
        
        def validate_config(self):
            return True
"""

from .base_detector import BaseDetector, DetectionResult, detector_registry

__all__ = ["BaseDetector", "DetectionResult", "detector_registry"]
__version__ = "0.1.0"