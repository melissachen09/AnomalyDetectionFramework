"""
Detection plugins for the Anomaly Detection Framework.

This package contains detector implementations that inherit from BaseDetector
and provide specific anomaly detection capabilities.

Available detectors:
- DbtTestDetector: Integrates with dbt tests for data quality monitoring
"""

from .base import BaseDetector, DetectionResult
from .dbt_detector import DbtTestDetector

__all__ = [
    'BaseDetector',
    'DetectionResult', 
    'DbtTestDetector'
]