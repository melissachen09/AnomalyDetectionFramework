"""
Detection package for Anomaly Detection Framework.

This package provides the core detection functionality including:
- Base detector interface and utilities
- Plugin manager for dynamic detector loading
- Specific detector implementations (threshold, statistical, etc.)
"""

from .plugin_manager import PluginManager
from .detectors.base_detector import BaseDetector, DetectionResult, register_detector, create_detector

__all__ = [
    'PluginManager',
    'BaseDetector', 
    'DetectionResult',
    'register_detector',
    'create_detector'
]