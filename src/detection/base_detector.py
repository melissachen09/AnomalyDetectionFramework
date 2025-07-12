"""
Base Detector Interface and Detection Result Classes

This module provides the abstract base class for all anomaly detectors
and the dataclass for representing detection results.

Classes:
    DetectionResult: Dataclass representing the result of an anomaly detection
    BaseDetector: Abstract base class for all detection plugins
    DetectorRegistry: Registry for managing detector plugins
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Type, Callable
import logging
import time
import functools


@dataclass
class DetectionResult:
    """
    Represents the result of an anomaly detection operation.
    
    This dataclass encapsulates all the information about a detected anomaly,
    including the metric values, deviation amount, severity level, and metadata.
    
    Attributes:
        metric_name: Name of the metric that was analyzed
        expected_value: The expected/baseline value for the metric
        actual_value: The actual observed value
        deviation_percentage: Percentage deviation from expected (0.0 to 1.0+)
        severity: Severity level ("critical", "high", "warning", "info")
        detection_method: Name of the detection method used
        timestamp: When the detection was performed
        alert_sent: Whether an alert has been sent for this result
        metadata: Optional additional information about the detection
    """
    
    metric_name: str
    expected_value: float
    actual_value: float
    deviation_percentage: float
    severity: str
    detection_method: str
    timestamp: datetime
    alert_sent: bool
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the DetectionResult after initialization."""
        valid_severities = {"critical", "high", "warning", "info"}
        if self.severity not in valid_severities:
            raise ValueError(
                f"Invalid severity level '{self.severity}'. "
                f"Must be one of: {', '.join(sorted(valid_severities))}"
            )
        
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_anomaly(self) -> bool:
        """
        Determine if this result represents an actual anomaly.
        
        Returns:
            True if the deviation percentage is above the anomaly threshold (0.1 = 10%)
        """
        return self.deviation_percentage > 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DetectionResult to a dictionary for serialization.
        
        Returns:
            Dictionary representation with timestamp converted to ISO format
        """
        result_dict = asdict(self)
        result_dict["timestamp"] = self.timestamp.isoformat()
        return result_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResult':
        """
        Create a DetectionResult from a dictionary.
        
        Args:
            data: Dictionary containing detection result data
            
        Returns:
            DetectionResult instance
        """
        # Convert ISO timestamp back to datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        return cls(**data)


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to time function execution and log the results.
    
    Args:
        func: Function to be timed
        
    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            self.logger.info(
                f"{func.__name__} execution took {execution_time:.3f} seconds"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                f"{func.__name__} failed after {execution_time:.3f} seconds: {str(e)}"
            )
            raise
    return wrapper


class BaseDetector(ABC):
    """
    Abstract base class for all anomaly detection plugins.
    
    This class provides the common interface and utility methods that all
    detection plugins must implement. It handles configuration management,
    logging setup, and provides timing decorators for performance monitoring.
    
    Attributes:
        config: Configuration dictionary for the detector
        name: Name of the detector class
        logger: Configured logger instance for the detector
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base detector with configuration.
        
        Args:
            config: Configuration dictionary containing detector parameters
        """
        self.config = config
        self.name = self.__class__.__name__
        self.logger = self._setup_logger()
        
        # Validate configuration on initialization
        if not self.validate_config():
            raise ValueError(f"Invalid configuration for {self.name}")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up a logger for this detector instance.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"anomaly_detection.{self.name}")
        
        # Only add handler if none exists (prevent duplicate handlers)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    @abstractmethod
    @timing_decorator
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        """
        Perform anomaly detection for the specified date range.
        
        This is the main detection method that must be implemented by all
        concrete detector classes.
        
        Args:
            start_date: Start date for detection analysis
            end_date: End date for detection analysis
            
        Returns:
            List of DetectionResult objects representing detected anomalies
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the detector configuration.
        
        This method should check that all required configuration parameters
        are present and have valid values.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with optional default.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def log_detection_summary(self, results: List[DetectionResult]) -> None:
        """
        Log a summary of detection results.
        
        Args:
            results: List of detection results to summarize
        """
        if not results:
            self.logger.info("No anomalies detected")
            return
        
        severity_counts = {}
        for result in results:
            severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1
        
        self.logger.info(
            f"Detected {len(results)} anomalies: {severity_counts}"
        )


class DetectorRegistry:
    """
    Registry for managing detector plugins.
    
    This class provides a way to register detector classes and retrieve them
    by name. It acts as a factory for creating detector instances.
    """
    
    def __init__(self):
        """Initialize an empty detector registry."""
        self._detectors: Dict[str, Type[BaseDetector]] = {}
    
    def register(self, name: str) -> Callable:
        """
        Decorator to register a detector class.
        
        Args:
            name: Name to register the detector under
            
        Returns:
            Decorator function that registers the class
            
        Example:
            @detector_registry.register("threshold")
            class ThresholdDetector(BaseDetector):
                pass
        """
        def decorator(detector_class: Type[BaseDetector]) -> Type[BaseDetector]:
            if name in self._detectors:
                raise ValueError(f"Detector '{name}' is already registered")
            
            if not issubclass(detector_class, BaseDetector):
                raise TypeError(f"Detector class must inherit from BaseDetector")
            
            self._detectors[name] = detector_class
            return detector_class
        
        return decorator
    
    def get_detector(self, name: str) -> Type[BaseDetector]:
        """
        Get a detector class by name.
        
        Args:
            name: Name of the detector to retrieve
            
        Returns:
            Detector class
            
        Raises:
            KeyError: If detector name is not found
        """
        if name not in self._detectors:
            raise KeyError(f"Detector '{name}' not found in registry")
        
        return self._detectors[name]
    
    def get_registered_detectors(self) -> List[str]:
        """
        Get a list of all registered detector names.
        
        Returns:
            List of registered detector names
        """
        return list(self._detectors.keys())
    
    def clear(self) -> None:
        """Clear all registered detectors (mainly for testing)."""
        self._detectors.clear()
    
    def create_detector(self, name: str, config: Dict[str, Any]) -> BaseDetector:
        """
        Create a detector instance with the given configuration.
        
        Args:
            name: Name of the detector to create
            config: Configuration for the detector
            
        Returns:
            Configured detector instance
        """
        detector_class = self.get_detector(name)
        return detector_class(config)


# Global detector registry instance
detector_registry = DetectorRegistry()