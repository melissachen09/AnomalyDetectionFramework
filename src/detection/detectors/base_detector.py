"""
Base Detector Interface for Anomaly Detection Framework.

This module provides the abstract base class for all detection plugins in the anomaly
detection system. It defines the contract that all detectors must implement and provides
common functionality for Snowflake connections and configuration management.

The BaseDetector follows the design patterns outlined in the anomaly detection framework
design document, providing a simple plugin architecture for detection logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Any, Optional, Type, Callable
import logging

# TODO: Implement actual Snowflake connection function
def get_snowflake_connection():
    """
    Get a Snowflake database connection.
    
    This is a placeholder function that should be implemented to return
    a proper Snowflake connection object based on configuration.
    
    Returns:
        Mock connection object for testing purposes.
    """
    # For now, return a mock connection for testing
    class MockConnection:
        def execute(self, query: str):
            return []
        
        def fetchall(self):
            return []
    
    return MockConnection()


# Registry for detector plugins - will be typed properly after BaseDetector definition
_DETECTOR_REGISTRY: Dict[str, Type['BaseDetector']] = {}


def register_detector(name: str) -> Callable[[Type['BaseDetector']], Type['BaseDetector']]:
    """
    Decorator to register a detector plugin in the detection framework.
    
    This decorator allows detector implementations to be automatically discovered
    and registered for use by the framework. The detector will be available
    by the specified name for configuration-based instantiation.
    
    Args:
        name: Unique name to register the detector under. Should match
              the detection_method used in configuration files.
    
    Returns:
        Decorator function that registers the detector class.
    
    Raises:
        ValueError: If a detector with the same name is already registered.
        TypeError: If the decorated class is not a BaseDetector subclass.
    
    Example:
        @register_detector("threshold")
        class ThresholdDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                # Implementation here
                return []
    """
    def decorator(detector_class: Type['BaseDetector']) -> Type['BaseDetector']:
        if not issubclass(detector_class, BaseDetector):
            raise TypeError(f"Detector {detector_class.__name__} must inherit from BaseDetector")
        
        if name in _DETECTOR_REGISTRY:
            raise ValueError(f"Detector with name '{name}' is already registered")
        
        _DETECTOR_REGISTRY[name] = detector_class
        return detector_class
    
    return decorator


def get_registered_detectors() -> Dict[str, Type['BaseDetector']]:
    """
    Get all registered detector classes.
    
    Returns:
        Dictionary mapping detector names to their classes.
    """
    return _DETECTOR_REGISTRY.copy()


def create_detector(name: str, config: Dict[str, Any]) -> 'BaseDetector':
    """
    Create a detector instance by name.
    
    Args:
        name: Name of the registered detector to create
        config: Configuration dictionary for the detector
    
    Returns:
        Instantiated detector object
    
    Raises:
        ValueError: If no detector is registered with the given name
    """
    if name not in _DETECTOR_REGISTRY:
        available = list(_DETECTOR_REGISTRY.keys())
        raise ValueError(f"No detector registered with name '{name}'. Available: {available}")
    
    detector_class = _DETECTOR_REGISTRY[name]
    return detector_class(config)


@dataclass
class DetectionResult:
    """
    Data structure representing the result of an anomaly detection operation.
    
    This class encapsulates all information about a detected anomaly, including
    the event details, metric values, deviation calculations, and metadata
    required for alerting and reporting.
    
    Attributes:
        event_type: Type of event being monitored (e.g., 'listing_views')
        metric_name: Name of the specific metric that triggered the anomaly
        detection_date: Date when the anomaly was detected
        expected_value: Expected value for the metric based on historical data
        actual_value: Actual observed value that triggered the detection
        deviation_percentage: Percentage deviation from expected value (0.0-1.0)
        severity: Severity level ('critical', 'high', 'warning')
        detection_method: Method used for detection ('threshold', 'statistical', etc.)
        alert_sent: Whether an alert has been sent for this anomaly
        details: Additional metadata about the detection
    """
    event_type: str
    metric_name: str
    detection_date: date
    expected_value: float
    actual_value: float
    deviation_percentage: float
    severity: str
    detection_method: str
    alert_sent: Optional[bool] = False
    details: Optional[Dict[str, Any]] = None


class BaseDetector(ABC):
    """
    Abstract base class for all anomaly detection plugins.
    
    This class defines the contract that all detection plugins must implement,
    providing common functionality for configuration management and database
    connections while requiring subclasses to implement the core detection logic.
    
    The BaseDetector follows the Template Method pattern, where the framework
    handles common operations (initialization, connection management) while
    delegating the specific detection algorithm to concrete implementations.
    
    Example:
        class ThresholdDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                # Implement threshold-based detection logic
                return detection_results
    
    Attributes:
        config: Dictionary containing configuration parameters for the detector
        snowflake_conn: Database connection object for data access
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detector with configuration.
        
        Args:
            config: Dictionary containing detector configuration parameters.
                   Cannot be None. Common configuration keys include:
                   - event_type: Type of events to monitor
                   - table: Source table for data
                   - metrics: List of metrics to analyze
                   - thresholds: Detection thresholds
        
        Raises:
            ValueError: If config is None
            TypeError: If config is not a dictionary
        """
        if config is None:
            raise ValueError("Configuration cannot be None")
        
        if not isinstance(config, dict):
            raise TypeError("Configuration must be a dictionary")
        
        self.config = config
        self.snowflake_conn = get_snowflake_connection()
        
        # Initialize logging for the detector
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @abstractmethod
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        """
        Run anomaly detection for the specified date range.
        
        This is the core method that must be implemented by all concrete detector
        classes. It should analyze data for the given date range and return any
        detected anomalies as DetectionResult objects.
        
        Args:
            start_date: Start date for detection analysis (inclusive)
            end_date: End date for detection analysis (inclusive)
        
        Returns:
            List of DetectionResult objects representing detected anomalies.
            Empty list if no anomalies are found.
        
        Raises:
            ValueError: If start_date is after end_date or dates are None
            NotImplementedError: If the method is not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement the detect method")
    
    def validate_date_range(self, start_date: date, end_date: date) -> None:
        """
        Validate that the provided date range is valid.
        
        Args:
            start_date: Start date to validate
            end_date: End date to validate
        
        Raises:
            ValueError: If dates are invalid or start_date > end_date
        """
        if start_date is None or end_date is None:
            raise ValueError("Date parameters cannot be None")
        
        if start_date > end_date:
            raise ValueError("start_date must be before or equal to end_date")
    
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
    
    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"{self.__class__.__name__}(config_keys={list(self.config.keys())})"