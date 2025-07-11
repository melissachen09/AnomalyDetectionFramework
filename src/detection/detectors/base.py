"""
Base detector interface for the Anomaly Detection Framework.

This module defines the base detector class that all specific detectors
should inherit from. It provides common functionality and ensures
consistent interface across all detector implementations.

Part of Epic ADF-3: Detection Plugin Architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


class DetectionResult:
    """Container for detection results."""
    
    def __init__(
        self,
        detector_name: str,
        test_id: str,
        status: str,
        message: str,
        execution_time: float = 0.0,
        failures: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.detector_name = detector_name
        self.test_id = test_id
        self.status = status
        self.message = message
        self.execution_time = execution_time
        self.failures = failures
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'detector_name': self.detector_name,
            'test_id': self.test_id,
            'status': self.status,
            'message': self.message,
            'execution_time': self.execution_time,
            'failures': self.failures,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class BaseDetector(ABC):
    """
    Base class for all anomaly detectors.
    
    Provides common functionality and interface that all detectors must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detector with configuration.
        
        Args:
            config: Dictionary containing detector configuration
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.timeout = config.get('timeout', 300)  # Default 5 minutes
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
    @abstractmethod
    def detect(self, **kwargs) -> List[DetectionResult]:
        """
        Run detection and return results.
        
        Returns:
            List of DetectionResult objects
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate detector configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        return True
    
    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about this detector.
        
        Returns:
            Dictionary with detector metadata
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'timeout': self.timeout,
            'config': self.config
        }