"""
Tests for BaseDetector interface and DetectionResult dataclass.

Following TDD approach - these tests define the expected behavior
before implementation.
"""

import pytest
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch
from dataclasses import asdict

import sys
import os
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

# Import the classes we'll implement
from detection.base_detector import BaseDetector, DetectionResult, detector_registry


class TestDetectionResult:
    """Test cases for DetectionResult dataclass."""
    
    def test_detection_result_creation_valid(self):
        """Test creating a valid DetectionResult instance."""
        result = DetectionResult(
            metric_name="test_metric",
            expected_value=100.0,
            actual_value=150.0,
            deviation_percentage=0.5,
            severity="high",
            detection_method="threshold",
            timestamp=datetime.now(),
            alert_sent=False
        )
        
        assert result.metric_name == "test_metric"
        assert result.expected_value == 100.0
        assert result.actual_value == 150.0
        assert result.deviation_percentage == 0.5
        assert result.severity == "high"
        assert result.detection_method == "threshold"
        assert result.alert_sent is False
        assert isinstance(result.timestamp, datetime)
    
    def test_detection_result_severity_validation(self):
        """Test that only valid severity levels are accepted."""
        valid_severities = ["critical", "high", "warning", "info"]
        
        for severity in valid_severities:
            result = DetectionResult(
                metric_name="test",
                expected_value=100.0,
                actual_value=150.0,
                deviation_percentage=0.5,
                severity=severity,
                detection_method="threshold",
                timestamp=datetime.now(),
                alert_sent=False
            )
            assert result.severity == severity
    
    def test_detection_result_invalid_severity(self):
        """Test that invalid severity levels raise ValueError."""
        with pytest.raises(ValueError, match="Invalid severity level"):
            DetectionResult(
                metric_name="test",
                expected_value=100.0,
                actual_value=150.0,
                deviation_percentage=0.5,
                severity="invalid",
                detection_method="threshold",
                timestamp=datetime.now(),
                alert_sent=False
            )
    
    def test_detection_result_is_anomaly_property(self):
        """Test the is_anomaly property based on deviation."""
        # No anomaly (deviation below threshold)
        result = DetectionResult(
            metric_name="test",
            expected_value=100.0,
            actual_value=105.0,
            deviation_percentage=0.05,
            severity="info",
            detection_method="threshold",
            timestamp=datetime.now(),
            alert_sent=False
        )
        assert not result.is_anomaly
        
        # Anomaly detected
        result_anomaly = DetectionResult(
            metric_name="test",
            expected_value=100.0,
            actual_value=150.0,
            deviation_percentage=0.5,
            severity="high",
            detection_method="threshold",
            timestamp=datetime.now(),
            alert_sent=False
        )
        assert result_anomaly.is_anomaly
    
    def test_detection_result_to_dict(self):
        """Test conversion to dictionary for serialization."""
        timestamp = datetime.now()
        result = DetectionResult(
            metric_name="test_metric",
            expected_value=100.0,
            actual_value=150.0,
            deviation_percentage=0.5,
            severity="high",
            detection_method="threshold",
            timestamp=timestamp,
            alert_sent=False
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["metric_name"] == "test_metric"
        assert result_dict["expected_value"] == 100.0
        assert result_dict["actual_value"] == 150.0
        assert result_dict["deviation_percentage"] == 0.5
        assert result_dict["severity"] == "high"
        assert result_dict["detection_method"] == "threshold"
        assert result_dict["timestamp"] == timestamp.isoformat()
        assert result_dict["alert_sent"] is False


class TestBaseDetector:
    """Test cases for BaseDetector abstract base class."""
    
    def test_base_detector_is_abstract(self):
        """Test that BaseDetector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDetector(config={})
    
    def test_base_detector_subclass_implementation(self):
        """Test that concrete implementation works correctly."""
        
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return [
                    DetectionResult(
                        metric_name="test_metric",
                        expected_value=100.0,
                        actual_value=150.0,
                        deviation_percentage=0.5,
                        severity="high",
                        detection_method="test",
                        timestamp=datetime.now(),
                        alert_sent=False
                    )
                ]
            
            def validate_config(self) -> bool:
                return True
        
        config = {"threshold": 0.2, "metric": "test_metric"}
        detector = TestDetector(config)
        
        assert detector.config == config
        assert detector.name == "TestDetector"
        assert detector.validate_config() is True
        
        # Test detection method
        results = detector.detect(date.today(), date.today())
        assert len(results) == 1
        assert isinstance(results[0], DetectionResult)
    
    def test_base_detector_initialization(self):
        """Test BaseDetector initialization with config."""
        
        class MockDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
            
            def validate_config(self) -> bool:
                return True
        
        config = {"threshold": 0.3, "metric": "views"}
        detector = MockDetector(config)
        
        assert detector.config == config
        assert detector.name == "MockDetector"
        assert hasattr(detector, 'logger')
    
    def test_base_detector_timing_decorator(self):
        """Test the timing functionality for detection methods."""
        
        class TimedDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                # Simulate some work
                import time
                time.sleep(0.01)
                return []
            
            def validate_config(self) -> bool:
                return True
        
        detector = TimedDetector({})
        
        with patch.object(detector.logger, 'info') as mock_logger:
            detector.detect(date.today(), date.today())
            mock_logger.assert_called()
            
            # Check that timing information was logged
            logged_calls = [str(call) for call in mock_logger.call_args_list]
            timing_logged = any("took" in call for call in logged_calls)
            assert timing_logged
    
    def test_base_detector_error_handling(self):
        """Test error handling in BaseDetector methods."""
        
        class ErrorDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                raise ValueError("Simulated detection error")
            
            def validate_config(self) -> bool:
                return True
        
        detector = ErrorDetector({})
        
        with patch.object(detector.logger, 'error') as mock_logger:
            with pytest.raises(ValueError):
                detector.detect(date.today(), date.today())
            
            mock_logger.assert_called()
    
    def test_base_detector_required_methods(self):
        """Test that subclasses must implement required abstract methods."""
        
        # Missing detect method
        class IncompleteDetector1(BaseDetector):
            def validate_config(self) -> bool:
                return True
        
        with pytest.raises(TypeError):
            IncompleteDetector1({})
        
        # Missing validate_config method
        class IncompleteDetector2(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        with pytest.raises(TypeError):
            IncompleteDetector2({})


class TestDetectorRegistry:
    """Test cases for detector registration decorator."""
    
    def test_detector_registration(self):
        """Test that detectors can be registered and retrieved."""
        
        # Clear registry for test
        detector_registry.clear()
        
        @detector_registry.register("test_detector")
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
            
            def validate_config(self) -> bool:
                return True
        
        assert "test_detector" in detector_registry.get_registered_detectors()
        assert detector_registry.get_detector("test_detector") == TestDetector
    
    def test_detector_registration_duplicate_name(self):
        """Test that duplicate detector names raise an error."""
        
        detector_registry.clear()
        
        @detector_registry.register("duplicate")
        class FirstDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
            
            def validate_config(self) -> bool:
                return True
        
        with pytest.raises(ValueError, match="already registered"):
            @detector_registry.register("duplicate")
            class SecondDetector(BaseDetector):
                def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                    return []
                
                def validate_config(self) -> bool:
                    return True
    
    def test_detector_registry_get_nonexistent(self):
        """Test that getting a non-existent detector raises an error."""
        
        detector_registry.clear()
        
        with pytest.raises(KeyError, match="not found"):
            detector_registry.get_detector("nonexistent")
    
    def test_detector_registry_list_all(self):
        """Test listing all registered detectors."""
        
        detector_registry.clear()
        
        @detector_registry.register("detector1")
        class Detector1(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
            
            def validate_config(self) -> bool:
                return True
        
        @detector_registry.register("detector2")
        class Detector2(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
            
            def validate_config(self) -> bool:
                return True
        
        registered = detector_registry.get_registered_detectors()
        assert "detector1" in registered
        assert "detector2" in registered
        assert len(registered) == 2


class TestDetectorIntegration:
    """Integration tests for the complete detector system."""
    
    def test_end_to_end_detection_workflow(self):
        """Test complete workflow from config to results."""
        
        detector_registry.clear()
        
        @detector_registry.register("integration_test")
        class IntegrationTestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                # Simulate detection logic
                if self.config.get("simulate_anomaly", False):
                    return [
                        DetectionResult(
                            metric_name=self.config["metric"],
                            expected_value=100.0,
                            actual_value=200.0,
                            deviation_percentage=1.0,
                            severity="critical",
                            detection_method="integration_test",
                            timestamp=datetime.now(),
                            alert_sent=False
                        )
                    ]
                return []
            
            def validate_config(self) -> bool:
                required_keys = ["metric", "threshold"]
                return all(key in self.config for key in required_keys)
        
        # Test with valid config - no anomaly
        config = {"metric": "views", "threshold": 0.2, "simulate_anomaly": False}
        detector = detector_registry.get_detector("integration_test")(config)
        
        assert detector.validate_config() is True
        results = detector.detect(date.today(), date.today())
        assert len(results) == 0
        
        # Test with anomaly simulation
        config["simulate_anomaly"] = True
        detector_with_anomaly = detector_registry.get_detector("integration_test")(config)
        
        results_with_anomaly = detector_with_anomaly.detect(date.today(), date.today())
        assert len(results_with_anomaly) == 1
        assert results_with_anomaly[0].is_anomaly
        assert results_with_anomaly[0].severity == "critical"