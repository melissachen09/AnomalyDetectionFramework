"""
Test cases for BaseDetector interface and abstract functionality.

This module implements the test suite for ADF-30: Write Test Cases for Base Detector Interface.
Tests ensure interface compliance, abstract method enforcement, and common functionality.
"""

import pytest
from abc import ABC, abstractmethod
from datetime import datetime, date
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional

# Import will fail initially - this follows TDD approach
try:
    from src.detection.detectors.base_detector import BaseDetector, DetectionResult
except ImportError:
    BaseDetector = None
    DetectionResult = None


class TestBaseDetectorInterface:
    """Test suite for BaseDetector interface compliance."""
    
    def test_base_detector_is_abstract_class(self):
        """Test that BaseDetector cannot be instantiated directly."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDetector(config={})
    
    def test_base_detector_inherits_from_abc(self):
        """Test that BaseDetector inherits from ABC."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        assert issubclass(BaseDetector, ABC)
    
    def test_detect_method_is_abstract(self):
        """Test that detect method is abstract and must be implemented."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        # Create a concrete class that doesn't implement detect
        class IncompleteDetector(BaseDetector):
            pass
            
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDetector(config={})
    
    def test_detect_method_signature(self):
        """Test that detect method has correct signature."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        # Create a valid implementation to test signature
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        detector = TestDetector(config={})
        
        # Test method exists and is callable
        assert hasattr(detector, 'detect')
        assert callable(detector.detect)
        
        # Test method can be called with correct parameters
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 2)
        result = detector.detect(start_date, end_date)
        
        assert isinstance(result, list)


class TestBaseDetectorInitialization:
    """Test suite for BaseDetector initialization and configuration."""
    
    def test_initialization_with_valid_config(self):
        """Test successful initialization with valid configuration."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        config = {
            "event_type": "listing_views",
            "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "metrics": ["total_views"]
        }
        
        detector = TestDetector(config)
        
        assert detector.config == config
        assert hasattr(detector, 'config')
    
    def test_initialization_with_empty_config(self):
        """Test initialization with empty configuration."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        detector = TestDetector(config={})
        
        assert detector.config == {}
    
    def test_initialization_with_none_config(self):
        """Test initialization with None configuration should raise error."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        with pytest.raises((TypeError, ValueError)):
            TestDetector(config=None)
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_snowflake_connection_initialization(self, mock_get_connection):
        """Test that Snowflake connection is properly initialized."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        mock_connection = Mock()
        mock_get_connection.return_value = mock_connection
        
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        detector = TestDetector(config={})
        
        mock_get_connection.assert_called_once()
        assert detector.snowflake_conn == mock_connection


class TestDetectionResult:
    """Test suite for DetectionResult data structure."""
    
    def test_detection_result_structure(self):
        """Test DetectionResult has required fields."""
        if DetectionResult is None:
            pytest.skip("DetectionResult not implemented yet")
            
        # Test that DetectionResult can be created with required fields
        result = DetectionResult(
            event_type="listing_views",
            metric_name="total_views",
            detection_date=date(2024, 1, 1),
            expected_value=10000.0,
            actual_value=5000.0,
            deviation_percentage=0.5,
            severity="critical",
            detection_method="threshold"
        )
        
        assert result.event_type == "listing_views"
        assert result.metric_name == "total_views"
        assert result.detection_date == date(2024, 1, 1)
        assert result.expected_value == 10000.0
        assert result.actual_value == 5000.0
        assert result.deviation_percentage == 0.5
        assert result.severity == "critical"
        assert result.detection_method == "threshold"
    
    def test_detection_result_optional_fields(self):
        """Test DetectionResult with optional fields."""
        if DetectionResult is None:
            pytest.skip("DetectionResult not implemented yet")
            
        result = DetectionResult(
            event_type="listing_views",
            metric_name="total_views",
            detection_date=date(2024, 1, 1),
            expected_value=10000.0,
            actual_value=5000.0,
            deviation_percentage=0.5,
            severity="critical",
            detection_method="threshold",
            alert_sent=True,
            details={"threshold_min": 8000, "threshold_max": 50000}
        )
        
        assert result.alert_sent is True
        assert result.details == {"threshold_min": 8000, "threshold_max": 50000}


class TestBaseDetectorValidation:
    """Test suite for input validation and error handling."""
    
    def test_invalid_date_range(self):
        """Test handling of invalid date ranges."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                self.validate_date_range(start_date, end_date)
                return []
        
        detector = TestDetector(config={})
        
        # Test invalid date range
        start_date = date(2024, 1, 2)
        end_date = date(2024, 1, 1)
        
        with pytest.raises(ValueError, match="start_date must be before or equal to end_date"):
            detector.detect(start_date, end_date)
    
    def test_none_date_parameters(self):
        """Test handling of None date parameters."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                self.validate_date_range(start_date, end_date)
                return []
        
        detector = TestDetector(config={})
        
        with pytest.raises(ValueError, match="Date parameters cannot be None"):
            detector.detect(None, date(2024, 1, 1))
        
        with pytest.raises(ValueError, match="Date parameters cannot be None"):
            detector.detect(date(2024, 1, 1), None)
    
    def test_equal_dates_are_valid(self):
        """Test that equal start and end dates are valid."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                self.validate_date_range(start_date, end_date)
                return []
        
        detector = TestDetector(config={})
        
        # Test equal dates (should be valid)
        test_date = date(2024, 1, 1)
        result = detector.detect(test_date, test_date)
        assert result == []
    
    def test_invalid_config_type(self):
        """Test initialization with invalid config types."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        with pytest.raises(TypeError, match="Configuration must be a dictionary"):
            TestDetector(config="not a dict")
        
        with pytest.raises(TypeError, match="Configuration must be a dictionary"):
            TestDetector(config=123)


class TestBaseDetectorCommonFunctionality:
    """Test suite for common functionality provided by BaseDetector."""
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_snowflake_connection_property(self, mock_get_connection):
        """Test that snowflake_conn property is accessible."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        mock_connection = Mock()
        mock_get_connection.return_value = mock_connection
        
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        detector = TestDetector(config={})
        
        assert hasattr(detector, 'snowflake_conn')
        assert detector.snowflake_conn == mock_connection
    
    def test_config_property_immutability(self):
        """Test that config property maintains reference to original config."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        original_config = {"event_type": "test"}
        detector = TestDetector(config=original_config)
        
        # Config should reference the original
        assert detector.config is original_config
        
        # Modifying original should be visible in detector
        original_config["new_key"] = "new_value"
        assert "new_key" in detector.config
    
    def test_get_config_value_with_default(self):
        """Test get_config_value method with defaults."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        config = {"existing_key": "existing_value"}
        detector = TestDetector(config=config)
        
        # Test existing key
        assert detector.get_config_value("existing_key") == "existing_value"
        
        # Test non-existing key with default
        assert detector.get_config_value("missing_key", "default") == "default"
        
        # Test non-existing key without default
        assert detector.get_config_value("missing_key") is None
    
    def test_detector_repr(self):
        """Test string representation of detector."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        config = {"key1": "value1", "key2": "value2"}
        detector = TestDetector(config=config)
        
        repr_str = repr(detector)
        assert "TestDetector" in repr_str
        assert "key1" in repr_str
        assert "key2" in repr_str
    
    def test_logger_initialization(self):
        """Test that logger is properly initialized."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class TestDetector(BaseDetector):
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                return []
        
        detector = TestDetector(config={})
        
        assert hasattr(detector, 'logger')
        assert detector.logger.name.endswith('TestDetector')


class TestPluginContractDocumentation:
    """Test suite to validate plugin contract documentation requirements."""
    
    def test_base_detector_docstring_exists(self):
        """Test that BaseDetector has comprehensive docstring."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        assert BaseDetector.__doc__ is not None
        assert len(BaseDetector.__doc__.strip()) > 50  # Ensure substantial documentation
    
    def test_detect_method_docstring_exists(self):
        """Test that detect method has docstring."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        # Get the abstract method from the class
        assert hasattr(BaseDetector, 'detect')
        detect_method = getattr(BaseDetector, 'detect')
        assert detect_method.__doc__ is not None
        assert len(detect_method.__doc__.strip()) > 20
    
    def test_detection_result_docstring_exists(self):
        """Test that DetectionResult has docstring."""
        if DetectionResult is None:
            pytest.skip("DetectionResult not implemented yet")
            
        assert DetectionResult.__doc__ is not None
        assert len(DetectionResult.__doc__.strip()) > 20


class TestConcreteteDetectorImplementation:
    """Test suite for concrete detector implementation example."""
    
    def test_concrete_detector_example(self):
        """Test a complete concrete detector implementation."""
        if BaseDetector is None:
            pytest.skip("BaseDetector not implemented yet")
            
        class ExampleDetector(BaseDetector):
            """Example concrete detector for testing."""
            
            def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
                """Detect anomalies in test data."""
                # Mock detection logic
                if self.config.get("simulate_anomaly", False):
                    return [DetectionResult(
                        event_type="test_event",
                        metric_name="test_metric",
                        detection_date=start_date,
                        expected_value=100.0,
                        actual_value=50.0,
                        deviation_percentage=0.5,
                        severity="high",
                        detection_method="test"
                    )]
                return []
        
        # Test normal operation
        detector = ExampleDetector(config={})
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 2))
        assert results == []
        
        # Test with anomaly simulation
        detector_with_anomaly = ExampleDetector(config={"simulate_anomaly": True})
        results = detector_with_anomaly.detect(date(2024, 1, 1), date(2024, 1, 2))
        assert len(results) == 1
        assert results[0].severity == "high"
        assert results[0].deviation_percentage == 0.5


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])