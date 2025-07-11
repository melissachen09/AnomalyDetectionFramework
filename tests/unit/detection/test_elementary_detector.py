"""
Test cases for Elementary Detector Integration.

This module implements the test suite for ADF-38: Write Test Cases for Elementary Integration.
Tests Elementary data quality check integration with API communication, result parsing,
error scenarios, and performance validation as specified in GADF-DETECT-009.

Sub-tasks covered:
- GADF-DETECT-009a: Mock Elementary API responses
- GADF-DETECT-009b: Test result transformation logic
- GADF-DETECT-009c: Verify error handling and retries
- GADF-DETECT-009d: Test configuration mapping
"""

import pytest
from datetime import date, datetime
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Any, Optional
import json
import time
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

# Import will fail initially - this follows TDD approach
try:
    from src.detection.detectors.elementary_detector import ElementaryDetector
    from src.detection.detectors.base_detector import BaseDetector, DetectionResult
except ImportError:
    ElementaryDetector = None
    BaseDetector = None
    DetectionResult = None


class TestElementaryDetectorInterface:
    """Test suite for Elementary detector interface compliance."""
    
    def test_elementary_detector_inherits_from_base_detector(self):
        """Test that ElementaryDetector inherits from BaseDetector."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        assert issubclass(ElementaryDetector, BaseDetector)
    
    def test_elementary_detector_implements_detect_method(self):
        """Test that ElementaryDetector implements required detect method."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        detector = ElementaryDetector(config={
            "elementary_endpoint": "http://localhost:8080",
            "event_type": "test_event"
        })
        
        assert hasattr(detector, 'detect')
        assert callable(detector.detect)
    
    def test_elementary_detector_initialization_with_config(self):
        """Test ElementaryDetector initialization with proper configuration."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        config = {
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_api_key",
            "event_type": "listing_views",
            "table_name": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "timeout": 30,
            "retry_attempts": 3
        }
        
        detector = ElementaryDetector(config)
        
        assert detector.config == config
        assert detector.get_config_value("elementary_endpoint") == "http://localhost:8080"
        assert detector.get_config_value("api_key") == "test_api_key"
        assert detector.get_config_value("event_type") == "listing_views"
    
    def test_elementary_detector_initialization_with_minimal_config(self):
        """Test ElementaryDetector initialization with minimal configuration."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        minimal_config = {
            "elementary_endpoint": "http://localhost:8080"
        }
        
        detector = ElementaryDetector(minimal_config)
        
        # Should use default values for optional settings
        assert detector.get_config_value("timeout", 30) == 30
        assert detector.get_config_value("retry_attempts", 3) == 3
    
    def test_elementary_detector_initialization_missing_endpoint(self):
        """Test ElementaryDetector initialization fails without elementary_endpoint."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        config = {
            "event_type": "listing_views"
        }
        
        with pytest.raises(ValueError, match="elementary_endpoint.*required"):
            ElementaryDetector(config)


class TestElementaryAPIResponseMocking:
    """Test suite for mocking Elementary API responses - GADF-DETECT-009a."""
    
    @pytest.fixture
    def sample_elementary_response(self):
        """Sample Elementary API response data."""
        return {
            "status": "success",
            "results": [
                {
                    "test_id": "test_001",
                    "table_name": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
                    "column_name": "NUMBEROFVIEWS",
                    "test_type": "anomaly_detection",
                    "status": "failed",
                    "detected_at": "2024-01-15T10:30:00Z",
                    "expected_value": 10000.0,
                    "actual_value": 5000.0,
                    "deviation": 0.5,
                    "severity": "high",
                    "description": "Significant drop in listing views detected"
                },
                {
                    "test_id": "test_002",
                    "table_name": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
                    "column_name": "NUMBEROFENQUIRIES",
                    "test_type": "threshold_check",
                    "status": "passed",
                    "detected_at": "2024-01-15T10:30:00Z",
                    "expected_value": 1000.0,
                    "actual_value": 1100.0,
                    "deviation": 0.1,
                    "severity": "low",
                    "description": "Enquiries within normal range"
                }
            ],
            "metadata": {
                "total_tests": 2,
                "failed_tests": 1,
                "execution_time": "2.5s",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    
    @pytest.fixture
    def sample_elementary_error_response(self):
        """Sample Elementary API error response."""
        return {
            "status": "error",
            "error_code": "INVALID_REQUEST",
            "message": "Invalid table name provided",
            "details": {
                "table_name": "INVALID_TABLE",
                "valid_tables": ["DATAMART.DD_LISTING_STATISTICS_BLENDED"]
            }
        }
    
    @patch('requests.get')
    def test_successful_api_response_mock(self, mock_get, sample_elementary_response):
        """Test successful Elementary API response mocking."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_elementary_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key"
        })
        
        # Execute API call
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        results = detector.detect(start_date, end_date)
        
        # Verify API was called
        mock_get.assert_called_once()
        
        # Verify results structure
        assert isinstance(results, list)
    
    @patch('requests.get')
    def test_api_error_response_mock(self, mock_get, sample_elementary_error_response):
        """Test Elementary API error response mocking."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup mock error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = sample_elementary_error_response
        mock_response.raise_for_status.side_effect = requests.HTTPError("400 Client Error")
        mock_get.return_value = mock_response
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key"
        })
        
        # Execute API call and expect error handling
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        # Should handle error gracefully
        results = detector.detect(start_date, end_date)
        assert isinstance(results, list)  # Should return empty list on error
    
    @patch('requests.get')
    def test_api_timeout_mock(self, mock_get):
        """Test Elementary API timeout scenario."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup mock timeout
        mock_get.side_effect = Timeout("Request timed out")
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key",
            "timeout": 5
        })
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        # Should handle timeout gracefully
        results = detector.detect(start_date, end_date)
        assert isinstance(results, list)
    
    @patch('requests.get')
    def test_api_connection_error_mock(self, mock_get):
        """Test Elementary API connection error scenario."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup mock connection error
        mock_get.side_effect = ConnectionError("Failed to establish connection")
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key"
        })
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        # Should handle connection error gracefully
        results = detector.detect(start_date, end_date)
        assert isinstance(results, list)


class TestElementaryResultTransformation:
    """Test suite for Elementary result transformation logic - GADF-DETECT-009b."""
    
    @pytest.fixture
    def elementary_api_result(self):
        """Sample Elementary API result to transform."""
        return {
            "test_id": "anomaly_test_001",
            "table_name": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "column_name": "NUMBEROFVIEWS",
            "test_type": "anomaly_detection",
            "status": "failed",
            "detected_at": "2024-01-15T10:30:00Z",
            "expected_value": 10000.0,
            "actual_value": 5000.0,
            "deviation": 0.5,
            "severity": "high",
            "description": "Significant drop in listing views detected",
            "additional_metadata": {
                "baseline_period": "7_days",
                "confidence_level": 0.95
            }
        }
    
    def test_transform_elementary_result_to_detection_result(self, elementary_api_result):
        """Test transformation of Elementary API result to DetectionResult."""
        if ElementaryDetector is None or DetectionResult is None:
            pytest.skip("ElementaryDetector or DetectionResult not implemented yet")
            
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "event_type": "listing_views"
        })
        
        # Transform the result
        detection_result = detector._transform_elementary_result(elementary_api_result)
        
        # Verify transformation
        assert isinstance(detection_result, DetectionResult)
        assert detection_result.event_type == "listing_views"
        assert detection_result.metric_name == "NUMBEROFVIEWS"
        assert detection_result.detection_date == date(2024, 1, 15)
        assert detection_result.expected_value == 10000.0
        assert detection_result.actual_value == 5000.0
        assert detection_result.deviation_percentage == 0.5
        assert detection_result.severity == "high"
        assert detection_result.detection_method == "elementary"
        assert detection_result.alert_sent is False
        
        # Verify details contain Elementary-specific information
        assert detection_result.details is not None
        assert "test_id" in detection_result.details
        assert "test_type" in detection_result.details
        assert "description" in detection_result.details
    
    def test_transform_elementary_result_with_missing_fields(self):
        """Test transformation of Elementary result with missing optional fields."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "event_type": "listing_views"
        })
        
        # Minimal Elementary result
        minimal_result = {
            "test_id": "test_001",
            "table_name": "TEST_TABLE",
            "column_name": "TEST_COLUMN",
            "status": "failed",
            "detected_at": "2024-01-15T10:30:00Z"
        }
        
        # Should handle missing fields gracefully
        detection_result = detector._transform_elementary_result(minimal_result)
        
        assert isinstance(detection_result, DetectionResult)
        assert detection_result.metric_name == "TEST_COLUMN"
        
        # Should use defaults for missing values
        assert detection_result.expected_value == 0.0
        assert detection_result.actual_value == 0.0
        assert detection_result.deviation_percentage == 0.0
        assert detection_result.severity == "warning"  # Default severity
    
    def test_transform_multiple_elementary_results(self):
        """Test transformation of multiple Elementary results."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "event_type": "listing_views"
        })
        
        # Multiple Elementary results
        elementary_results = [
            {
                "test_id": "test_001",
                "table_name": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
                "column_name": "NUMBEROFVIEWS",
                "status": "failed",
                "detected_at": "2024-01-15T10:30:00Z",
                "expected_value": 10000.0,
                "actual_value": 5000.0,
                "deviation": 0.5,
                "severity": "critical"
            },
            {
                "test_id": "test_002",
                "table_name": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
                "column_name": "NUMBEROFENQUIRIES",
                "status": "failed",
                "detected_at": "2024-01-15T10:30:00Z",
                "expected_value": 1000.0,
                "actual_value": 800.0,
                "deviation": 0.2,
                "severity": "high"
            }
        ]
        
        # Transform all results
        detection_results = detector._transform_elementary_results(elementary_results)
        
        assert len(detection_results) == 2
        assert all(isinstance(result, DetectionResult) for result in detection_results)
        assert detection_results[0].severity == "critical"
        assert detection_results[1].severity == "high"
        assert detection_results[0].metric_name == "NUMBEROFVIEWS"
        assert detection_results[1].metric_name == "NUMBEROFENQUIRIES"
    
    def test_filter_passed_tests_in_transformation(self):
        """Test that passed Elementary tests are filtered out."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "event_type": "listing_views"
        })
        
        # Mix of passed and failed tests
        elementary_results = [
            {
                "test_id": "test_001",
                "status": "failed",
                "detected_at": "2024-01-15T10:30:00Z",
                "column_name": "NUMBEROFVIEWS"
            },
            {
                "test_id": "test_002",
                "status": "passed",
                "detected_at": "2024-01-15T10:30:00Z",
                "column_name": "NUMBEROFENQUIRIES"
            },
            {
                "test_id": "test_003",
                "status": "failed",
                "detected_at": "2024-01-15T10:30:00Z",
                "column_name": "NUMBEROFCLICKS"
            }
        ]
        
        # Transform results - should only include failed tests
        detection_results = detector._transform_elementary_results(elementary_results)
        
        assert len(detection_results) == 2  # Only failed tests
        assert all(result.details["test_id"] in ["test_001", "test_003"] for result in detection_results)


class TestElementaryErrorHandlingAndRetries:
    """Test suite for Elementary error handling and retry logic - GADF-DETECT-009c."""
    
    @patch('requests.get')
    def test_retry_on_temporary_failure(self, mock_get):
        """Test retry mechanism on temporary API failures."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup mock to fail twice then succeed
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"status": "success", "results": []}
        mock_response_success.raise_for_status.return_value = None
        
        mock_get.side_effect = [
            ConnectionError("Connection failed"),
            Timeout("Request timed out"),
            mock_response_success
        ]
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key",
            "retry_attempts": 3,
            "retry_delay": 0.1  # Short delay for testing
        })
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        # Should retry and eventually succeed
        results = detector.detect(start_date, end_date)
        
        # Verify 3 attempts were made
        assert mock_get.call_count == 3
        assert isinstance(results, list)
    
    @patch('requests.get')
    def test_retry_exhaustion(self, mock_get):
        """Test behavior when all retry attempts are exhausted."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup mock to always fail
        mock_get.side_effect = ConnectionError("Connection failed")
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key",
            "retry_attempts": 2,
            "retry_delay": 0.1
        })
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        # Should return empty results after exhausting retries
        results = detector.detect(start_date, end_date)
        
        # Verify all attempts were made
        assert mock_get.call_count == 2
        assert isinstance(results, list)
        assert len(results) == 0
    
    @patch('requests.get')
    def test_no_retry_on_client_error(self, mock_get):
        """Test that client errors (4xx) don't trigger retries."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup mock to return 400 error
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"status": "error", "message": "Bad request"}
        mock_response.raise_for_status.side_effect = requests.HTTPError("400 Client Error")
        mock_get.return_value = mock_response
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key",
            "retry_attempts": 3
        })
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        # Should not retry on client error
        results = detector.detect(start_date, end_date)
        
        # Verify only one attempt was made
        assert mock_get.call_count == 1
        assert isinstance(results, list)
    
    @patch('requests.get')
    def test_retry_on_server_error(self, mock_get):
        """Test retry mechanism on server errors (5xx)."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup mock to return 500 error then succeed
        mock_response_error = Mock()
        mock_response_error.status_code = 500
        mock_response_error.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"status": "success", "results": []}
        mock_response_success.raise_for_status.return_value = None
        
        mock_get.side_effect = [mock_response_error, mock_response_success]
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key",
            "retry_attempts": 2,
            "retry_delay": 0.1
        })
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        # Should retry on server error and succeed
        results = detector.detect(start_date, end_date)
        
        # Verify 2 attempts were made
        assert mock_get.call_count == 2
        assert isinstance(results, list)
    
    @patch('time.sleep')
    @patch('requests.get')
    def test_exponential_backoff(self, mock_get, mock_sleep):
        """Test exponential backoff in retry mechanism."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup mock to fail then succeed
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"status": "success", "results": []}
        mock_response_success.raise_for_status.return_value = None
        
        mock_get.side_effect = [
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed"),
            mock_response_success
        ]
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key",
            "retry_attempts": 3,
            "retry_delay": 1,
            "exponential_backoff": True
        })
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        results = detector.detect(start_date, end_date)
        
        # Verify exponential backoff delays
        assert mock_sleep.call_count == 2
        sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
        
        # Should use exponential backoff: 1s, 2s
        assert sleep_calls[0] == 1
        assert sleep_calls[1] == 2
    
    @patch('requests.get')
    def test_circuit_breaker_functionality(self, mock_get):
        """Test circuit breaker to prevent cascading failures."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup mock to always fail
        mock_get.side_effect = ConnectionError("Connection failed")
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key",
            "retry_attempts": 2,
            "circuit_breaker_threshold": 3,
            "circuit_breaker_timeout": 0.1
        })
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        # First few calls should attempt API
        results1 = detector.detect(start_date, end_date)
        results2 = detector.detect(start_date, end_date)
        results3 = detector.detect(start_date, end_date)
        
        # Circuit breaker should open after threshold failures
        # Next call should fail fast
        results4 = detector.detect(start_date, end_date)
        
        # Verify circuit breaker behavior
        assert all(isinstance(result, list) for result in [results1, results2, results3, results4])
        
        # Should have attempted API calls but then stopped
        assert mock_get.call_count <= 6  # 3 failures * 2 retries each


class TestElementaryConfigurationMapping:
    """Test suite for Elementary configuration mapping - GADF-DETECT-009d."""
    
    def test_configuration_mapping_basic(self):
        """Test basic configuration mapping to Elementary API parameters."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        config = {
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_api_key",
            "event_type": "listing_views",
            "table_name": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "columns": ["NUMBEROFVIEWS", "NUMBEROFENQUIRIES"],
            "test_types": ["anomaly_detection", "threshold_check"],
            "severity_levels": ["critical", "high", "warning"]
        }
        
        detector = ElementaryDetector(config)
        
        # Test configuration mapping
        api_params = detector._build_api_parameters(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        
        assert "table_name" in api_params
        assert "start_date" in api_params
        assert "end_date" in api_params
        assert "columns" in api_params
        assert "test_types" in api_params
        
        assert api_params["table_name"] == "DATAMART.DD_LISTING_STATISTICS_BLENDED"
        assert api_params["start_date"] == "2024-01-01"
        assert api_params["end_date"] == "2024-01-31"
    
    def test_configuration_mapping_with_filters(self):
        """Test configuration mapping with filtering parameters."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        config = {
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_api_key",
            "event_type": "listing_views",
            "filters": {
                "min_severity": "high",
                "exclude_columns": ["DEPRECATED_COLUMN"],
                "only_failed_tests": True
            },
            "custom_parameters": {
                "confidence_level": 0.95,
                "baseline_days": 14
            }
        }
        
        detector = ElementaryDetector(config)
        
        api_params = detector._build_api_parameters(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        
        # Verify filter parameters are included
        assert "filters" in api_params
        assert api_params["filters"]["min_severity"] == "high"
        assert api_params["filters"]["only_failed_tests"] is True
        
        # Verify custom parameters
        assert "custom_parameters" in api_params
        assert api_params["custom_parameters"]["confidence_level"] == 0.95
    
    def test_configuration_mapping_with_authentication(self):
        """Test configuration mapping with different authentication methods."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Test API key authentication
        config_api_key = {
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_api_key"
        }
        
        detector_api_key = ElementaryDetector(config_api_key)
        headers_api_key = detector_api_key._build_request_headers()
        
        assert "Authorization" in headers_api_key or "X-API-Key" in headers_api_key
        
        # Test bearer token authentication
        config_bearer = {
            "elementary_endpoint": "http://localhost:8080",
            "bearer_token": "test_bearer_token"
        }
        
        detector_bearer = ElementaryDetector(config_bearer)
        headers_bearer = detector_bearer._build_request_headers()
        
        assert "Authorization" in headers_bearer
        assert headers_bearer["Authorization"].startswith("Bearer ")
    
    def test_configuration_validation(self):
        """Test validation of Elementary configuration parameters."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Test invalid endpoint URL
        with pytest.raises(ValueError, match="Invalid endpoint URL"):
            ElementaryDetector({
                "elementary_endpoint": "not-a-valid-url"
            })
        
        # Test missing required configuration
        with pytest.raises(ValueError, match="elementary_endpoint.*required"):
            ElementaryDetector({
                "api_key": "test_key"
            })
        
        # Test invalid timeout values
        with pytest.raises(ValueError, match="timeout.*positive"):
            ElementaryDetector({
                "elementary_endpoint": "http://localhost:8080",
                "timeout": -5
            })
        
        # Test invalid retry attempts
        with pytest.raises(ValueError, match="retry_attempts.*non-negative"):
            ElementaryDetector({
                "elementary_endpoint": "http://localhost:8080",
                "retry_attempts": -1
            })
    
    def test_configuration_defaults(self):
        """Test that configuration uses appropriate defaults."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        minimal_config = {
            "elementary_endpoint": "http://localhost:8080"
        }
        
        detector = ElementaryDetector(minimal_config)
        
        # Verify default values
        assert detector.get_config_value("timeout", 30) == 30
        assert detector.get_config_value("retry_attempts", 3) == 3
        assert detector.get_config_value("retry_delay", 1) == 1
        assert detector.get_config_value("exponential_backoff", True) is True
        assert detector.get_config_value("circuit_breaker_threshold", 5) == 5
    
    def test_configuration_override(self):
        """Test that custom configuration values override defaults."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        custom_config = {
            "elementary_endpoint": "http://localhost:8080",
            "timeout": 60,
            "retry_attempts": 5,
            "retry_delay": 2,
            "exponential_backoff": False,
            "circuit_breaker_threshold": 10
        }
        
        detector = ElementaryDetector(custom_config)
        
        # Verify custom values are used
        assert detector.get_config_value("timeout") == 60
        assert detector.get_config_value("retry_attempts") == 5
        assert detector.get_config_value("retry_delay") == 2
        assert detector.get_config_value("exponential_backoff") is False
        assert detector.get_config_value("circuit_breaker_threshold") == 10


class TestElementaryPerformanceValidation:
    """Test suite for Elementary detector performance validation."""
    
    @patch('requests.get')
    def test_api_call_performance(self, mock_get):
        """Test that Elementary API calls complete within reasonable time."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup mock response with delay
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "results": []}
        mock_response.raise_for_status.return_value = None
        
        def slow_response(*args, **kwargs):
            time.sleep(0.1)  # Simulate 100ms response time
            return mock_response
        
        mock_get.side_effect = slow_response
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key",
            "timeout": 5
        })
        
        start_time = time.time()
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 31))
        execution_time = time.time() - start_time
        
        # Verify performance is acceptable
        assert execution_time < 1.0  # Should complete within 1 second
        assert isinstance(results, list)
    
    @patch('requests.get')
    def test_large_result_set_handling(self, mock_get):
        """Test handling of large result sets from Elementary API."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Generate large result set
        large_results = []
        for i in range(1000):
            large_results.append({
                "test_id": f"test_{i:04d}",
                "table_name": "LARGE_TABLE",
                "column_name": f"COLUMN_{i % 10}",
                "status": "failed" if i % 5 == 0 else "passed",
                "detected_at": "2024-01-15T10:30:00Z",
                "expected_value": 1000.0,
                "actual_value": 800.0 if i % 5 == 0 else 1000.0,
                "deviation": 0.2 if i % 5 == 0 else 0.0,
                "severity": "warning"
            })
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "results": large_results}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key"
        })
        
        start_time = time.time()
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 31))
        execution_time = time.time() - start_time
        
        # Verify large result sets are handled efficiently
        assert execution_time < 2.0  # Should process 1000 results within 2 seconds
        assert isinstance(results, list)
        
        # Only failed tests should be returned (200 out of 1000)
        failed_results = [r for r in large_results if r["status"] == "failed"]
        assert len(results) == len(failed_results)
    
    def test_memory_usage_with_large_datasets(self):
        """Test memory usage remains reasonable with large datasets."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        detector = ElementaryDetector({
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "test_key"
        })
        
        # Simulate processing large dataset
        large_elementary_results = []
        for i in range(10000):
            large_elementary_results.append({
                "test_id": f"test_{i:05d}",
                "status": "failed",
                "detected_at": "2024-01-15T10:30:00Z",
                "column_name": f"COLUMN_{i % 100}",
                "expected_value": 1000.0,
                "actual_value": 800.0,
                "deviation": 0.2
            })
        
        # Process results in batches to test memory efficiency
        batch_size = 100
        total_results = []
        
        for i in range(0, len(large_elementary_results), batch_size):
            batch = large_elementary_results[i:i + batch_size]
            batch_results = detector._transform_elementary_results(batch)
            total_results.extend(batch_results)
        
        # Verify all results were processed
        assert len(total_results) == len(large_elementary_results)
        assert all(isinstance(result, DetectionResult) for result in total_results)


class TestElementaryIntegrationEndToEnd:
    """End-to-end integration tests for Elementary detector."""
    
    @patch('requests.get')
    def test_complete_detection_workflow(self, mock_get):
        """Test complete detection workflow from API call to results."""
        if ElementaryDetector is None:
            pytest.skip("ElementaryDetector not implemented yet")
            
        # Setup realistic Elementary API response
        api_response = {
            "status": "success",
            "results": [
                {
                    "test_id": "anomaly_test_001",
                    "table_name": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
                    "column_name": "NUMBEROFVIEWS",
                    "test_type": "anomaly_detection",
                    "status": "failed",
                    "detected_at": "2024-01-15T10:30:00Z",
                    "expected_value": 10000.0,
                    "actual_value": 5000.0,
                    "deviation": 0.5,
                    "severity": "critical",
                    "description": "Significant drop in listing views detected"
                }
            ],
            "metadata": {
                "total_tests": 1,
                "failed_tests": 1,
                "execution_time": "1.2s"
            }
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = api_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Initialize detector with realistic configuration
        detector = ElementaryDetector({
            "elementary_endpoint": "http://elementary.company.com/api/v1",
            "api_key": "prod_api_key_123",
            "event_type": "listing_views",
            "table_name": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "columns": ["NUMBEROFVIEWS", "NUMBEROFENQUIRIES"],
            "severity_levels": ["critical", "high", "warning"],
            "timeout": 30,
            "retry_attempts": 3
        })
        
        # Execute detection
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        results = detector.detect(start_date, end_date)
        
        # Verify API was called correctly
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        
        # Verify URL and headers
        # call_args[0] contains positional args, call_args[1] contains keyword args
        # URL is passed as first positional argument, so it's in call_args[0][0]
        assert "elementary.company.com" in call_args[0][0]
        assert "headers" in call_args[1]
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        
        assert isinstance(result, DetectionResult)
        assert result.event_type == "listing_views"
        assert result.metric_name == "NUMBEROFVIEWS"
        assert result.severity == "critical"
        assert result.detection_method == "elementary"
        assert result.deviation_percentage == 0.5
        assert result.expected_value == 10000.0
        assert result.actual_value == 5000.0


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])