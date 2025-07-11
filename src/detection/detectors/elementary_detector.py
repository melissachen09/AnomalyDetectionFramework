"""
Elementary Detector Implementation for Anomaly Detection Framework.

This module implements the Elementary data quality API integration for anomaly detection.
Part of ADF-39: Implement Elementary Detector (GADF-DETECT-010).

The ElementaryDetector provides:
- Elementary API integration with authentication
- Robust error handling with retries and circuit breaker
- Result transformation to framework standards
- Caching layer for performance optimization
- Comprehensive configuration validation
"""

import json
import time
from datetime import date, datetime
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urljoin, urlparse
import logging
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError

from .base_detector import BaseDetector, DetectionResult, register_detector


class ElementaryAPIClient:
    """
    Client for communicating with Elementary data quality API.
    
    Handles authentication, retries, circuit breaker pattern, and response parsing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Elementary API client.
        
        Args:
            config: Configuration dictionary containing API settings
        """
        self.endpoint = config["elementary_endpoint"]
        self.api_key = config.get("api_key")
        self.bearer_token = config.get("bearer_token")
        self.timeout = config.get("timeout", 30)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 1)
        self.exponential_backoff = config.get("exponential_backoff", True)
        
        # Circuit breaker settings
        self.circuit_breaker_threshold = config.get("circuit_breaker_threshold", 5)
        self.circuit_breaker_timeout = config.get("circuit_breaker_timeout", 60)
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = None
        
        self.logger = logging.getLogger(__name__)
    
    def _build_request_headers(self) -> Dict[str, str]:
        """Build HTTP headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        elif self.api_key:
            headers["X-API-Key"] = self.api_key
        
        return headers
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_breaker_failures < self.circuit_breaker_threshold:
            return False
        
        if self._circuit_breaker_last_failure is None:
            return False
        
        time_since_failure = time.time() - self._circuit_breaker_last_failure
        return time_since_failure < self.circuit_breaker_timeout
    
    def _record_failure(self) -> None:
        """Record API failure for circuit breaker."""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = time.time()
    
    def _record_success(self) -> None:
        """Record API success for circuit breaker."""
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = None
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if request should be retried based on exception type."""
        # Retry on network errors and server errors, but not client errors
        if isinstance(exception, (ConnectionError, Timeout)):
            return True
        
        if isinstance(exception, HTTPError):
            # Check if response is available
            if hasattr(exception, 'response') and exception.response is not None:
                return exception.response.status_code >= 500
            
            # If response not available, check the error message for status code
            error_msg = str(exception)
            if '4' in error_msg and 'Client Error' in error_msg:
                return False  # Don't retry client errors
            elif '5' in error_msg and 'Server Error' in error_msg:
                return True   # Retry server errors
            
            # Default to not retrying if we can't determine
            return False
        
        return False
    
    def call_api(self, endpoint_path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make API call to Elementary service with retry logic.
        
        Args:
            endpoint_path: API endpoint path
            params: Query parameters
            
        Returns:
            API response data or None if call fails
        """
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            self.logger.warning("Circuit breaker is open, skipping API call")
            return None
        
        url = urljoin(self.endpoint, endpoint_path)
        headers = self._build_request_headers()
        
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                self._record_success()
                return response.json()
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                
                if not self._should_retry(e):
                    self.logger.error(f"Non-retryable error: {e}")
                    break
                
                if attempt < self.retry_attempts - 1:
                    # Calculate delay with exponential backoff
                    delay = self.retry_delay
                    if self.exponential_backoff:
                        delay *= (2 ** attempt)
                    
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        
        # All attempts failed
        self._record_failure()
        self.logger.error(f"All API attempts failed. Last error: {last_exception}")
        return None


@register_detector("elementary")
class ElementaryDetector(BaseDetector):
    """
    Elementary data quality API integration detector.
    
    Integrates with Elementary data quality monitoring service to detect
    anomalies using existing data quality tests. Provides robust error
    handling, caching, and result transformation.
    
    Configuration Parameters:
        elementary_endpoint (str): Base URL for Elementary API
        api_key (str, optional): API key for authentication
        bearer_token (str, optional): Bearer token for authentication
        timeout (int): Request timeout in seconds (default: 30)
        retry_attempts (int): Number of retry attempts (default: 3)
        retry_delay (int): Base delay between retries (default: 1)
        exponential_backoff (bool): Use exponential backoff (default: True)
        circuit_breaker_threshold (int): Failure threshold for circuit breaker (default: 5)
        circuit_breaker_timeout (int): Circuit breaker timeout in seconds (default: 60)
        cache_ttl (int): Cache time-to-live in seconds (default: 300)
        
    Example Configuration:
        {
            "elementary_endpoint": "http://localhost:8080",
            "api_key": "your-api-key",
            "event_type": "listing_views",
            "table_name": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "timeout": 30,
            "retry_attempts": 3,
            "cache_ttl": 300
        }
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ElementaryDetector with configuration validation.
        
        Args:
            config: Configuration dictionary containing detector parameters
            
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        super().__init__(config)
        
        self._validate_elementary_config()
        self.api_client = ElementaryAPIClient(config)
        self._cache = {}
        self.cache_ttl = config.get("cache_ttl", 300)
        
        self.logger.info(f"ElementaryDetector initialized for endpoint: {self.get_config_value('elementary_endpoint')}")
    
    def _build_request_headers(self) -> Dict[str, str]:
        """Build HTTP headers for API requests - exposed for testing."""
        return self.api_client._build_request_headers()
    
    def _validate_elementary_config(self) -> None:
        """
        Validate Elementary configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required endpoint
        endpoint = self.get_config_value("elementary_endpoint")
        if not endpoint:
            raise ValueError("elementary_endpoint is required")
        
        # Validate endpoint URL format
        try:
            parsed = urlparse(endpoint)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid endpoint URL format")
        except Exception as e:
            raise ValueError(f"Invalid endpoint URL: {e}")
        
        # Validate timeout
        timeout = self.get_config_value("timeout")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        
        # Validate retry attempts
        retry_attempts = self.get_config_value("retry_attempts")
        if retry_attempts is not None and retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")
        
        # Validate cache TTL
        cache_ttl = self.get_config_value("cache_ttl")
        if cache_ttl is not None and cache_ttl < 0:
            raise ValueError("cache_ttl must be non-negative")
    
    def _build_api_parameters(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Build API parameters for Elementary request.
        
        Args:
            start_date: Start date for detection
            end_date: End date for detection
            
        Returns:
            Dictionary of API parameters
        """
        params = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        
        # Add table name if specified
        if self.get_config_value("table_name"):
            params["table_name"] = self.get_config_value("table_name")
        
        # Add column filters if specified
        if self.get_config_value("columns"):
            params["columns"] = self.get_config_value("columns")
        
        # Add test type filters if specified
        if self.get_config_value("test_types"):
            params["test_types"] = self.get_config_value("test_types")
        
        # Add custom filters
        filters = self.get_config_value("filters", {})
        if filters:
            params["filters"] = filters
        
        # Add custom parameters
        custom_params = self.get_config_value("custom_parameters", {})
        if custom_params:
            params["custom_parameters"] = custom_params
        
        return params
    
    def _get_cache_key(self, start_date: date, end_date: date) -> str:
        """Generate cache key for API results."""
        return f"{self.get_config_value('elementary_endpoint')}_{start_date}_{end_date}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached API result if still valid."""
        if cache_key not in self._cache:
            return None
        
        cached_data, timestamp = self._cache[cache_key]
        if time.time() - timestamp > self.cache_ttl:
            del self._cache[cache_key]
            return None
        
        return cached_data
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache API result with timestamp."""
        self._cache[cache_key] = (result, time.time())
    
    def _transform_elementary_result(self, elementary_result: Dict[str, Any]) -> DetectionResult:
        """
        Transform Elementary API result to DetectionResult.
        
        Args:
            elementary_result: Individual Elementary test result
            
        Returns:
            DetectionResult object
        """
        # Parse detection date
        detected_at = elementary_result.get("detected_at", datetime.now().isoformat())
        if isinstance(detected_at, str):
            detection_date = datetime.fromisoformat(detected_at.replace('Z', '+00:00')).date()
        else:
            detection_date = date.today()
        
        # Extract values with defaults
        expected_value = elementary_result.get("expected_value", 0.0)
        actual_value = elementary_result.get("actual_value", 0.0)
        deviation = elementary_result.get("deviation", 0.0)
        severity = elementary_result.get("severity", "warning")
        
        # Build details dictionary
        details = {
            "test_id": elementary_result.get("test_id"),
            "test_type": elementary_result.get("test_type"),
            "description": elementary_result.get("description"),
            "table_name": elementary_result.get("table_name"),
            "status": elementary_result.get("status")
        }
        
        # Add any additional metadata
        if "additional_metadata" in elementary_result:
            details.update(elementary_result["additional_metadata"])
        
        return DetectionResult(
            event_type=self.get_config_value("event_type", "unknown"),
            metric_name=elementary_result.get("column_name", "unknown"),
            detection_date=detection_date,
            expected_value=expected_value,
            actual_value=actual_value,
            deviation_percentage=deviation,
            severity=severity,
            detection_method="elementary",
            alert_sent=False,
            details=details
        )
    
    def _transform_elementary_results(self, elementary_results: List[Dict[str, Any]]) -> List[DetectionResult]:
        """
        Transform multiple Elementary API results to DetectionResults.
        
        Args:
            elementary_results: List of Elementary test results
            
        Returns:
            List of DetectionResult objects (only failed tests)
        """
        detection_results = []
        
        for result in elementary_results:
            # Only process failed tests
            if result.get("status") == "failed":
                detection_results.append(self._transform_elementary_result(result))
        
        return detection_results
    
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        """
        Run Elementary anomaly detection for the specified date range.
        
        Args:
            start_date: Start date for detection analysis (inclusive)
            end_date: End date for detection analysis (inclusive)
        
        Returns:
            List of DetectionResult objects representing detected anomalies.
            Empty list if no anomalies are found or if API call fails.
        """
        # Validate date range
        self.validate_date_range(start_date, end_date)
        
        # Check cache first
        cache_key = self._get_cache_key(start_date, end_date)
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            self.logger.info(f"Using cached results for {start_date} to {end_date}")
            return self._transform_elementary_results(cached_result.get("results", []))
        
        # Build API parameters
        params = self._build_api_parameters(start_date, end_date)
        
        # Make API call
        self.logger.info(f"Calling Elementary API for {start_date} to {end_date}")
        api_response = self.api_client.call_api("/api/v1/tests", params)
        
        if not api_response:
            self.logger.warning("Elementary API call failed, returning empty results")
            return []
        
        # Cache successful response
        self._cache_result(cache_key, api_response)
        
        # Transform results
        elementary_results = api_response.get("results", [])
        detection_results = self._transform_elementary_results(elementary_results)
        
        self.logger.info(f"Elementary detection completed: {len(detection_results)} anomalies found")
        return detection_results