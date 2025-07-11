"""
Test cases for ThresholdDetector - ADF-32: Write Test Cases for Threshold Detector.

This module implements comprehensive test cases for threshold-based anomaly detection
following TDD approach. Tests cover all threshold types, edge cases, performance
validation, and configuration error handling as specified in GADF-DETECT-003.

Test Coverage:
- GADF-DETECT-003a: Test min/max threshold violations
- GADF-DETECT-003b: Verify percentage change calculations  
- GADF-DETECT-003c: Test null and zero value handling
- GADF-DETECT-003d: Validate multi-metric threshold detection
"""

import pytest
import pandas as pd
from datetime import date, datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import numpy as np

# Following TDD - import will initially fail
try:
    from src.detection.detectors.base_detector import BaseDetector, DetectionResult
    from src.detection.detectors.threshold_detector import ThresholdDetector
except ImportError:
    BaseDetector = None
    DetectionResult = None
    ThresholdDetector = None


class TestThresholdDetectorInterface:
    """Test ThresholdDetector interface compliance and inheritance."""
    
    def test_threshold_detector_inherits_from_base_detector(self):
        """Test that ThresholdDetector inherits from BaseDetector."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        assert issubclass(ThresholdDetector, BaseDetector)
    
    def test_threshold_detector_implements_detect_method(self):
        """Test that ThresholdDetector implements the abstract detect method."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        config = {
            "event_type": "test_event",
            "table": "TEST_TABLE",
            "metrics": [{"column": "test_metric", "alias": "metric"}],
            "thresholds": {"metric": {"min_value": 100, "max_value": 1000}}
        }
        
        detector = ThresholdDetector(config)
        
        # Should be able to call detect method without TypeError
        assert hasattr(detector, 'detect')
        assert callable(detector.detect)
    
    def test_threshold_detector_registration(self):
        """Test that ThresholdDetector can be registered with the framework."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        from src.detection.detectors.base_detector import register_detector, get_registered_detectors
        
        # Test registration decorator usage (will be implemented later)
        initial_count = len(get_registered_detectors())
        
        @register_detector("test_threshold")
        class TestThresholdDetector(ThresholdDetector):
            pass
        
        registered = get_registered_detectors()
        assert len(registered) == initial_count + 1
        assert "test_threshold" in registered


class TestThresholdDetectorInitialization:
    """Test ThresholdDetector initialization and configuration validation."""
    
    def test_initialization_with_valid_threshold_config(self):
        """Test successful initialization with valid threshold configuration."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "date_column": "STATISTIC_DATE",
            "metrics": [
                {"column": "NUMBEROFVIEWS", "alias": "total_views"},
                {"column": "NUMBEROFENQUIRIES", "alias": "total_enquiries"}
            ],
            "thresholds": {
                "total_views": {"min_value": 10000, "max_value": 1000000},
                "total_enquiries": {"min_value": 100, "max_value": 50000}
            }
        }
        
        detector = ThresholdDetector(config)
        
        assert detector.config == config
        assert detector.get_config_value("event_type") == "listing_views"
        assert detector.get_config_value("table") == "DATAMART.DD_LISTING_STATISTICS_BLENDED"
    
    def test_initialization_with_missing_thresholds(self):
        """Test initialization error when thresholds are missing."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "metrics": [{"column": "test_metric", "alias": "metric"}]
            # Missing thresholds
        }
        
        with pytest.raises(ValueError, match="thresholds.*required"):
            ThresholdDetector(config)
    
    def test_initialization_with_invalid_threshold_values(self):
        """Test initialization error with invalid threshold values."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        config = {
            "event_type": "test_event",
            "table": "TEST_TABLE", 
            "metrics": [{"column": "test_metric", "alias": "metric"}],
            "thresholds": {
                "metric": {"min_value": 1000, "max_value": 100}  # min > max
            }
        }
        
        with pytest.raises(ValueError, match="min_value.*max_value"):
            ThresholdDetector(config)
    
    def test_initialization_with_missing_metrics(self):
        """Test initialization error when metrics configuration is missing."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        config = {
            "event_type": "test_event",
            "table": "TEST_TABLE",
            "thresholds": {"metric": {"min_value": 100, "max_value": 1000}}
            # Missing metrics
        }
        
        with pytest.raises(ValueError, match="metrics.*required"):
            ThresholdDetector(config)


class TestMinMaxThresholdViolations:
    """Test min/max threshold violation detection - GADF-DETECT-003a."""
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_min_threshold_violation_detection(self, mock_get_connection):
        """Test detection of values below minimum threshold."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        # Mock Snowflake connection and data
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Simulate data below minimum threshold
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1), date(2024, 1, 2)],
            'total_views': [5000, 8000]  # Below min_value of 10000
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = test_data
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {"min_value": 10000, "max_value": 1000000}
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 2))
        
        # Should detect 2 anomalies
        assert len(results) == 2
        
        # Verify first anomaly details
        anomaly = results[0]
        assert anomaly.event_type == "listing_views"
        assert anomaly.metric_name == "total_views"
        assert anomaly.detection_date == date(2024, 1, 1)
        assert anomaly.actual_value == 5000
        assert anomaly.expected_value >= 10000  # Should be min threshold
        assert anomaly.severity in ["critical", "high", "warning"]
        assert anomaly.detection_method == "threshold"
        
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_max_threshold_violation_detection(self, mock_get_connection):
        """Test detection of values above maximum threshold."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Simulate data above maximum threshold
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1)],
            'total_views': [1500000]  # Above max_value of 1000000
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = test_data
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE", 
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {"min_value": 10000, "max_value": 1000000}
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 1))
        
        assert len(results) == 1
        
        anomaly = results[0]
        assert anomaly.actual_value == 1500000
        assert anomaly.expected_value <= 1000000  # Should be max threshold
        assert anomaly.severity in ["critical", "high", "warning"]
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_values_within_threshold_range(self, mock_get_connection):
        """Test that values within threshold range don't trigger anomalies."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Simulate data within thresholds
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1), date(2024, 1, 2)],
            'total_views': [50000, 100000]  # Within 10000-1000000 range
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = test_data
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {"min_value": 10000, "max_value": 1000000}
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 2))
        
        # Should not detect any anomalies
        assert len(results) == 0
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_boundary_value_handling(self, mock_get_connection):
        """Test handling of exact threshold boundary values."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Test exact boundary values
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1), date(2024, 1, 2)],
            'total_views': [10000, 1000000]  # Exact min and max values
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = test_data
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {"min_value": 10000, "max_value": 1000000}
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 2))
        
        # Boundary values should be considered valid (inclusive bounds)
        assert len(results) == 0


class TestPercentageChangeCalculations:
    """Test percentage change threshold detection - GADF-DETECT-003b."""
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_percentage_increase_detection(self, mock_get_connection):
        """Test detection of significant percentage increases."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Mock historical data for baseline calculation
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1), date(2024, 1, 2)],
            'total_views': [10000, 10000]  # Baseline average: 10000
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 3)],
            'total_views': [18000]  # 80% increase from baseline
        })
        
        # Mock sequential calls for historical and current data
        mock_conn.execute.return_value.fetch_pandas_all.side_effect = [
            historical_data, current_data, current_data  # Add extra data for absolute check
        ]
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {
                    "percentage_change_max": 0.5,  # 50% max increase
                    "baseline_days": 2
                }
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 3), date(2024, 1, 3))
        
        assert len(results) == 1
        
        anomaly = results[0]
        assert anomaly.actual_value == 18000
        assert anomaly.expected_value == 10000  # Baseline average
        assert anomaly.deviation_percentage == 0.8  # 80% increase
        assert anomaly.severity in ["critical", "high", "warning"]
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_percentage_decrease_detection(self, mock_get_connection):
        """Test detection of significant percentage decreases."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Mock data showing significant decrease
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1), date(2024, 1, 2)],
            'total_views': [20000, 20000]  # Baseline: 20000
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 3)],
            'total_views': [8000]  # 60% decrease from baseline
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.side_effect = [
            historical_data, current_data, current_data  # Add extra data for absolute check
        ]
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {
                    "percentage_change_min": -0.3,  # 30% max decrease
                    "baseline_days": 2
                }
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 3), date(2024, 1, 3))
        
        assert len(results) == 1
        
        anomaly = results[0]
        assert anomaly.actual_value == 8000
        assert anomaly.expected_value == 20000
        assert anomaly.deviation_percentage == -0.6  # 60% decrease
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_insufficient_historical_data_handling(self, mock_get_connection):
        """Test handling when insufficient historical data is available."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Only 1 day of historical data when 7 days requested
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1)],
            'total_views': [10000]
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 8)],
            'total_views': [15000]
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.side_effect = [
            historical_data, current_data, current_data  # Add extra data for absolute check
        ]
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {
                    "percentage_change_max": 0.3,
                    "baseline_days": 7  # Request 7 days but only 1 available
                }
            }
        }
        
        detector = ThresholdDetector(config)
        
        # Should handle gracefully - either skip detection or use available data
        results = detector.detect(date(2024, 1, 8), date(2024, 1, 8))
        
        # Implementation choice: could return empty results or use available data
        # Test should verify it doesn't crash
        assert isinstance(results, list)
    
    def test_percentage_change_calculation_accuracy(self):
        """Test accuracy of percentage change calculations."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        # Test the percentage calculation logic directly
        config = {
            "event_type": "test",
            "table": "TEST_TABLE",
            "metrics": [{"column": "test", "alias": "test"}],
            "thresholds": {"test": {"percentage_change_max": 0.5}}
        }
        
        detector = ThresholdDetector(config)
        
        # Test various percentage calculations
        test_cases = [
            (100, 150, 0.5),    # 50% increase
            (200, 100, -0.5),   # 50% decrease  
            (100, 200, 1.0),    # 100% increase
            (100, 0, -1.0),     # 100% decrease
            (100, 100, 0.0),    # No change
        ]
        
        for baseline, current, expected_pct in test_cases:
            # This tests the calculation method when implemented
            calculated_pct = detector._calculate_percentage_change(baseline, current)
            assert abs(calculated_pct - expected_pct) < 0.001  # Allow for float precision


class TestNullAndZeroValueHandling:
    """Test null and zero value handling - GADF-DETECT-003c."""
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_null_value_handling(self, mock_get_connection):
        """Test handling of null/NaN values in metrics data."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Data with null values
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            'total_views': [10000, np.nan, 15000]  # Middle value is null
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = test_data
        
        config = {
            "event_type": "listing_views", 
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {"min_value": 5000, "max_value": 50000}
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 3))
        
        # Should handle null values gracefully - either skip or flag as anomaly
        # Verify no exceptions are raised
        assert isinstance(results, list)
        
        # If null values are flagged as anomalies, verify the detection
        null_anomalies = [r for r in results if pd.isna(r.actual_value)]
        if null_anomalies:
            assert len(null_anomalies) == 1
            assert null_anomalies[0].detection_date == date(2024, 1, 2)
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_zero_value_handling(self, mock_get_connection):
        """Test handling of zero values in metrics."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Data with zero values
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1), date(2024, 1, 2)],
            'total_views': [0, 0]  # All zero values
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = test_data
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE", 
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {"min_value": 1000, "max_value": 50000}
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 2))
        
        # Zero values below minimum should be detected as anomalies
        assert len(results) == 2
        for result in results:
            assert result.actual_value == 0
            assert result.severity in ["critical", "high", "warning"]
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_zero_baseline_percentage_calculation(self, mock_get_connection):
        """Test percentage calculation when baseline is zero."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Historical data with zero baseline
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1), date(2024, 1, 2)],
            'total_views': [0, 0]  # Zero baseline
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 3)],
            'total_views': [1000]  # Non-zero current value
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.side_effect = [
            historical_data, current_data, current_data  # Add extra data for absolute check
        ]
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE", 
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {
                    "percentage_change_max": 2.0,  # 200% max increase
                    "baseline_days": 2
                }
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 3), date(2024, 1, 3))
        
        # Should handle zero baseline gracefully 
        # Implementation choice: could use absolute threshold or skip percentage
        assert isinstance(results, list)
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_mixed_null_zero_valid_data(self, mock_get_connection):
        """Test handling of mixed null, zero, and valid values."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Mixed data types
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [
                date(2024, 1, 1), date(2024, 1, 2), 
                date(2024, 1, 3), date(2024, 1, 4)
            ],
            'total_views': [10000, 0, np.nan, 25000]  # Valid, zero, null, valid
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = test_data
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {"min_value": 5000, "max_value": 50000}
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 4))
        
        # Should process all rows and handle each appropriately
        assert isinstance(results, list)
        
        # Valid values within threshold should not trigger anomalies
        # Zero and null values should be handled based on implementation


class TestMultiMetricThresholdDetection:
    """Test multi-metric threshold detection - GADF-DETECT-003d."""
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_multiple_metrics_detection(self, mock_get_connection):
        """Test detection across multiple metrics simultaneously."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Multi-metric data with violations in different metrics
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1), date(2024, 1, 2)],
            'total_views': [5000, 15000],      # Day 1: below min, Day 2: normal
            'total_enquiries': [200, 60000]    # Day 1: normal, Day 2: above max
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = test_data
        
        config = {
            "event_type": "listing_activity",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [
                {"column": "NUMBEROFVIEWS", "alias": "total_views"},
                {"column": "NUMBEROFENQUIRIES", "alias": "total_enquiries"}
            ],
            "thresholds": {
                "total_views": {"min_value": 10000, "max_value": 100000},
                "total_enquiries": {"min_value": 100, "max_value": 50000}
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 2))
        
        # Should detect 2 anomalies: views below min on day 1, enquiries above max on day 2
        assert len(results) == 2
        
        # Verify detection details
        views_anomaly = next(r for r in results if r.metric_name == "total_views")
        enquiries_anomaly = next(r for r in results if r.metric_name == "total_enquiries")
        
        assert views_anomaly.detection_date == date(2024, 1, 1)
        assert views_anomaly.actual_value == 5000
        
        assert enquiries_anomaly.detection_date == date(2024, 1, 2)
        assert enquiries_anomaly.actual_value == 60000
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_independent_metric_threshold_configurations(self, mock_get_connection):
        """Test that each metric can have independent threshold configurations."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1)],
            'page_views': [500],      # Should trigger percentage check
            'click_through': [0.15]   # Should trigger absolute threshold check
        })
        
        # Historical data for percentage calculation
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2023, 12, 31)],
            'page_views': [1000],     # Baseline for percentage
            'click_through': [0.12]
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.side_effect = [
            historical_data, test_data, test_data, test_data  # Extra mock data to handle multiple calls
        ]
        
        config = {
            "event_type": "web_metrics",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [
                {"column": "PAGE_VIEWS", "alias": "page_views"},
                {"column": "CLICK_THROUGH_RATE", "alias": "click_through"}
            ],
            "thresholds": {
                "page_views": {
                    "percentage_change_min": -0.25,  # 25% decrease threshold
                    "baseline_days": 1
                },
                "click_through": {
                    "min_value": 0.05,   # Absolute threshold
                    "max_value": 0.20
                }
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 1))
        
        # Should detect page_views decrease (50% drop) but not click_through (within range)
        assert len(results) == 1
        assert results[0].metric_name == "page_views"
        assert results[0].deviation_percentage == -0.5  # 50% decrease
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_metric_alias_handling(self, mock_get_connection):
        """Test proper handling of metric aliases vs column names."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1)],
            'listing_views': [5000]  # Column uses alias name in result
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = test_data
        
        config = {
            "event_type": "listing_activity",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [
                # Column name differs from alias
                {"column": "NUMBEROFVIEWS", "alias": "listing_views"}
            ],
            "thresholds": {
                "listing_views": {"min_value": 10000}  # Uses alias in threshold config
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 1))
        
        assert len(results) == 1
        assert results[0].metric_name == "listing_views"  # Should use alias
        assert results[0].actual_value == 5000
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_missing_threshold_configuration_for_metric(self, mock_get_connection):
        """Test handling when a metric is defined but has no threshold configuration."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1)],
            'metric_with_threshold': [5000],
            'metric_without_threshold': [1000]
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = test_data
        
        config = {
            "event_type": "test_event",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [
                {"column": "METRIC1", "alias": "metric_with_threshold"},
                {"column": "METRIC2", "alias": "metric_without_threshold"}
            ],
            "thresholds": {
                "metric_with_threshold": {"min_value": 10000}
                # metric_without_threshold has no threshold config
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 1))
        
        # Should only check metrics with threshold configurations
        assert len(results) == 1
        assert results[0].metric_name == "metric_with_threshold"


class TestThresholdDetectorPerformance:
    """Test performance characteristics and optimization."""
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_large_dataset_performance(self, mock_get_connection):
        """Test performance with large datasets."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Large dataset simulation
        import random
        from datetime import timedelta
        start_date = date(2024, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(100)]  # 100 days across multiple months
        views = [random.randint(8000, 12000) for _ in range(100)]  # Most within threshold
        
        # Add some anomalies
        views[10] = 5000   # Below threshold
        views[50] = 15000  # Above threshold
        
        large_data = pd.DataFrame({
            'STATISTIC_DATE': dates,
            'total_views': views
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = large_data
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {"min_value": 6000, "max_value": 14000}
            }
        }
        
        detector = ThresholdDetector(config)
        
        # Measure execution time
        import time
        start_time = time.time()
        results = detector.detect(start_date, start_date + timedelta(days=99))
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 5.0  # 5 seconds max
        
        # Should detect the 2 anomalies
        assert len(results) == 2
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_memory_efficiency_with_large_data(self, mock_get_connection):
        """Test memory usage with large datasets."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        # This test would require memory profiling tools in a real implementation
        # For now, verify the detector handles large data without crashing
        
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Simulate very large dataset
        large_data = pd.DataFrame({
            'STATISTIC_DATE': [date(2024, 1, 1)] * 10000,  # 10k rows
            'total_views': list(range(10000))
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = large_data
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {"min_value": 5000, "max_value": 8000}
            }
        }
        
        detector = ThresholdDetector(config)
        
        # Should handle large dataset without memory errors
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 1))
        assert isinstance(results, list)


class TestThresholdDetectorErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_database_connection_error_handling(self, mock_get_connection):
        """Test handling of database connection errors."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        # Mock connection that raises an error
        mock_conn = Mock()
        mock_conn.execute.side_effect = Exception("Database connection failed")
        mock_get_connection.return_value = mock_conn
        
        config = {
            "event_type": "test_event",
            "table": "TEST_TABLE",
            "metrics": [{"column": "test", "alias": "test"}],
            "thresholds": {"test": {"min_value": 100}}
        }
        
        detector = ThresholdDetector(config)
        
        # Should handle database errors gracefully
        with pytest.raises(Exception, match="Database connection failed"):
            detector.detect(date(2024, 1, 1), date(2024, 1, 2))
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_empty_dataset_handling(self, mock_get_connection):
        """Test handling of empty result sets."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Empty dataset
        empty_data = pd.DataFrame(columns=['STATISTIC_DATE', 'total_views'])
        mock_conn.execute.return_value.fetch_pandas_all.return_value = empty_data
        
        config = {
            "event_type": "listing_views",
            "table": "TEST_TABLE",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "thresholds": {
                "total_views": {"min_value": 1000}
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 2))
        
        # Should return empty results, not crash
        assert results == []
    
    def test_invalid_threshold_combination(self):
        """Test validation of invalid threshold combinations."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        # Test conflicting threshold configurations
        invalid_configs = [
            # Min greater than max
            {
                "event_type": "test",
                "table": "TEST_TABLE",
                "metrics": [{"column": "test", "alias": "test"}],
                "thresholds": {"test": {"min_value": 1000, "max_value": 500}}
            },
            # Negative percentage change range
            {
                "event_type": "test", 
                "table": "TEST_TABLE",
                "metrics": [{"column": "test", "alias": "test"}],
                "thresholds": {"test": {"percentage_change_min": 0.5, "percentage_change_max": -0.5}}
            }
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                ThresholdDetector(config)


class TestThresholdDetectorIntegration:
    """Integration tests for complete threshold detection workflow."""
    
    @patch('src.detection.detectors.base_detector.get_snowflake_connection')
    def test_complete_detection_workflow(self, mock_get_connection):
        """Test complete end-to-end detection workflow."""
        if ThresholdDetector is None:
            pytest.skip("ThresholdDetector not implemented yet")
            
        mock_conn = Mock()
        mock_get_connection.return_value = mock_conn
        
        # Realistic data with multiple types of anomalies
        test_data = pd.DataFrame({
            'STATISTIC_DATE': [
                date(2024, 1, 1), date(2024, 1, 2), 
                date(2024, 1, 3), date(2024, 1, 4)
            ],
            'page_views': [100000, 45000, 150000, 120000],  # Day 2: below min, Day 3: above max
            'conversions': [500, 600, 2500, 550]            # Day 3: above max
        })
        
        mock_conn.execute.return_value.fetch_pandas_all.return_value = test_data
        
        config = {
            "event_type": "web_analytics",
            "table": "ANALYTICS.DAILY_METRICS",
            "date_column": "STATISTIC_DATE",
            "metrics": [
                {"column": "PAGE_VIEWS", "alias": "page_views"},
                {"column": "CONVERSIONS", "alias": "conversions"}
            ],
            "thresholds": {
                "page_views": {"min_value": 50000, "max_value": 140000},
                "conversions": {"min_value": 100, "max_value": 2000}
            }
        }
        
        detector = ThresholdDetector(config)
        results = detector.detect(date(2024, 1, 1), date(2024, 1, 4))
        
        # Should detect 3 anomalies total
        assert len(results) == 3
        
        # Verify all anomalies have required fields
        for result in results:
            assert hasattr(result, 'event_type')
            assert hasattr(result, 'metric_name')
            assert hasattr(result, 'detection_date')
            assert hasattr(result, 'actual_value')
            assert hasattr(result, 'expected_value')
            assert hasattr(result, 'deviation_percentage')
            assert hasattr(result, 'severity')
            assert hasattr(result, 'detection_method')
            
            assert result.event_type == "web_analytics"
            assert result.detection_method == "threshold"
            assert result.severity in ["critical", "high", "warning"]


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])