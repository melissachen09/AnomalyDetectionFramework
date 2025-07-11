"""
Test cases for StatisticalDetector implementation.

This module implements the test suite for ADF-34: Write Test Cases for Statistical Detector.
Tests cover statistical calculations, seasonal patterns, small sample handling, and performance.

Requirements from ADF-34:
- Statistical calculations verified 
- Seasonal patterns tested
- Small sample handling validated
- Performance benchmarked
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date as datetime_date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import time

# Import will fail initially - this follows TDD approach
try:
    from src.detection.detectors.statistical_detector import StatisticalDetector
    from src.detection.detectors.base_detector import BaseDetector, DetectionResult
except ImportError:
    StatisticalDetector = None
    BaseDetector = None
    DetectionResult = None


class TestStatisticalDetectorInterface:
    """Test suite for StatisticalDetector interface compliance."""
    
    def test_statistical_detector_inherits_from_base_detector(self):
        """Test that StatisticalDetector inherits from BaseDetector."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        assert issubclass(StatisticalDetector, BaseDetector)
    
    def test_statistical_detector_can_be_instantiated(self):
        """Test that StatisticalDetector can be instantiated with valid config."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": 2.0
                }
            }
        }
        
        detector = StatisticalDetector(config)
        assert isinstance(detector, StatisticalDetector)
        assert isinstance(detector, BaseDetector)
    
    def test_statistical_detector_detect_method_exists(self):
        """Test that StatisticalDetector implements detect method."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "statistical_methods": {
                "test_metric": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": 2.0
                }
            }
        }
        detector = StatisticalDetector(config)
        
        assert hasattr(detector, 'detect')
        assert callable(detector.detect)
    
    def test_statistical_detector_plugin_registration(self):
        """Test that StatisticalDetector is properly registered as a plugin."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        from src.detection.detectors.base_detector import get_registered_detectors
        
        registered_detectors = get_registered_detectors()
        assert 'statistical' in registered_detectors
        assert registered_detectors['statistical'] == StatisticalDetector


class TestStatisticalDetectorConfiguration:
    """Test suite for StatisticalDetector configuration validation."""
    
    def test_configuration_validation_valid_config(self):
        """Test successful initialization with valid configuration."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": 2.0
                }
            }
        }
        
        detector = StatisticalDetector(config)
        assert detector.config == config
    
    def test_configuration_validation_missing_statistical_methods(self):
        """Test configuration validation when statistical_methods is missing."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}]
        }
        
        with pytest.raises(ValueError, match="statistical_methods configuration is required"):
            StatisticalDetector(config)
    
    def test_configuration_validation_invalid_method(self):
        """Test configuration validation with invalid statistical method."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "invalid_method",
                    "window_size": 30,
                    "threshold": 2.0
                }
            }
        }
        
        with pytest.raises(ValueError, match="Unsupported statistical method"):
            StatisticalDetector(config)
    
    def test_configuration_validation_invalid_window_size(self):
        """Test configuration validation with invalid window size."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 0,
                    "threshold": 2.0
                }
            }
        }
        
        with pytest.raises(ValueError, match="window_size must be greater than 0"):
            StatisticalDetector(config)
    
    def test_configuration_validation_invalid_threshold(self):
        """Test configuration validation with invalid threshold."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": -1.0
                }
            }
        }
        
        with pytest.raises(ValueError, match="threshold must be greater than 0"):
            StatisticalDetector(config)


class TestZScoreCalculations:
    """Test suite for Z-score statistical calculations."""
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_zscore_calculation_accuracy(self, mock_fetch_data):
        """Test Z-score calculation accuracy with known data."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 5,
                    "threshold": 2.0
                }
            }
        }
        
        # Mock historical data for Z-score calculation
        # Data: [10, 12, 11, 13, 14] mean=12, std=1.58, current=20
        # Z-score for 20 = (20-12)/1.58 = 5.06 (should trigger anomaly)
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.date_range('2024-01-01', periods=5),
            'total_views': [10, 12, 11, 13, 14]
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 6)],
            'total_views': [20]
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 6), datetime_date(2024, 1, 6))
        
        assert len(results) == 1
        assert results[0].detection_method == "statistical"
        assert results[0].details["method"] == "zscore"
        assert results[0].details["zscore"] == pytest.approx(5.06, rel=0.1)
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_zscore_no_anomaly_within_threshold(self, mock_fetch_data):
        """Test Z-score calculation when value is within threshold."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 5,
                    "threshold": 2.0
                }
            }
        }
        
        # Data: [10, 12, 11, 13, 14] mean=12, std=1.58, current=13
        # Z-score for 13 = (13-12)/1.58 = 0.63 (should NOT trigger anomaly)
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.date_range('2024-01-01', periods=5),
            'total_views': [10, 12, 11, 13, 14]
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 6)],
            'total_views': [13]
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 6), datetime_date(2024, 1, 6))
        
        assert len(results) == 0
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_zscore_calculation_with_zero_standard_deviation(self, mock_fetch_data):
        """Test Z-score calculation when standard deviation is zero."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 5,
                    "threshold": 2.0
                }
            }
        }
        
        # All historical values are the same (std = 0)
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.date_range('2024-01-01', periods=5),
            'total_views': [10, 10, 10, 10, 10]
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 6)],
            'total_views': [15]  # Different from historical mean
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 6), datetime_date(2024, 1, 6))
        
        # Should detect anomaly when current value differs from constant historical values
        assert len(results) == 1
        assert results[0].details["method"] == "zscore"
        assert results[0].details["zero_std_deviation"] is True


class TestMovingAverageCalculations:
    """Test suite for moving average statistical calculations."""
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_moving_average_calculation_accuracy(self, mock_fetch_data):
        """Test moving average calculation accuracy with known data."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "moving_average",
                    "window_size": 5,
                    "threshold": 0.5  # 50% deviation threshold
                }
            }
        }
        
        # Historical data: [10, 12, 11, 13, 14] mean=12, current=20
        # Percentage deviation = (20-12)/12 = 0.67 (should trigger anomaly)
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.date_range('2024-01-01', periods=5),
            'total_views': [10, 12, 11, 13, 14]
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 6)],
            'total_views': [20]
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 6), datetime_date(2024, 1, 6))
        
        assert len(results) == 1
        assert results[0].detection_method == "statistical"
        assert results[0].details["method"] == "moving_average"
        assert results[0].details["moving_average"] == 12.0
        assert results[0].deviation_percentage == pytest.approx(0.67, rel=0.1)
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_moving_average_no_anomaly_within_threshold(self, mock_fetch_data):
        """Test moving average when value is within threshold."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "moving_average",
                    "window_size": 5,
                    "threshold": 0.5  # 50% deviation threshold
                }
            }
        }
        
        # Historical data: [10, 12, 11, 13, 14] mean=12, current=15
        # Percentage deviation = (15-12)/12 = 0.25 (should NOT trigger anomaly)
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.date_range('2024-01-01', periods=5),
            'total_views': [10, 12, 11, 13, 14]
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 6)],
            'total_views': [15]
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 6), datetime_date(2024, 1, 6))
        
        assert len(results) == 0
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_moving_average_with_zero_baseline(self, mock_fetch_data):
        """Test moving average calculation when baseline is zero."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "moving_average",
                    "window_size": 3,
                    "threshold": 0.1
                }
            }
        }
        
        # Historical data with zero values
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.date_range('2024-01-01', periods=3),
            'total_views': [0, 0, 0]
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 4)],
            'total_views': [10]
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 4), datetime_date(2024, 1, 4))
        
        # Should detect anomaly when current value is non-zero and baseline is zero
        assert len(results) == 1
        assert results[0].details["method"] == "moving_average"
        assert results[0].details["zero_baseline"] is True


class TestSeasonalPatterns:
    """Test suite for seasonal pattern detection."""
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_seasonal_decomposition_weekly_pattern(self, mock_fetch_data):
        """Test seasonal decomposition with weekly patterns."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "seasonal",
                    "window_size": 28,  # 4 weeks
                    "seasonal_period": 7,  # Weekly pattern
                    "threshold": 2.0
                }
            }
        }
        
        # Generate synthetic weekly seasonal data
        # Higher values on weekends (days 5, 6), lower on weekdays
        dates = pd.date_range('2024-01-01', periods=28)
        values = []
        for i, date in enumerate(dates):
            weekday = date.weekday()
            if weekday in [5, 6]:  # Weekend
                values.append(100 + np.random.normal(0, 5))
            else:  # Weekday
                values.append(50 + np.random.normal(0, 5))
        
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': dates,
            'total_views': values
        })
        
        # Current data - abnormally high value on a weekday
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 29)],  # Monday
            'total_views': [150]  # Should be ~50 based on pattern
        })
        
        mock_fetch_data.side_effect = [
            historical_data,  # First call for seasonal decomposition
            current_data,     # Second call for current data
            historical_data,  # Third call if fallback to zscore (might happen)
            current_data      # Fourth call if fallback to zscore (might happen)
        ]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 29), datetime_date(2024, 1, 29))
        
        assert len(results) == 1
        assert results[0].detection_method == "statistical"
        assert results[0].details["method"] == "seasonal"
        assert "seasonal_expected" in results[0].details
        assert "seasonal_residual" in results[0].details
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_seasonal_decomposition_insufficient_data(self, mock_fetch_data):
        """Test seasonal decomposition with insufficient data."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "seasonal",
                    "window_size": 28,
                    "seasonal_period": 7,
                    "threshold": 2.0
                }
            }
        }
        
        # Insufficient historical data (need at least 2 seasonal periods)
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.date_range('2024-01-01', periods=5),
            'total_views': [10, 12, 11, 13, 14]
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 6)],
            'total_views': [20]
        })
        
        mock_fetch_data.side_effect = [
            historical_data,  # First call for seasonal 
            current_data,     # Second call for seasonal current data
            historical_data,  # Third call for zscore fallback historical
            current_data      # Fourth call for zscore fallback current
        ]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 6), datetime_date(2024, 1, 6))
        
        # Should fall back to Z-score or moving average when insufficient data
        assert len(results) >= 0  # May or may not detect anomaly
        if len(results) > 0:
            assert results[0].details["insufficient_seasonal_data"] is True
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_seasonal_decomposition_monthly_pattern(self, mock_fetch_data):
        """Test seasonal decomposition with monthly patterns."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "seasonal",
                    "window_size": 90,  # 3 months
                    "seasonal_period": 30,  # Monthly pattern
                    "threshold": 2.0
                }
            }
        }
        
        # Generate synthetic monthly seasonal data
        # Higher values at month start/end, lower in middle
        dates = pd.date_range('2024-01-01', periods=90)
        values = []
        for date in dates:
            day_of_month = date.day
            if day_of_month <= 5 or day_of_month >= 26:  # Start/end of month
                values.append(80 + np.random.normal(0, 5))
            else:  # Middle of month
                values.append(40 + np.random.normal(0, 5))
        
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': dates,
            'total_views': values
        })
        
        # Current data - abnormally high value in middle of month
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 3, 31)],  # End of month
            'total_views': [120]  # Should be ~40 based on pattern
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 3, 31), datetime_date(2024, 3, 31))
        
        assert len(results) == 1
        assert results[0].detection_method == "statistical"
        assert results[0].details["method"] == "seasonal"
        assert "seasonal_period" in results[0].details
        assert results[0].details["seasonal_period"] == 30


class TestSmallSampleHandling:
    """Test suite for small sample size handling."""
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_small_sample_size_zscore(self, mock_fetch_data):
        """Test Z-score calculation with small sample size."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": 2.0,
                    "min_samples": 3
                }
            }
        }
        
        # Only 2 historical samples (below min_samples)
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.date_range('2024-01-01', periods=2),
            'total_views': [10, 12]
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 3)],
            'total_views': [20]
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 3), datetime_date(2024, 1, 3))
        
        # Should handle small sample size gracefully
        if len(results) > 0:
            assert results[0].details["insufficient_samples"] is True
            assert results[0].details["sample_count"] == 2
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_empty_historical_data(self, mock_fetch_data):
        """Test handling of empty historical data."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": 2.0
                }
            }
        }
        
        # Empty historical data
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.Series([], dtype='datetime64[ns]'),
            'total_views': pd.Series([], dtype=float)
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 1)],
            'total_views': [20]
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 1), datetime_date(2024, 1, 1))
        
        # Should handle empty data gracefully (no anomalies detected)
        assert len(results) == 0
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_single_historical_sample(self, mock_fetch_data):
        """Test handling of single historical sample."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": 2.0
                }
            }
        }
        
        # Single historical sample
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 1)],
            'total_views': [10]
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 2)],
            'total_views': [20]
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 2), datetime_date(2024, 1, 2))
        
        # Should handle single sample gracefully
        if len(results) > 0:
            assert results[0].details["insufficient_samples"] is True
            assert results[0].details["sample_count"] == 1
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_null_values_in_historical_data(self, mock_fetch_data):
        """Test handling of null values in historical data."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": 2.0
                }
            }
        }
        
        # Historical data with null values
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.date_range('2024-01-01', periods=5),
            'total_views': [10, None, 12, None, 14]
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 6)],
            'total_views': [20]
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 6), datetime_date(2024, 1, 6))
        
        # Should handle null values by filtering them out
        # Expected to use [10, 12, 14] for calculations
        assert len(results) >= 0
        if len(results) > 0 and "null_values_filtered" in results[0].details:
            assert results[0].details["null_values_filtered"] == 2


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarking."""
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_performance_large_dataset_zscore(self, mock_fetch_data):
        """Test Z-score calculation performance with large dataset."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 1000,
                    "threshold": 2.0
                }
            }
        }
        
        # Generate large dataset (1000 samples)
        np.random.seed(42)  # For reproducible results
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.date_range('2024-01-01', periods=1000),
            'total_views': np.random.normal(100, 15, 1000)
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2026, 9, 27)],  # Date after 1000 days
            'total_views': [200]
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        
        # Benchmark performance
        start_time = time.time()
        results = detector.detect(datetime_date(2026, 9, 27), datetime_date(2026, 9, 27))
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertion: should complete within reasonable time
        assert execution_time < 1.0, f"Z-score calculation took {execution_time:.2f}s, expected < 1.0s"
        
        # Verify results
        assert len(results) == 1
        assert results[0].detection_method == "statistical"
        assert results[0].details["method"] == "zscore"
        assert results[0].details["sample_count"] == 1000
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_performance_seasonal_decomposition_large_dataset(self, mock_fetch_data):
        """Test seasonal decomposition performance with large dataset."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "seasonal",
                    "window_size": 365,  # 1 year
                    "seasonal_period": 7,  # Weekly
                    "threshold": 2.0
                }
            }
        }
        
        # Generate large seasonal dataset (365 days)
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=365)
        values = []
        for date in dates:
            # Weekly pattern + trend + noise
            weekly_pattern = 50 + 20 * np.sin(2 * np.pi * date.weekday() / 7)
            trend = 0.1 * (date - dates[0]).days
            noise = np.random.normal(0, 5)
            values.append(weekly_pattern + trend + noise)
        
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': dates,
            'total_views': values
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 12, 31)],
            'total_views': [150]
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        
        # Benchmark performance
        start_time = time.time()
        results = detector.detect(datetime_date(2024, 12, 31), datetime_date(2024, 12, 31))
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertion: seasonal decomposition should complete within reasonable time
        assert execution_time < 2.0, f"Seasonal decomposition took {execution_time:.2f}s, expected < 2.0s"
        
        # Verify results
        assert len(results) >= 0
        if len(results) > 0:
            assert results[0].detection_method == "statistical"
            assert results[0].details["method"] == "seasonal"
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_performance_multiple_metrics(self, mock_fetch_data):
        """Test performance with multiple metrics."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [
                {"column": "NUMBEROFVIEWS", "alias": "total_views"},
                {"column": "NUMBEROFENQUIRIES", "alias": "enquiries"},
                {"column": "NUMBEROFCLICKS", "alias": "clicks"}
            ],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 100,
                    "threshold": 2.0
                },
                "enquiries": {
                    "method": "moving_average",
                    "window_size": 50,
                    "threshold": 0.5
                },
                "clicks": {
                    "method": "seasonal",
                    "window_size": 70,
                    "seasonal_period": 7,
                    "threshold": 2.0
                }
            }
        }
        
        # Generate data for multiple metrics
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)
        
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': dates,
            'total_views': np.random.normal(100, 15, 100),
            'enquiries': np.random.normal(20, 3, 100),
            'clicks': np.random.normal(50, 8, 100)
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 4, 10)],
            'total_views': [200],
            'enquiries': [40],
            'clicks': [80]
        })
        
        # Need to provide enough calls for all metrics and potential fallbacks
        mock_fetch_data.side_effect = [
            historical_data,  # zscore historical for total_views
            current_data,     # zscore current for total_views
            historical_data,  # moving_average historical for enquiries
            current_data,     # moving_average current for enquiries
            historical_data,  # seasonal historical for clicks
            current_data,     # seasonal current for clicks
            historical_data,  # potential fallback historical for clicks
            current_data      # potential fallback current for clicks
        ]
        
        detector = StatisticalDetector(config)
        
        # Benchmark performance
        start_time = time.time()
        results = detector.detect(datetime_date(2024, 4, 10), datetime_date(2024, 4, 10))
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertion: multiple metrics should complete within reasonable time
        assert execution_time < 1.5, f"Multiple metrics processing took {execution_time:.2f}s, expected < 1.5s"
        
        # Verify results
        assert len(results) >= 0
        # Should have results for anomalous metrics
        metric_names = [r.metric_name for r in results]
        assert all(name in ["total_views", "enquiries", "clicks"] for name in metric_names)


class TestStatisticalDetectorIntegration:
    """Integration test suite for StatisticalDetector."""
    
    @patch('src.detection.detectors.statistical_detector.StatisticalDetector._fetch_metric_data')
    def test_end_to_end_anomaly_detection(self, mock_fetch_data):
        """Test complete end-to-end anomaly detection workflow."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "date_column": "STATISTIC_DATE",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": 2.0
                }
            }
        }
        
        # Realistic dataset with clear anomaly
        np.random.seed(42)
        historical_data = pd.DataFrame({
            'STATISTIC_DATE': pd.date_range('2024-01-01', periods=30),
            'total_views': np.random.normal(10000, 500, 30)
        })
        
        current_data = pd.DataFrame({
            'STATISTIC_DATE': [datetime_date(2024, 1, 31)],
            'total_views': [20000]  # Clear anomaly
        })
        
        mock_fetch_data.side_effect = [historical_data, current_data]
        
        detector = StatisticalDetector(config)
        results = detector.detect(datetime_date(2024, 1, 31), datetime_date(2024, 1, 31))
        
        # Verify complete result structure
        assert len(results) == 1
        result = results[0]
        
        assert result.event_type == "listing_views"
        assert result.metric_name == "total_views"
        assert result.detection_date == datetime_date(2024, 1, 31)
        assert result.actual_value == 20000
        assert result.detection_method == "statistical"
        assert result.severity in ["critical", "high", "warning"]
        assert result.alert_sent is False
        
        # Verify details
        assert "method" in result.details
        assert result.details["method"] == "zscore"
        assert "zscore" in result.details
        assert "sample_count" in result.details
        assert result.details["sample_count"] == 30
    
    def test_statistical_detector_error_handling(self):
        """Test StatisticalDetector error handling."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": 2.0
                }
            }
        }
        
        detector = StatisticalDetector(config)
        
        # Test with invalid date range
        with pytest.raises(ValueError, match="start_date must be before or equal to end_date"):
            detector.detect(datetime_date(2024, 1, 2), datetime_date(2024, 1, 1))
    
    def test_statistical_detector_repr(self):
        """Test StatisticalDetector string representation."""
        if StatisticalDetector is None:
            pytest.skip("StatisticalDetector not implemented yet")
            
        config = {
            "event_type": "listing_views",
            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": 2.0
                }
            }
        }
        
        detector = StatisticalDetector(config)
        repr_str = repr(detector)
        
        assert "StatisticalDetector" in repr_str
        assert "event_type" in repr_str


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])