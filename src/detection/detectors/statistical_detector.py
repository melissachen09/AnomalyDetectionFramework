"""
Statistical Detector Implementation for Anomaly Detection Framework.

This module implements statistical-based anomaly detection using Z-score analysis,
moving averages, and seasonal decomposition methods.
Part of ADF-34: Write Test Cases for Statistical Detector (GADF-DETECT-005).

The StatisticalDetector provides:
- Z-score anomaly detection with configurable thresholds
- Moving average baseline comparisons
- Seasonal pattern decomposition and analysis
- Robust handling of small sample sizes
- Performance optimizations for large datasets
- Multiple statistical method support per metric
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

from .base_detector import BaseDetector, DetectionResult, register_detector


@register_detector("statistical")
class StatisticalDetector(BaseDetector):
    """
    Statistical-based anomaly detector.
    
    Detects anomalies using statistical methods including Z-score analysis,
    moving averages, and seasonal decomposition. Supports multiple methods
    per metric and handles edge cases like small sample sizes gracefully.
    
    Configuration Parameters:
        statistical_methods (dict): Per-metric statistical configurations containing:
            - method (str): Statistical method ('zscore', 'moving_average', 'seasonal')
            - window_size (int): Size of historical window for analysis
            - threshold (float): Threshold for anomaly detection
            - seasonal_period (int): Period for seasonal decomposition (seasonal method only)
            - min_samples (int): Minimum samples required for reliable detection
        
    Example Configuration:
        {
            "event_type": "listing_views",
            "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "date_column": "STATISTIC_DATE",
            "metrics": [
                {"column": "NUMBEROFVIEWS", "alias": "total_views"}
            ],
            "statistical_methods": {
                "total_views": {
                    "method": "zscore",
                    "window_size": 30,
                    "threshold": 2.0,
                    "min_samples": 3
                }
            }
        }
    """
    
    # Supported statistical methods
    SUPPORTED_METHODS = {"zscore", "moving_average", "seasonal"}
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize StatisticalDetector with configuration validation.
        
        Args:
            config: Configuration dictionary containing detector parameters
            
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        super().__init__(config)
        
        self._validate_statistical_config()
        self.logger.info(f"StatisticalDetector initialized for event_type: {self.get_config_value('event_type')}")
    
    def _validate_statistical_config(self) -> None:
        """
        Validate statistical configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required configuration sections
        if not self.get_config_value("statistical_methods"):
            raise ValueError("statistical_methods configuration is required")
        
        statistical_methods = self.get_config_value("statistical_methods")
        
        # Validate each metric's statistical configuration
        for metric_alias, method_config in statistical_methods.items():
            if not isinstance(method_config, dict):
                raise ValueError(f"Statistical configuration for metric '{metric_alias}' must be a dictionary")
            
            # Validate statistical method
            method = method_config.get("method")
            if method not in self.SUPPORTED_METHODS:
                raise ValueError(f"Unsupported statistical method '{method}' for metric '{metric_alias}'. "
                               f"Supported methods: {self.SUPPORTED_METHODS}")
            
            # Validate window size
            window_size = method_config.get("window_size", 30)
            if window_size <= 0:
                raise ValueError(f"window_size must be greater than 0 for metric '{metric_alias}'")
            
            # Validate threshold
            threshold = method_config.get("threshold", 2.0)
            if threshold <= 0:
                raise ValueError(f"threshold must be greater than 0 for metric '{metric_alias}'")
            
            # Validate seasonal-specific parameters
            if method == "seasonal":
                seasonal_period = method_config.get("seasonal_period", 7)
                if seasonal_period <= 0:
                    raise ValueError(f"seasonal_period must be greater than 0 for metric '{metric_alias}'")
                
                if window_size < 2 * seasonal_period:
                    self.logger.warning(f"window_size ({window_size}) should be at least 2x seasonal_period ({seasonal_period}) "
                                      f"for reliable seasonal decomposition for metric '{metric_alias}'")
    
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        """
        Run statistical anomaly detection for the specified date range.
        
        Args:
            start_date: Start date for detection analysis (inclusive)
            end_date: End date for detection analysis (inclusive)
        
        Returns:
            List of DetectionResult objects representing detected anomalies
        """
        self.validate_date_range(start_date, end_date)
        
        self.logger.info(f"Running statistical detection from {start_date} to {end_date}")
        
        anomalies = []
        
        # Process each metric
        for metric_config in self.get_config_value("metrics", []):
            metric_alias = metric_config.get("alias")
            
            if not metric_alias or metric_alias not in self.get_config_value("statistical_methods", {}):
                self.logger.debug(f"Skipping metric '{metric_alias}' - no statistical configuration")
                continue
            
            method_config = self.get_config_value("statistical_methods")[metric_alias]
            
            # Apply the configured statistical method
            anomalies.extend(self._detect_metric_anomalies(
                metric_alias, method_config, start_date, end_date
            ))
        
        self.logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies
    
    def _detect_metric_anomalies(
        self,
        metric_alias: str,
        method_config: Dict[str, Any],
        start_date: date,
        end_date: date
    ) -> List[DetectionResult]:
        """
        Detect anomalies for a single metric using configured statistical method.
        
        Args:
            metric_alias: Name of the metric to analyze
            method_config: Statistical method configuration
            start_date: Start date for current period
            end_date: End date for current period
            
        Returns:
            List of anomalies detected for this metric
        """
        method = method_config.get("method")
        
        if method == "zscore":
            return self._detect_zscore_anomalies(metric_alias, method_config, start_date, end_date)
        elif method == "moving_average":
            return self._detect_moving_average_anomalies(metric_alias, method_config, start_date, end_date)
        elif method == "seasonal":
            return self._detect_seasonal_anomalies(metric_alias, method_config, start_date, end_date)
        else:
            self.logger.error(f"Unknown statistical method: {method}")
            return []
    
    def _detect_zscore_anomalies(
        self,
        metric_alias: str,
        method_config: Dict[str, Any],
        start_date: date,
        end_date: date
    ) -> List[DetectionResult]:
        """
        Detect anomalies using Z-score analysis.
        
        Args:
            metric_alias: Name of the metric to analyze
            method_config: Z-score method configuration
            start_date: Start date for current period
            end_date: End date for current period
            
        Returns:
            List of Z-score anomalies detected
        """
        anomalies = []
        window_size = method_config.get("window_size", 30)
        threshold = method_config.get("threshold", 2.0)
        min_samples = method_config.get("min_samples", 3)
        
        # Get historical data for Z-score calculation
        hist_end = start_date - timedelta(days=1)
        hist_start = hist_end - timedelta(days=window_size - 1)
        
        historical_data = self._fetch_metric_data(hist_start, hist_end)
        
        if historical_data.empty:
            self.logger.warning(f"No historical data available for Z-score calculation for {metric_alias}")
            return anomalies
        
        # Get current period data
        current_data = self._fetch_metric_data(start_date, end_date)
        
        if current_data.empty:
            self.logger.warning(f"No current data available for Z-score calculation for {metric_alias}")
            return anomalies
        
        # Filter out null values from historical data
        historical_values = historical_data[metric_alias].dropna()
        
        if len(historical_values) < min_samples:
            self.logger.warning(f"Insufficient historical samples for Z-score calculation for {metric_alias}: "
                              f"{len(historical_values)} < {min_samples}")
            return self._create_insufficient_data_anomalies(
                metric_alias, current_data, len(historical_values), method_config
            )
        
        if len(historical_values) == 0:
            return anomalies
        
        # Calculate historical statistics
        historical_mean = historical_values.mean()
        historical_std = historical_values.std()
        
        date_column = self.get_config_value("date_column", "STATISTIC_DATE")
        
        # Analyze each current value
        for _, row in current_data.iterrows():
            metric_value = row[metric_alias]
            detection_date = row[date_column]
            
            if pd.isna(metric_value):
                continue
            
            # Handle zero standard deviation case
            if historical_std == 0:
                if metric_value != historical_mean:
                    # Any deviation from constant historical values is an anomaly
                    anomalies.append(self._create_statistical_anomaly_result(
                        metric_alias=metric_alias,
                        detection_date=detection_date,
                        actual_value=metric_value,
                        expected_value=historical_mean,
                        method_config=method_config,
                        method="zscore",
                        details={
                            "zscore": float('inf') if metric_value > historical_mean else float('-inf'),
                            "historical_mean": historical_mean,
                            "historical_std": historical_std,
                            "zero_std_deviation": True,
                            "sample_count": len(historical_values)
                        }
                    ))
                continue
            
            # Calculate Z-score
            zscore = (metric_value - historical_mean) / historical_std
            
            # Check if Z-score exceeds threshold
            if abs(zscore) > threshold:
                anomalies.append(self._create_statistical_anomaly_result(
                    metric_alias=metric_alias,
                    detection_date=detection_date,
                    actual_value=metric_value,
                    expected_value=historical_mean,
                    method_config=method_config,
                    method="zscore",
                    details={
                        "zscore": zscore,
                        "historical_mean": historical_mean,
                        "historical_std": historical_std,
                        "sample_count": len(historical_values)
                    }
                ))
        
        return anomalies
    
    def _detect_moving_average_anomalies(
        self,
        metric_alias: str,
        method_config: Dict[str, Any],
        start_date: date,
        end_date: date
    ) -> List[DetectionResult]:
        """
        Detect anomalies using moving average analysis.
        
        Args:
            metric_alias: Name of the metric to analyze
            method_config: Moving average method configuration
            start_date: Start date for current period
            end_date: End date for current period
            
        Returns:
            List of moving average anomalies detected
        """
        anomalies = []
        window_size = method_config.get("window_size", 30)
        threshold = method_config.get("threshold", 0.5)  # 50% deviation threshold
        
        # Get historical data for moving average calculation
        hist_end = start_date - timedelta(days=1)
        hist_start = hist_end - timedelta(days=window_size - 1)
        
        historical_data = self._fetch_metric_data(hist_start, hist_end)
        
        if historical_data.empty:
            self.logger.warning(f"No historical data available for moving average calculation for {metric_alias}")
            return anomalies
        
        # Get current period data
        current_data = self._fetch_metric_data(start_date, end_date)
        
        if current_data.empty:
            self.logger.warning(f"No current data available for moving average calculation for {metric_alias}")
            return anomalies
        
        # Calculate moving average from historical data
        historical_values = historical_data[metric_alias].dropna()
        
        if len(historical_values) == 0:
            return anomalies
        
        moving_average = historical_values.mean()
        
        # Handle zero baseline case
        if moving_average == 0:
            date_column = self.get_config_value("date_column", "STATISTIC_DATE")
            for _, row in current_data.iterrows():
                metric_value = row[metric_alias]
                detection_date = row[date_column]
                
                if pd.isna(metric_value):
                    continue
                
                if metric_value != 0:
                    # Non-zero value when baseline is zero is an anomaly
                    anomalies.append(self._create_statistical_anomaly_result(
                        metric_alias=metric_alias,
                        detection_date=detection_date,
                        actual_value=metric_value,
                        expected_value=moving_average,
                        method_config=method_config,
                        method="moving_average",
                        details={
                            "moving_average": moving_average,
                            "zero_baseline": True,
                            "sample_count": len(historical_values)
                        }
                    ))
            return anomalies
        
        date_column = self.get_config_value("date_column", "STATISTIC_DATE")
        
        # Analyze each current value against moving average
        for _, row in current_data.iterrows():
            metric_value = row[metric_alias]
            detection_date = row[date_column]
            
            if pd.isna(metric_value):
                continue
            
            # Calculate percentage deviation
            percentage_deviation = abs((metric_value - moving_average) / moving_average)
            
            # Check if deviation exceeds threshold
            if percentage_deviation > threshold:
                anomalies.append(self._create_statistical_anomaly_result(
                    metric_alias=metric_alias,
                    detection_date=detection_date,
                    actual_value=metric_value,
                    expected_value=moving_average,
                    method_config=method_config,
                    method="moving_average",
                    details={
                        "moving_average": moving_average,
                        "percentage_deviation": percentage_deviation,
                        "sample_count": len(historical_values)
                    },
                    deviation_percentage=percentage_deviation
                ))
        
        return anomalies
    
    def _detect_seasonal_anomalies(
        self,
        metric_alias: str,
        method_config: Dict[str, Any],
        start_date: date,
        end_date: date
    ) -> List[DetectionResult]:
        """
        Detect anomalies using seasonal decomposition analysis.
        
        Args:
            metric_alias: Name of the metric to analyze
            method_config: Seasonal method configuration
            start_date: Start date for current period
            end_date: End date for current period
            
        Returns:
            List of seasonal anomalies detected
        """
        anomalies = []
        window_size = method_config.get("window_size", 90)
        seasonal_period = method_config.get("seasonal_period", 7)
        threshold = method_config.get("threshold", 2.0)
        
        # Need at least 2 seasonal periods for reliable decomposition
        min_required_samples = 2 * seasonal_period
        
        # Get historical data for seasonal decomposition
        hist_end = start_date - timedelta(days=1)
        hist_start = hist_end - timedelta(days=window_size - 1)
        
        historical_data = self._fetch_metric_data(hist_start, hist_end)
        
        if historical_data.empty or len(historical_data) < min_required_samples:
            self.logger.warning(f"Insufficient data for seasonal decomposition for {metric_alias}: "
                              f"{len(historical_data) if not historical_data.empty else 0} < {min_required_samples}")
            
            # Fall back to Z-score method
            return self._detect_zscore_anomalies_with_fallback(
                metric_alias, method_config, start_date, end_date, "insufficient_seasonal_data"
            )
        
        # Get current period data
        current_data = self._fetch_metric_data(start_date, end_date)
        
        if current_data.empty:
            self.logger.warning(f"No current data available for seasonal analysis for {metric_alias}")
            return anomalies
        
        try:
            # Prepare data for seasonal decomposition
            historical_values = historical_data[metric_alias].dropna()
            
            if len(historical_values) < min_required_samples:
                return self._detect_zscore_anomalies_with_fallback(
                    metric_alias, method_config, start_date, end_date, "insufficient_seasonal_data"
                )
            
            # Suppress statsmodels warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Perform seasonal decomposition
                decomposition = seasonal_decompose(
                    historical_values.values,
                    model='additive',
                    period=seasonal_period,
                    extrapolate_trend='freq'
                )
            
            # Calculate seasonal component statistics
            seasonal_component = decomposition.seasonal[-seasonal_period:]  # Last period's pattern
            residual_std = np.std(decomposition.resid[~np.isnan(decomposition.resid)])
            
            date_column = self.get_config_value("date_column", "STATISTIC_DATE")
            
            # Analyze each current value
            for _, row in current_data.iterrows():
                metric_value = row[metric_alias]
                detection_date = row[date_column]
                
                if pd.isna(metric_value):
                    continue
                
                # Determine position in seasonal cycle
                days_since_start = (detection_date - hist_start).days
                seasonal_index = days_since_start % seasonal_period
                
                # Get expected seasonal value
                seasonal_expected = seasonal_component[seasonal_index] + decomposition.trend[-1]
                
                # Calculate residual
                residual = metric_value - seasonal_expected
                
                # Check if residual exceeds threshold (in terms of standard deviations)
                if residual_std > 0 and abs(residual) > threshold * residual_std:
                    anomalies.append(self._create_statistical_anomaly_result(
                        metric_alias=metric_alias,
                        detection_date=detection_date,
                        actual_value=metric_value,
                        expected_value=seasonal_expected,
                        method_config=method_config,
                        method="seasonal",
                        details={
                            "seasonal_expected": seasonal_expected,
                            "seasonal_residual": residual,
                            "residual_std": residual_std,
                            "seasonal_period": seasonal_period,
                            "seasonal_index": seasonal_index,
                            "sample_count": len(historical_values)
                        }
                    ))
        
        except Exception as e:
            self.logger.error(f"Error in seasonal decomposition for {metric_alias}: {str(e)}")
            # Fall back to Z-score method
            return self._detect_zscore_anomalies_with_fallback(
                metric_alias, method_config, start_date, end_date, "seasonal_decomposition_error"
            )
        
        return anomalies
    
    def _detect_zscore_anomalies_with_fallback(
        self,
        metric_alias: str,
        method_config: Dict[str, Any],
        start_date: date,
        end_date: date,
        fallback_reason: str
    ) -> List[DetectionResult]:
        """
        Fallback to Z-score detection when seasonal decomposition fails.
        
        Args:
            metric_alias: Name of the metric to analyze
            method_config: Original method configuration
            start_date: Start date for current period
            end_date: End date for current period
            fallback_reason: Reason for fallback
            
        Returns:
            List of anomalies detected using Z-score fallback
        """
        # Create temporary Z-score config
        zscore_config = {
            "method": "zscore",
            "window_size": method_config.get("window_size", 30),
            "threshold": method_config.get("threshold", 2.0),
            "min_samples": method_config.get("min_samples", 3)
        }
        
        anomalies = self._detect_zscore_anomalies(metric_alias, zscore_config, start_date, end_date)
        
        # Add fallback information to details
        for anomaly in anomalies:
            anomaly.details[fallback_reason] = True
        
        return anomalies
    
    def _create_insufficient_data_anomalies(
        self,
        metric_alias: str,
        current_data: pd.DataFrame,
        sample_count: int,
        method_config: Dict[str, Any]
    ) -> List[DetectionResult]:
        """
        Create anomaly results for insufficient data scenarios.
        
        Args:
            metric_alias: Name of the metric
            current_data: Current period data
            sample_count: Number of historical samples available
            method_config: Method configuration
            
        Returns:
            List of anomalies indicating insufficient data
        """
        anomalies = []
        date_column = self.get_config_value("date_column", "STATISTIC_DATE")
        
        for _, row in current_data.iterrows():
            metric_value = row[metric_alias]
            detection_date = row[date_column]
            
            if pd.isna(metric_value):
                continue
            
            anomalies.append(self._create_statistical_anomaly_result(
                metric_alias=metric_alias,
                detection_date=detection_date,
                actual_value=metric_value,
                expected_value=0.0,  # Unknown expected value
                method_config=method_config,
                method=method_config.get("method", "unknown"),
                details={
                    "insufficient_samples": True,
                    "sample_count": sample_count,
                    "min_required": method_config.get("min_samples", 3)
                },
                severity="warning"  # Lower severity for insufficient data
            ))
        
        return anomalies
    
    def _fetch_metric_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch metric data from Snowflake for the specified date range.
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            DataFrame containing metric data
        """
        table = self.get_config_value("table")
        date_column = self.get_config_value("date_column", "STATISTIC_DATE")
        metrics = self.get_config_value("metrics", [])
        
        # Build SELECT clause with metric aliases
        select_clauses = [f"{date_column}"]
        for metric in metrics:
            column = metric.get("column")
            alias = metric.get("alias")
            if column and alias:
                select_clauses.append(f"{column} AS {alias}")
        
        select_clause = ", ".join(select_clauses)
        
        query = f"""
        SELECT {select_clause}
        FROM {table}
        WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY {date_column}
        """
        
        self.logger.debug(f"Executing query: {query}")
        
        try:
            cursor = self.snowflake_conn.execute(query)
            return cursor.fetch_pandas_all()
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def _create_statistical_anomaly_result(
        self,
        metric_alias: str,
        detection_date: date,
        actual_value: float,
        expected_value: float,
        method_config: Dict[str, Any],
        method: str,
        details: Dict[str, Any],
        deviation_percentage: Optional[float] = None,
        severity: Optional[str] = None
    ) -> DetectionResult:
        """
        Create a DetectionResult object for a statistical anomaly.
        
        Args:
            metric_alias: Name of the metric
            detection_date: Date when anomaly was detected
            actual_value: Actual metric value
            expected_value: Expected value (baseline)
            method_config: Statistical method configuration
            method: Statistical method used
            details: Method-specific details
            deviation_percentage: Percentage deviation (optional)
            severity: Override severity level (optional)
            
        Returns:
            DetectionResult object
        """
        # Calculate deviation percentage if not provided
        if deviation_percentage is None:
            if expected_value != 0:
                deviation_percentage = abs((actual_value - expected_value) / expected_value)
            else:
                deviation_percentage = 1.0  # 100% deviation for zero expected
        
        # Determine severity based on deviation magnitude or method-specific logic
        if severity is None:
            severity = self._calculate_severity(abs(deviation_percentage), method, details)
        
        # Build complete details dictionary
        complete_details = {
            "method": method,
            "threshold": method_config.get("threshold"),
            "window_size": method_config.get("window_size"),
            "table": self.get_config_value("table"),
            **details
        }
        
        return DetectionResult(
            event_type=self.get_config_value("event_type"),
            metric_name=metric_alias,
            detection_date=detection_date,
            expected_value=expected_value,
            actual_value=actual_value,
            deviation_percentage=deviation_percentage,
            severity=severity,
            detection_method="statistical",
            alert_sent=False,
            details=complete_details
        )
    
    def _calculate_severity(self, deviation_magnitude: float, method: str, details: Dict[str, Any]) -> str:
        """
        Calculate severity level based on deviation magnitude and method-specific criteria.
        
        Args:
            deviation_magnitude: Absolute deviation percentage
            method: Statistical method used
            details: Method-specific details for severity calculation
            
        Returns:
            Severity level string
        """
        # Method-specific severity calculation
        if method == "zscore":
            zscore_abs = abs(details.get("zscore", 0))
            if zscore_abs >= 3.0:  # 3+ standard deviations
                return "critical"
            elif zscore_abs >= 2.5:  # 2.5-3 standard deviations
                return "high"
            else:  # 2-2.5 standard deviations
                return "warning"
        
        # General percentage-based severity for other methods
        if deviation_magnitude >= 0.5:  # 50% or more deviation
            return "critical"
        elif deviation_magnitude >= 0.3:  # 30-49% deviation
            return "high"
        else:  # Less than 30% deviation
            return "warning"