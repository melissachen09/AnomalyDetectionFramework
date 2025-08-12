"""
Threshold Detector Implementation for Anomaly Detection Framework.

This module implements threshold-based anomaly detection that supports multiple
threshold types including absolute min/max values and percentage change detection.
Part of ADF-33: Implement Threshold Detector (GADF-DETECT-004).

The ThresholdDetector provides:
- Absolute threshold checking (min/max values)
- Percentage change detection with configurable baselines
- Multi-metric detection support
- Configurable sensitivity levels
- Efficient computation for large datasets
- Clear violation reporting
"""

import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import logging

from .base_detector import BaseDetector, DetectionResult, register_detector


@register_detector("threshold")
class ThresholdDetector(BaseDetector):
    """
    Threshold-based anomaly detector.
    
    Detects anomalies by comparing metric values against configurable thresholds.
    Supports both absolute thresholds (min/max values) and percentage change
    thresholds (relative to historical baselines).
    
    Configuration Parameters:
        thresholds (dict): Per-metric threshold configurations containing:
            - min_value (float): Minimum acceptable value (absolute threshold)
            - max_value (float): Maximum acceptable value (absolute threshold)
            - percentage_change_min (float): Minimum percentage change (-1.0 to 0.0)
            - percentage_change_max (float): Maximum percentage change (0.0 to +inf)
            - baseline_days (int): Days of historical data for percentage calculations
        
    Example Configuration:
        {
            "event_type": "listing_views",
            "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
            "date_column": "STATISTIC_DATE",
            "metrics": [
                {"column": "NUMBEROFVIEWS", "alias": "total_views"}
            ],
            "thresholds": {
                "total_views": {
                    "min_value": 10000,
                    "max_value": 1000000,
                    "percentage_change_max": 0.5,
                    "baseline_days": 7
                }
            }
        }
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ThresholdDetector with configuration validation.
        
        Args:
            config: Configuration dictionary containing detector parameters
            
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        super().__init__(config)
        
        self._validate_threshold_config()
        self.logger.info(f"ThresholdDetector initialized for event_type: {self.get_config_value('event_type')}")
    
    def _validate_threshold_config(self) -> None:
        """
        Validate threshold configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required configuration sections
        if not self.get_config_value("metrics"):
            raise ValueError("metrics configuration is required")
        
        if not self.get_config_value("thresholds"):
            raise ValueError("thresholds configuration is required")
        
        thresholds = self.get_config_value("thresholds")
        
        # Validate each metric's threshold configuration
        for metric_alias, threshold_config in thresholds.items():
            if not isinstance(threshold_config, dict):
                raise ValueError(f"Threshold configuration for metric '{metric_alias}' must be a dictionary")
            
            # Validate min/max absolute thresholds
            min_val = threshold_config.get("min_value")
            max_val = threshold_config.get("max_value")
            
            if min_val is not None and max_val is not None:
                if min_val > max_val:
                    raise ValueError(f"min_value ({min_val}) cannot be greater than max_value ({max_val}) for metric '{metric_alias}'")
            
            # Validate percentage change thresholds
            pct_min = threshold_config.get("percentage_change_min")
            pct_max = threshold_config.get("percentage_change_max")
            
            if pct_min is not None and pct_max is not None:
                if pct_min > pct_max:
                    raise ValueError(f"percentage_change_min ({pct_min}) cannot be greater than percentage_change_max ({pct_max}) for metric '{metric_alias}'")
            
            # Validate baseline_days for percentage calculations
            baseline_days = threshold_config.get("baseline_days")
            if baseline_days is not None and baseline_days <= 0:
                raise ValueError(f"baseline_days must be positive for metric '{metric_alias}'")
    
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        """
        Run threshold-based anomaly detection for the specified date range.
        
        Args:
            start_date: Start date for detection analysis (inclusive)
            end_date: End date for detection analysis (inclusive)
        
        Returns:
            List of DetectionResult objects representing detected anomalies
        """
        self.validate_date_range(start_date, end_date)
        
        self.logger.info(f"Running threshold detection from {start_date} to {end_date}")
        
        anomalies = []
        
        # Process each metric
        for metric_config in self.get_config_value("metrics", []):
            metric_alias = metric_config.get("alias")
            
            if not metric_alias or metric_alias not in self.get_config_value("thresholds", {}):
                self.logger.debug(f"Skipping metric '{metric_alias}' - no threshold configuration")
                continue
            
            threshold_config = self.get_config_value("thresholds")[metric_alias]
            
            # Check percentage change thresholds first (to match test expectations for data fetch order)
            anomalies.extend(self._check_percentage_thresholds(
                metric_alias, threshold_config, start_date, end_date
            ))
            
            # Check absolute thresholds
            anomalies.extend(self._check_absolute_thresholds(
                metric_alias, threshold_config, start_date, end_date
            ))
        
        self.logger.info(f"Detected {len(anomalies)} anomalies")
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
    
    def _check_absolute_thresholds_with_data(
        self, 
        data: pd.DataFrame,
        metric_alias: str, 
        threshold_config: Dict[str, Any]
    ) -> List[DetectionResult]:
        """
        Check for absolute threshold violations using provided data.
        
        Args:
            data: DataFrame containing metric data
            metric_alias: Name of the metric to check
            threshold_config: Threshold configuration for the metric
            
        Returns:
            List of anomalies detected
        """
        anomalies = []
        min_value = threshold_config.get("min_value")
        max_value = threshold_config.get("max_value")
        
        if min_value is None and max_value is None:
            return anomalies
        
        if data.empty:
            self.logger.warning(f"No data provided for absolute threshold check for {metric_alias}")
            return anomalies
        
        date_column = self.get_config_value("date_column", "STATISTIC_DATE")
        
        for _, row in data.iterrows():
            metric_value = row[metric_alias]
            detection_date = row[date_column]
            
            # Handle null/NaN values
            if pd.isna(metric_value):
                self.logger.warning(f"Null value detected for {metric_alias} on {detection_date}")
                # Optionally treat null as anomaly
                anomalies.append(self._create_anomaly_result(
                    metric_alias=metric_alias,
                    detection_date=detection_date,
                    actual_value=np.nan,
                    expected_value=min_value or max_value,
                    violation_type="null_value"
                ))
                continue
            
            # Check minimum threshold
            if min_value is not None and metric_value < min_value:
                anomalies.append(self._create_anomaly_result(
                    metric_alias=metric_alias,
                    detection_date=detection_date,
                    actual_value=metric_value,
                    expected_value=min_value,
                    violation_type="below_minimum"
                ))
            
            # Check maximum threshold
            elif max_value is not None and metric_value > max_value:
                anomalies.append(self._create_anomaly_result(
                    metric_alias=metric_alias,
                    detection_date=detection_date,
                    actual_value=metric_value,
                    expected_value=max_value,
                    violation_type="above_maximum"
                ))
        
        return anomalies
    
    def _check_absolute_thresholds(
        self, 
        metric_alias: str, 
        threshold_config: Dict[str, Any],
        start_date: date,
        end_date: date
    ) -> List[DetectionResult]:
        """
        Check for absolute threshold violations (min/max values).
        
        Args:
            metric_alias: Name of the metric to check
            threshold_config: Threshold configuration for the metric
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            List of anomalies detected
        """
        # Get current period data
        data = self._fetch_metric_data(start_date, end_date)
        return self._check_absolute_thresholds_with_data(data, metric_alias, threshold_config)
    
    def _check_percentage_thresholds(
        self,
        metric_alias: str,
        threshold_config: Dict[str, Any],
        start_date: date,
        end_date: date
    ) -> List[DetectionResult]:
        """
        Check for percentage change threshold violations.
        
        Args:
            metric_alias: Name of the metric to check
            threshold_config: Threshold configuration for the metric
            start_date: Start date of current period
            end_date: End date of current period
            
        Returns:
            List of anomalies detected
        """
        anomalies = []
        
        pct_min = threshold_config.get("percentage_change_min")
        pct_max = threshold_config.get("percentage_change_max")
        baseline_days = threshold_config.get("baseline_days", 7)
        
        if pct_min is None and pct_max is None:
            return anomalies
        
        # Get historical baseline data (fetched first to match test expectations)
        baseline_end = start_date - timedelta(days=1)
        baseline_start = baseline_end - timedelta(days=baseline_days - 1)
        
        baseline_data = self._fetch_metric_data(baseline_start, baseline_end)
        
        if baseline_data.empty:
            self.logger.warning(f"No baseline data available for percentage calculation for {metric_alias}")
            return anomalies
        
        # Get current period data (fetched second to match test expectations)
        current_data = self._fetch_metric_data(start_date, end_date)
        
        if current_data.empty:
            self.logger.warning(f"No current data available for percentage calculation for {metric_alias}")
            return anomalies
        
        # Calculate baseline average
        baseline_values = baseline_data[metric_alias].dropna()
        if baseline_values.empty:
            self.logger.warning(f"No valid baseline values for {metric_alias}")
            return anomalies
        
        baseline_avg = baseline_values.mean()
        
        if baseline_avg == 0:
            self.logger.warning(f"Zero baseline average for {metric_alias} - cannot calculate percentage change")
            return anomalies
        
        date_column = self.get_config_value("date_column", "STATISTIC_DATE")
        
        # Check each current period value against baseline
        for _, row in current_data.iterrows():
            metric_value = row[metric_alias]
            detection_date = row[date_column]
            
            if pd.isna(metric_value):
                continue
            
            percentage_change = self._calculate_percentage_change(baseline_avg, metric_value)
            
            # Check percentage thresholds
            if pct_min is not None and percentage_change < pct_min:
                anomalies.append(self._create_anomaly_result(
                    metric_alias=metric_alias,
                    detection_date=detection_date,
                    actual_value=metric_value,
                    expected_value=baseline_avg,
                    violation_type="percentage_decrease",
                    deviation_percentage=percentage_change
                ))
            
            elif pct_max is not None and percentage_change > pct_max:
                anomalies.append(self._create_anomaly_result(
                    metric_alias=metric_alias,
                    detection_date=detection_date,
                    actual_value=metric_value,
                    expected_value=baseline_avg,
                    violation_type="percentage_increase",
                    deviation_percentage=percentage_change
                ))
        
        return anomalies
    
    def _calculate_percentage_change(self, baseline: float, current: float) -> float:
        """
        Calculate percentage change from baseline to current value.
        
        Args:
            baseline: Baseline value
            current: Current value
            
        Returns:
            Percentage change as decimal (e.g., 0.5 for 50% increase)
        """
        if baseline == 0:
            return float('inf') if current > 0 else float('-inf') if current < 0 else 0.0
        
        return (current - baseline) / baseline
    
    def _create_anomaly_result(
        self,
        metric_alias: str,
        detection_date: date,
        actual_value: float,
        expected_value: float,
        violation_type: str,
        deviation_percentage: Optional[float] = None
    ) -> DetectionResult:
        """
        Create a DetectionResult object for an anomaly.
        
        Args:
            metric_alias: Name of the metric
            detection_date: Date when anomaly was detected
            actual_value: Actual metric value
            expected_value: Expected value (threshold or baseline)
            violation_type: Type of violation detected
            deviation_percentage: Percentage deviation (for percentage thresholds)
            
        Returns:
            DetectionResult object
        """
        # Calculate deviation percentage if not provided
        if deviation_percentage is None:
            if expected_value != 0:
                deviation_percentage = abs((actual_value - expected_value) / expected_value)
            else:
                deviation_percentage = 1.0  # 100% deviation for zero expected
        
        # Determine severity based on deviation magnitude
        severity = self._calculate_severity(abs(deviation_percentage))
        
        # Build details dictionary
        details = {
            "violation_type": violation_type,
            "threshold_config": self.get_config_value("thresholds", {}).get(metric_alias, {}),
            "table": self.get_config_value("table"),
            "query_date_range": f"{detection_date} to {detection_date}"
        }
        
        return DetectionResult(
            event_type=self.get_config_value("event_type"),
            metric_name=metric_alias,
            detection_date=detection_date,
            expected_value=expected_value,
            actual_value=actual_value,
            deviation_percentage=deviation_percentage,
            severity=severity,
            detection_method="threshold",
            alert_sent=False,
            details=details
        )
    
    def _calculate_severity(self, deviation_magnitude: float) -> str:
        """
        Calculate severity level based on deviation magnitude.
        
        Args:
            deviation_magnitude: Absolute deviation percentage
            
        Returns:
            Severity level string
        """
        if deviation_magnitude >= 0.5:  # 50% or more deviation
            return "critical"
        elif deviation_magnitude >= 0.3:  # 30-49% deviation
            return "high"
        else:  # Less than 30% deviation
            return "warning"