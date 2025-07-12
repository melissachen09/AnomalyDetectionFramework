# Base Detector Interface Implementation

## Overview

This module provides the foundational interface and utilities for the Anomaly Detection Framework's plugin architecture. It includes:

- **BaseDetector**: Abstract base class that all detection plugins must inherit from
- **DetectionResult**: Dataclass for standardizing detection results  
- **DetectorRegistry**: Registry system for managing and discovering detector plugins
- **Timing and logging utilities**: Common functionality for all detectors

## Architecture

### BaseDetector Abstract Base Class

The `BaseDetector` class provides a standardized interface that all detection plugins must implement:

```python
from detection.base_detector import BaseDetector, DetectionResult
from datetime import date
from typing import List, Dict, Any

class MyDetector(BaseDetector):
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        # Implement detection logic
        pass
    
    def validate_config(self) -> bool:
        # Validate configuration parameters
        return True
```

#### Required Methods

- **`detect(start_date, end_date)`**: Core detection logic that returns list of anomalies
- **`validate_config()`**: Validates the detector's configuration

#### Built-in Features

- **Automatic logging**: Each detector gets a configured logger
- **Timing decorators**: Automatic execution time tracking
- **Configuration management**: Standardized config access patterns
- **Error handling**: Structured error logging

### DetectionResult Dataclass

Standardizes the output format for all detectors:

```python
result = DetectionResult(
    metric_name="listing_views",
    expected_value=10000.0,
    actual_value=5000.0,
    deviation_percentage=0.5,  # 50% deviation
    severity="critical",
    detection_method="threshold",
    timestamp=datetime.now(),
    alert_sent=False
)
```

#### Validation

- **Severity levels**: Only accepts "critical", "high", "warning", "info"
- **Data consistency**: Validates all required fields are present
- **Serialization**: Built-in methods for converting to/from dictionaries

### Detector Registry

Manages discovery and instantiation of detector plugins:

```python
from detection.base_detector import detector_registry

# Register a detector
@detector_registry.register("threshold")
class ThresholdDetector(BaseDetector):
    # Implementation...

# Use registered detectors
detector_class = detector_registry.get_detector("threshold")
detector = detector_class(config)

# Or create directly
detector = detector_registry.create_detector("threshold", config)
```

## Usage Examples

### 1. Creating a Simple Threshold Detector

```python
@detector_registry.register("threshold")
class ThresholdDetector(BaseDetector):
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        # Get current metric value (implementation would query data source)
        current_value = self.get_metric_value(start_date, end_date)
        expected_value = self.config["expected_value"]
        threshold = self.config["threshold"]
        
        deviation = abs(current_value - expected_value) / expected_value
        
        if deviation > threshold:
            severity = "critical" if deviation > 0.5 else "high"
            return [DetectionResult(
                metric_name=self.config["metric_name"],
                expected_value=expected_value,
                actual_value=current_value,
                deviation_percentage=deviation,
                severity=severity,
                detection_method="threshold",
                timestamp=datetime.now(),
                alert_sent=False
            )]
        
        return []
    
    def validate_config(self) -> bool:
        required = ["metric_name", "expected_value", "threshold"]
        return all(key in self.config for key in required)
```

### 2. Using in Detection Pipeline

```python
# Configuration for multiple detectors
detector_configs = {
    "listing_views": {
        "detector_type": "threshold",
        "metric_name": "listing_views",
        "expected_value": 50000,
        "threshold": 0.3
    },
    "enquiries": {
        "detector_type": "statistical",
        "metric_name": "enquiries", 
        "lookback_days": 7,
        "z_score_threshold": 2.5
    }
}

# Run detection pipeline
all_results = []
for metric, config in detector_configs.items():
    detector_type = config.pop("detector_type")
    detector = detector_registry.create_detector(detector_type, config)
    
    results = detector.detect(yesterday, today)
    all_results.extend(results)

# Process results
critical_alerts = [r for r in all_results if r.severity == "critical"]
```

## Configuration Standards

### Required Configuration Keys

All detectors should support these standard configuration keys:

- **`metric_name`**: Name of the metric being monitored
- **`data_source`**: Information about where to get the data
- **`detection_frequency`**: How often detection should run ("daily", "hourly", etc.)

### Detector-Specific Configuration

Each detector type can define additional required configuration:

```python
# Threshold detector
{
    "metric_name": "views",
    "expected_value": 10000,
    "threshold": 0.2,  # 20% deviation threshold
    "min_value": 1000,  # Optional minimum bound
    "max_value": 100000  # Optional maximum bound
}

# Statistical detector  
{
    "metric_name": "enquiries",
    "lookback_days": 14,
    "z_score_threshold": 3.0,
    "seasonal_adjustment": True
}
```

## Integration with Airflow

The base detector interface is designed to work seamlessly with Airflow DAGs:

```python
from airflow.operators.python import PythonOperator
from detection.base_detector import detector_registry

def run_detection_task(**context):
    config = context['dag_run'].conf
    detector_name = config['detector_type']
    
    detector = detector_registry.create_detector(detector_name, config)
    results = detector.detect(
        start_date=context['ds'],
        end_date=context['ds']
    )
    
    # Store results in Snowflake or trigger alerts
    return len(results)

detect_task = PythonOperator(
    task_id='detect_anomalies',
    python_callable=run_detection_task,
    dag=dag
)
```

## Error Handling

The base detector provides structured error handling:

```python
# Errors are automatically logged with context
class MyDetector(BaseDetector):
    def detect(self, start_date, end_date):
        try:
            # Detection logic that might fail
            return self.run_analysis()
        except DataSourceError as e:
            self.logger.error(f"Data source unavailable: {e}")
            raise  # Re-raise for Airflow to handle
        except ConfigurationError as e:
            self.logger.error(f"Configuration issue: {e}")
            raise
```

## Testing

### Unit Testing Detectors

```python
def test_threshold_detector():
    config = {
        "metric_name": "test_metric",
        "expected_value": 100,
        "threshold": 0.2
    }
    
    detector = detector_registry.create_detector("threshold", config)
    
    # Mock the data source
    with patch.object(detector, 'get_metric_value', return_value=150):
        results = detector.detect(date.today(), date.today())
        
        assert len(results) == 1
        assert results[0].severity == "high"
        assert results[0].deviation_percentage == 0.5
```

### Integration Testing

```python
def test_detector_pipeline():
    # Test complete pipeline with real configuration
    config = load_test_config("listing_views.yaml")
    detector = detector_registry.create_detector("threshold", config)
    
    # Test with known data
    results = detector.detect(
        date(2024, 1, 1), 
        date(2024, 1, 1)
    )
    
    # Verify results structure
    for result in results:
        assert isinstance(result, DetectionResult)
        assert result.metric_name in config["monitored_metrics"]
```

## Performance Considerations

- **Timing**: All detection methods are automatically timed for monitoring
- **Memory**: DetectionResult objects are lightweight and serializable
- **Caching**: Configuration can be cached at the detector level
- **Parallel execution**: Detectors are stateless and thread-safe

## Extension Points

### Custom Utility Methods

```python
class BaseDetector(ABC):
    def get_historical_baseline(self, metric: str, days: int) -> float:
        """Get historical baseline for comparison."""
        # Implementation for querying historical data
        pass
    
    def calculate_seasonal_adjustment(self, value: float, date: date) -> float:
        """Apply seasonal adjustments to values."""
        # Implementation for seasonal normalization
        pass
```

### Plugin Discovery

The registry can be extended to support automatic plugin discovery:

```python
# Future enhancement: Auto-discover detector plugins
detector_registry.discover_plugins("detection/plugins/")
```

## Next Steps

This base detector interface enables:

1. **Threshold Detector** (ADF-32): Simple min/max and percentage change detection
2. **Statistical Detector** (ADF-33): Z-score and moving average analysis  
3. **Elementary Integration** (ADF-34): Wrapper for existing data quality tests
4. **dbt Test Integration** (ADF-35): Execute and parse dbt test results

Each concrete detector will inherit from `BaseDetector` and implement the required abstract methods while leveraging the common utilities provided.