# BaseDetector Plugin Contract Documentation

## Overview

This document describes the plugin contract for the Anomaly Detection Framework's detection plugins. All detection plugins must inherit from the `BaseDetector` abstract base class and implement the required interface methods.

## Base Class: `BaseDetector`

The `BaseDetector` abstract base class provides the foundation for all detection plugins in the anomaly detection system. It handles common functionality such as configuration management, database connections, and validation while requiring concrete implementations to provide the core detection logic.

### Location
```
src/detection/detectors/base_detector.py
```

### Inheritance
```python
from abc import ABC, abstractmethod
class BaseDetector(ABC):
    # Abstract base class implementation
```

## Required Implementation

### Abstract Methods

#### `detect(start_date: date, end_date: date) -> List[DetectionResult]`

**Purpose**: Core detection method that must be implemented by all concrete detector classes.

**Parameters**:
- `start_date` (date): Start date for detection analysis (inclusive)
- `end_date` (date): End date for detection analysis (inclusive)

**Returns**: 
- `List[DetectionResult]`: List of detected anomalies, empty list if no anomalies found

**Validation**:
- Both parameters must not be None
- `start_date` must be <= `end_date`
- Use `self.validate_date_range(start_date, end_date)` for validation

**Example Implementation**:
```python
class ThresholdDetector(BaseDetector):
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        # Validate input parameters
        self.validate_date_range(start_date, end_date)
        
        # Implement detection logic
        anomalies = []
        
        # Query data using self.snowflake_conn
        # Apply detection algorithm
        # Create DetectionResult objects for anomalies
        
        return anomalies
```

## Required Constructor

### `__init__(config: Dict[str, Any])`

**Purpose**: Initialize the detector with configuration parameters.

**Parameters**:
- `config` (Dict[str, Any]): Configuration dictionary, cannot be None

**Behavior**:
- Validates config is not None and is a dictionary
- Stores config in `self.config`
- Initializes Snowflake connection in `self.snowflake_conn`
- Sets up logger in `self.logger`

**Example**:
```python
detector = ThresholdDetector(config={
    "event_type": "listing_views",
    "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
    "metrics": ["total_views"],
    "thresholds": {"min": 1000, "max": 100000}
})
```

## Data Structures

### `DetectionResult`

**Purpose**: Represents a detected anomaly with all required metadata.

**Required Fields**:
- `event_type` (str): Type of event being monitored
- `metric_name` (str): Name of the metric that triggered the anomaly
- `detection_date` (date): Date when the anomaly was detected
- `expected_value` (float): Expected value based on historical data
- `actual_value` (float): Actual observed value
- `deviation_percentage` (float): Percentage deviation (0.0-1.0)
- `severity` (str): Severity level ('critical', 'high', 'warning')
- `detection_method` (str): Detection method used

**Optional Fields**:
- `alert_sent` (bool): Whether alert was sent (default: False)
- `details` (Dict[str, Any]): Additional metadata

**Example**:
```python
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
```

## Common Properties and Methods

### Properties
- `self.config`: Configuration dictionary passed during initialization
- `self.snowflake_conn`: Database connection object for data access
- `self.logger`: Logger instance for the detector class

### Utility Methods

#### `validate_date_range(start_date: date, end_date: date) -> None`
Validates date parameters and raises ValueError if invalid.

#### `get_config_value(key: str, default: Any = None) -> Any`
Safely retrieves configuration values with optional defaults.

### String Representation
The `__repr__` method provides helpful debugging information:
```python
repr(detector)  # Returns: "ThresholdDetector(config_keys=['event_type', 'table', 'metrics'])"
```

## Implementation Requirements

### 1. Interface Compliance
- Must inherit from `BaseDetector`
- Must implement the `detect` method
- Must call parent `__init__` with valid config

### 2. Error Handling
- Use `validate_date_range()` to validate input dates
- Handle database connection errors gracefully
- Log errors using `self.logger`
- Raise appropriate exceptions with descriptive messages

### 3. Performance
- Process data efficiently for large date ranges
- Use batch processing when appropriate
- Implement connection pooling if needed

### 4. Configuration
- Validate configuration parameters in `__init__`
- Use `get_config_value()` for safe config access
- Document expected configuration keys

### 5. Testing
- All concrete detectors must have comprehensive test coverage
- Test both normal operation and error conditions
- Mock external dependencies (database connections)

## Example Complete Implementation

```python
from datetime import date
from typing import List, Dict, Any
from src.detection.detectors.base_detector import BaseDetector, DetectionResult

class ExampleDetector(BaseDetector):
    """Example detector demonstrating the plugin contract."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with required configuration."""
        super().__init__(config)
        
        # Validate required config keys
        required_keys = ['event_type', 'table', 'metrics']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        """Detect anomalies in the specified date range."""
        # Validate input parameters
        self.validate_date_range(start_date, end_date)
        
        # Get configuration
        event_type = self.get_config_value('event_type')
        table = self.get_config_value('table')
        metrics = self.get_config_value('metrics', [])
        
        anomalies = []
        
        try:
            # Query data using self.snowflake_conn
            # Implement detection logic
            # Create DetectionResult objects for anomalies
            
            self.logger.info(f"Processed {event_type} from {start_date} to {end_date}")
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            raise
        
        return anomalies
```

## Testing Contract

### Required Test Coverage
- Abstract class instantiation prevention
- Concrete implementation works correctly
- Configuration validation
- Date range validation
- Error handling
- Method signatures and return types
- Common functionality (config access, logging, etc.)

### Test Structure
```python
class TestExampleDetector:
    def test_initialization_with_valid_config(self):
        # Test successful initialization
        
    def test_initialization_with_invalid_config(self):
        # Test configuration validation
        
    def test_detect_with_valid_dates(self):
        # Test normal detection operation
        
    def test_detect_with_invalid_dates(self):
        # Test date validation
        
    def test_error_handling(self):
        # Test error scenarios
```

## Compliance Checklist

When implementing a new detector plugin, ensure:

- [ ] Inherits from `BaseDetector`
- [ ] Implements `detect` method with correct signature
- [ ] Calls `super().__init__(config)` in constructor
- [ ] Validates required configuration parameters
- [ ] Uses `validate_date_range()` for input validation
- [ ] Returns `List[DetectionResult]` from detect method
- [ ] Handles errors gracefully with logging
- [ ] Has comprehensive test coverage (>80%)
- [ ] Includes docstrings for class and methods
- [ ] Follows project coding standards

## Integration Points

### With Framework
- Detector plugins are loaded dynamically by the orchestration system
- Configuration is passed from YAML config files
- Results are collected and processed by the alert management system

### With Database
- Use `self.snowflake_conn` for all database operations
- Follow SQL best practices for performance
- Handle connection errors and retries

### With Logging
- Use `self.logger` for all logging operations
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Include context in log messages

This contract ensures consistency, reliability, and maintainability across all detection plugins in the anomaly detection framework.