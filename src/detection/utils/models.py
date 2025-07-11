"""Data models for anomaly detection results.

This module defines data classes for anomaly detection results and write operations,
providing type safety and validation for the ResultsWriter component.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Any, List, Optional, Union
import json


@dataclass
class AnomalyResult:
    """Represents a single anomaly detection result.
    
    This dataclass encapsulates all information about a detected anomaly,
    including the detection metadata, severity, and context information.
    """
    
    detection_date: date
    event_type: str
    metric_name: str
    expected_value: Optional[float]
    actual_value: float
    deviation_percentage: Optional[float]
    severity: str  # 'critical', 'high', 'warning'
    detection_method: str
    detector_config: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    
    # Optional fields with defaults
    alert_sent: bool = False
    created_at: Optional[datetime] = field(default_factory=datetime.now)
    detection_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if self.severity not in ['critical', 'high', 'warning']:
            raise ValueError(f"Invalid severity: {self.severity}")
        
        if not self.event_type or not self.event_type.strip():
            raise ValueError("event_type cannot be empty")
        
        if not self.metric_name or not self.metric_name.strip():
            raise ValueError("metric_name cannot be empty")
        
        if not self.detection_method or not self.detection_method.strip():
            raise ValueError("detection_method cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'detection_date': self.detection_date.isoformat(),
            'event_type': self.event_type,
            'metric_name': self.metric_name,
            'expected_value': self.expected_value,
            'actual_value': self.actual_value,
            'deviation_percentage': self.deviation_percentage,
            'severity': self.severity,
            'detection_method': self.detection_method,
            'detector_config': json.dumps(self.detector_config) if self.detector_config else None,
            'metadata': json.dumps(self.metadata) if self.metadata else None,
            'alert_sent': self.alert_sent,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'detection_id': self.detection_id
        }
    
    def get_unique_key(self) -> tuple:
        """Get unique key for duplicate detection."""
        return (self.detection_date, self.event_type, self.metric_name)


@dataclass
class WriteResult:
    """Result of a single write operation."""
    
    success: bool
    rows_affected: int
    error: Optional[Exception] = None
    execution_time_ms: Optional[float] = None
    
    def __str__(self) -> str:
        if self.success:
            return f"WriteResult(success=True, rows_affected={self.rows_affected})"
        else:
            return f"WriteResult(success=False, error={self.error})"


@dataclass
class BatchInsertResult:
    """Result of a batch insert operation."""
    
    success: bool
    total_rows: int
    successful_rows: int
    failed_rows: int
    errors: List[Exception] = field(default_factory=list)
    execution_time_ms: Optional[float] = None
    batches_processed: int = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_rows == 0:
            return 0.0
        return (self.failed_rows / self.total_rows) * 100.0
    
    def __str__(self) -> str:
        if self.success:
            return f"BatchInsertResult(success=True, total={self.total_rows}, successful={self.successful_rows})"
        else:
            return f"BatchInsertResult(success=False, total={self.total_rows}, failed={self.failed_rows}, errors={len(self.errors)})"