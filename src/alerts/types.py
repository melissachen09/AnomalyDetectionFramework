"""
Type definitions for Alert Classification System.

This module defines the data structures used throughout the alert classification
system, implementing GADF-ALERT-002 requirements.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Alert:
    """Represents an anomaly alert to be classified."""
    metric_name: str
    value: float
    threshold: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BusinessImpact:
    """Represents the business impact assessment of an alert."""
    impact_score: float  # 0.0 to 1.0
    affected_metrics: List[str]
    revenue_impact: Optional[float] = None
    customer_experience_impact: Optional[float] = None
    operational_impact: Optional[float] = None


@dataclass
class HistoricalContext:
    """Represents historical context for an alert."""
    similar_alerts: List[Dict[str, Any]]
    frequency: float  # alerts per day
    last_occurrence: Optional[datetime] = None
    average_severity: Optional[str] = None


@dataclass
class ClassificationResult:
    """Result of alert classification."""
    severity: AlertSeverity
    confidence: float  # 0.0 to 1.0
    explanation: str
    business_impact: Optional[BusinessImpact] = None
    historical_context: Optional[HistoricalContext] = None
    severity_score: Optional[float] = None
    rules_triggered: Optional[List[str]] = None