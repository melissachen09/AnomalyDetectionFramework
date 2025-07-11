"""Alert test fixtures and data models for alert classification testing."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Union
import pytest


@dataclass
class Alert:
    """Represents an anomaly detection alert."""
    event_type: str
    metric_name: str
    actual_value: float
    expected_value: float
    deviation_percentage: float
    detection_date: date
    detector_method: str
    raw_data: Optional[Dict] = None
    
    # Classification fields (to be set by classifier)
    severity: Optional[str] = None
    business_impact_score: Optional[float] = None
    explanation: Optional[str] = None
    confidence_score: Optional[float] = None


@dataclass
class AlertClassificationResult:
    """Result of alert classification process."""
    alert: Alert
    severity: str  # 'critical', 'high', 'warning'
    business_impact_score: float
    explanation: str
    confidence_score: float
    classification_factors: Dict
    classification_timestamp: datetime


@dataclass
class AlertClassificationConfig:
    """Configuration for alert classification."""
    severity_thresholds: Dict
    business_impact_weights: Dict
    classification_rules: Dict
    escalation_settings: Dict


class AlertFixtures:
    """Test fixture data for alert classification testing."""
    
    @staticmethod
    def get_critical_alert() -> Alert:
        """Create a critical severity alert for testing."""
        return Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=5000.0,
            expected_value=50000.0,
            deviation_percentage=0.9,  # 90% drop
            detection_date=date.today(),
            detector_method="threshold",
            raw_data={"dimension": "total", "hour": 14}
        )
    
    @staticmethod
    def get_high_alert() -> Alert:
        """Create a high severity alert for testing."""
        return Alert(
            event_type="enquiries",
            metric_name="total_enquiries",
            actual_value=1000.0,
            expected_value=1500.0,
            deviation_percentage=0.33,  # 33% drop
            detection_date=date.today(),
            detector_method="statistical",
            raw_data={"dimension": "mobile", "hour": 10}
        )
    
    @staticmethod
    def get_warning_alert() -> Alert:
        """Create a warning severity alert for testing."""
        return Alert(
            event_type="clicks",
            metric_name="click_through_rate",
            actual_value=0.18,
            expected_value=0.22,
            deviation_percentage=0.18,  # 18% drop
            detection_date=date.today(),
            detector_method="percentage_change",
            raw_data={"dimension": "desktop", "hour": 16}
        )
    
    @staticmethod
    def get_multiple_alerts() -> List[Alert]:
        """Create multiple alerts for batch testing."""
        return [
            AlertFixtures.get_critical_alert(),
            AlertFixtures.get_high_alert(),
            AlertFixtures.get_warning_alert(),
            Alert(
                event_type="listing_views",
                metric_name="mobile_views",
                actual_value=8000.0,
                expected_value=10000.0,
                deviation_percentage=0.2,
                detection_date=date.today(),
                detector_method="threshold"
            ),
            Alert(
                event_type="listing_views",
                metric_name="desktop_views",
                actual_value=42000.0,
                expected_value=40000.0,
                deviation_percentage=0.05,  # 5% increase
                detection_date=date.today(),
                detector_method="statistical"
            )
        ]
    
    @staticmethod
    def get_edge_case_alerts() -> List[Alert]:
        """Create edge case alerts for boundary testing."""
        return [
            # Zero values
            Alert(
                event_type="clicks",
                metric_name="zero_clicks",
                actual_value=0.0,
                expected_value=100.0,
                deviation_percentage=1.0,
                detection_date=date.today(),
                detector_method="threshold"
            ),
            # Negative values
            Alert(
                event_type="revenue",
                metric_name="net_change",
                actual_value=-500.0,
                expected_value=1000.0,
                deviation_percentage=1.5,
                detection_date=date.today(),
                detector_method="statistical"
            ),
            # Very small deviation
            Alert(
                event_type="views",
                metric_name="page_views",
                actual_value=9999.0,
                expected_value=10000.0,
                deviation_percentage=0.0001,
                detection_date=date.today(),
                detector_method="threshold"
            ),
            # Exactly at threshold
            Alert(
                event_type="enquiries",
                metric_name="form_submissions",
                actual_value=700.0,
                expected_value=1000.0,
                deviation_percentage=0.3,  # Exactly 30%
                detection_date=date.today(),
                detector_method="percentage_change"
            )
        ]
    
    @staticmethod
    def get_performance_test_alerts(count: int = 1000) -> List[Alert]:
        """Generate large number of alerts for performance testing."""
        import random
        
        alerts = []
        event_types = ["listing_views", "enquiries", "clicks", "searches"]
        metrics = ["total", "mobile", "desktop", "tablet"]
        detectors = ["threshold", "statistical", "percentage_change"]
        
        for i in range(count):
            deviation = random.uniform(0.1, 1.0)
            expected_val = random.uniform(1000, 100000)
            actual_val = expected_val * (1 - deviation) if random.choice([True, False]) else expected_val * (1 + deviation)
            
            alerts.append(Alert(
                event_type=random.choice(event_types),
                metric_name=f"{random.choice(metrics)}_{random.choice(event_types)}",
                actual_value=actual_val,
                expected_value=expected_val,
                deviation_percentage=abs(deviation),
                detection_date=date.today(),
                detector_method=random.choice(detectors),
                raw_data={"test_id": i}
            ))
        
        return alerts


class ConfigFixtures:
    """Test configuration fixtures for alert classification."""
    
    @staticmethod
    def get_default_classification_config() -> AlertClassificationConfig:
        """Get default alert classification configuration."""
        return AlertClassificationConfig(
            severity_thresholds={
                "critical": {
                    "deviation_threshold": 0.5,
                    "business_impact_min": 8.0
                },
                "high": {
                    "deviation_threshold": 0.3,
                    "business_impact_min": 5.0
                },
                "warning": {
                    "deviation_threshold": 0.2,
                    "business_impact_min": 2.0
                }
            },
            business_impact_weights={
                "listing_views": 10.0,  # High business impact
                "enquiries": 9.0,
                "clicks": 7.0,
                "searches": 6.0,
                "page_loads": 4.0
            },
            classification_rules={
                "confidence_threshold": 0.7,
                "multi_factor_weights": {
                    "deviation": 0.4,
                    "business_impact": 0.3,
                    "historical_context": 0.2,
                    "time_context": 0.1
                }
            },
            escalation_settings={
                "critical_immediate": True,
                "high_batch_window": 300,  # 5 minutes
                "warning_batch_window": 3600  # 1 hour
            }
        )
    
    @staticmethod
    def get_test_thresholds() -> Dict:
        """Get test-specific threshold configurations."""
        return {
            "strict": {
                "critical": 0.7,
                "high": 0.4,
                "warning": 0.1
            },
            "lenient": {
                "critical": 0.9,
                "high": 0.6,
                "warning": 0.3
            },
            "custom": {
                "listing_views": {"critical": 0.5, "high": 0.3, "warning": 0.15},
                "enquiries": {"critical": 0.6, "high": 0.35, "warning": 0.2},
                "clicks": {"critical": 0.8, "high": 0.5, "warning": 0.25}
            }
        }


# Pytest fixtures
@pytest.fixture
def critical_alert():
    """Pytest fixture for critical alert."""
    return AlertFixtures.get_critical_alert()


@pytest.fixture
def high_alert():
    """Pytest fixture for high severity alert."""
    return AlertFixtures.get_high_alert()


@pytest.fixture
def warning_alert():
    """Pytest fixture for warning alert."""
    return AlertFixtures.get_warning_alert()


@pytest.fixture
def multiple_alerts():
    """Pytest fixture for multiple alerts."""
    return AlertFixtures.get_multiple_alerts()


@pytest.fixture
def edge_case_alerts():
    """Pytest fixture for edge case alerts."""
    return AlertFixtures.get_edge_case_alerts()


@pytest.fixture
def default_config():
    """Pytest fixture for default classification configuration."""
    return ConfigFixtures.get_default_classification_config()


@pytest.fixture
def test_thresholds():
    """Pytest fixture for test threshold configurations."""
    return ConfigFixtures.get_test_thresholds()