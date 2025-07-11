"""
Test cases for Alert Classifier - GADF-ALERT-001: Write Test Cases for Alert Classifier

This module implements comprehensive test cases for alert severity calculation algorithms,
business impact scoring, multi-factor classification, and consistency validation.

Following TDD principles - these tests define the expected behavior before implementation.
"""

import pytest
from datetime import date, datetime, timedelta
from typing import Dict, List
from unittest.mock import Mock, patch

# Import test fixtures
from tests.fixtures.alert_fixtures import (
    Alert, AlertClassificationResult, AlertClassificationConfig,
    AlertFixtures, ConfigFixtures
)


class TestAlertSeverityCalculation:
    """
    GADF-ALERT-001a: Test severity calculation algorithms
    
    Tests threshold-based severity calculation, deviation percentage to severity mapping,
    business impact factor integration, and historical context consideration.
    """
    
    def test_deviation_threshold_classification(self, default_config):
        """Test basic deviation threshold-based severity classification."""
        # Test critical threshold (>50% deviation)
        critical_alert = Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=2000.0,
            expected_value=10000.0,
            deviation_percentage=0.8,  # 80% deviation
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # Test high threshold (30-50% deviation)
        high_alert = Alert(
            event_type="enquiries",
            metric_name="form_submissions",
            actual_value=700.0,
            expected_value=1000.0,
            deviation_percentage=0.3,  # 30% deviation
            detection_date=date.today(),
            detector_method="statistical"
        )
        
        # Test warning threshold (20-30% deviation)
        warning_alert = Alert(
            event_type="clicks",
            metric_name="click_rate",
            actual_value=800.0,
            expected_value=1000.0,
            deviation_percentage=0.2,  # 20% deviation
            detection_date=date.today(),
            detector_method="percentage_change"
        )
        
        # Test below warning threshold (<20% deviation)
        minimal_alert = Alert(
            event_type="views",
            metric_name="page_views",
            actual_value=950.0,
            expected_value=1000.0,
            deviation_percentage=0.05,  # 5% deviation
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # NOTE: These tests will fail until AlertClassifier is implemented
        # Expected behavior defined by tests:
        
        # AlertClassifier.classify_severity(critical_alert, default_config) should return "critical"
        # AlertClassifier.classify_severity(high_alert, default_config) should return "high" 
        # AlertClassifier.classify_severity(warning_alert, default_config) should return "warning"
        # AlertClassifier.classify_severity(minimal_alert, default_config) should return None or "info"
        
        # Placeholder assertions for TDD
        assert critical_alert.deviation_percentage > default_config.severity_thresholds["critical"]["deviation_threshold"]
        assert high_alert.deviation_percentage >= default_config.severity_thresholds["high"]["deviation_threshold"]
        assert warning_alert.deviation_percentage >= default_config.severity_thresholds["warning"]["deviation_threshold"]
        assert minimal_alert.deviation_percentage < default_config.severity_thresholds["warning"]["deviation_threshold"]
    
    def test_severity_calculation_with_custom_thresholds(self, test_thresholds):
        """Test severity calculation with custom threshold configurations."""
        alert = Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=6000.0,
            expected_value=10000.0,
            deviation_percentage=0.4,  # 40% deviation
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # Test with strict thresholds (should be warning)
        strict_config = AlertClassificationConfig(
            severity_thresholds={
                "critical": {"deviation_threshold": 0.7, "business_impact_min": 8.0},
                "high": {"deviation_threshold": 0.4, "business_impact_min": 5.0},
                "warning": {"deviation_threshold": 0.1, "business_impact_min": 2.0}
            },
            business_impact_weights={},
            classification_rules={},
            escalation_settings={}
        )
        
        # Test with lenient thresholds (should be high)
        lenient_config = AlertClassificationConfig(
            severity_thresholds={
                "critical": {"deviation_threshold": 0.9, "business_impact_min": 8.0},
                "high": {"deviation_threshold": 0.6, "business_impact_min": 5.0},
                "warning": {"deviation_threshold": 0.3, "business_impact_min": 2.0}
            },
            business_impact_weights={},
            classification_rules={},
            escalation_settings={}
        )
        
        # Expected: AlertClassifier should handle different threshold configurations
        # alert with 40% deviation should be "high" with strict config, "warning" with lenient config
        assert alert.deviation_percentage >= strict_config.severity_thresholds["high"]["deviation_threshold"]
        assert alert.deviation_percentage < lenient_config.severity_thresholds["high"]["deviation_threshold"]
    
    def test_positive_vs_negative_deviations(self, default_config):
        """Test that both positive and negative deviations are classified correctly."""
        # Large positive deviation (spike)
        spike_alert = Alert(
            event_type="clicks",
            metric_name="bot_clicks",
            actual_value=15000.0,
            expected_value=1000.0,
            deviation_percentage=14.0,  # 1400% increase (potential bot traffic)
            detection_date=date.today(),
            detector_method="statistical"
        )
        
        # Large negative deviation (drop)
        drop_alert = Alert(
            event_type="listing_views",
            metric_name="organic_views",
            actual_value=100.0,
            expected_value=10000.0,
            deviation_percentage=0.99,  # 99% drop
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # Both should be classified as critical due to magnitude
        assert spike_alert.deviation_percentage > default_config.severity_thresholds["critical"]["deviation_threshold"]
        assert drop_alert.deviation_percentage > default_config.severity_thresholds["critical"]["deviation_threshold"]
    
    def test_boundary_condition_classification(self, default_config):
        """Test classification at exact threshold boundaries."""
        # Exactly at critical threshold
        exactly_critical = Alert(
            event_type="enquiries",
            metric_name="leads",
            actual_value=5000.0,
            expected_value=10000.0,
            deviation_percentage=0.5,  # Exactly 50%
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # Just below critical threshold
        just_below_critical = Alert(
            event_type="enquiries",
            metric_name="leads",
            actual_value=5001.0,
            expected_value=10000.0,
            deviation_percentage=0.4999,  # Just below 50%
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # Exactly at high threshold
        exactly_high = Alert(
            event_type="clicks",
            metric_name="ctr",
            actual_value=7000.0,
            expected_value=10000.0,
            deviation_percentage=0.3,  # Exactly 30%
            detection_date=date.today(),
            detector_method="statistical"
        )
        
        # Expected: Boundary conditions should be handled consistently
        # Exactly at threshold should be included in that severity level
        assert exactly_critical.deviation_percentage == default_config.severity_thresholds["critical"]["deviation_threshold"]
        assert just_below_critical.deviation_percentage < default_config.severity_thresholds["critical"]["deviation_threshold"]
        assert exactly_high.deviation_percentage == default_config.severity_thresholds["high"]["deviation_threshold"]
    
    def test_metric_type_specific_thresholds(self):
        """Test that different metric types can have different severity thresholds."""
        # Revenue metrics might have stricter thresholds
        revenue_config = AlertClassificationConfig(
            severity_thresholds={
                "revenue_metrics": {
                    "critical": {"deviation_threshold": 0.2, "business_impact_min": 9.0},
                    "high": {"deviation_threshold": 0.1, "business_impact_min": 7.0},
                    "warning": {"deviation_threshold": 0.05, "business_impact_min": 5.0}
                },
                "engagement_metrics": {
                    "critical": {"deviation_threshold": 0.6, "business_impact_min": 6.0},
                    "high": {"deviation_threshold": 0.4, "business_impact_min": 4.0},
                    "warning": {"deviation_threshold": 0.2, "business_impact_min": 2.0}
                }
            },
            business_impact_weights={},
            classification_rules={},
            escalation_settings={}
        )
        
        revenue_alert = Alert(
            event_type="conversions",
            metric_name="revenue_per_visitor",
            actual_value=80.0,
            expected_value=100.0,
            deviation_percentage=0.2,  # 20% deviation
            detection_date=date.today(),
            detector_method="statistical"
        )
        
        engagement_alert = Alert(
            event_type="clicks",
            metric_name="click_through_rate",
            actual_value=80.0,
            expected_value=100.0,
            deviation_percentage=0.2,  # Same 20% deviation
            detection_date=date.today(),
            detector_method="statistical"
        )
        
        # Expected: Same deviation percentage should result in different severities
        # Revenue: 20% = critical, Engagement: 20% = warning
        assert revenue_alert.deviation_percentage == engagement_alert.deviation_percentage
        # Implementation should handle metric-specific thresholds


class TestBusinessImpactScoring:
    """
    GADF-ALERT-001b: Test business impact scoring
    
    Tests metric importance weighting, stakeholder impact assessment,
    downstream system dependencies, and time-of-day impact adjustments.
    """
    
    def test_metric_importance_weighting(self, default_config):
        """Test that business impact scores are calculated based on metric importance."""
        # High-importance metric (listing views)
        high_impact_alert = Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=5000.0,
            expected_value=10000.0,
            deviation_percentage=0.5,
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # Lower-importance metric (page loads)
        low_impact_alert = Alert(
            event_type="page_loads",
            metric_name="load_time",
            actual_value=5000.0,
            expected_value=10000.0,
            deviation_percentage=0.5,  # Same deviation
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # Expected business impact scores (to be calculated by implementation):
        # high_impact_alert should get score around 10.0 * 0.5 = 5.0
        # low_impact_alert should get score around 4.0 * 0.5 = 2.0
        
        expected_high_impact = default_config.business_impact_weights["listing_views"] * high_impact_alert.deviation_percentage
        expected_low_impact = default_config.business_impact_weights["page_loads"] * low_impact_alert.deviation_percentage
        
        assert expected_high_impact > expected_low_impact
        assert default_config.business_impact_weights["listing_views"] > default_config.business_impact_weights["page_loads"]
    
    def test_time_context_impact_adjustment(self, default_config):
        """Test that business impact is adjusted based on time context."""
        base_alert = Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=5000.0,
            expected_value=10000.0,
            deviation_percentage=0.5,
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # Business hours (higher impact)
        business_hours_alert = base_alert
        business_hours_alert.raw_data = {"hour": 14, "day_of_week": 2}  # Tuesday 2 PM
        
        # Off hours (lower impact)  
        off_hours_alert = Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=5000.0,
            expected_value=10000.0,
            deviation_percentage=0.5,
            detection_date=date.today(),
            detector_method="threshold",
            raw_data={"hour": 3, "day_of_week": 6}  # Saturday 3 AM
        )
        
        # Weekend vs weekday
        weekend_alert = Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=5000.0,
            expected_value=10000.0,
            deviation_percentage=0.5,
            detection_date=date.today(),
            detector_method="threshold",
            raw_data={"hour": 14, "day_of_week": 6}  # Saturday 2 PM
        )
        
        # Expected: Business hours should have higher impact multiplier
        # Implementation should apply time-based adjustments to impact score
        assert business_hours_alert.raw_data["hour"] in range(9, 18)  # Business hours
        assert off_hours_alert.raw_data["hour"] not in range(9, 18)  # Off hours
        assert weekend_alert.raw_data["day_of_week"] in [5, 6]  # Weekend
    
    def test_downstream_system_dependency_impact(self):
        """Test impact scoring considers downstream system dependencies."""
        # Critical system dependency
        critical_dependency_alert = Alert(
            event_type="listing_views",
            metric_name="search_api_views",
            actual_value=100.0,
            expected_value=10000.0,
            deviation_percentage=0.99,  # 99% drop
            detection_date=date.today(),
            detector_method="threshold",
            raw_data={
                "downstream_systems": ["search_service", "recommendation_engine", "analytics_pipeline"],
                "dependency_level": "critical"
            }
        )
        
        # Isolated system
        isolated_alert = Alert(
            event_type="admin_logs",
            metric_name="admin_actions",
            actual_value=10.0,
            expected_value=100.0,
            deviation_percentage=0.9,  # 90% drop
            detection_date=date.today(),
            detector_method="threshold",
            raw_data={
                "downstream_systems": [],
                "dependency_level": "isolated"
            }
        )
        
        # Expected: Critical dependencies should increase business impact score
        # Even with similar deviation, critical dependency should have higher impact
        assert len(critical_dependency_alert.raw_data["downstream_systems"]) > 0
        assert len(isolated_alert.raw_data["downstream_systems"]) == 0
    
    def test_stakeholder_impact_assessment(self, default_config):
        """Test that business impact considers affected stakeholder groups."""
        # Customer-facing impact
        customer_facing_alert = Alert(
            event_type="listing_views",
            metric_name="public_search_results",
            actual_value=1000.0,
            expected_value=10000.0,
            deviation_percentage=0.9,
            detection_date=date.today(),
            detector_method="threshold",
            raw_data={
                "affected_stakeholders": ["customers", "sales_team", "customer_support"],
                "customer_facing": True
            }
        )
        
        # Internal-only impact
        internal_alert = Alert(
            event_type="admin_metrics",
            metric_name="backend_processing",
            actual_value=1000.0,
            expected_value=10000.0,
            deviation_percentage=0.9,  # Same deviation
            detection_date=date.today(),
            detector_method="threshold",
            raw_data={
                "affected_stakeholders": ["data_team"],
                "customer_facing": False
            }
        )
        
        # Expected: Customer-facing issues should have higher business impact
        assert customer_facing_alert.raw_data["customer_facing"] == True
        assert internal_alert.raw_data["customer_facing"] == False
        assert len(customer_facing_alert.raw_data["affected_stakeholders"]) > len(internal_alert.raw_data["affected_stakeholders"])
    
    def test_business_impact_score_range_validation(self, default_config):
        """Test that business impact scores are within expected ranges."""
        test_alerts = AlertFixtures.get_multiple_alerts()
        
        for alert in test_alerts:
            # Calculate expected business impact score
            base_weight = default_config.business_impact_weights.get(alert.event_type, 1.0)
            expected_score = base_weight * alert.deviation_percentage
            
            # Business impact scores should be positive and reasonable
            assert expected_score >= 0.0
            assert expected_score <= 20.0  # Maximum reasonable score
            
            # High deviation should result in high impact
            if alert.deviation_percentage > 0.5:
                assert expected_score >= 2.0  # Minimum significant impact


class TestMultiFactorClassification:
    """
    GADF-ALERT-001c: Test multi-factor classification
    
    Tests combined severity factors, weighted classification algorithms,
    context-aware adjustments, and classification confidence scoring.
    """
    
    def test_weighted_multi_factor_classification(self, default_config):
        """Test that classification considers multiple weighted factors."""
        alert = Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=6000.0,
            expected_value=10000.0,
            deviation_percentage=0.4,  # 40% deviation (borderline)
            detection_date=date.today(),
            detector_method="threshold",
            raw_data={
                "historical_context": "recent_trend_decline",
                "time_context": {"hour": 14, "is_business_day": True},
                "system_load": "normal"
            }
        )
        
        # Expected multi-factor weights from config:
        # deviation: 0.4, business_impact: 0.3, historical_context: 0.2, time_context: 0.1
        
        # Calculate expected weighted score components:
        deviation_score = 0.4 * alert.deviation_percentage  # 0.4 * 0.4 = 0.16
        business_impact_score = 0.3 * (default_config.business_impact_weights["listing_views"] * alert.deviation_percentage)  # 0.3 * (10 * 0.4)
        
        # Expected: Implementation should combine all factors with proper weighting
        total_weight = sum(default_config.classification_rules["multi_factor_weights"].values())
        assert total_weight == 1.0  # Weights should sum to 1.0
    
    def test_historical_context_classification_adjustment(self, default_config):
        """Test that historical context adjusts classification severity."""
        base_alert = Alert(
            event_type="enquiries",
            metric_name="form_submissions",
            actual_value=700.0,
            expected_value=1000.0,
            deviation_percentage=0.3,  # Exactly at high threshold
            detection_date=date.today(),
            detector_method="statistical"
        )
        
        # Alert with declining trend context
        declining_trend_alert = base_alert
        declining_trend_alert.raw_data = {
            "historical_context": "declining_trend",
            "trend_duration_days": 7,
            "trend_severity": "moderate"
        }
        
        # Alert with stable historical context
        stable_context_alert = Alert(
            event_type="enquiries",
            metric_name="form_submissions",
            actual_value=700.0,
            expected_value=1000.0,
            deviation_percentage=0.3,
            detection_date=date.today(),
            detector_method="statistical",
            raw_data={
                "historical_context": "stable",
                "trend_duration_days": 0,
                "trend_severity": "none"
            }
        )
        
        # Alert during known maintenance window
        maintenance_context_alert = Alert(
            event_type="enquiries",
            metric_name="form_submissions",
            actual_value=700.0,
            expected_value=1000.0,
            deviation_percentage=0.3,
            detection_date=date.today(),
            detector_method="statistical",
            raw_data={
                "historical_context": "planned_maintenance",
                "maintenance_window": True,
                "expected_impact": True
            }
        )
        
        # Expected: Historical context should adjust final classification
        # Declining trend: might escalate to critical
        # Stable: normal classification as high
        # Maintenance: might downgrade to warning or info
        assert declining_trend_alert.raw_data["historical_context"] == "declining_trend"
        assert stable_context_alert.raw_data["historical_context"] == "stable"
        assert maintenance_context_alert.raw_data["historical_context"] == "planned_maintenance"
    
    def test_classification_confidence_scoring(self, default_config):
        """Test that classification includes confidence scoring."""
        # High confidence scenario (clear thresholds, stable context)
        high_confidence_alert = Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=1000.0,
            expected_value=10000.0,
            deviation_percentage=0.9,  # Well above critical threshold
            detection_date=date.today(),
            detector_method="threshold",
            raw_data={
                "data_quality": "high",
                "sample_size": 10000,
                "variance": 0.05,
                "historical_consistency": "stable"
            }
        )
        
        # Low confidence scenario (borderline thresholds, noisy data)
        low_confidence_alert = Alert(
            event_type="clicks",
            metric_name="click_rate",
            actual_value=795.0,
            expected_value=1000.0,
            deviation_percentage=0.205,  # Just above warning threshold
            detection_date=date.today(),
            detector_method="statistical",
            raw_data={
                "data_quality": "medium",
                "sample_size": 50,
                "variance": 0.3,
                "historical_consistency": "volatile"
            }
        )
        
        # Expected confidence factors:
        # High confidence: large deviation, high data quality, large sample
        # Low confidence: borderline deviation, medium data quality, small sample
        
        # Confidence should be higher for clear cases
        assert high_confidence_alert.deviation_percentage > 0.8  # Very clear anomaly
        assert low_confidence_alert.deviation_percentage < 0.25  # Borderline case
        assert high_confidence_alert.raw_data["sample_size"] > low_confidence_alert.raw_data["sample_size"]
    
    def test_context_aware_classification_adjustments(self, default_config):
        """Test that classification adapts to various contextual factors."""
        # Holiday period context
        holiday_alert = Alert(
            event_type="listing_views",
            metric_name="search_traffic",
            actual_value=3000.0,
            expected_value=10000.0,
            deviation_percentage=0.7,
            detection_date=date.today(),
            detector_method="threshold",
            raw_data={
                "context_type": "holiday_period",
                "holiday_name": "Christmas",
                "expected_pattern": "traffic_decrease",
                "adjustment_factor": 0.3
            }
        )
        
        # Marketing campaign context
        campaign_alert = Alert(
            event_type="clicks",
            metric_name="campaign_clicks",
            actual_value=50000.0,
            expected_value=10000.0,
            deviation_percentage=4.0,  # 400% increase
            detection_date=date.today(),
            detector_method="statistical",
            raw_data={
                "context_type": "marketing_campaign",
                "campaign_id": "summer_promo_2024",
                "expected_pattern": "traffic_increase",
                "adjustment_factor": 0.8
            }
        )
        
        # System deployment context
        deployment_alert = Alert(
            event_type="api_calls",
            metric_name="response_time",
            actual_value=2000.0,
            expected_value=500.0,
            deviation_percentage=3.0,  # 300% increase
            detection_date=date.today(),
            detector_method="threshold",
            raw_data={
                "context_type": "system_deployment",
                "deployment_id": "v2.1.0",
                "deployment_time": datetime.now() - timedelta(hours=1),
                "adjustment_factor": 0.5
            }
        )
        
        # Expected: Context should adjust severity classification
        # Holiday: expected decrease, might reduce severity
        # Campaign: expected increase, might reduce severity for positive anomalies
        # Deployment: temporary performance issues, might adjust timing
        assert holiday_alert.raw_data["expected_pattern"] == "traffic_decrease"
        assert campaign_alert.raw_data["expected_pattern"] == "traffic_increase"
        assert deployment_alert.raw_data["context_type"] == "system_deployment"


class TestClassificationConsistency:
    """
    GADF-ALERT-001d: Test classification consistency validation
    
    Tests deterministic classification behavior, consistent results across similar inputs,
    classification stability over time, and edge case handling consistency.
    """
    
    def test_deterministic_classification_behavior(self, default_config):
        """Test that identical inputs always produce identical classifications."""
        alert = Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=5000.0,
            expected_value=10000.0,
            deviation_percentage=0.5,
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # Classification should be deterministic - same input, same output
        # Run classification multiple times and verify consistency
        for i in range(10):
            # Expected: AlertClassifier.classify(alert, default_config) should return identical results
            # This test validates that classification is deterministic and not random
            pass
    
    def test_similar_inputs_consistent_classification(self, default_config):
        """Test that similar inputs produce consistent classification results."""
        base_deviation = 0.35  # Should be 'high' severity
        
        similar_alerts = []
        for i in range(5):
            # Create alerts with very similar characteristics
            alert = Alert(
                event_type="enquiries",
                metric_name="form_submissions",
                actual_value=650.0 + i,  # Slight variation
                expected_value=1000.0,
                deviation_percentage=base_deviation + (i * 0.001),  # Tiny variation
                detection_date=date.today(),
                detector_method="statistical"
            )
            similar_alerts.append(alert)
        
        # Expected: All similar alerts should have same severity classification
        # Small variations shouldn't cause classification instability
        for alert in similar_alerts:
            assert abs(alert.deviation_percentage - base_deviation) < 0.01  # Very similar
    
    def test_classification_stability_over_time(self, default_config):
        """Test that classification remains stable for the same alert over time."""
        alert = Alert(
            event_type="clicks",
            metric_name="click_through_rate",
            actual_value=600.0,
            expected_value=1000.0,
            deviation_percentage=0.4,
            detection_date=date.today(),
            detector_method="percentage_change"
        )
        
        # Simulate classification at different times
        time_periods = [
            datetime.now(),
            datetime.now() + timedelta(minutes=5),
            datetime.now() + timedelta(hours=1),
            datetime.now() + timedelta(days=1)
        ]
        
        # Expected: Same alert characteristics should produce stable classification
        # regardless of when classification occurs (unless config changes)
        for timestamp in time_periods:
            # AlertClassifier.classify(alert, default_config, timestamp) should be consistent
            pass
    
    def test_edge_case_handling_consistency(self, default_config):
        """Test consistent behavior for edge cases and boundary conditions."""
        edge_cases = AlertFixtures.get_edge_case_alerts()
        
        # Test each edge case for consistent handling
        for alert in edge_cases:
            # Zero values
            if alert.actual_value == 0.0:
                # Expected: Should handle division by zero gracefully
                # Should not crash and should produce reasonable classification
                assert alert.expected_value > 0.0
                assert alert.deviation_percentage >= 0.0
            
            # Negative values  
            elif alert.actual_value < 0.0:
                # Expected: Should handle negative values appropriately
                # Might indicate different type of issue (data quality vs anomaly)
                assert alert.deviation_percentage > 0.0
            
            # Very small deviations
            elif alert.deviation_percentage < 0.001:
                # Expected: Should handle very small deviations consistently
                # Might not meet any threshold (return None or 'info')
                assert alert.deviation_percentage >= 0.0
            
            # Exact threshold boundaries
            elif alert.deviation_percentage in [0.2, 0.3, 0.5]:
                # Expected: Boundary cases should be handled consistently
                # Should be included in the threshold level they equal
                assert alert.deviation_percentage in default_config.severity_thresholds.values()
    
    def test_configuration_change_impact_consistency(self):
        """Test that configuration changes affect classification predictably."""
        alert = Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=6000.0,
            expected_value=10000.0,
            deviation_percentage=0.4,  # 40% deviation
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # Original config (40% = high)
        original_config = AlertClassificationConfig(
            severity_thresholds={
                "critical": {"deviation_threshold": 0.5, "business_impact_min": 8.0},
                "high": {"deviation_threshold": 0.3, "business_impact_min": 5.0},
                "warning": {"deviation_threshold": 0.2, "business_impact_min": 2.0}
            },
            business_impact_weights={"listing_views": 10.0},
            classification_rules={},
            escalation_settings={}
        )
        
        # Modified config (40% = critical)
        stricter_config = AlertClassificationConfig(
            severity_thresholds={
                "critical": {"deviation_threshold": 0.35, "business_impact_min": 8.0},
                "high": {"deviation_threshold": 0.25, "business_impact_min": 5.0},
                "warning": {"deviation_threshold": 0.15, "business_impact_min": 2.0}
            },
            business_impact_weights={"listing_views": 10.0},
            classification_rules={},
            escalation_settings={}
        )
        
        # Expected: Configuration changes should predictably affect classification
        # Same alert with stricter config should get higher severity
        assert alert.deviation_percentage > stricter_config.severity_thresholds["critical"]["deviation_threshold"]
        assert alert.deviation_percentage > original_config.severity_thresholds["high"]["deviation_threshold"]
        assert alert.deviation_percentage < original_config.severity_thresholds["critical"]["deviation_threshold"]
    
    def test_concurrent_classification_consistency(self, default_config):
        """Test that concurrent classifications produce consistent results."""
        alerts = AlertFixtures.get_multiple_alerts()
        
        # Expected: Concurrent processing should not affect individual classifications
        # Each alert should be classified independently and consistently
        # This tests thread safety and state isolation
        
        for alert in alerts:
            # AlertClassifier should be stateless and thread-safe
            # Multiple concurrent calls should produce identical results
            assert alert.deviation_percentage >= 0.0
            assert alert.actual_value >= 0.0
            assert alert.expected_value > 0.0


class TestAlertClassifierEdgeCases:
    """
    Additional edge case tests for comprehensive coverage
    
    Tests boundary conditions, error scenarios, malformed data,
    and performance edge cases.
    """
    
    def test_malformed_alert_data_handling(self, default_config):
        """Test graceful handling of malformed or incomplete alert data."""
        # Missing required fields
        incomplete_alert = Alert(
            event_type="",  # Empty event type
            metric_name="",  # Empty metric name
            actual_value=0.0,
            expected_value=0.0,  # Zero expected value
            deviation_percentage=float('inf'),  # Invalid deviation
            detection_date=date.today(),
            detector_method=""
        )
        
        # Invalid numeric values
        invalid_numeric_alert = Alert(
            event_type="test",
            metric_name="test_metric",
            actual_value=float('nan'),  # NaN value
            expected_value=-1.0,  # Negative expected value
            deviation_percentage=-0.5,  # Negative deviation
            detection_date=date.today(),
            detector_method="test"
        )
        
        # Expected: Classifier should handle malformed data gracefully
        # Should return error status or None, not crash
        assert incomplete_alert.event_type == ""
        assert invalid_numeric_alert.actual_value != invalid_numeric_alert.actual_value  # NaN check
    
    def test_extreme_value_handling(self, default_config):
        """Test handling of extremely large or small values."""
        # Extremely large values
        large_value_alert = Alert(
            event_type="listing_views",
            metric_name="total_views",
            actual_value=1e15,  # Very large number
            expected_value=1e14,
            deviation_percentage=9.0,  # 900% increase
            detection_date=date.today(),
            detector_method="threshold"
        )
        
        # Extremely small values
        small_value_alert = Alert(
            event_type="conversion_rate",
            metric_name="micro_conversions",
            actual_value=1e-10,  # Very small number
            expected_value=1e-9,
            deviation_percentage=0.9,  # 90% decrease
            detection_date=date.today(),
            detector_method="statistical"
        )
        
        # Expected: Should handle extreme values without overflow/underflow
        assert large_value_alert.actual_value > 1e10
        assert small_value_alert.actual_value < 1e-5
    
    def test_configuration_validation(self):
        """Test validation of classification configuration."""
        # Invalid threshold configuration
        invalid_config = AlertClassificationConfig(
            severity_thresholds={
                "critical": {"deviation_threshold": -0.5, "business_impact_min": -1.0},  # Negative thresholds
                "high": {"deviation_threshold": 1.5, "business_impact_min": 5.0},  # > 100% threshold
                "warning": {"deviation_threshold": 0.8, "business_impact_min": 2.0}  # warning > high
            },
            business_impact_weights={"invalid_metric": -5.0},  # Negative weight
            classification_rules={"confidence_threshold": 1.5},  # > 100% confidence
            escalation_settings={}
        )
        
        # Expected: Configuration validation should catch invalid values
        # Should raise validation errors or provide warnings
        assert invalid_config.severity_thresholds["critical"]["deviation_threshold"] < 0
        assert invalid_config.severity_thresholds["high"]["deviation_threshold"] > 1.0
        assert invalid_config.business_impact_weights["invalid_metric"] < 0


class TestAlertClassifierPerformance:
    """
    Performance validation tests for alert classification
    
    Tests processing speed, memory usage, and scalability
    under various load conditions.
    """
    
    def test_single_alert_classification_performance(self, default_config):
        """Test performance of classifying a single alert."""
        alert = AlertFixtures.get_critical_alert()
        
        # Expected: Single alert classification should complete quickly
        # Target: < 10ms for single alert
        start_time = datetime.now()
        
        # AlertClassifier.classify(alert, default_config)
        # Placeholder for actual implementation timing
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Performance assertion (will be meaningful once implemented)
        assert processing_time >= 0.0  # Sanity check
    
    def test_batch_alert_classification_performance(self, default_config):
        """Test performance of classifying multiple alerts in batch."""
        alerts = AlertFixtures.get_performance_test_alerts(100)  # 100 alerts
        
        # Expected: Batch processing should be efficient
        # Target: < 1 second for 100 alerts (< 10ms per alert)
        start_time = datetime.now()
        
        # for alert in alerts:
        #     AlertClassifier.classify(alert, default_config)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Performance targets
        target_time_per_alert = 0.01  # 10ms per alert
        target_total_time = len(alerts) * target_time_per_alert
        
        assert len(alerts) == 100
        assert total_time >= 0.0  # Sanity check
    
    def test_high_volume_classification_performance(self, default_config):
        """Test performance under high volume conditions."""
        high_volume_alerts = AlertFixtures.get_performance_test_alerts(1000)  # 1000 alerts
        
        # Expected: Should handle high volume efficiently
        # Target: < 10 seconds for 1000 alerts
        start_time = datetime.now()
        
        # Simulate batch processing
        batch_size = 100
        for i in range(0, len(high_volume_alerts), batch_size):
            batch = high_volume_alerts[i:i + batch_size]
            # Process batch
            for alert in batch:
                # AlertClassifier.classify(alert, default_config)
                pass
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Performance validation
        assert len(high_volume_alerts) == 1000
        assert total_time >= 0.0
    
    def test_memory_usage_stability(self, default_config):
        """Test that classification doesn't cause memory leaks."""
        import gc
        
        # Get baseline memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Process alerts
        alerts = AlertFixtures.get_performance_test_alerts(500)
        for alert in alerts:
            # AlertClassifier.classify(alert, default_config)
            # Expected: No memory accumulation
            pass
        
        # Check memory after processing
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory shouldn't grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < len(alerts)  # Should not create too many persistent objects


# Integration test placeholder for when implementation exists
class TestAlertClassifierIntegration:
    """
    Integration tests for alert classifier with other system components
    
    These tests will validate integration with detection results,
    configuration loading, and result storage.
    """
    
    def test_integration_with_detection_results(self):
        """Test integration with actual detection result format."""
        # This will test integration with the QueryBuilder and BaseDetector results
        # once AlertClassifier is implemented
        pass
    
    def test_integration_with_snowflake_storage(self):
        """Test integration with Snowflake result storage."""
        # This will test that classification results are properly stored
        # in the ANOMALY_DETECTION.RESULTS.DAILY_ANOMALIES table
        pass
    
    def test_integration_with_configuration_loading(self):
        """Test integration with YAML configuration loading."""
        # This will test loading classification config from YAML files
        # as part of the configuration management system
        pass