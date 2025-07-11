"""
Alert Classifier Implementation.

This module implements the intelligent alert classification engine with multi-factor
severity calculation, business impact assessment, historical context, and explainable
classifications as specified in GADF-ALERT-002.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import math

from .types import Alert, AlertSeverity, BusinessImpact, HistoricalContext, ClassificationResult


logger = logging.getLogger(__name__)


class AlertClassifier:
    """
    Intelligent alert classification engine.
    
    Implements multi-factor severity calculation, business impact assessment,
    historical context integration, and explainable classifications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AlertClassifier with configuration.
        
        Args:
            config: Configuration dictionary with severity thresholds,
                   business impact weights, and other settings
        """
        self.config = config
        self._validate_config()
        
        # Cache for historical data to avoid repeated queries
        self._historical_cache = {}
        
        logger.info("AlertClassifier initialized with config keys: %s", 
                   list(config.keys()))
    
    def _validate_config(self):
        """Validate classifier configuration."""
        required_keys = ["severity_thresholds", "business_impact_weights"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate severity thresholds
        thresholds = self.config["severity_thresholds"]
        for severity in ["critical", "high", "medium", "low"]:
            if severity not in thresholds:
                raise ValueError(f"Missing severity threshold: {severity}")
            if not 0.0 <= thresholds[severity] <= 1.0:
                raise ValueError(f"Invalid threshold for {severity}: {thresholds[severity]}")
    
    def classify(self, alert: Alert) -> ClassificationResult:
        """
        Classify an alert and determine its severity.
        
        Args:
            alert: Alert to classify
            
        Returns:
            ClassificationResult with severity, confidence, and explanation
        """
        logger.debug("Classifying alert for metric: %s", alert.metric_name)
        
        # Calculate multi-factor severity score
        severity_score = self.calculate_severity_score(alert)
        
        # Determine severity level
        severity = self._score_to_severity(severity_score)
        
        # Assess business impact
        business_impact = self.assess_business_impact(alert)
        
        # Get historical context
        historical_context = self.get_historical_context(alert)
        
        # Calculate confidence
        confidence = self._calculate_confidence(alert, severity_score, historical_context)
        
        # Apply rules engine if configured
        rules_triggered = []
        if "rules" in self.config:
            severity, rules_triggered = self._apply_rules(alert, severity)
        
        # Generate explanation
        explanation = self._generate_explanation(
            alert, severity, severity_score, business_impact, 
            historical_context, rules_triggered
        )
        
        result = ClassificationResult(
            severity=severity,
            confidence=confidence,
            explanation=explanation,
            business_impact=business_impact,
            historical_context=historical_context,
            severity_score=severity_score,
            rules_triggered=rules_triggered
        )
        
        logger.info("Alert classified: %s (confidence: %.2f)", severity.value, confidence)
        return result
    
    def calculate_severity_score(self, alert: Alert) -> float:
        """
        Calculate multi-factor severity score.
        
        Considers deviation magnitude, business hours, trend direction,
        data quality, and other factors.
        
        Args:
            alert: Alert to score
            
        Returns:
            Severity score between 0.0 and 1.0
        """
        score = 0.0
        weights_sum = 0.0
        
        # Factor 1: Deviation magnitude (primary factor)
        deviation = alert.metadata.get("deviation_percentage", 0.0)
        if isinstance(deviation, (int, float)):
            score += deviation * 0.4
            weights_sum += 0.4
        
        # Factor 2: Business hours impact
        business_hours = alert.metadata.get("business_hours", True)
        if business_hours:
            score += 0.2
            weights_sum += 0.2
        else:
            score += 0.05  # Reduced impact after hours
            weights_sum += 0.2
        
        # Factor 3: Trend direction
        trend = alert.metadata.get("trend_direction", "unknown")
        if trend == "declining":
            score += 0.15
        elif trend == "volatile":
            score += 0.1
        elif trend == "stable":
            score += 0.05
        weights_sum += 0.15
        
        # Factor 4: Data quality
        data_quality = alert.metadata.get("data_quality_score", 0.8)
        if isinstance(data_quality, (int, float)):
            # Higher data quality increases confidence in the score
            quality_factor = min(data_quality, 1.0)
            score += 0.1 * quality_factor
            weights_sum += 0.1
        
        # Factor 5: Geographic/scope impact
        affected_regions = alert.metadata.get("affected_regions", [])
        if isinstance(affected_regions, list):
            if "all" in affected_regions or len(affected_regions) > 3:
                score += 0.1
            elif len(affected_regions) > 1:
                score += 0.05
            weights_sum += 0.1
        
        # Factor 6: Customer-facing impact
        customer_facing = alert.metadata.get("customer_facing", False)
        if customer_facing:
            score += 0.05
        weights_sum += 0.05
        
        # Normalize score if we have weights
        if weights_sum > 0:
            score = score / weights_sum
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def assess_business_impact(self, alert: Alert) -> BusinessImpact:
        """
        Assess business impact of the alert.
        
        Args:
            alert: Alert to assess
            
        Returns:
            BusinessImpact with calculated impact score and affected metrics
        """
        weights = self.config["business_impact_weights"]
        
        # Get impact factors from alert metadata
        revenue_impact = alert.metadata.get("revenue_impact", 0.0)
        customer_exp_impact = alert.metadata.get("customer_experience", 0.0)
        operational_impact = alert.metadata.get("operational_impact", 0.0)
        
        # If not provided in metadata, calculate based on metric type and deviation
        if revenue_impact == 0.0:
            revenue_impact = self._estimate_revenue_impact(alert)
        
        if customer_exp_impact == 0.0:
            customer_exp_impact = self._estimate_customer_experience_impact(alert)
        
        if operational_impact == 0.0:
            operational_impact = self._estimate_operational_impact(alert)
        
        # Calculate weighted impact score
        impact_score = (
            revenue_impact * weights.get("revenue_impact", 0.4) +
            customer_exp_impact * weights.get("customer_experience", 0.3) +
            operational_impact * weights.get("operational_impact", 0.3)
        )
        
        # Determine affected metrics
        affected_metrics = alert.metadata.get("affected_metrics", [alert.metric_name])
        if not isinstance(affected_metrics, list):
            affected_metrics = [alert.metric_name]
        
        return BusinessImpact(
            impact_score=max(0.0, min(1.0, impact_score)),
            affected_metrics=affected_metrics,
            revenue_impact=revenue_impact,
            customer_experience_impact=customer_exp_impact,
            operational_impact=operational_impact
        )
    
    def get_historical_context(self, alert: Alert) -> HistoricalContext:
        """
        Get historical context for the alert.
        
        Args:
            alert: Alert to get context for
            
        Returns:
            HistoricalContext with similar alerts and frequency
        """
        cache_key = f"{alert.metric_name}_{alert.timestamp.date()}"
        
        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]
        
        # Query historical alerts (mock implementation for now)
        similar_alerts = self._query_historical_alerts(alert)
        
        # Calculate frequency (alerts per day)
        lookback_days = self.config.get("historical_lookback_days", 30)
        frequency = len(similar_alerts) / max(lookback_days, 1)
        
        # Find last occurrence and average severity
        last_occurrence = None
        severity_sum = 0
        severity_count = 0
        
        for hist_alert in similar_alerts:
            if "timestamp" in hist_alert:
                timestamp = hist_alert["timestamp"]
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if last_occurrence is None or timestamp > last_occurrence:
                    last_occurrence = timestamp
            
            if "severity" in hist_alert:
                severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                severity_sum += severity_map.get(hist_alert["severity"], 2)
                severity_count += 1
        
        average_severity = None
        if severity_count > 0:
            avg_score = severity_sum / severity_count
            severity_map_reverse = {4: "critical", 3: "high", 2: "medium", 1: "low"}
            average_severity = severity_map_reverse[round(avg_score)]
        
        context = HistoricalContext(
            similar_alerts=similar_alerts,
            frequency=frequency,
            last_occurrence=last_occurrence,
            average_severity=average_severity
        )
        
        # Cache the result
        self._historical_cache[cache_key] = context
        
        return context
    
    def _score_to_severity(self, score: float) -> AlertSeverity:
        """Convert severity score to AlertSeverity enum."""
        thresholds = self.config["severity_thresholds"]
        
        if score >= thresholds["critical"]:
            return AlertSeverity.CRITICAL
        elif score >= thresholds["high"]:
            return AlertSeverity.HIGH
        elif score >= thresholds["medium"]:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _calculate_confidence(self, alert: Alert, severity_score: float, 
                            historical_context: HistoricalContext) -> float:
        """Calculate confidence in the classification."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence if we have good data quality
        data_quality = alert.metadata.get("data_quality_score", 0.8)
        confidence += (data_quality - 0.5) * 0.3
        
        # Higher confidence if we have historical precedent
        if historical_context.frequency > 0:
            confidence += 0.2
            
        # Higher confidence for clear deviations
        if alert.metadata.get("clear_deviation", False):
            confidence += 0.2
        
        # Lower confidence for edge cases
        if severity_score < 0.1 or severity_score > 0.95:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _apply_rules(self, alert: Alert, base_severity: AlertSeverity) -> tuple:
        """Apply rules engine to modify severity."""
        rules_triggered = []
        current_severity = base_severity
        
        for rule in self.config.get("rules", []):
            if self._rule_matches(alert, rule):
                rules_triggered.append(rule["name"])
                
                # Apply rule modifications
                if "severity" in rule:
                    rule_severity = AlertSeverity(rule["severity"])
                    # Take the higher severity
                    if self._severity_to_numeric(rule_severity) > self._severity_to_numeric(current_severity):
                        current_severity = rule_severity
                
                if "severity_modifier" in rule:
                    # Apply modifier to severity score and recalculate
                    modifier = rule["severity_modifier"]
                    current_score = self._severity_to_numeric(current_severity) / 4.0
                    new_score = max(0.0, min(1.0, current_score + modifier))
                    current_severity = self._score_to_severity(new_score)
        
        return current_severity, rules_triggered
    
    def _rule_matches(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if a rule matches the alert."""
        conditions = rule.get("conditions", {})
        
        for condition, expected_value in conditions.items():
            if condition == "metric_name":
                if alert.metric_name != expected_value:
                    return False
            elif condition == "deviation_threshold":
                deviation = alert.metadata.get("deviation_percentage", 0.0)
                if deviation < expected_value:
                    return False
            elif condition in alert.metadata:
                if alert.metadata[condition] != expected_value:
                    return False
        
        return True
    
    def _severity_to_numeric(self, severity: AlertSeverity) -> int:
        """Convert severity to numeric value for comparison."""
        mapping = {
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4
        }
        return mapping[severity]
    
    def _estimate_revenue_impact(self, alert: Alert) -> float:
        """Estimate revenue impact based on metric and deviation."""
        revenue_metrics = ["revenue", "sales", "conversion", "checkout"]
        
        if any(metric in alert.metric_name.lower() for metric in revenue_metrics):
            deviation = alert.metadata.get("deviation_percentage", 0.0)
            return min(1.0, deviation * 1.2)  # Amplify revenue impact
        
        return 0.3  # Default moderate impact
    
    def _estimate_customer_experience_impact(self, alert: Alert) -> float:
        """Estimate customer experience impact."""
        customer_metrics = ["page_views", "session", "bounce", "error", "latency"]
        
        if any(metric in alert.metric_name.lower() for metric in customer_metrics):
            deviation = alert.metadata.get("deviation_percentage", 0.0)
            return min(1.0, deviation)
        
        return 0.2  # Default low impact
    
    def _estimate_operational_impact(self, alert: Alert) -> float:
        """Estimate operational impact."""
        operational_metrics = ["system", "database", "api", "service", "infrastructure"]
        
        if any(metric in alert.metric_name.lower() for metric in operational_metrics):
            deviation = alert.metadata.get("deviation_percentage", 0.0)
            return min(1.0, deviation * 0.8)
        
        return 0.25  # Default moderate impact
    
    def _query_historical_alerts(self, alert: Alert) -> List[Dict[str, Any]]:
        """
        Query historical alerts similar to the current one.
        
        This is a mock implementation. In a real system, this would
        query Snowflake for historical alert data.
        """
        # Mock historical data for testing
        lookback_days = self.config.get("historical_lookback_days", 30)
        base_date = alert.timestamp - timedelta(days=lookback_days)
        
        # Generate some mock historical alerts
        historical_alerts = []
        
        # Add some similar alerts based on metric name
        if "revenue" in alert.metric_name.lower():
            historical_alerts.extend([
                {
                    "alert_id": "hist_1",
                    "timestamp": base_date + timedelta(days=5),
                    "severity": "high",
                    "metric_name": alert.metric_name
                },
                {
                    "alert_id": "hist_2", 
                    "timestamp": base_date + timedelta(days=15),
                    "severity": "medium",
                    "metric_name": alert.metric_name
                }
            ])
        elif "views" in alert.metric_name.lower():
            historical_alerts.extend([
                {
                    "alert_id": "hist_3",
                    "timestamp": base_date + timedelta(days=2),
                    "severity": "low",
                    "metric_name": alert.metric_name
                }
            ])
        
        return historical_alerts
    
    def _generate_explanation(self, alert: Alert, severity: AlertSeverity, 
                            severity_score: float, business_impact: BusinessImpact,
                            historical_context: HistoricalContext,
                            rules_triggered: List[str]) -> str:
        """Generate human-readable explanation for the classification."""
        explanation_parts = []
        
        # Start with severity and confidence
        explanation_parts.append(
            f"Alert classified as {severity.value.upper()} severity "
            f"(score: {severity_score:.2f}) for metric '{alert.metric_name}'"
        )
        
        # Add deviation information
        deviation = alert.metadata.get("deviation_percentage", 0.0)
        if deviation > 0:
            explanation_parts.append(
                f"showing {deviation*100:.1f}% deviation from expected threshold"
            )
        
        # Add business impact details
        if business_impact.impact_score > 0.3:
            impact_level = "high" if business_impact.impact_score > 0.7 else "moderate"
            explanation_parts.append(
                f"with {impact_level} business impact (score: {business_impact.impact_score:.2f})"
            )
        
        # Add historical context
        if historical_context.frequency > 0:
            explanation_parts.append(
                f"Historical analysis shows {historical_context.frequency:.1f} similar alerts per day"
            )
            if historical_context.average_severity:
                explanation_parts.append(
                    f"with average historical severity of {historical_context.average_severity}"
                )
        
        # Add rules information
        if rules_triggered:
            explanation_parts.append(
                f"Triggered classification rules: {', '.join(rules_triggered)}"
            )
        
        # Add timing context
        business_hours = alert.metadata.get("business_hours", True)
        time_context = "during business hours" if business_hours else "outside business hours"
        explanation_parts.append(f"Alert occurred {time_context}")
        
        # Add affected metrics
        if len(business_impact.affected_metrics) > 1:
            explanation_parts.append(
                f"Potentially affecting metrics: {', '.join(business_impact.affected_metrics)}"
            )
        
        return ". ".join(explanation_parts) + "."