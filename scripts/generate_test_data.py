#!/usr/bin/env python3
"""
Simple test data generator for Snowflake development environment
ADF-22: Configure Snowflake Development Environment

This script generates synthetic test data using basic SQL that can be
executed through the MCP Snowflake tool.
"""

import random
import datetime
from typing import List, Dict, Any

def generate_test_data_sql() -> List[str]:
    """Generate SQL statements for test data creation"""
    
    # Generate 30 days of sample data
    base_date = datetime.date.today()
    sql_statements = []
    
    # Generate normal baseline data for listing views
    for i in range(25):
        event_date = base_date - datetime.timedelta(days=i+1)
        normal_value = 50000 + random.randint(-5000, 5000)
        
        sql = f"""
        INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('{event_date}', 'listing_views', 'total_views', {normal_value}, FALSE, 'Normal baseline data')
        """
        sql_statements.append(sql.strip())
    
    # Add known anomalies
    anomalies = [
        {
            'date': base_date - datetime.timedelta(days=3),
            'value': 100000,
            'type': 'spike',
            'magnitude': 1.0,
            'description': 'Simulated traffic spike from viral listing'
        },
        {
            'date': base_date - datetime.timedelta(days=5),
            'value': 15000,
            'type': 'drop',
            'magnitude': -0.7,
            'description': 'Simulated system outage impact'
        },
        {
            'date': base_date - datetime.timedelta(days=7),
            'value': 75000,
            'type': 'drift',
            'magnitude': 0.5,
            'description': 'Gradual increase trend'
        },
        {
            'date': base_date - datetime.timedelta(days=10),
            'value': 62000,
            'type': 'spike',
            'magnitude': 0.24,
            'description': 'Minor traffic increase'
        }
    ]
    
    for anomaly in anomalies:
        sql = f"""
        INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, anomaly_type, anomaly_magnitude, description)
        VALUES 
        ('{anomaly['date']}', 'listing_views', 'total_views', {anomaly['value']}, TRUE, 
         '{anomaly['type']}', {anomaly['magnitude']}, '{anomaly['description']}')
        """
        sql_statements.append(sql.strip())
    
    return sql_statements

def generate_config_data_sql() -> List[str]:
    """Generate SQL statements for configuration data"""
    
    sql_statements = []
    
    # Event types
    event_types = [
        ('listing_views', 'Property Listing Views', 'Daily views of property listings', 'DATAMART.DD_LISTING_STATISTICS_BLENDED', 'STATISTIC_DATE'),
        ('listing_enquiries', 'Property Enquiries', 'Daily enquiries for property listings', 'DATAMART.DD_LISTING_STATISTICS_BLENDED', 'STATISTIC_DATE'),
        ('page_views', 'Website Page Views', 'Daily website page view metrics', 'DATAMART.FACT_LISTING_STATISTICS_UTM', 'STATISTIC_DATE')
    ]
    
    for event_type, display_name, description, source_table, date_column in event_types:
        sql = f"""
        INSERT INTO CONFIG.EVENT_TYPES 
        (event_type, display_name, description, source_table, date_column)
        VALUES 
        ('{event_type}', '{display_name}', '{description}', '{source_table}', '{date_column}')
        """
        sql_statements.append(sql.strip())
    
    # Detection rules
    detection_rules = [
        ('listing_views', 'total_views', 'threshold', '{"min_value": 10000, "max_value": 1000000, "percentage_change_threshold": 0.3}'),
        ('listing_views', 'total_views', 'statistical', '{"method": "zscore", "threshold": 3.0, "lookback_days": 30}'),
        ('listing_enquiries', 'enquiry_count', 'threshold', '{"min_value": 500, "max_value": 50000, "percentage_change_threshold": 0.5}')
    ]
    
    for event_type, metric_name, detector_type, config_json in detection_rules:
        sql = f"""
        INSERT INTO CONFIG.DETECTION_RULES 
        (event_type, metric_name, detector_type, rule_config)
        VALUES 
        ('{event_type}', '{metric_name}', '{detector_type}', PARSE_JSON('{config_json}'))
        """
        sql_statements.append(sql.strip())
    
    return sql_statements

def generate_alert_config_sql() -> List[str]:
    """Generate SQL statements for alert configuration"""
    
    sql_statements = []
    
    # Alert routing rules
    alert_configs = [
        ('critical', 'email', '{"recipients": ["director-bi@company.com"], "template": "critical_alert"}'),
        ('critical', 'slack', '{"channel": "#alerts-critical", "webhook_url": "https://hooks.slack.com/critical"}'),
        ('high', 'email', '{"recipients": ["data-team@company.com"], "template": "high_alert"}'),
        ('high', 'slack', '{"channel": "#data-alerts", "webhook_url": "https://hooks.slack.com/data"}'),
        ('warning', 'dashboard', '{"dashboard_id": "anomaly_dashboard", "update_frequency": "daily"}')
    ]
    
    for severity, channel_type, config_json in alert_configs:
        sql = f"""
        INSERT INTO ALERTS.ALERT_ROUTING 
        (event_type, severity, channel_type, recipient_config)
        VALUES 
        (NULL, '{severity}', '{channel_type}', PARSE_JSON('{config_json}'))
        """
        sql_statements.append(sql.strip())
    
    return sql_statements

def main():
    """Generate all SQL statements for manual execution"""
    
    print("=== Snowflake Development Environment Test Data ===")
    print("Generated SQL statements for manual execution:")
    print()
    
    print("-- 1. Configuration Data")
    print("-- Execute these first to set up event types and detection rules")
    for sql in generate_config_data_sql():
        print(sql)
        print()
    
    print("-- 2. Alert Configuration")  
    print("-- Execute these to set up alert routing")
    for sql in generate_alert_config_sql():
        print(sql)
        print()
    
    print("-- 3. Test Data Generation")
    print("-- Execute these to create synthetic test data with known anomalies")
    for sql in generate_test_data_sql():
        print(sql)
        print()
    
    print("-- 4. Verification Queries")
    print("-- Run these to verify the setup")
    verification_queries = [
        "SELECT COUNT(*) as event_types_count FROM CONFIG.EVENT_TYPES;",
        "SELECT COUNT(*) as detection_rules_count FROM CONFIG.DETECTION_RULES;",
        "SELECT COUNT(*) as alert_routing_count FROM ALERTS.ALERT_ROUTING;",
        "SELECT COUNT(*) as test_events_count FROM TESTING.TEST_EVENTS;",
        "SELECT COUNT(*) as anomalies_count FROM TESTING.TEST_EVENTS WHERE is_anomaly = TRUE;",
        """
        SELECT 
            event_type,
            COUNT(*) as total_events,
            SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count,
            AVG(metric_value) as avg_value,
            MIN(metric_value) as min_value,
            MAX(metric_value) as max_value
        FROM TESTING.TEST_EVENTS
        GROUP BY event_type
        ORDER BY event_type;
        """
    ]
    
    for query in verification_queries:
        print(query)
        print()

if __name__ == "__main__":
    main()