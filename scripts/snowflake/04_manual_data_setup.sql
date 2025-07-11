=== Snowflake Development Environment Test Data ===
Generated SQL statements for manual execution:

-- 1. Configuration Data
-- Execute these first to set up event types and detection rules
INSERT INTO CONFIG.EVENT_TYPES 
        (event_type, display_name, description, source_table, date_column)
        VALUES 
        ('listing_views', 'Property Listing Views', 'Daily views of property listings', 'DATAMART.DD_LISTING_STATISTICS_BLENDED', 'STATISTIC_DATE')

INSERT INTO CONFIG.EVENT_TYPES 
        (event_type, display_name, description, source_table, date_column)
        VALUES 
        ('listing_enquiries', 'Property Enquiries', 'Daily enquiries for property listings', 'DATAMART.DD_LISTING_STATISTICS_BLENDED', 'STATISTIC_DATE')

INSERT INTO CONFIG.EVENT_TYPES 
        (event_type, display_name, description, source_table, date_column)
        VALUES 
        ('page_views', 'Website Page Views', 'Daily website page view metrics', 'DATAMART.FACT_LISTING_STATISTICS_UTM', 'STATISTIC_DATE')

INSERT INTO CONFIG.DETECTION_RULES 
        (event_type, metric_name, detector_type, rule_config)
        VALUES 
        ('listing_views', 'total_views', 'threshold', PARSE_JSON('{"min_value": 10000, "max_value": 1000000, "percentage_change_threshold": 0.3}'))

INSERT INTO CONFIG.DETECTION_RULES 
        (event_type, metric_name, detector_type, rule_config)
        VALUES 
        ('listing_views', 'total_views', 'statistical', PARSE_JSON('{"method": "zscore", "threshold": 3.0, "lookback_days": 30}'))

INSERT INTO CONFIG.DETECTION_RULES 
        (event_type, metric_name, detector_type, rule_config)
        VALUES 
        ('listing_enquiries', 'enquiry_count', 'threshold', PARSE_JSON('{"min_value": 500, "max_value": 50000, "percentage_change_threshold": 0.5}'))

-- 2. Alert Configuration
-- Execute these to set up alert routing
INSERT INTO ALERTS.ALERT_ROUTING 
        (event_type, severity, channel_type, recipient_config)
        VALUES 
        (NULL, 'critical', 'email', PARSE_JSON('{"recipients": ["director-bi@company.com"], "template": "critical_alert"}'))

INSERT INTO ALERTS.ALERT_ROUTING 
        (event_type, severity, channel_type, recipient_config)
        VALUES 
        (NULL, 'critical', 'slack', PARSE_JSON('{"channel": "#alerts-critical", "webhook_url": "https://hooks.slack.com/critical"}'))

INSERT INTO ALERTS.ALERT_ROUTING 
        (event_type, severity, channel_type, recipient_config)
        VALUES 
        (NULL, 'high', 'email', PARSE_JSON('{"recipients": ["data-team@company.com"], "template": "high_alert"}'))

INSERT INTO ALERTS.ALERT_ROUTING 
        (event_type, severity, channel_type, recipient_config)
        VALUES 
        (NULL, 'high', 'slack', PARSE_JSON('{"channel": "#data-alerts", "webhook_url": "https://hooks.slack.com/data"}'))

INSERT INTO ALERTS.ALERT_ROUTING 
        (event_type, severity, channel_type, recipient_config)
        VALUES 
        (NULL, 'warning', 'dashboard', PARSE_JSON('{"dashboard_id": "anomaly_dashboard", "update_frequency": "daily"}'))

-- 3. Test Data Generation
-- Execute these to create synthetic test data with known anomalies
INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-07-11', 'listing_views', 'total_views', 47933, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-07-10', 'listing_views', 'total_views', 48380, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-07-09', 'listing_views', 'total_views', 48588, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-07-08', 'listing_views', 'total_views', 46671, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-07-07', 'listing_views', 'total_views', 50381, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-07-06', 'listing_views', 'total_views', 54784, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-07-05', 'listing_views', 'total_views', 48784, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-07-04', 'listing_views', 'total_views', 54775, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-07-03', 'listing_views', 'total_views', 53438, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-07-02', 'listing_views', 'total_views', 46497, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-07-01', 'listing_views', 'total_views', 45565, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-30', 'listing_views', 'total_views', 52387, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-29', 'listing_views', 'total_views', 50186, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-28', 'listing_views', 'total_views', 50051, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-27', 'listing_views', 'total_views', 54955, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-26', 'listing_views', 'total_views', 49842, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-25', 'listing_views', 'total_views', 51142, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-24', 'listing_views', 'total_views', 46929, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-23', 'listing_views', 'total_views', 47948, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-22', 'listing_views', 'total_views', 46978, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-21', 'listing_views', 'total_views', 52961, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-20', 'listing_views', 'total_views', 47872, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-19', 'listing_views', 'total_views', 48060, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-18', 'listing_views', 'total_views', 48395, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, description)
        VALUES 
        ('2025-06-17', 'listing_views', 'total_views', 50778, FALSE, 'Normal baseline data')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, anomaly_type, anomaly_magnitude, description)
        VALUES 
        ('2025-07-09', 'listing_views', 'total_views', 100000, TRUE, 
         'spike', 1.0, 'Simulated traffic spike from viral listing')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, anomaly_type, anomaly_magnitude, description)
        VALUES 
        ('2025-07-07', 'listing_views', 'total_views', 15000, TRUE, 
         'drop', -0.7, 'Simulated system outage impact')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, anomaly_type, anomaly_magnitude, description)
        VALUES 
        ('2025-07-05', 'listing_views', 'total_views', 75000, TRUE, 
         'drift', 0.5, 'Gradual increase trend')

INSERT INTO TESTING.TEST_EVENTS 
        (event_date, event_type, metric_name, metric_value, is_anomaly, anomaly_type, anomaly_magnitude, description)
        VALUES 
        ('2025-07-02', 'listing_views', 'total_views', 62000, TRUE, 
         'spike', 0.24, 'Minor traffic increase')

-- 4. Verification Queries
-- Run these to verify the setup
SELECT COUNT(*) as event_types_count FROM CONFIG.EVENT_TYPES;

SELECT COUNT(*) as detection_rules_count FROM CONFIG.DETECTION_RULES;

SELECT COUNT(*) as alert_routing_count FROM ALERTS.ALERT_ROUTING;

SELECT COUNT(*) as test_events_count FROM TESTING.TEST_EVENTS;

SELECT COUNT(*) as anomalies_count FROM TESTING.TEST_EVENTS WHERE is_anomaly = TRUE;


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
        

