-- Sample Test Data Generation
-- ADF-22: Configure Snowflake Development Environment
--
-- This script populates the development environment with synthetic test data
-- including known anomalies for validation and testing purposes.

USE DATABASE ANOMALY_DETECTION_DEV;

-- =====================================================
-- Insert Event Type Configurations
-- =====================================================
USE SCHEMA CONFIG;

INSERT INTO EVENT_TYPES (event_type, display_name, description, source_table, date_column) VALUES
('listing_views', 'Property Listing Views', 'Daily views of property listings', 'DATAMART.DD_LISTING_STATISTICS_BLENDED', 'STATISTIC_DATE'),
('listing_enquiries', 'Property Enquiries', 'Daily enquiries for property listings', 'DATAMART.DD_LISTING_STATISTICS_BLENDED', 'STATISTIC_DATE'),
('page_views', 'Website Page Views', 'Daily website page view metrics', 'DATAMART.FACT_LISTING_STATISTICS_UTM', 'STATISTIC_DATE'),
('user_registrations', 'User Registrations', 'Daily new user registrations', 'DATAMART.USER_ACTIVITY_DAILY', 'ACTIVITY_DATE'),
('search_queries', 'Search Queries', 'Daily search query volumes', 'DATAMART.SEARCH_METRICS_DAILY', 'SEARCH_DATE');

-- Insert sample detection rules
INSERT INTO DETECTION_RULES (event_type, metric_name, detector_type, rule_config) VALUES
('listing_views', 'total_views', 'threshold', 
    PARSE_JSON('{"min_value": 10000, "max_value": 1000000, "percentage_change_threshold": 0.3}')),
('listing_views', 'total_views', 'statistical', 
    PARSE_JSON('{"method": "zscore", "threshold": 3.0, "lookback_days": 30}')),
('listing_enquiries', 'enquiry_count', 'threshold', 
    PARSE_JSON('{"min_value": 500, "max_value": 50000, "percentage_change_threshold": 0.5}')),
('page_views', 'total_page_views', 'statistical', 
    PARSE_JSON('{"method": "moving_average", "window_size": 7, "threshold": 0.25}')),
('user_registrations', 'registration_count', 'threshold', 
    PARSE_JSON('{"min_value": 100, "max_value": 10000, "percentage_change_threshold": 0.4}');

-- =====================================================
-- Insert Alert Routing Configuration  
-- =====================================================
USE SCHEMA ALERTS;

INSERT INTO ALERT_ROUTING (event_type, severity, channel_type, recipient_config) VALUES
(NULL, 'critical', 'email', PARSE_JSON('{"recipients": ["director-bi@company.com"], "template": "critical_alert"}')),
(NULL, 'critical', 'slack', PARSE_JSON('{"channel": "#alerts-critical", "webhook_url": "https://hooks.slack.com/critical"}')),
(NULL, 'high', 'email', PARSE_JSON('{"recipients": ["data-team@company.com"], "template": "high_alert"}')),
(NULL, 'high', 'slack', PARSE_JSON('{"channel": "#data-alerts", "webhook_url": "https://hooks.slack.com/data"}')),
(NULL, 'warning', 'dashboard', PARSE_JSON('{"dashboard_id": "anomaly_dashboard", "update_frequency": "daily"}'));

-- =====================================================
-- Generate Synthetic Test Data with Known Anomalies
-- =====================================================
USE SCHEMA TESTING;

-- Generate baseline normal data for listing views (30 days)
INSERT INTO TEST_EVENTS (event_date, event_type, metric_name, metric_value, is_anomaly, description)
SELECT 
    DATEADD(day, -seq, CURRENT_DATE()) as event_date,
    'listing_views' as event_type,
    'total_views' as metric_name,
    -- Normal baseline: 50k +/- random variation
    50000 + (RANDOM() * 10000 - 5000) as metric_value,
    FALSE as is_anomaly,
    'Normal baseline data' as description
FROM (SELECT ROW_NUMBER() OVER (ORDER BY NULL) as seq FROM TABLE(GENERATOR(ROWCOUNT => 25)));

-- Insert known anomalies for testing
INSERT INTO TEST_EVENTS (event_date, event_type, metric_name, metric_value, is_anomaly, anomaly_type, anomaly_magnitude, description) VALUES
-- Critical spike (100% increase)
(DATEADD(day, -5, CURRENT_DATE()), 'listing_views', 'total_views', 100000, TRUE, 'spike', 1.0, 'Simulated traffic spike from viral listing'),
-- Critical drop (70% decrease)  
(DATEADD(day, -3, CURRENT_DATE()), 'listing_views', 'total_views', 15000, TRUE, 'drop', -0.7, 'Simulated system outage impact'),
-- High severity gradual increase
(DATEADD(day, -10, CURRENT_DATE()), 'listing_views', 'total_views', 75000, TRUE, 'drift', 0.5, 'Gradual increase trend'),
-- Warning level minor spike
(DATEADD(day, -7, CURRENT_DATE()), 'listing_views', 'total_views', 62000, TRUE, 'spike', 0.24, 'Minor traffic increase');

-- Generate test data for enquiries
INSERT INTO TEST_EVENTS (event_date, event_type, metric_name, metric_value, is_anomaly, description)
SELECT 
    DATEADD(day, -seq, CURRENT_DATE()) as event_date,
    'listing_enquiries' as event_type,
    'enquiry_count' as metric_name,
    -- Normal baseline: 2k +/- random variation
    2000 + (RANDOM() * 400 - 200) as metric_value,
    FALSE as is_anomaly,
    'Normal enquiry baseline' as description
FROM (SELECT ROW_NUMBER() OVER (ORDER BY NULL) as seq FROM TABLE(GENERATOR(ROWCOUNT => 25)));

-- Enquiry anomalies
INSERT INTO TEST_EVENTS (event_date, event_type, metric_name, metric_value, is_anomaly, anomaly_type, anomaly_magnitude, description) VALUES
(DATEADD(day, -4, CURRENT_DATE()), 'listing_enquiries', 'enquiry_count', 4500, TRUE, 'spike', 1.25, 'High enquiry volume day'),
(DATEADD(day, -8, CURRENT_DATE()), 'listing_enquiries', 'enquiry_count', 800, TRUE, 'drop', -0.6, 'Low enquiry volume day');

-- Generate page view test data
INSERT INTO TEST_EVENTS (event_date, event_type, metric_name, metric_value, is_anomaly, description)
SELECT 
    DATEADD(day, -seq, CURRENT_DATE()) as event_date,
    'page_views' as event_type,
    'total_page_views' as metric_name,
    -- Normal baseline: 150k +/- random variation
    150000 + (RANDOM() * 20000 - 10000) as metric_value,
    FALSE as is_anomaly,
    'Normal page view baseline' as description
FROM (SELECT ROW_NUMBER() OVER (ORDER BY NULL) as seq FROM TABLE(GENERATOR(ROWCOUNT => 25)));

-- Page view anomalies
INSERT INTO TEST_EVENTS (event_date, event_type, metric_name, metric_value, is_anomaly, anomaly_type, anomaly_magnitude, description) VALUES
(DATEADD(day, -6, CURRENT_DATE()), 'page_views', 'total_page_views', 280000, TRUE, 'spike', 0.87, 'Major page view spike'),
(DATEADD(day, -2, CURRENT_DATE()), 'page_views', 'total_page_views', 95000, TRUE, 'drop', -0.37, 'Page view drop');

-- =====================================================
-- Create Views for Easy Data Access
-- =====================================================

-- Summary view of test anomalies by type
CREATE OR REPLACE VIEW TESTING.ANOMALY_SUMMARY AS
SELECT 
    event_type,
    anomaly_type,
    COUNT(*) as anomaly_count,
    AVG(anomaly_magnitude) as avg_magnitude,
    MIN(event_date) as first_anomaly,
    MAX(event_date) as last_anomaly
FROM TEST_EVENTS 
WHERE is_anomaly = TRUE
GROUP BY event_type, anomaly_type
ORDER BY event_type, anomaly_type;

-- Daily test data summary
CREATE OR REPLACE VIEW TESTING.DAILY_TEST_SUMMARY AS
SELECT 
    event_date,
    event_type,
    COUNT(*) as total_events,
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value
FROM TEST_EVENTS
GROUP BY event_date, event_type
ORDER BY event_date DESC, event_type;

-- View showing configuration completeness
CREATE OR REPLACE VIEW CONFIG.CONFIG_COMPLETENESS AS
SELECT 
    et.event_type,
    et.display_name,
    et.is_active,
    COUNT(dr.rule_id) as rule_count,
    COUNT(DISTINCT dr.detector_type) as detector_types,
    LISTAGG(DISTINCT dr.detector_type, ', ') as detector_list
FROM EVENT_TYPES et
LEFT JOIN DETECTION_RULES dr ON et.event_type = dr.event_type AND dr.is_active = TRUE
GROUP BY et.event_type, et.display_name, et.is_active
ORDER BY et.event_type;