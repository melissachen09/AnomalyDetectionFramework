-- Core Tables for Anomaly Detection Framework
-- ADF-22: Configure Snowflake Development Environment
--
-- This script creates the base tables needed for anomaly detection results,
-- configuration metadata, and alert tracking.

USE DATABASE ANOMALY_DETECTION_DEV;

-- =====================================================
-- DETECTION SCHEMA - Core anomaly detection results
-- =====================================================
USE SCHEMA DETECTION;

-- Main anomaly detection results table
CREATE TABLE IF NOT EXISTS DAILY_ANOMALIES (
    detection_id STRING DEFAULT UUID_STRING() PRIMARY KEY,
    detection_date DATE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    expected_value FLOAT,
    actual_value FLOAT NOT NULL,
    deviation_percentage FLOAT,
    deviation_absolute FLOAT,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'high', 'warning')),
    detection_method VARCHAR(50) NOT NULL,
    detector_config VARIANT,
    alert_sent BOOLEAN DEFAULT FALSE,
    alert_channels VARIANT,
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
) COMMENT = 'Daily anomaly detection results with severity and alert status';

-- Detector execution log
CREATE TABLE IF NOT EXISTS DETECTOR_EXECUTION_LOG (
    execution_id STRING DEFAULT UUID_STRING() PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    detector_type VARCHAR(50) NOT NULL,
    execution_date DATE NOT NULL,
    start_time TIMESTAMP_LTZ NOT NULL,
    end_time TIMESTAMP_LTZ,
    status VARCHAR(20) NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'timeout')),
    anomalies_found INTEGER DEFAULT 0,
    records_processed INTEGER DEFAULT 0,
    error_message TEXT,
    performance_metrics VARIANT,
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
) COMMENT = 'Log of detector execution with performance metrics';

-- =====================================================
-- CONFIG SCHEMA - Configuration metadata
-- =====================================================
USE SCHEMA CONFIG;

-- Event type configurations
CREATE TABLE IF NOT EXISTS EVENT_TYPES (
    event_type VARCHAR(100) PRIMARY KEY,
    display_name VARCHAR(200) NOT NULL,
    description TEXT,
    source_table VARCHAR(500) NOT NULL,
    date_column VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    config_file_path VARCHAR(500),
    last_updated TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
) COMMENT = 'Registry of configured event types and their data sources';

-- Detection rules metadata
CREATE TABLE IF NOT EXISTS DETECTION_RULES (
    rule_id STRING DEFAULT UUID_STRING() PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    detector_type VARCHAR(50) NOT NULL,
    rule_config VARIANT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (event_type) REFERENCES EVENT_TYPES(event_type)
) COMMENT = 'Detection rules configuration for each event type and metric';

-- =====================================================
-- ALERTS SCHEMA - Alert management
-- =====================================================
USE SCHEMA ALERTS;

-- Alert routing configuration
CREATE TABLE IF NOT EXISTS ALERT_ROUTING (
    routing_id STRING DEFAULT UUID_STRING() PRIMARY KEY,
    event_type VARCHAR(100),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'high', 'warning')),
    channel_type VARCHAR(50) NOT NULL CHECK (channel_type IN ('email', 'slack', 'dashboard')),
    recipient_config VARIANT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
) COMMENT = 'Configuration for alert routing by event type and severity';

-- Notification delivery log
CREATE TABLE IF NOT EXISTS NOTIFICATION_LOG (
    notification_id STRING DEFAULT UUID_STRING() PRIMARY KEY,
    detection_id STRING NOT NULL,
    channel_type VARCHAR(50) NOT NULL,
    recipient VARCHAR(500) NOT NULL,
    subject VARCHAR(500),
    message_content TEXT,
    sent_at TIMESTAMP_LTZ,
    delivery_status VARCHAR(50) NOT NULL CHECK (delivery_status IN ('pending', 'sent', 'failed', 'bounced')),
    error_message TEXT,
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (detection_id) REFERENCES DETECTION.DAILY_ANOMALIES(detection_id)
) COMMENT = 'Log of notification delivery attempts and status';

-- =====================================================
-- TESTING SCHEMA - Test data and validation
-- =====================================================
USE SCHEMA TESTING;

-- Test event data with known anomalies
CREATE TABLE IF NOT EXISTS TEST_EVENTS (
    test_id STRING DEFAULT UUID_STRING() PRIMARY KEY,
    event_date DATE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_type VARCHAR(50), -- 'spike', 'drop', 'drift', etc.
    anomaly_magnitude FLOAT, -- How much of an anomaly (percentage)
    description TEXT,
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
) COMMENT = 'Synthetic test data with known anomalies for validation';

-- Validation results
CREATE TABLE IF NOT EXISTS VALIDATION_RESULTS (
    validation_id STRING DEFAULT UUID_STRING() PRIMARY KEY,
    test_run_date DATE NOT NULL,
    detector_type VARCHAR(50) NOT NULL,
    total_tests INTEGER NOT NULL,
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_negatives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    notes TEXT,
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
) COMMENT = 'Results of detector validation against test data';

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS IDX_DAILY_ANOMALIES_DATE_EVENT 
ON DETECTION.DAILY_ANOMALIES (detection_date, event_type);

CREATE INDEX IF NOT EXISTS IDX_DAILY_ANOMALIES_SEVERITY 
ON DETECTION.DAILY_ANOMALIES (severity, detection_date);

CREATE INDEX IF NOT EXISTS IDX_EXECUTION_LOG_DATE_TYPE 
ON DETECTION.DETECTOR_EXECUTION_LOG (execution_date, detector_type);

CREATE INDEX IF NOT EXISTS IDX_TEST_EVENTS_DATE_TYPE 
ON TESTING.TEST_EVENTS (event_date, event_type);