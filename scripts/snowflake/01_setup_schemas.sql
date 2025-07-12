-- Snowflake Development Environment Setup
-- ADF-22: Configure Snowflake Development Environment
-- 
-- This script creates the required schemas for the anomaly detection framework
-- development environment.

USE DATABASE ANOMALY_DETECTION_DEV;

-- Create core schemas for anomaly detection framework
CREATE SCHEMA IF NOT EXISTS CONFIG 
    COMMENT = 'Schema for configuration tables and metadata';

CREATE SCHEMA IF NOT EXISTS DETECTION 
    COMMENT = 'Schema for detection results and anomaly data';

CREATE SCHEMA IF NOT EXISTS ALERTS 
    COMMENT = 'Schema for alert management and notification tracking';

CREATE SCHEMA IF NOT EXISTS TESTING 
    COMMENT = 'Schema for test data and validation';

-- Grant usage to the service account role
GRANT USAGE ON SCHEMA CONFIG TO ROLE ACCOUNTADMIN;
GRANT USAGE ON SCHEMA DETECTION TO ROLE ACCOUNTADMIN;
GRANT USAGE ON SCHEMA ALERTS TO ROLE ACCOUNTADMIN;
GRANT USAGE ON SCHEMA TESTING TO ROLE ACCOUNTADMIN;

-- Grant full privileges for development
GRANT ALL PRIVILEGES ON SCHEMA CONFIG TO ROLE ACCOUNTADMIN;
GRANT ALL PRIVILEGES ON SCHEMA DETECTION TO ROLE ACCOUNTADMIN;
GRANT ALL PRIVILEGES ON SCHEMA ALERTS TO ROLE ACCOUNTADMIN;
GRANT ALL PRIVILEGES ON SCHEMA TESTING TO ROLE ACCOUNTADMIN;

-- List all schemas to verify creation
SELECT SCHEMA_NAME, COMMENT 
FROM INFORMATION_SCHEMA.SCHEMATA 
WHERE CATALOG_NAME = 'ANOMALY_DETECTION_DEV'
ORDER BY SCHEMA_NAME;