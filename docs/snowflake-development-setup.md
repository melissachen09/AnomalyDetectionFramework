# Snowflake Development Environment Setup Guide

## Overview

This guide provides complete setup instructions for the Anomaly Detection Framework's Snowflake development environment. The setup includes database schemas, tables, sample data, and connection configuration.

## Prerequisites

- Access to Snowflake account: `ya78352.ap-southeast-2`
- Service account: `APP_CLAUDE` 
- Snowflake Web UI or SnowSQL client access
- Python 3.9+ (for local development)

## Environment Details

### Connection Information
```
Account: ya78352.ap-southeast-2
Database: ANOMALY_DETECTION_DEV
User: APP_CLAUDE
Role: ACCOUNTADMIN
Warehouse: COMPUTE_WH
```

### Environment Variables
```bash
export SNOWFLAKE_ACCOUNT=ya78352.ap-southeast-2
export SNOWFLAKE_USER=APP_CLAUDE
export SNOWFLAKE_PASSWORD=B9Dbz@jNCiWu@111
export SNOWFLAKE_WAREHOUSE=COMPUTE_WH
export SNOWFLAKE_DATABASE=ANOMALY_DETECTION_DEV
export SNOWFLAKE_ROLE=ACCOUNTADMIN
```

## Setup Steps

### 1. Verify Database Access

First, verify connection to the development database:

```sql
SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_SCHEMA();
```

Expected output:
- User: APP_CLAUDE
- Role: ACCOUNTADMIN
- Database: ANOMALY_DETECTION_DEV
- Schema: PUBLIC

### 2. Create Schemas

Execute the following commands in Snowflake Web UI or SnowSQL:

```sql
USE DATABASE ANOMALY_DETECTION_DEV;

-- Core schemas for anomaly detection framework
CREATE SCHEMA IF NOT EXISTS CONFIG 
    COMMENT = 'Schema for configuration tables and metadata';

CREATE SCHEMA IF NOT EXISTS DETECTION 
    COMMENT = 'Schema for detection results and anomaly data';

CREATE SCHEMA IF NOT EXISTS ALERTS 
    COMMENT = 'Schema for alert management and notification tracking';

CREATE SCHEMA IF NOT EXISTS TESTING 
    COMMENT = 'Schema for test data and validation';
```

### 3. Create Tables

Execute the table creation scripts located in `scripts/snowflake/`:

1. **Schema setup**: `01_setup_schemas.sql`
2. **Table creation**: `02_create_tables.sql`
3. **Sample data**: `03_populate_test_data.sql`

### 4. Core Tables Structure

#### CONFIG Schema
- **EVENT_TYPES**: Registry of configured event types and data sources
- **DETECTION_RULES**: Detection rules configuration for each metric

#### DETECTION Schema  
- **DAILY_ANOMALIES**: Main anomaly detection results
- **DETECTOR_EXECUTION_LOG**: Performance and execution tracking

#### ALERTS Schema
- **ALERT_ROUTING**: Alert routing configuration by severity
- **NOTIFICATION_LOG**: Notification delivery tracking

#### TESTING Schema
- **TEST_EVENTS**: Synthetic test data with known anomalies
- **VALIDATION_RESULTS**: Detector validation metrics

### 5. Populate Sample Data

Use the generated SQL statements in `scripts/snowflake/04_manual_data_setup.sql` to:

- Configure event types (listing_views, listing_enquiries, page_views)
- Set up detection rules (threshold, statistical)
- Configure alert routing (email, Slack, dashboard)
- Generate synthetic test data with known anomalies

### 6. Verification

After setup, verify the environment:

```sql
-- Check schemas
SELECT SCHEMA_NAME, COMMENT 
FROM INFORMATION_SCHEMA.SCHEMATA 
WHERE CATALOG_NAME = 'ANOMALY_DETECTION_DEV'
ORDER BY SCHEMA_NAME;

-- Check tables
SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE, COMMENT
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_CATALOG = 'ANOMALY_DETECTION_DEV'
AND TABLE_SCHEMA IN ('CONFIG', 'DETECTION', 'ALERTS', 'TESTING')
ORDER BY TABLE_SCHEMA, TABLE_NAME;

-- Verify sample data
SELECT COUNT(*) as event_types_count FROM CONFIG.EVENT_TYPES;
SELECT COUNT(*) as test_events_count FROM TESTING.TEST_EVENTS;
SELECT COUNT(*) as anomalies_count FROM TESTING.TEST_EVENTS WHERE is_anomaly = TRUE;
```

## Service Account Configuration

The `APP_CLAUDE` service account is pre-configured with:

✅ **Permissions Granted:**
- ACCOUNTADMIN role (full database access)
- COMPUTE_WH warehouse usage
- All schema privileges in ANOMALY_DETECTION_DEV

✅ **Access Verified:**
- Database connection established
- Query execution permissions confirmed
- Read/write access to all schemas

## Security Configuration

### Role-Based Access Control
- Service account uses ACCOUNTADMIN for development
- Production should use more restrictive roles
- Regular password rotation recommended

### Network Security
- Connection uses TLS encryption
- IP whitelisting configured in Snowflake account
- MFA enabled for admin accounts

## Development Workflow

### 1. Local Development Setup

Create Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install snowflake-connector-python pandas pyyaml
```

### 2. Connection Testing

Test connection with Python:
```python
import snowflake.connector

conn = snowflake.connector.connect(
    account='ya78352.ap-southeast-2',
    user='APP_CLAUDE',
    password='B9Dbz@jNCiWu@111',
    warehouse='COMPUTE_WH',
    database='ANOMALY_DETECTION_DEV',
    role='ACCOUNTADMIN'
)

cursor = conn.cursor()
cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_USER()")
result = cursor.fetchone()
print(f"Connected to: {result[0]} as {result[1]}")
```

### 3. Test Data Queries

Verify test data setup:
```sql
-- View test events summary
SELECT 
    event_type,
    COUNT(*) as total_events,
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count,
    AVG(metric_value) as avg_value
FROM TESTING.TEST_EVENTS
GROUP BY event_type;

-- View known anomalies
SELECT 
    event_date,
    event_type,
    metric_name,
    metric_value,
    anomaly_type,
    anomaly_magnitude,
    description
FROM TESTING.TEST_EVENTS
WHERE is_anomaly = TRUE
ORDER BY event_date DESC;
```

## Performance Optimization

### Indexes Created
- `IDX_DAILY_ANOMALIES_DATE_EVENT`: Query optimization for date/event filters
- `IDX_DAILY_ANOMALIES_SEVERITY`: Alert severity lookups
- `IDX_EXECUTION_LOG_DATE_TYPE`: Performance monitoring queries
- `IDX_TEST_EVENTS_DATE_TYPE`: Test data analysis

### Query Performance
- Use appropriate WHERE clauses on indexed columns
- Leverage Snowflake's automatic clustering
- Monitor query execution plans
- Use result caching for repeated queries

## Troubleshooting

### Common Issues

**Connection Timeout**
- Check network connectivity
- Verify firewall rules allow HTTPS traffic
- Confirm account name spelling and region

**Authentication Failed**
- Verify username and password
- Check if account is locked or expired
- Confirm role assignments

**Permission Denied**
- Verify ACCOUNTADMIN role is granted
- Check schema-level permissions
- Ensure warehouse access is granted

**Database Not Found**
- Confirm database name case sensitivity
- Verify database exists in account
- Check USE DATABASE statements

### Performance Issues

**Slow Query Execution**
- Check warehouse size (COMPUTE_WH is XS)
- Consider scaling up for large datasets
- Review query execution plans
- Add appropriate indexes

**High Compute Costs**
- Monitor warehouse auto-suspend settings
- Use appropriate warehouse sizes
- Implement query result caching
- Optimize JOIN operations

### Data Issues

**Missing Test Data**
- Re-run data population scripts
- Check for truncated inserts
- Verify schema permissions
- Review error logs

**Inconsistent Results**
- Check for duplicate data
- Verify date ranges in queries
- Review test data generation logic
- Validate anomaly calculations

## Monitoring and Maintenance

### Health Checks
Run these queries regularly to monitor environment health:

```sql
-- Check recent execution logs
SELECT 
    execution_date,
    detector_type,
    status,
    anomalies_found,
    records_processed
FROM DETECTION.DETECTOR_EXECUTION_LOG
WHERE execution_date >= CURRENT_DATE - 7
ORDER BY execution_date DESC;

-- Monitor alert delivery
SELECT 
    DATE(sent_at) as send_date,
    channel_type,
    delivery_status,
    COUNT(*) as count
FROM ALERTS.NOTIFICATION_LOG
WHERE sent_at >= CURRENT_DATE - 7
GROUP BY send_date, channel_type, delivery_status
ORDER BY send_date DESC;
```

### Maintenance Tasks
- Weekly: Review query performance metrics
- Monthly: Analyze storage usage and costs
- Quarterly: Review and rotate service account credentials
- Annually: Audit role assignments and permissions

## Next Steps

1. **Complete Manual Setup**: Execute all SQL scripts in Snowflake Web UI
2. **Test Queries**: Verify data access and basic operations
3. **Deploy Detection Scripts**: Set up Airflow DAGs and Docker containers
4. **Configure Alerts**: Test email and Slack notification channels
5. **Performance Testing**: Validate with realistic data volumes

## Support and Resources

### Documentation
- [Snowflake Documentation](https://docs.snowflake.com/)
- [Python Connector Guide](https://docs.snowflake.com/en/user-guide/python-connector.html)
- [SQL Reference](https://docs.snowflake.com/en/sql-reference.html)

### Contacts
- **Development Environment**: APP_CLAUDE service account
- **Production Support**: Contact system administrator
- **Emergency Issues**: Use Snowflake support portal

---

**Last Updated**: 2025-01-11  
**Version**: 1.0  
**Reviewer**: ADF Development Team