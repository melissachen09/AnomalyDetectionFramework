# Snowflake Setup Scripts

This directory contains SQL scripts and Python utilities for setting up the Snowflake development environment for the Anomaly Detection Framework.

## Files Overview

### SQL Scripts
- **`01_setup_schemas.sql`** - Creates core database schemas (CONFIG, DETECTION, ALERTS, TESTING)
- **`02_create_tables.sql`** - Creates all required tables with proper data types and constraints
- **`03_populate_test_data.sql`** - Populates tables with sample configuration and test data
- **`04_manual_data_setup.sql`** - Generated SQL statements for manual execution (auto-generated)

### Python Scripts
- **`setup_snowflake_dev.py`** - Complete automated setup script (requires snowflake-connector-python)
- **`generate_test_data.py`** - Generates synthetic test data with known anomalies

## Quick Setup

### Option 1: Manual Setup (Recommended)
1. Open Snowflake Web UI
2. Connect to ANOMALY_DETECTION_DEV database
3. Execute scripts in order:
   ```sql
   -- Execute each script in sequence
   \copy 01_setup_schemas.sql
   \copy 02_create_tables.sql
   \copy 03_populate_test_data.sql
   ```

### Option 2: Automated Setup
```bash
# Install dependencies (requires virtual environment)
python3 -m venv venv
source venv/bin/activate
pip install snowflake-connector-python

# Run setup script
python3 setup_snowflake_dev.py
```

## Database Structure

```
ANOMALY_DETECTION_DEV/
├── CONFIG/
│   ├── EVENT_TYPES
│   └── DETECTION_RULES
├── DETECTION/
│   ├── DAILY_ANOMALIES
│   └── DETECTOR_EXECUTION_LOG
├── ALERTS/
│   ├── ALERT_ROUTING
│   └── NOTIFICATION_LOG
└── TESTING/
    ├── TEST_EVENTS
    └── VALIDATION_RESULTS
```

## Configuration Data

### Event Types
- `listing_views` - Property listing page views
- `listing_enquiries` - Property enquiries
- `page_views` - Website page views

### Detection Rules
- Threshold detectors with min/max bounds
- Statistical detectors with Z-score analysis
- Percentage change detection

### Alert Routing
- Critical alerts → Email + Slack
- High priority → Data team email + Slack
- Warnings → Dashboard only

## Test Data

### Normal Baseline Data
- 25 days of normal metric values with random variation
- Baseline around 50,000 daily views
- Natural fluctuation within ±10%

### Known Anomalies
- **Traffic Spike**: 100% increase (100,000 views)
- **System Outage**: 70% decrease (15,000 views)
- **Gradual Drift**: 50% increase trend
- **Minor Spike**: 24% increase

## Verification Queries

After setup, verify with these queries:

```sql
-- Check schema creation
SELECT SCHEMA_NAME, COMMENT 
FROM INFORMATION_SCHEMA.SCHEMATA 
WHERE CATALOG_NAME = 'ANOMALY_DETECTION_DEV';

-- Check table creation
SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_CATALOG = 'ANOMALY_DETECTION_DEV'
AND TABLE_SCHEMA IN ('CONFIG', 'DETECTION', 'ALERTS', 'TESTING');

-- Verify test data
SELECT 
    event_type,
    COUNT(*) as total_events,
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count
FROM TESTING.TEST_EVENTS
GROUP BY event_type;
```

## Troubleshooting

### Permission Issues
- Ensure ACCOUNTADMIN role is active
- Verify database access: `USE DATABASE ANOMALY_DETECTION_DEV;`
- Check warehouse state: `USE WAREHOUSE COMPUTE_WH;`

### Script Execution Issues
- Execute scripts in order (dependencies exist)
- Check for syntax errors in SQL statements
- Verify schema exists before creating tables

### Data Population Issues
- Clear existing data if re-running: `TRUNCATE TABLE schema.table;`
- Check for constraint violations
- Verify date formats in INSERT statements

## Next Steps

1. **Test Connectivity**: Verify MCP Snowflake tool can query tables
2. **Deploy Detection Logic**: Create Python detection scripts
3. **Configure Airflow**: Set up DAGs for orchestration
4. **Test Alerting**: Validate email and Slack notifications

## Support

For issues with these scripts:
1. Check Snowflake query history for error details
2. Review execution logs in Web UI
3. Verify connection parameters and credentials
4. Consult the main setup documentation in `docs/snowflake-development-setup.md`