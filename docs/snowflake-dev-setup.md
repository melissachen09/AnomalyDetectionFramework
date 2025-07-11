# Snowflake Development Environment Configuration

## Connection Details
- **Database**: ANOMALY_DETECTION_DEV
- **Service Account**: APP_CLAUDE  
- **Role**: ACCOUNTADMIN
- **Warehouse**: COMPUTE_WH
- **Account**: YA78352.ap-southeast-2

## Manual Setup Steps

Since the MCP Snowflake tool has limitations on DDL operations, the schemas and tables need to be created manually using a Snowflake client. The following SQL scripts have been prepared:

### 1. Schema Creation
```sql
-- Execute in Snowflake Web UI or SnowSQL
USE DATABASE ANOMALY_DETECTION_DEV;

CREATE SCHEMA IF NOT EXISTS CONFIG 
    COMMENT = 'Schema for configuration tables and metadata';

CREATE SCHEMA IF NOT EXISTS DETECTION 
    COMMENT = 'Schema for detection results and anomaly data';

CREATE SCHEMA IF NOT EXISTS ALERTS 
    COMMENT = 'Schema for alert management and notification tracking';

CREATE SCHEMA IF NOT EXISTS TESTING 
    COMMENT = 'Schema for test data and validation';
```

### 2. Table Creation
Execute the SQL scripts in order:
- `scripts/snowflake/01_setup_schemas.sql`
- `scripts/snowflake/02_create_tables.sql`  
- `scripts/snowflake/03_populate_test_data.sql`

### 3. Verification
After running the setup scripts, verify with:
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
```

## Service Account Configuration

The APP_CLAUDE service account is already configured with:
- ✅ ACCOUNTADMIN role (full permissions)
- ✅ Access to ANOMALY_DETECTION_DEV database
- ✅ COMPUTE_WH warehouse access

## Connection Testing

You can test the connection using the MCP Snowflake tool:
```sql
SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_SCHEMA();
```

## Environment Variables

For applications connecting to this development environment:
```bash
SNOWFLAKE_ACCOUNT=ya78352.ap-southeast-2
SNOWFLAKE_USER=APP_CLAUDE
SNOWFLAKE_PASSWORD=B9Dbz@jNCiWu@111
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=ANOMALY_DETECTION_DEV
SNOWFLAKE_ROLE=ACCOUNTADMIN
```

## Next Steps

1. Execute the schema creation scripts manually
2. Run the table creation scripts
3. Populate with sample data
4. Test anomaly detection queries
5. Verify alert routing configuration

## Troubleshooting

### Common Issues
- **Connection timeout**: Check network connectivity and firewall rules
- **Authentication failed**: Verify credentials and account name
- **Permission denied**: Ensure ACCOUNTADMIN role is granted
- **Database not found**: Verify database name spelling and case sensitivity

### Performance Optimization
- Use appropriate warehouse size for workload
- Create indexes on frequently queried columns
- Consider clustering keys for large tables
- Use result caching for repeated queries

### Security Considerations
- Rotate service account password regularly
- Use role-based access control
- Enable MFA for admin accounts
- Audit query and login activity
- Encrypt sensitive data at rest and in transit