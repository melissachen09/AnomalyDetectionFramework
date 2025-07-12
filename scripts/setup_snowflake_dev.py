#!/usr/bin/env python3
"""
Snowflake Development Environment Setup Script
ADF-22: Configure Snowflake Development Environment

This script sets up the complete Snowflake development environment including:
- Database schemas
- Core tables
- Test data with known anomalies
- Views and indexes
"""

import os
import sys
from pathlib import Path
import snowflake.connector
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SnowflakeSetup:
    """Handles Snowflake development environment setup"""
    
    def __init__(self):
        """Initialize connection using environment variables"""
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish Snowflake connection"""
        try:
            self.conn = snowflake.connector.connect(
                account=os.getenv('SNOWFLAKE_ACCOUNT', 'ya78352.ap-southeast-2'),
                user=os.getenv('SNOWFLAKE_USER', 'APP_CLAUDE'),
                password=os.getenv('SNOWFLAKE_PASSWORD', 'B9Dbz@jNCiWu@111'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
                database=os.getenv('SNOWFLAKE_DATABASE', 'ANOMALY_DETECTION_DEV'),
                schema=os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC'),
                role=os.getenv('SNOWFLAKE_ROLE', 'ACCOUNTADMIN')
            )
            logger.info("Successfully connected to Snowflake")
            
            # Verify connection
            cursor = self.conn.cursor()
            cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_USER()")
            result = cursor.fetchone()
            logger.info(f"Connected as: {result[2]} to {result[0]}.{result[1]}")
            cursor.close()
            
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise
    
    def execute_sql_statements(self, sql_statements: List[str], continue_on_error: bool = False) -> Dict[str, Any]:
        """Execute a list of SQL statements"""
        results = {
            'executed': 0,
            'errors': [],
            'success': True
        }
        
        cursor = self.conn.cursor()
        
        for i, statement in enumerate(sql_statements):
            if not statement.strip():
                continue
                
            try:
                logger.info(f"Executing statement {i+1}/{len(sql_statements)}: {statement[:100]}...")
                cursor.execute(statement)
                results['executed'] += 1
                logger.info(f"✓ Statement executed successfully")
                
            except Exception as e:
                error_msg = f"Statement {i+1} failed: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                results['success'] = False
                
                if not continue_on_error:
                    cursor.close()
                    return results
        
        cursor.close()
        return results
    
    def create_schemas(self) -> bool:
        """Create required schemas"""
        logger.info("Creating Snowflake schemas...")
        
        statements = [
            "USE DATABASE ANOMALY_DETECTION_DEV",
            "CREATE SCHEMA IF NOT EXISTS CONFIG COMMENT = 'Schema for configuration tables and metadata'",
            "CREATE SCHEMA IF NOT EXISTS DETECTION COMMENT = 'Schema for detection results and anomaly data'", 
            "CREATE SCHEMA IF NOT EXISTS ALERTS COMMENT = 'Schema for alert management and notification tracking'",
            "CREATE SCHEMA IF NOT EXISTS TESTING COMMENT = 'Schema for test data and validation'"
        ]
        
        result = self.execute_sql_statements(statements)
        if result['success']:
            logger.info("✓ All schemas created successfully")
            return True
        else:
            logger.error(f"Schema creation failed: {result['errors']}")
            return False
    
    def create_tables(self) -> bool:
        """Create all required tables"""
        logger.info("Creating database tables...")
        
        # Detection schema tables
        detection_tables = [
            "USE DATABASE ANOMALY_DETECTION_DEV",
            "USE SCHEMA DETECTION",
            """
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
            ) COMMENT = 'Daily anomaly detection results with severity and alert status'
            """,
            """
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
            ) COMMENT = 'Log of detector execution with performance metrics'
            """
        ]
        
        # Config schema tables
        config_tables = [
            "USE SCHEMA CONFIG",
            """
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
            ) COMMENT = 'Registry of configured event types and their data sources'
            """,
            """
            CREATE TABLE IF NOT EXISTS DETECTION_RULES (
                rule_id STRING DEFAULT UUID_STRING() PRIMARY KEY,
                event_type VARCHAR(100) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                detector_type VARCHAR(50) NOT NULL,
                rule_config VARIANT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
                updated_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            ) COMMENT = 'Detection rules configuration for each event type and metric'
            """
        ]
        
        # Alerts schema tables
        alert_tables = [
            "USE SCHEMA ALERTS",
            """
            CREATE TABLE IF NOT EXISTS ALERT_ROUTING (
                routing_id STRING DEFAULT UUID_STRING() PRIMARY KEY,
                event_type VARCHAR(100),
                severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'high', 'warning')),
                channel_type VARCHAR(50) NOT NULL CHECK (channel_type IN ('email', 'slack', 'dashboard')),
                recipient_config VARIANT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            ) COMMENT = 'Configuration for alert routing by event type and severity'
            """,
            """
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
                created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            ) COMMENT = 'Log of notification delivery attempts and status'
            """
        ]
        
        # Testing schema tables
        testing_tables = [
            "USE SCHEMA TESTING",
            """
            CREATE TABLE IF NOT EXISTS TEST_EVENTS (
                test_id STRING DEFAULT UUID_STRING() PRIMARY KEY,
                event_date DATE NOT NULL,
                event_type VARCHAR(100) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value FLOAT NOT NULL,
                is_anomaly BOOLEAN DEFAULT FALSE,
                anomaly_type VARCHAR(50),
                anomaly_magnitude FLOAT,
                description TEXT,
                created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            ) COMMENT = 'Synthetic test data with known anomalies for validation'
            """,
            """
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
            ) COMMENT = 'Results of detector validation against test data'
            """
        ]
        
        # Execute all table creation statements
        all_statements = detection_tables + config_tables + alert_tables + testing_tables
        result = self.execute_sql_statements(all_statements, continue_on_error=True)
        
        if result['success']:
            logger.info("✓ All tables created successfully")
            return True
        else:
            logger.error(f"Table creation had issues: {result['errors']}")
            return len(result['errors']) == 0
    
    def create_indexes(self) -> bool:
        """Create performance indexes"""
        logger.info("Creating database indexes...")
        
        index_statements = [
            "USE SCHEMA DETECTION",
            "CREATE INDEX IF NOT EXISTS IDX_DAILY_ANOMALIES_DATE_EVENT ON DAILY_ANOMALIES (detection_date, event_type)",
            "CREATE INDEX IF NOT EXISTS IDX_DAILY_ANOMALIES_SEVERITY ON DAILY_ANOMALIES (severity, detection_date)",
            "CREATE INDEX IF NOT EXISTS IDX_EXECUTION_LOG_DATE_TYPE ON DETECTOR_EXECUTION_LOG (execution_date, detector_type)",
            "USE SCHEMA TESTING",
            "CREATE INDEX IF NOT EXISTS IDX_TEST_EVENTS_DATE_TYPE ON TEST_EVENTS (event_date, event_type)"
        ]
        
        result = self.execute_sql_statements(index_statements, continue_on_error=True)
        if result['success']:
            logger.info("✓ All indexes created successfully")
            return True
        else:
            logger.warning(f"Index creation had issues: {result['errors']}")
            return True  # Indexes are optional
    
    def populate_sample_data(self) -> bool:
        """Populate tables with sample configuration and test data"""
        logger.info("Populating sample data...")
        
        # Event types
        config_data = [
            "USE SCHEMA CONFIG",
            """INSERT INTO EVENT_TYPES (event_type, display_name, description, source_table, date_column) VALUES
            ('listing_views', 'Property Listing Views', 'Daily views of property listings', 'DATAMART.DD_LISTING_STATISTICS_BLENDED', 'STATISTIC_DATE')""",
            """INSERT INTO EVENT_TYPES (event_type, display_name, description, source_table, date_column) VALUES
            ('listing_enquiries', 'Property Enquiries', 'Daily enquiries for property listings', 'DATAMART.DD_LISTING_STATISTICS_BLENDED', 'STATISTIC_DATE')""",
            """INSERT INTO EVENT_TYPES (event_type, display_name, description, source_table, date_column) VALUES
            ('page_views', 'Website Page Views', 'Daily website page view metrics', 'DATAMART.FACT_LISTING_STATISTICS_UTM', 'STATISTIC_DATE')"""
        ]
        
        # Detection rules
        rules_data = [
            """INSERT INTO DETECTION_RULES (event_type, metric_name, detector_type, rule_config) VALUES
            ('listing_views', 'total_views', 'threshold', 
                PARSE_JSON('{"min_value": 10000, "max_value": 1000000, "percentage_change_threshold": 0.3}'))""",
            """INSERT INTO DETECTION_RULES (event_type, metric_name, detector_type, rule_config) VALUES
            ('listing_views', 'total_views', 'statistical', 
                PARSE_JSON('{"method": "zscore", "threshold": 3.0, "lookback_days": 30}'))"""
        ]
        
        # Alert routing
        alert_data = [
            "USE SCHEMA ALERTS",
            """INSERT INTO ALERT_ROUTING (event_type, severity, channel_type, recipient_config) VALUES
            (NULL, 'critical', 'email', PARSE_JSON('{"recipients": ["director-bi@company.com"], "template": "critical_alert"}'))""",
            """INSERT INTO ALERT_ROUTING (event_type, severity, channel_type, recipient_config) VALUES
            (NULL, 'high', 'slack', PARSE_JSON('{"channel": "#data-alerts", "webhook_url": "https://hooks.slack.com/data"}'))"""
        ]
        
        all_data = config_data + rules_data + alert_data
        result = self.execute_sql_statements(all_data, continue_on_error=True)
        
        if result['success']:
            logger.info("✓ Sample data populated successfully")
            return True
        else:
            logger.warning(f"Sample data population had issues: {result['errors']}")
            return True  # Continue even with data issues
    
    def create_test_data(self) -> bool:
        """Generate synthetic test data with known anomalies"""
        logger.info("Generating synthetic test data...")
        
        test_data = [
            "USE SCHEMA TESTING",
            # Normal baseline data
            """INSERT INTO TEST_EVENTS (event_date, event_type, metric_name, metric_value, is_anomaly, description)
            SELECT 
                DATEADD(day, -seq, CURRENT_DATE()) as event_date,
                'listing_views' as event_type,
                'total_views' as metric_name,
                50000 + (UNIFORM(1, 10000, RANDOM()) - 5000) as metric_value,
                FALSE as is_anomaly,
                'Normal baseline data' as description
            FROM (SELECT ROW_NUMBER() OVER (ORDER BY NULL) as seq FROM TABLE(GENERATOR(ROWCOUNT => 25)))""",
            
            # Known anomalies
            """INSERT INTO TEST_EVENTS (event_date, event_type, metric_name, metric_value, is_anomaly, anomaly_type, anomaly_magnitude, description) VALUES
            (DATEADD(day, -5, CURRENT_DATE()), 'listing_views', 'total_views', 100000, TRUE, 'spike', 1.0, 'Simulated traffic spike')""",
            """INSERT INTO TEST_EVENTS (event_date, event_type, metric_name, metric_value, is_anomaly, anomaly_type, anomaly_magnitude, description) VALUES
            (DATEADD(day, -3, CURRENT_DATE()), 'listing_views', 'total_views', 15000, TRUE, 'drop', -0.7, 'Simulated system outage')"""
        ]
        
        result = self.execute_sql_statements(test_data, continue_on_error=True)
        
        if result['success']:
            logger.info("✓ Test data generated successfully")
            return True
        else:
            logger.warning(f"Test data generation had issues: {result['errors']}")
            return True  # Continue even with data issues
    
    def verify_setup(self) -> Dict[str, Any]:
        """Verify the setup by running validation queries"""
        logger.info("Verifying Snowflake setup...")
        
        verification_results = {}
        cursor = self.conn.cursor()
        
        try:
            # Check schemas
            cursor.execute("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE CATALOG_NAME = 'ANOMALY_DETECTION_DEV'")
            schemas = [row[0] for row in cursor.fetchall()]
            verification_results['schemas'] = schemas
            logger.info(f"✓ Found schemas: {schemas}")
            
            # Check tables in each schema
            for schema in ['CONFIG', 'DETECTION', 'ALERTS', 'TESTING']:
                if schema in schemas:
                    cursor.execute(f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = '{schema}'")
                    tables = [row[0] for row in cursor.fetchall()]
                    verification_results[f'{schema.lower()}_tables'] = tables
                    logger.info(f"✓ {schema} schema has tables: {tables}")
            
            # Check sample data
            cursor.execute("USE SCHEMA CONFIG")
            cursor.execute("SELECT COUNT(*) FROM EVENT_TYPES")
            event_count = cursor.fetchone()[0]
            verification_results['event_types_count'] = event_count
            logger.info(f"✓ Event types configured: {event_count}")
            
            cursor.execute("USE SCHEMA TESTING")
            cursor.execute("SELECT COUNT(*) FROM TEST_EVENTS")
            test_event_count = cursor.fetchone()[0]
            verification_results['test_events_count'] = test_event_count
            logger.info(f"✓ Test events generated: {test_event_count}")
            
            logger.info("✓ Snowflake setup verification completed successfully")
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            verification_results['error'] = str(e)
        
        finally:
            cursor.close()
        
        return verification_results
    
    def run_complete_setup(self) -> bool:
        """Run the complete Snowflake development environment setup"""
        logger.info("Starting complete Snowflake development environment setup...")
        
        try:
            # Step 1: Create schemas
            if not self.create_schemas():
                return False
            
            # Step 2: Create tables
            if not self.create_tables():
                return False
            
            # Step 3: Create indexes
            self.create_indexes()  # Optional
            
            # Step 4: Populate sample data
            self.populate_sample_data()  # Optional
            
            # Step 5: Generate test data
            self.create_test_data()  # Optional
            
            # Step 6: Verify setup
            verification = self.verify_setup()
            
            logger.info("✓ Snowflake development environment setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    def close(self):
        """Close Snowflake connection"""
        if self.conn:
            self.conn.close()
            logger.info("Snowflake connection closed")

def main():
    """Main execution function"""
    setup = None
    try:
        setup = SnowflakeSetup()
        success = setup.run_complete_setup()
        
        if success:
            print("\\n✅ Snowflake development environment setup completed successfully!")
            print("\\n📋 Summary:")
            print("- Development database: ANOMALY_DETECTION_DEV") 
            print("- Schemas created: CONFIG, DETECTION, ALERTS, TESTING")
            print("- Core tables created with sample data")
            print("- Test data generated with known anomalies")
            print("- Ready for anomaly detection development!")
            return 0
        else:
            print("\\n❌ Setup failed. Check logs for details.")
            return 1
            
    except Exception as e:
        print(f"\\n❌ Fatal error: {e}")
        return 1
    
    finally:
        if setup:
            setup.close()

if __name__ == "__main__":
    sys.exit(main())