# Anomaly Detection Framework - Design Document

## 1. Executive Summary

### 1.1 Problem Statement
GroupStats data quality issues frequently occur, including bot traffic, upstream code changes without notification, new event types without configuration, and data anomalies. These issues primarily affect downstream reporting and require a simple, maintainable solution using existing technology stack.

### 1.2 Solution Overview
Design a lightweight anomaly detection framework that leverages existing infrastructure, focusing on automated detection and alerting without adding unnecessary complexity. The system will use Airflow, Snowflake, Python, and Docker containers.

## 2. Architecture Overview

### 2.1 High-Level Architecture
The system uses a simplified architecture:
- **Data Source**: GroupStats raw events (existing)
- **Processing**: Existing ELT process loads data into Snowflake tables
- **Detection**: Airflow-orchestrated Python scripts in Docker containers
- **Alerting**: Snowflake email function and Slack webhooks
- **Reporting**: Snowflake built-in dashboards

### 2.2 Data Flow
```
GroupStats Events → ELT Process (Existing) → Snowflake Tables → 
Anomaly Detection (Airflow) → Alert Classification → 
Email/Slack Notifications → Snowflake Dashboard
```

### 2.3 Key Design Principles
- Use existing technology stack only
- No new APIs or frontend development
- Leverage Snowflake's built-in capabilities
- Simple configuration via files, not UI
- Focus on downstream reporting impact

## 3. Core Components

### 3.1 Data Ingestion (Existing)
**Status**: Already implemented, no changes needed

**Current Process**:
- GroupStats events are collected daily
- ELT process transforms raw data into structured tables
- Key tables include:
  - `DATAMART.DD_LISTING_STATISTICS_BLENDED`
  - `DATAMART.FACT_LISTING_STATISTICS_UTM`
  - Other metric-specific tables

**Note**: Page views are one of many metrics captured in these events, not a separate log.

### 3.2 Configuration Management
**Purpose**: Manage detection rules and stakeholder mappings

**Implementation**:
- Configuration is stored as version-controlled YAML files in a Git repository.
- Configurations are read directly from the filesystem. The CI/CD pipeline builds Docker images by copying the configuration files into the image. This eliminates the need for an intermediate storage layer and simplifies the architecture.
- Engineers update detection logic by modifying these files and submitting pull requests.

**Configuration Schema**:
```yaml
event_type: "listing_views"
source_table: "DATAMART.DD_LISTING_STATISTICS_BLENDED"
metrics:
  - name: "NUMBEROFVIEWS"
    detection_rules:
      - type: "threshold"
        min: 1000
        max: 1000000
      - type: "percentage_change"
        threshold: 0.5
reporting_frequency: "daily"  # Data updates daily
alert_frequency: "weekly"     # Stakeholders prefer weekly summaries
stakeholders:
  critical:
    - email: "director-bi@company.com"
  high:
    - email: "data-team@company.com"
    - slack: "#data-alerts"
```

### 3.3 Detection Orchestrator
**Purpose**: Execute detection logic via Airflow

**Airflow DAG Structure**:
```python
# Main Detection DAG - Runs daily after ELT completes
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

# This function would read from the config files included in the Airflow deployment
def get_event_types_to_process(**kwargs):
    # Logic to scan the /configs directory and return a list of event types
    # e.g., ['listing_views', 'enquiries', 'clicks']
    event_configs = load_all_event_configs_from_files()
    return [config['name'] for config in event_configs]

with DAG('anomaly_detection_daily', 
         schedule_interval='0 6 * * *',  # 6 AM daily
         catchup=False) as dag:
    
    # 1. Get a list of all event types from configuration files
    get_event_types = PythonOperator(
        task_id='get_event_types',
        python_callable=get_event_types_to_process
    )
    
    # 2. Dynamically map a detection task for each event type
    # The DockerOperator will be expanded into parallel tasks based on the list from the previous step.
    detect_anomalies = DockerOperator.expand(
        task_id='detect_anomaly_for_event',
        image='anomaly-detector:latest',
        command='python detect.py --event-type {{ task_instance.xcom_pull(task_ids="get_event_types", key="return_value") }}',
        docker_url='unix://var/run/docker.sock'
    )
    
    # 3. Aggregate results from all dynamic tasks
    aggregate = PythonOperator(
        task_id='aggregate_results',
        python_callable=aggregate_detection_results,
        trigger_rule='all_done' # Run even if some detection tasks fail
    )
    
    # 4. Send notifications
    notify = PythonOperator(
        task_id='send_notifications',
        python_callable=route_and_send_alerts
    )
    
    # 5. Update dashboard tables
    update_dashboard = SnowflakeOperator(
        task_id='update_dashboard_tables',
        sql='CALL update_anomaly_dashboard_tables()'
    )

    get_event_types >> detect_anomalies >> aggregate >> [notify, update_dashboard]
```

### 3.4 Detection Plugins

#### 3.4.1 Plugin Architecture
Simple Python classes that implement detection logic:

```python
class BaseDetector:
    def __init__(self, config):
        self.config = config
        self.snowflake_conn = get_snowflake_connection()
    
    def detect(self, start_date, end_date):
        """Run detection and return anomalies"""
        raise NotImplementedError
```

#### 3.4.2 Included Detectors

**1. Threshold Detector**
- Simple min/max bounds checking
- Percentage change detection
- Row count validation

**2. Statistical Detector**
- Z-score based outlier detection
- Moving average comparison
- Seasonal adjustment using historical data

**3. Elementary Integration**
- Wrapper for existing Elementary tests
- Reuses already configured data quality checks

**4. dbt Test Integration**
- Executes existing dbt tests
- Parses test results for anomalies

### 3.5 Alert Management

#### 3.5.1 Alert Classification
Based on business impact and deviation severity:

**Level 1: Critical**
- Definition: An issue with major business impact, as defined in the alert condition for the specific event.
- Response: Immediate email to Director of BI
- Example: Total listing views drop by 70%

**Level 2: High Priority**
- Definition: A significant anomaly that affects data integrity or key dimensions, as defined by its configuration.
- Response: Email to data team, Slack notification
- Example: Mobile device views spike by 40%

**Level 3: Warning**
- Definition: A minor deviation that should be logged for awareness but doesn't require immediate action.
- Response: Log to dashboard only, weekly summary
- Example: Minor fluctuation in less-used features

#### 3.5.2 Notification Logic
```python
def route_alerts(anomalies):
    # Group by severity
    critical = [a for a in anomalies if a.severity == 'critical']
    high = [a for a in anomalies if a.severity == 'high']
    warnings = [a for a in anomalies if a.severity == 'warning']
    
    # Send immediate alerts for critical
    if critical:
        send_critical_alert(critical)
    
    # Batch high priority alerts
    if high:
        send_high_priority_summary(high)
    
    # Warnings go to dashboard only
    update_warning_dashboard(warnings)
    
    # Weekly/monthly summaries for stakeholders
    if is_report_day():
        send_stakeholder_summary(anomalies)
```

### 3.6 Notification Implementation

#### 3.6.1 Snowflake Email Function
```sql
CREATE OR REPLACE PROCEDURE send_anomaly_email(
    recipient_list ARRAY,
    subject VARCHAR,
    anomaly_data VARIANT
)
RETURNS STRING
LANGUAGE JAVASCRIPT
AS $$
    var email_body = build_email_template(ANOMALY_DATA);
    
    var result = snowflake.execute({
        sqlText: `CALL SYSTEM$SEND_EMAIL(
            'anomaly_alerts',
            :1,
            :2,
            :3
        )`,
        binds: [RECIPIENT_LIST, SUBJECT, email_body]
    });
    
    return 'Email sent successfully';
$$;
```

#### 3.6.2 Slack Integration
```python
def send_slack_alert(channel, anomalies):
    webhook_url = SLACK_WEBHOOKS[channel]
    
    blocks = [{
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*Anomaly Alert - {datetime.now().date()}*\n"
                    f"Found {len(anomalies)} anomalies"
        }
    }]
    
    for anomaly in anomalies[:5]:  # Top 5 only
        blocks.append({
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Event:* {anomaly.event_type}"},
                {"type": "mrkdwn", "text": f"*Severity:* {anomaly.severity}"},
                {"type": "mrkdwn", "text": f"*Deviation:* {anomaly.deviation:.1%}"},
                {"type": "mrkdwn", "text": f"*Metric:* {anomaly.metric}"}
            ]
        })
    
    requests.post(webhook_url, json={"blocks": blocks})
```

### 3.7 Results Storage
All detection results stored in Snowflake for dashboard access:

```sql
-- Anomaly detection results
CREATE TABLE IF NOT EXISTS ANOMALY_DETECTION.RESULTS.DAILY_ANOMALIES (
    detection_date DATE,
    event_type VARCHAR,
    metric_name VARCHAR,
    expected_value FLOAT,
    actual_value FLOAT,
    deviation_percentage FLOAT,
    severity VARCHAR,
    detection_method VARCHAR,
    alert_sent BOOLEAN,
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Aggregated weekly/monthly views for reporting
CREATE VIEW ANOMALY_DETECTION.RESULTS.WEEKLY_SUMMARY AS
SELECT 
    DATE_TRUNC('week', detection_date) as week,
    event_type,
    COUNT(*) as anomaly_count,
    AVG(deviation_percentage) as avg_deviation,
    MAX(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as has_critical
FROM DAILY_ANOMALIES
GROUP BY 1, 2;
```

## 4. Deployment

### 4.1 Container Management
All detection code packaged in Docker containers:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY detection/ ./detection/
COPY config/ ./config/

CMD ["python", "detection/main.py"]
```

### 4.2 Deployment Process
1. Build Docker images with detection code
2. Push to internal container registry
3. Update Airflow DAGs to use new image versions
4. No API deployment needed

## 5. Reporting

### 5.1 Snowflake Dashboard
Use Snowflake's built-in dashboard capabilities:

**Dashboard Components**:
1. **Daily Anomaly Summary**: Table of all detected anomalies
2. **Trend Charts**: Weekly/monthly anomaly trends by event type
3. **Severity Distribution**: Pie chart of anomaly severities
4. **Top Affected Metrics**: Bar chart of most anomalous metrics
5. **Historical Comparison**: Time series of key metrics with anomaly markers

**Dashboard Queries**:
```sql
-- Daily summary for dashboard
CREATE OR REPLACE VIEW ANOMALY_DASHBOARD.DAILY_SUMMARY AS
SELECT 
    detection_date,
    event_type,
    metric_name,
    actual_value,
    expected_value,
    ROUND(deviation_percentage * 100, 1) || '%' as deviation,
    severity,
    CASE 
        WHEN alert_sent THEN 'Sent'
        ELSE 'Dashboard Only'
    END as notification_status
FROM ANOMALY_DETECTION.RESULTS.DAILY_ANOMALIES
WHERE detection_date >= CURRENT_DATE - 30
ORDER BY detection_date DESC, severity;
```

### 5.2 Stakeholder Reports
- **Daily**: Technical team receives detailed anomaly alerts
- **Weekly**: Business stakeholders receive summary reports
- **Monthly**: Executive dashboard with trend analysis

## 6. Monitoring

### 6.1 Existing Schema Monitoring
- Current simple monitoring for data schema exists but is not maintained
- Will not be modified or replaced in this project
- Focus remains on anomaly detection, not schema validation

### 6.2 Detection System Monitoring
Basic monitoring via Airflow:
- DAG execution status
- Task duration tracking
- Failure alerts via Airflow's built-in notifications

## 7. Configuration Examples

### 7.1 Event Configuration
```yaml
# configs/events/listing_views.yaml
event_config:
  name: "listing_views"
  description: "Property listing page views"
  
  data_source:
    table: "DATAMART.DD_LISTING_STATISTICS_BLENDED"
    date_column: "STATISTIC_DATE"
    metrics:
      - column: "NUMBEROFVIEWS"
        alias: "total_views"
      - column: "NUMBEROFENQUIRIES"
        alias: "enquiries"
    
  detection:
    daily_checks:
      - detector: "threshold"
        metric: "total_views"
        min_value: 10000
        max_value: 1000000
      
      - detector: "percentage_change"
        metric: "total_views"
        lookback_days: 7
        threshold: 0.3
    
    weekly_checks:
      - detector: "statistical"
        metric: "enquiries"
        method: "zscore"
        threshold: 3.0
  
  alerting:
    critical:
      condition: "deviation > 0.5"
      recipients:
        email: ["director-bi@company.com"]
    
    high:
      condition: "deviation > 0.3"
      recipients:
        email: ["data-team@company.com"]
        slack: ["#data-alerts"]
    
    warning:
      condition: "deviation > 0.2"
      dashboard_only: true
```

### 7.2 Docker Compose for Local Testing
```yaml
version: '3.8'
services:
  anomaly-detector:
    build: .
    environment:
      - SNOWFLAKE_ACCOUNT=${SNOWFLAKE_ACCOUNT}
      - SNOWFLAKE_USER=${SNOWFLAKE_USER}
      - SNOWFLAKE_PASSWORD=${SNOWFLAKE_PASSWORD}
      - SNOWFLAKE_WAREHOUSE=${SNOWFLAKE_WAREHOUSE}
    volumes:
      - ./configs:/app/configs
    command: python detection/main.py --event-type listing_views
```

## 8. Implementation Timeline

### Phase 1: Core Detection (Weeks 1-4)
- Set up Docker containers
- Implement basic threshold detection
- Create Snowflake email notifications
- Test with 3-5 critical event types

### Phase 2: Enhanced Detection (Weeks 5-8)
- Add statistical detectors
- Integrate Elementary tests
- Implement Slack notifications
- Expand to 20+ event types

### Phase 3: Reporting & Optimization (Weeks 9-12)
- Build Snowflake dashboards
- Create weekly/monthly summary views
- Optimize detection algorithms
- Document configuration process

## 9. Success Metrics

### Technical Metrics
- Detection latency < 30 minutes after data arrival
- False positive rate < 10%
- System availability > 99%

### Business Metrics
- Reduce undetected data issues by 80%
- Decrease manual investigation time by 75%
- Improve stakeholder confidence in data quality

## 10. Maintenance & Operations

### 10.1 Adding New Event Types
1. Create YAML configuration file
2. Test detection locally using Docker
3. Deploy configuration to production
4. Monitor for first week

### 10.2 Updating Detection Rules
1. Modify YAML configuration
2. Test changes in staging environment
3. Deploy during maintenance window
4. No service restart required

### 10.3 Troubleshooting
- Check Airflow logs for execution issues
- Query anomaly results table for detection history
- Review Snowflake dashboard for patterns
- Adjust thresholds based on feedback