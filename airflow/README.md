# Airflow Local Development Setup

This directory contains the configuration for running Apache Airflow locally using Docker Compose for the Anomaly Detection Framework.

## Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available for Docker
- Port 8080 available (for Airflow webserver)

## Quick Start

1. **Copy environment configuration:**
   ```bash
   cp .env.example .env
   ```

2. **Update Snowflake credentials in `.env` file:**
   ```bash
   # Edit .env file with your actual Snowflake credentials
   SNOWFLAKE_ACCOUNT=your-account.snowflakecomputing.com
   SNOWFLAKE_USER=your-username
   SNOWFLAKE_PASSWORD=your-password
   ```

3. **Start Airflow services:**
   ```bash
   docker-compose up
   ```

4. **Access Airflow Web UI:**
   - URL: http://localhost:8080
   - Username: airflow
   - Password: airflow

## Services

The docker-compose setup includes:

- **airflow-webserver**: Web UI for Airflow (port 8080)
- **airflow-scheduler**: Task scheduler
- **postgres**: Database backend for Airflow metadata
- **airflow-init**: Initialization service for setup

## Directory Structure

```
airflow/
├── dags/           # Airflow DAG files
├── plugins/        # Custom Airflow plugins
└── logs/           # Airflow logs (auto-created)

docker/
└── airflow/
    └── Dockerfile  # Custom Airflow image with required packages
```

## Snowflake Integration

The Airflow environment is pre-configured with Snowflake connection variables:

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`
- `SNOWFLAKE_WAREHOUSE`

These can be used in DAGs to connect to Snowflake for anomaly detection workflows.

## Volume Mounts

- `./airflow/dags` → `/opt/airflow/dags` (DAG files)
- `./airflow/plugins` → `/opt/airflow/plugins` (Custom plugins)
- `./logs` → `/opt/airflow/logs` (Log files)

## Development

### Adding New DAGs

Place your DAG files in the `airflow/dags/` directory. They will be automatically picked up by Airflow.

### Custom Plugins

Add custom operators, hooks, or sensors to the `airflow/plugins/` directory.

### Managing Dependencies

Update the `docker/airflow/Dockerfile` to install additional Python packages.

## Stopping Services

```bash
docker-compose down
```

To remove all volumes (including database):
```bash
docker-compose down --volumes
```

## Troubleshooting

1. **Permission Issues**: Ensure proper AIRFLOW_UID is set in .env file
2. **Memory Issues**: Ensure Docker has at least 4GB RAM allocated
3. **Port Conflicts**: Ensure port 8080 is not in use by other services

## Testing

Run the test suite to verify the Airflow environment setup:

```bash
python3 -m unittest tests.ADF-16.test_airflow_environment -v
```