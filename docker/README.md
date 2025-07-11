# Docker Configuration

This directory contains Docker configuration files for the Anomaly Detection Framework.

## Files

- `Dockerfile` - Main application Docker image
- `docker-compose.yml` - Local development environment
- `docker-compose.prod.yml` - Production deployment
- `airflow/` - Airflow-specific Docker configurations
- `scripts/` - Docker build and deployment scripts

## Usage

```bash
# Build and run locally
docker-compose up --build

# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```