# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Claude AI Instructions for Anomaly Detection Framework

Always call necessary tool calls in parallel for optimal performance.

## Project Context

**Project Type**: Data Processing Framework / Monitoring System
**Technology Stack**: Python, Airflow, Snowflake, Docker, YAML
**Domain**: Data Quality & Analytics
**Stage**: Design Complete / Implementation Phase

## Core Architecture

### High-Level System Design

This is a **lightweight anomaly detection framework** that leverages existing data infrastructure:

- **Data Source**: GroupStats events → ELT process → Snowflake tables
- **Detection**: Airflow-orchestrated Python scripts in Docker containers
- **Configuration**: Version-controlled YAML files (file-based, no UI)
- **Alerting**: Snowflake email functions + Slack webhooks
- **Reporting**: Snowflake built-in dashboards

### Key Architectural Principles

- **Use existing technology stack only** - No new APIs or frontend development
- **Configuration-as-Code** - All detection rules in YAML files, version controlled
- **Container-based execution** - Detection logic runs in Docker containers via Airflow
- **Leverage Snowflake capabilities** - Email, dashboards, and data processing
- **Simple maintenance** - File-based configuration, no complex UI

## Development Environment

### Current State
- **No build system yet** - Project is in design phase, ready for implementation
- **No package.json, requirements.txt, or Makefile** - These need to be created
- **No source code directories** - Implementation starts from design documents

### Implementation Roadmap
Based on `specs/anomaly-detection-design-doc.md`, the implementation should follow:

1. **Phase 1**: Core detection system with Docker containers
2. **Phase 2**: Enhanced statistical detectors and notifications  
3. **Phase 3**: Snowflake dashboards and reporting

## Core Responsibilities

### Primary Role
You are a senior data engineer with expertise in Airflow, Snowflake, and data quality systems. Your responsibilities:

- Implement detection plugins based on the design specifications
- Create Docker containers for detection logic
- Build Airflow DAGs for orchestration
- Configure YAML-based detection rules
- Set up Snowflake email and dashboard systems
- Follow Test-Driven Development practices

### Project-Specific Behaviors

#### Code Standards
- Follow universal code standards in `specs/code-standards.md`
- Prioritize **simplicity and maintainability** over complex features
- Use **Test-Driven Development** - write tests before implementation
- Implement comprehensive error handling and logging

#### Technology Decisions
- **Orchestration**: Airflow DAGs with Docker operators
- **Data Processing**: Snowflake SQL and Python pandas
- **Configuration**: YAML files in `configs/` directory structure
- **Containerization**: Docker for detection scripts
- **Notifications**: Snowflake email functions + Python Slack webhooks

## Development Workflow

### Before Starting Development
1. **Load Context**: Use `/prime` command to load project specifications
2. **Check Priorities**: Review `specs/project_plan.md` for current phase
3. **Follow Design**: Implement according to `specs/anomaly-detection-design-doc.md`
4. **JIRA Tasks**: Load tasks from JIRA with status TO DO
5. **Review Work Plan**: Follow epic structure in `specs/work-plan.md`

### TDD Implementation Process
Based on `specs/work-plan.md`, development follows strict Test-Driven Development:

**Epic Progression**:
1. **GADF-ENV**: Environment & Project Setup (Weeks 1-2)
2. **GADF-CONFIG**: Configuration Management System (Weeks 3-4)
3. **GADF-DETECT**: Detection Plugin Architecture (Weeks 5-6)
4. **GADF-SNOW**: Snowflake Integration Layer (Weeks 7-8)
5. **GADF-ALERT**: Alert Classification & Routing (Weeks 9-10)
6. **GADF-NOTIFY**: Notification Channels (Week 11)
7. **GADF-ORCH**: Airflow Orchestration (Week 12)
8. **GADF-DASH**: Dashboard & Reporting (Week 13)
9. **GADF-E2E**: Integration Testing & E2E Validation (Week 14)
10. **GADF-DOCS**: Documentation & Knowledge Transfer (Week 15)

**Each Epic Follows Pattern**:
1. Write comprehensive test cases first
2. Implement to make tests pass
3. Refactor while keeping tests green
4. Move to next task only when tests pass

**Task Naming Convention**: GADF-{EPIC}-{NUMBER} (e.g., GADF-ENV-001, GADF-CONFIG-001)

**Implementation Notes**:
- Each epic has test-first tasks followed by implementation tasks
- 15-week timeline with specific story point estimates in `specs/work-plan.md`
- Comprehensive project completion checklist with quality gates
- Focus on configuration-as-code and container-based architecture

### Implementation Guidelines
1. **JIRA Integration**: Use AnomalyDetectionFramework project (Key: ADF, Cloud ID: 5c7e1404-fb71-4e23-b0e7-5a45d3c7db8c)
2. **Conventional Commits**: Include JIRA references in commit messages
3. **Feature Branches**: Never commit directly to main
4. **Pull Requests**: Use `gh pr create` with JIRA issue references
5. **Test Coverage**: Implement unit tests for all detection logic

### Git Workflow
- Work on feature branches: `feature/ADF-XXX-description`
- Commit with references: `feat(detection): implement threshold detector [ADF-XXX]`
- Push and create PRs: `git push origin feature/ADF-XXX && gh pr create`

## Domain Knowledge

### Business Context
- **GroupStats**: Existing data source with daily events (listing views, enquiries, etc.)
- **Data Quality Issues**: Bot traffic, upstream changes, new events without config
- **Stakeholder Impact**: Affects downstream reporting and business decisions

### Key Data Sources
- `DATAMART.DD_LISTING_STATISTICS_BLENDED` - Primary listing statistics
- `DATAMART.FACT_LISTING_STATISTICS_UTM` - UTM-tracked metrics
- Daily batch processing via existing ELT pipeline

### Alert Severity Levels
- **Critical**: Major business impact (>50% deviation) → Director of BI email
- **High**: Significant anomalies (>30% deviation) → Data team + Slack
- **Warning**: Minor deviations (>20% deviation) → Dashboard only

## Configuration Architecture

### YAML Configuration Structure
```yaml
event_config:
  name: "listing_views"
  data_source:
    table: "DATAMART.DD_LISTING_STATISTICS_BLENDED"
    date_column: "STATISTIC_DATE"
    metrics:
      - column: "NUMBEROFVIEWS"
        alias: "total_views"
  
  detection:
    daily_checks:
      - detector: "threshold"
        metric: "total_views"
        min_value: 10000
        max_value: 1000000
    
  alerting:
    critical:
      condition: "deviation > 0.5"
      recipients:
        email: ["director-bi@company.com"]
```

### Detection Plugin Types
- **Threshold Detector**: Min/max bounds and percentage change
- **Statistical Detector**: Z-score and moving average analysis
- **Elementary Integration**: Wrapper for existing data quality tests
- **dbt Test Integration**: Execute and parse existing dbt tests

## MCP Integration

### Available Tools
- **GitHub**: Repository management and PR workflows
- **Snowflake**: Direct database access for detection and alerting
- **Playwright**: Browser automation for testing dashboards
- **Atlassian**: JIRA integration for project management

### Snowflake Connection
- Full read/write access to production data warehouse
- Use for implementing detection queries and email notifications
- Database: PRODDOMAINDW, Schema: CLEAN

## Implementation Commands

### Current Project State
- **Design Phase Complete**: All architecture and specs documented in `specs/`
- **Ready for Implementation**: Follow TDD approach from `specs/work-plan.md`
- **Epic Structure**: 10 epics covering Environment Setup → Documentation/Training

### Development Commands (To Be Created)

1. **Initial Project Setup** (Epic 1: GADF-ENV):
   ```bash
   mkdir -p src/{detection/{detectors,utils,config},notification,orchestration}
   mkdir -p tests/{unit,integration,e2e}
   mkdir -p configs/events
   mkdir -p docker/{airflow,detector}
   ```

2. **Python Environment**:
   ```bash
   # requirements.txt (based on work plan)
   pandas>=1.5.0
   snowflake-connector-python>=3.0.0
   pyyaml>=6.0
   pytest>=7.0.0
   pytest-cov>=4.0.0
   pydantic>=2.0.0
   docker>=6.0.0
   airflow>=2.7.0
   ```

3. **Testing Commands** (TDD Required):
   ```bash
   pytest tests/                    # Run all tests
   pytest tests/unit/              # Unit tests only
   pytest tests/integration/       # Integration tests
   pytest -k "test_threshold"      # Specific test pattern
   pytest --cov=src --cov-report=html  # Coverage report
   ```

4. **Docker Development**:
   ```bash
   make build                      # Build all containers
   make test                       # Run tests in containers
   make airflow-up                 # Start local Airflow
   ```

5. **Development Workflow**:
   ```bash
   # Follow TDD: Write test → Implement → Refactor
   pytest tests/unit/test_threshold_detector.py -v  # Run specific tests
   docker-compose up -d            # Local Airflow + dependencies
   ```

### Airflow Integration
- Detection scripts run as DockerOperator tasks
- Configuration loaded from filesystem within containers
- Results stored in Snowflake tables for dashboard access

## Security Considerations

### Data Access
- All detection runs against production Snowflake data
- Implement input validation for all configuration parameters
- Never log sensitive data or credentials
- Use environment variables for all connection details

### Configuration Security
- YAML files are version controlled and not sensitive
- Snowflake credentials via environment variables only
- Slack webhook URLs should be environment-specific

---

## AI-Assisted Development Features

This project includes specialized Claude commands in `.claude/commands/`:

- **`/prime`**: Load full project context and specifications
- **`/work_on_ticket_engineer`**: Technical implementation workflow
- **`/technicalManager`**: Strategic architecture decisions
- **Git Worktree Scripts**: For parallel development on multiple features

Use `/prime` at the start of each development session to load current project state and priorities.