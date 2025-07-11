# Test Suite

This directory contains all test files for the Anomaly Detection Framework.

## Test Structure

- `ADF-{task-id}/` - Tests organized by JIRA task ID
- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for component interactions
- `e2e/` - End-to-end tests for full workflows
- `fixtures/` - Test data and fixtures
- `utils/` - Test utilities and helpers

## Task-Based Test Organization

Tests are organized by JIRA task ID to maintain traceability:

- `ADF-11/` - Initialize Project Repository tests
  - `test_project_structure.py` - Directory structure validation (ADF-12)

## Running Tests

```bash
# Run all tests
python3 -m unittest discover tests/ -v

# Run tests for specific task
python3 -m unittest discover tests/ADF-11/ -v

# Run specific test file
python3 tests/ADF-11/test_project_structure.py
```