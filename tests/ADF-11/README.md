# ADF-11: Initialize Project Repository - Tests

This directory contains test files for ADF-11 task: "Initialize Project Repository".

## Test Files

- `test_project_structure.py` - Tests for project directory structure validation (ADF-12 subtask)

## Parent Task

**ADF-11**: Initialize Project Repository
- **Type**: Setup  
- **Story Points**: 3
- **Epic**: ADF-1 (Environment & Project Setup)

## Subtasks

1. **ADF-12**: Create directory structure and initialize Git ✅
2. **ADF-13**: Configure .gitignore for Python, Docker, Airflow, and IDE files
3. **ADF-14**: Write comprehensive README.md with badges and quick start  
4. **ADF-15**: Set up requirements.txt with version pinning

## Running Tests

```bash
# Run all ADF-11 tests
python3 -m unittest discover tests/ADF-11/ -v

# Run specific test file
python3 tests/ADF-11/test_project_structure.py
```