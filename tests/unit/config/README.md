# Configuration Loader Test Suite

This directory contains comprehensive test cases for the configuration loading functionality of the Anomaly Detection Framework.

## Test Structure

### Test Categories

- **File System Operations** (`test_config_loader.py` - TestConfigLoaderFileSystemOperations)
  - Single and bulk file loading
  - Directory scanning with glob patterns
  - Recursive directory traversal
  - File modification time tracking
  - Permission handling

- **YAML Parsing** (`test_config_loader.py` - TestConfigLoaderYAMLParsing)
  - Valid YAML with various data types
  - YAML anchors and references
  - Empty files and null values
  - Unicode content handling

- **Error Handling** (`test_config_loader.py` - TestConfigLoaderErrorHandling)
  - Missing file scenarios
  - Invalid YAML syntax
  - Dangerous/unsafe YAML content
  - Corrupted binary files
  - Large file limits
  - Directory vs file handling
  - Network timeout simulation

- **Performance Testing** (`test_performance.py`)
  - Single config load benchmarks
  - Bulk loading scaling performance
  - Memory usage monitoring
  - Concurrent loading performance
  - Large file handling
  - Cache performance optimization
  - Stress testing scenarios

- **Caching** (`test_config_loader.py` - TestConfigLoaderCaching, `test_performance.py` - TestConfigLoaderCachingPerformance)
  - Cache hit behavior
  - Cache invalidation on file modification
  - Cache size limits
  - Memory efficiency

## Test Fixtures

### Key Fixtures (`test_fixtures.py`)

- `temp_config_dir`: Provides isolated temporary directories
- `sample_valid_config`: Standard valid configuration structure
- `sample_invalid_configs`: Various invalid configuration examples
- `config_factory`: Factory for creating test configuration files
- `ConfigFileFactory`: Utility class for programmatic config file creation

### Configuration Files

The test suite automatically creates and manages temporary configuration files for testing:

- Valid configurations with various complexity levels
- Invalid configurations for error testing
- Large configurations for performance testing
- Binary and corrupted files for error handling

## Running Tests

### All Tests
```bash
pytest tests/unit/config/ -v
```

### By Category
```bash
# File system operations
pytest tests/unit/config/ -m filesystem -v

# YAML parsing tests
pytest tests/unit/config/ -m yaml_parsing -v

# Error handling tests
pytest tests/unit/config/ -m error_handling -v

# Performance tests
pytest tests/unit/config/ -m performance -v

# Caching tests
pytest tests/unit/config/ -m caching -v
```

### Excluding Slow Tests
```bash
pytest tests/unit/config/ -v -m "not slow"
```

### Performance Benchmarks Only
```bash
pytest tests/unit/config/test_performance.py -v
```

## Test Markers

- `filesystem`: Tests involving file system operations
- `yaml_parsing`: Tests focused on YAML parsing
- `error_handling`: Tests verifying error scenarios
- `performance`: Tests measuring performance characteristics
- `caching`: Tests verifying caching behavior
- `slow`: Tests that take longer to execute

## Coverage Requirements

- **Minimum Coverage**: 80% (enforced by pytest-cov)
- **Target Areas**: All public methods of ConfigLoader class
- **Critical Paths**: Error handling, file parsing, caching logic

## Performance Benchmarks

### Expected Performance Characteristics

- **Single Config Load**: < 100ms
- **Bulk Loading (100 files)**: < 5 seconds
- **Cache Hit Improvement**: > 5x faster than cold load
- **Memory Usage**: < 10MB increase for 500 configs
- **Concurrent Safety**: No degradation under 4 concurrent threads

### Performance Test Types

1. **Benchmark Tests**: Measure absolute performance
2. **Scaling Tests**: Verify linear scaling with file count
3. **Memory Tests**: Monitor memory usage patterns
4. **Stress Tests**: High-frequency and mixed workload scenarios
5. **Cache Tests**: Verify caching effectiveness

## Test Data Management

### Automatic Cleanup
- All test files are automatically created and cleaned up
- Temporary directories are isolated per test
- No persistent test data between runs

### Factory Pattern
- `ConfigFileFactory` creates various test scenarios
- Supports valid, invalid, large, and binary file creation
- Parameterized content generation for scale testing

## Dependencies

### Core Testing
- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `pytest-mock`: Mocking utilities
- `pytest-xdist`: Parallel test execution

### Test Utilities
- `PyYAML`: YAML processing
- `psutil`: Memory monitoring (optional)
- `pathlib`: Path handling
- `tempfile`: Temporary file management

### Development Tools
- `pytest-html`: HTML test reports
- `pytest-benchmark`: Performance benchmarking

## Implementation Notes

### Test-Driven Development (TDD)
These tests are written before the actual ConfigLoader implementation, following TDD principles:

1. **Red Phase**: Tests fail initially (ConfigLoader not implemented)
2. **Green Phase**: Implement ConfigLoader to make tests pass
3. **Refactor Phase**: Improve implementation while keeping tests green

### Mock Implementation
- `MockConfigLoader` provides placeholder functionality for testing the test suite itself
- Real implementation will replace mock behavior
- Test structure validates the expected interface

### Error Simulation
- Comprehensive error scenarios covered
- Network timeouts, permission issues, corrupted files
- Realistic error conditions for production robustness

## Future Enhancements

### Additional Test Scenarios
- Remote configuration loading (HTTP/HTTPS)
- Configuration validation against schema
- Hot-reload functionality
- Configuration versioning and migration

### Integration Testing
- Integration with actual Snowflake connections
- End-to-end configuration loading in Airflow DAGs
- Real-world configuration file scenarios

### Performance Monitoring
- Continuous performance regression testing
- Memory leak detection over extended runs
- Profiling integration for optimization guidance