"""
Pytest configuration and shared fixtures for configuration loader tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from tests.unit.config.test_fixtures import ConfigFileFactory


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide a session-wide test data directory."""
    temp_dir = tempfile.mkdtemp(prefix="anomaly_detection_tests_")
    test_dir = Path(temp_dir)
    
    yield test_dir
    
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def isolated_config_dir(test_data_dir):
    """Provide an isolated config directory for each test."""
    config_dir = test_data_dir / "isolated_configs"
    config_dir.mkdir(exist_ok=True)
    
    yield config_dir
    
    # Cleanup after each test
    if config_dir.exists():
        shutil.rmtree(config_dir, ignore_errors=True)


@pytest.fixture
def config_factory_instance(isolated_config_dir):
    """Provide a config factory instance for each test."""
    factory = ConfigFileFactory(isolated_config_dir)
    yield factory
    factory.cleanup()


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "filesystem: tests that involve file system operations"
    )
    config.addinivalue_line(
        "markers", "yaml_parsing: tests that focus on YAML parsing"
    )
    config.addinivalue_line(
        "markers", "error_handling: tests that verify error scenarios"
    )
    config.addinivalue_line(
        "markers", "performance: tests that measure performance characteristics"
    )
    config.addinivalue_line(
        "markers", "caching: tests that verify caching behavior"
    )
    config.addinivalue_line(
        "markers", "slow: tests that are slow to execute"
    )


# Test collection modifications
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file names
        if "test_config_loader" in item.nodeid:
            if "performance" in item.name.lower():
                item.add_marker(pytest.mark.performance)
            if "cache" in item.name.lower():
                item.add_marker(pytest.mark.caching)
            if "error" in item.name.lower():
                item.add_marker(pytest.mark.error_handling)
            if "yaml" in item.name.lower():
                item.add_marker(pytest.mark.yaml_parsing)
            if any(x in item.name.lower() for x in ["file", "directory", "load"]):
                item.add_marker(pytest.mark.filesystem)
            if any(x in item.name.lower() for x in ["large", "bulk", "concurrent"]):
                item.add_marker(pytest.mark.slow)