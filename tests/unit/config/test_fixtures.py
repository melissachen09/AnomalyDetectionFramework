"""
Test fixtures and utilities for configuration loader tests.

This module provides reusable test fixtures, mock data, and utility functions
for testing the configuration loading system.
"""

import tempfile
import yaml
import pytest
from pathlib import Path
from typing import Dict, Any, List


@pytest.fixture
def temp_config_dir():
    """Provide a temporary directory for configuration files."""
    temp_dir = tempfile.mkdtemp()
    config_dir = Path(temp_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    yield config_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_valid_config():
    """Provide a sample valid configuration structure."""
    return {
        "event_config": {
            "name": "sample_event",
            "enabled": True,
            "data_source": {
                "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
                "date_column": "STATISTIC_DATE",
                "metrics": [
                    {
                        "column": "NUMBEROFVIEWS",
                        "alias": "total_views"
                    },
                    {
                        "column": "NUMBEROFENQUIRIES", 
                        "alias": "total_enquiries"
                    }
                ]
            },
            "detection": {
                "daily_checks": [
                    {
                        "detector": "threshold",
                        "metric": "total_views",
                        "min_value": 1000,
                        "max_value": 1000000,
                        "percentage_change_threshold": 0.5
                    },
                    {
                        "detector": "statistical",
                        "metric": "total_enquiries",
                        "z_score_threshold": 3.0,
                        "moving_average_window": 7
                    }
                ]
            },
            "alerting": {
                "critical": {
                    "condition": "deviation > 0.5",
                    "recipients": {
                        "email": ["director-bi@company.com"],
                        "slack": ["#data-alerts"]
                    }
                },
                "warning": {
                    "condition": "deviation > 0.2",
                    "recipients": {
                        "slack": ["#data-team"]
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_invalid_configs():
    """Provide samples of invalid configuration structures."""
    return {
        "missing_required_field": {
            "event_config": {
                "data_source": {
                    "table": "TEST_TABLE"
                }
                # Missing 'name' field
            }
        },
        "invalid_data_types": {
            "event_config": {
                "name": "invalid_types",
                "enabled": "not_a_boolean",  # Should be boolean
                "data_source": {
                    "table": 12345,  # Should be string
                    "date_column": "DATE_COL"
                }
            }
        },
        "empty_detection_rules": {
            "event_config": {
                "name": "empty_rules",
                "data_source": {
                    "table": "TEST_TABLE",
                    "date_column": "DATE_COL"
                },
                "detection": {
                    "daily_checks": []  # Empty detection rules
                }
            }
        }
    }


class ConfigFileFactory:
    """Factory class for creating test configuration files."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.created_files = []
    
    def create_config_file(self, filename: str, content: Dict[str, Any], 
                          subdir: str = None) -> Path:
        """Create a configuration file with given content."""
        if subdir:
            config_dir = self.base_dir / subdir
            config_dir.mkdir(parents=True, exist_ok=True)
        else:
            config_dir = self.base_dir
        
        config_path = config_dir / filename
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(content, f, default_flow_style=False, allow_unicode=True)
        
        self.created_files.append(config_path)
        return config_path
    
    def create_invalid_yaml_file(self, filename: str, content: str) -> Path:
        """Create a file with invalid YAML content."""
        config_path = self.base_dir / filename
        with open(config_path, 'w') as f:
            f.write(content)
        
        self.created_files.append(config_path)
        return config_path
    
    def create_binary_file(self, filename: str, size: int = 1024) -> Path:
        """Create a binary file that should not be parsed as YAML."""
        config_path = self.base_dir / filename
        with open(config_path, 'wb') as f:
            f.write(b'\x00\x01\x02\x03' * (size // 4))
        
        self.created_files.append(config_path)
        return config_path
    
    def create_large_config_file(self, filename: str, size_multiplier: int = 1000) -> Path:
        """Create a large configuration file for performance testing."""
        content = {
            "event_config": {
                "name": f"large_config_{size_multiplier}",
                "data_source": {
                    "table": "LARGE_TEST_TABLE",
                    "date_column": "DATE_COL"
                },
                "detection": {
                    "daily_checks": [
                        {
                            "detector": "threshold",
                            "metric": f"metric_{i}",
                            "min_value": i,
                            "max_value": i * 10
                        }
                        for i in range(size_multiplier)
                    ]
                },
                "metadata": {
                    f"key_{i}": f"value_{i}" * 100  # Make it actually large
                    for i in range(size_multiplier)
                }
            }
        }
        return self.create_config_file(filename, content)
    
    def cleanup(self):
        """Remove all created files."""
        for file_path in self.created_files:
            try:
                file_path.unlink()
            except FileNotFoundError:
                pass  # File already removed


@pytest.fixture
def config_factory(temp_config_dir):
    """Provide a configuration file factory."""
    factory = ConfigFileFactory(temp_config_dir)
    yield factory
    factory.cleanup()


class MockConfigLoader:
    """Mock implementation of ConfigLoader for testing purposes."""
    
    def __init__(self, enable_cache=False, max_cache_size=100, max_file_size_mb=10):
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        self.max_file_size_mb = max_file_size_mb
        self.cache = {}
        self.load_times = {}
        self.cache_stats = {"hits": 0, "misses": 0, "size": 0}
    
    def load_single_config(self, file_path: str):
        """Mock implementation of single config loading."""
        # This would be replaced with actual implementation
        if self.enable_cache and file_path in self.cache:
            self.cache_stats["hits"] += 1
            return self.cache[file_path]
        
        self.cache_stats["misses"] += 1
        
        # Simulate loading (placeholder)
        result = {"loaded_from": file_path}
        
        if self.enable_cache:
            self.cache[file_path] = result
            self.cache_stats["size"] = len(self.cache)
        
        return result
    
    def get_cache_stats(self):
        """Return cache statistics."""
        return self.cache_stats.copy()


# Performance testing utilities
def measure_execution_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def assert_performance_threshold(execution_time: float, threshold: float, 
                                operation_name: str):
    """Assert that execution time is within acceptable threshold."""
    assert execution_time < threshold, (
        f"{operation_name} took {execution_time:.3f}s, "
        f"which exceeds threshold of {threshold:.3f}s"
    )


# Error simulation utilities
class SimulatedError:
    """Context manager for simulating various error conditions."""
    
    def __init__(self, error_type: str):
        self.error_type = error_type
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def create_permission_denied_file(file_path: Path) -> Path:
    """Create a file with restricted permissions (Unix only)."""
    file_path.touch()
    import os
    if os.name != 'nt':  # Skip on Windows
        os.chmod(file_path, 0o000)
    return file_path


# Validation utilities
def validate_config_structure(config: Dict[str, Any]) -> List[str]:
    """Validate configuration structure and return list of errors."""
    errors = []
    
    if "event_config" not in config:
        errors.append("Missing required 'event_config' section")
        return errors
    
    event_config = config["event_config"]
    
    # Check required fields
    required_fields = ["name", "data_source"]
    for field in required_fields:
        if field not in event_config:
            errors.append(f"Missing required field: event_config.{field}")
    
    # Validate data source
    if "data_source" in event_config:
        data_source = event_config["data_source"]
        if not isinstance(data_source.get("table"), str):
            errors.append("data_source.table must be a string")
        if not isinstance(data_source.get("date_column"), str):
            errors.append("data_source.date_column must be a string")
    
    return errors