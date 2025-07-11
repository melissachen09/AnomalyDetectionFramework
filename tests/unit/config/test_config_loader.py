"""
Test suite for YAMLConfigLoader class.

This module contains comprehensive tests for the configuration loader functionality
including file system operations, YAML parsing, error scenarios, and performance validation.

Following TDD principles, these tests are written before implementation.
"""

import pytest
from pathlib import Path
import tempfile
import os
import yaml
from unittest.mock import Mock, patch, MagicMock
import time
from typing import Dict, Any, List

# Import the class that we're going to implement
# Note: This will fail initially as we haven't implemented it yet (TDD Red phase)
try:
    from src.detection.config.yaml_config_loader import YAMLConfigLoader
    from src.detection.config.exceptions import ConfigurationError, ValidationError
except ImportError:
    # Expected during TDD Red phase
    pass


class TestYAMLConfigLoader:
    """Test suite for YAMLConfigLoader class."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with test configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test directory structure
            events_dir = temp_path / "events"
            events_dir.mkdir()
            
            alerts_dir = temp_path / "alerts"
            alerts_dir.mkdir()
            
            # Create valid test config files
            valid_config = {
                "event_config": {
                    "name": "listing_views",
                    "description": "Property listing page views",
                    "data_source": {
                        "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
                        "date_column": "STATISTIC_DATE",
                        "metrics": [
                            {
                                "column": "NUMBEROFVIEWS",
                                "alias": "total_views"
                            }
                        ]
                    },
                    "detection": {
                        "daily_checks": [
                            {
                                "detector": "threshold",
                                "metric": "total_views",
                                "min_value": 10000,
                                "max_value": 1000000
                            }
                        ]
                    }
                }
            }
            
            # Write valid config file
            with open(events_dir / "listing_views.yaml", "w") as f:
                yaml.dump(valid_config, f)
            
            # Create another valid config
            enquiry_config = {
                "event_config": {
                    "name": "enquiries",
                    "description": "Property enquiries",
                    "data_source": {
                        "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
                        "date_column": "STATISTIC_DATE",
                        "metrics": [
                            {
                                "column": "NUMBEROFENQUIRIES",
                                "alias": "total_enquiries"
                            }
                        ]
                    }
                }
            }
            
            with open(events_dir / "enquiries.yaml", "w") as f:
                yaml.dump(enquiry_config, f)
            
            # Create invalid YAML file
            with open(events_dir / "invalid.yaml", "w") as f:
                f.write("invalid_yaml: [\n  - item1\n  - item2\n  missing_bracket")
            
            # Create empty file
            (events_dir / "empty.yaml").touch()
            
            # Create non-YAML file
            with open(events_dir / "not_yaml.txt", "w") as f:
                f.write("This is not a YAML file")
            
            yield temp_path

    @pytest.fixture
    def config_loader(self, temp_config_dir):
        """Create a YAMLConfigLoader instance for testing."""
        return YAMLConfigLoader(config_dir=temp_config_dir)

    def test_init_with_valid_directory(self, temp_config_dir):
        """Test YAMLConfigLoader initialization with valid directory."""
        loader = YAMLConfigLoader(config_dir=temp_config_dir)
        assert loader.config_dir == temp_config_dir
        assert loader.cache_enabled is True
        assert loader.cache_size == 100  # default cache size

    def test_init_with_invalid_directory(self):
        """Test YAMLConfigLoader initialization with invalid directory."""
        with pytest.raises(ConfigurationError) as exc_info:
            YAMLConfigLoader(config_dir="/nonexistent/path")
        assert "Configuration directory does not exist" in str(exc_info.value)

    def test_init_with_file_instead_of_directory(self, temp_config_dir):
        """Test YAMLConfigLoader initialization with file path instead of directory."""
        file_path = temp_config_dir / "events" / "listing_views.yaml"
        with pytest.raises(ConfigurationError) as exc_info:
            YAMLConfigLoader(config_dir=file_path)
        assert "Configuration directory path is not a directory" in str(exc_info.value)

    def test_init_with_custom_cache_settings(self, temp_config_dir):
        """Test YAMLConfigLoader initialization with custom cache settings."""
        loader = YAMLConfigLoader(
            config_dir=temp_config_dir,
            cache_enabled=False,
            cache_size=50
        )
        assert loader.cache_enabled is False
        assert loader.cache_size == 50

    def test_load_single_file_success(self, config_loader, temp_config_dir):
        """Test successful single file loading."""
        file_path = temp_config_dir / "events" / "listing_views.yaml"
        config = config_loader.load_single_file(file_path)
        
        assert config is not None
        assert "event_config" in config
        assert config["event_config"]["name"] == "listing_views"
        assert config["event_config"]["description"] == "Property listing page views"

    def test_load_single_file_nonexistent(self, config_loader, temp_config_dir):
        """Test loading non-existent file."""
        file_path = temp_config_dir / "events" / "nonexistent.yaml"
        with pytest.raises(ConfigurationError) as exc_info:
            config_loader.load_single_file(file_path)
        assert "Configuration file does not exist" in str(exc_info.value)

    def test_load_single_file_invalid_yaml(self, config_loader, temp_config_dir):
        """Test loading invalid YAML file."""
        file_path = temp_config_dir / "events" / "invalid.yaml"
        with pytest.raises(ConfigurationError) as exc_info:
            config_loader.load_single_file(file_path)
        assert "Failed to parse YAML file" in str(exc_info.value)

    def test_load_single_file_empty_file(self, config_loader, temp_config_dir):
        """Test loading empty file."""
        file_path = temp_config_dir / "events" / "empty.yaml"
        config = config_loader.load_single_file(file_path)
        assert config is None or config == {}

    def test_scan_directory_with_yaml_files(self, config_loader, temp_config_dir):
        """Test directory scanning with YAML files."""
        files = config_loader.scan_directory(temp_config_dir / "events")
        yaml_files = [f for f in files if f.suffix in ['.yaml', '.yml']]
        
        assert len(yaml_files) >= 2  # At least listing_views.yaml and enquiries.yaml
        assert any(f.name == "listing_views.yaml" for f in yaml_files)
        assert any(f.name == "enquiries.yaml" for f in yaml_files)

    def test_scan_directory_with_pattern(self, config_loader, temp_config_dir):
        """Test directory scanning with specific pattern."""
        files = config_loader.scan_directory(temp_config_dir / "events", pattern="*.yaml")
        
        assert all(f.suffix == ".yaml" for f in files)
        assert len(files) >= 2

    def test_scan_directory_recursive(self, config_loader, temp_config_dir):
        """Test recursive directory scanning."""
        files = config_loader.scan_directory(temp_config_dir, recursive=True)
        
        # Should find files in both events and alerts directories
        assert len(files) >= 2
        assert any("events" in str(f) for f in files)

    def test_scan_directory_nonexistent(self, config_loader, temp_config_dir):
        """Test scanning non-existent directory."""
        with pytest.raises(ConfigurationError) as exc_info:
            config_loader.scan_directory(temp_config_dir / "nonexistent")
        assert "Directory does not exist" in str(exc_info.value)

    def test_load_all_configs_success(self, config_loader):
        """Test loading all configuration files successfully."""
        configs = config_loader.load_all_configs()
        
        assert isinstance(configs, dict)
        assert len(configs) >= 2
        assert "listing_views" in configs
        assert "enquiries" in configs

    def test_load_all_configs_with_validation_errors(self, config_loader, temp_config_dir):
        """Test loading configs with validation errors."""
        # Create a config with validation errors
        invalid_config = {
            "event_config": {
                "name": "",  # Empty name should cause validation error
                "description": "Invalid config"
            }
        }
        
        with open(temp_config_dir / "events" / "invalid_config.yaml", "w") as f:
            yaml.dump(invalid_config, f)
        
        # Should handle validation errors gracefully
        with pytest.raises(ValidationError) as exc_info:
            config_loader.load_all_configs()
        
        assert "Validation failed" in str(exc_info.value)

    def test_load_all_configs_partial_failure(self, config_loader, temp_config_dir):
        """Test loading configs with some files failing."""
        # By default, should collect all errors and return what it can
        configs = config_loader.load_all_configs(fail_on_error=False)
        
        # Should still load valid configs despite some invalid ones
        assert len(configs) >= 2
        assert "listing_views" in configs
        assert "enquiries" in configs

    def test_get_config_by_name(self, config_loader):
        """Test getting configuration by name."""
        config = config_loader.get_config("listing_views")
        
        assert config is not None
        assert config["event_config"]["name"] == "listing_views"

    def test_get_config_by_name_nonexistent(self, config_loader):
        """Test getting non-existent configuration."""
        config = config_loader.get_config("nonexistent")
        assert config is None

    def test_list_config_names(self, config_loader):
        """Test listing all configuration names."""
        names = config_loader.list_config_names()
        
        assert isinstance(names, list)
        assert "listing_views" in names
        assert "enquiries" in names

    def test_reload_configs(self, config_loader, temp_config_dir):
        """Test reloading configurations."""
        # Load initial configs
        initial_configs = config_loader.load_all_configs()
        initial_count = len(initial_configs)
        
        # Add a new config file
        new_config = {
            "event_config": {
                "name": "new_metric",
                "description": "New metric for testing"
            }
        }
        
        with open(temp_config_dir / "events" / "new_metric.yaml", "w") as f:
            yaml.dump(new_config, f)
        
        # Reload configs
        config_loader.reload_configs()
        updated_configs = config_loader.load_all_configs()
        
        assert len(updated_configs) == initial_count + 1
        assert "new_metric" in updated_configs

    def test_cache_functionality(self, config_loader, temp_config_dir):
        """Test that caching works correctly."""
        file_path = temp_config_dir / "events" / "listing_views.yaml"
        
        # First load
        start_time = time.time()
        config1 = config_loader.load_single_file(file_path)
        first_load_time = time.time() - start_time
        
        # Second load (should be faster due to caching)
        start_time = time.time()
        config2 = config_loader.load_single_file(file_path)
        second_load_time = time.time() - start_time
        
        assert config1 == config2
        # Second load should be faster (cached)
        assert second_load_time < first_load_time

    def test_cache_invalidation(self, config_loader, temp_config_dir):
        """Test cache invalidation when file is modified."""
        file_path = temp_config_dir / "events" / "listing_views.yaml"
        
        # First load
        config1 = config_loader.load_single_file(file_path)
        
        # Modify the file
        time.sleep(0.1)  # Ensure different modification time
        modified_config = config1.copy()
        modified_config["event_config"]["description"] = "Modified description"
        
        with open(file_path, "w") as f:
            yaml.dump(modified_config, f)
        
        # Second load should get updated content
        config2 = config_loader.load_single_file(file_path)
        assert config2["event_config"]["description"] == "Modified description"

    def test_cache_size_limit(self, temp_config_dir):
        """Test cache size limit functionality."""
        # Create loader with small cache size
        loader = YAMLConfigLoader(config_dir=temp_config_dir, cache_size=2)
        
        # Create multiple config files
        for i in range(5):
            config = {
                "event_config": {
                    "name": f"test_config_{i}",
                    "description": f"Test config {i}"
                }
            }
            
            with open(temp_config_dir / "events" / f"test_config_{i}.yaml", "w") as f:
                yaml.dump(config, f)
        
        # Load all configs
        for i in range(5):
            file_path = temp_config_dir / "events" / f"test_config_{i}.yaml"
            loader.load_single_file(file_path)
        
        # Cache should only contain the last 2 items
        assert len(loader._cache) <= 2

    def test_disabled_cache(self, temp_config_dir):
        """Test functionality with disabled cache."""
        loader = YAMLConfigLoader(config_dir=temp_config_dir, cache_enabled=False)
        file_path = temp_config_dir / "events" / "listing_views.yaml"
        
        # Multiple loads should not use cache
        config1 = loader.load_single_file(file_path)
        config2 = loader.load_single_file(file_path)
        
        assert config1 == config2
        assert len(loader._cache) == 0

    def test_concurrent_access(self, config_loader, temp_config_dir):
        """Test thread-safe concurrent access."""
        import threading
        import queue
        
        file_path = temp_config_dir / "events" / "listing_views.yaml"
        results = queue.Queue()
        
        def load_config():
            try:
                config = config_loader.load_single_file(file_path)
                results.put(config)
            except Exception as e:
                results.put(e)
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=load_config)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        configs = []
        while not results.empty():
            result = results.get()
            assert not isinstance(result, Exception)
            configs.append(result)
        
        assert len(configs) == 10
        # All configs should be identical
        for config in configs[1:]:
            assert config == configs[0]

    def test_performance_with_large_configs(self, config_loader, temp_config_dir):
        """Test performance with large configuration files."""
        # Create a large config file
        large_config = {
            "event_config": {
                "name": "large_config",
                "description": "Large configuration for performance testing",
                "data_source": {
                    "table": "LARGE_TABLE",
                    "metrics": []
                }
            }
        }
        
        # Add many metrics
        for i in range(1000):
            large_config["event_config"]["data_source"]["metrics"].append({
                "column": f"METRIC_{i}",
                "alias": f"metric_{i}"
            })
        
        large_file_path = temp_config_dir / "events" / "large_config.yaml"
        with open(large_file_path, "w") as f:
            yaml.dump(large_config, f)
        
        # Test loading performance
        start_time = time.time()
        config = config_loader.load_single_file(large_file_path)
        load_time = time.time() - start_time
        
        assert config is not None
        assert len(config["event_config"]["data_source"]["metrics"]) == 1000
        # Should load within reasonable time (adjust threshold as needed)
        assert load_time < 5.0

    def test_memory_usage(self, config_loader, temp_config_dir):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Load many configs
        for i in range(100):
            config = {
                "event_config": {
                    "name": f"memory_test_{i}",
                    "description": f"Memory test config {i}"
                }
            }
            
            with open(temp_config_dir / "events" / f"memory_test_{i}.yaml", "w") as f:
                yaml.dump(config, f)
            
            file_path = temp_config_dir / "events" / f"memory_test_{i}.yaml"
            config_loader.load_single_file(file_path)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 50 * 1024 * 1024  # 50MB threshold

    def test_error_aggregation(self, config_loader, temp_config_dir):
        """Test error aggregation for multiple failed files."""
        # Create multiple invalid files
        for i in range(3):
            with open(temp_config_dir / "events" / f"invalid_{i}.yaml", "w") as f:
                f.write(f"invalid_yaml_{i}: [\n  - item1\n  missing_bracket")
        
        # Should aggregate all errors
        with pytest.raises(ValidationError) as exc_info:
            config_loader.load_all_configs()
        
        error_message = str(exc_info.value)
        assert "invalid_0.yaml" in error_message
        assert "invalid_1.yaml" in error_message
        assert "invalid_2.yaml" in error_message

    def test_path_resolution(self, config_loader, temp_config_dir):
        """Test proper path resolution for different input types."""
        # Test with string path
        file_path_str = str(temp_config_dir / "events" / "listing_views.yaml")
        config1 = config_loader.load_single_file(file_path_str)
        
        # Test with Path object
        file_path_obj = temp_config_dir / "events" / "listing_views.yaml"
        config2 = config_loader.load_single_file(file_path_obj)
        
        assert config1 == config2

    def test_yaml_safe_loading(self, config_loader, temp_config_dir):
        """Test that YAML loading is safe and doesn't execute arbitrary code."""
        # Create a potentially dangerous YAML file
        dangerous_yaml = """
        !!python/object/apply:os.system
        args: ['echo "This should not execute"']
        """
        
        dangerous_file = temp_config_dir / "events" / "dangerous.yaml"
        with open(dangerous_file, "w") as f:
            f.write(dangerous_yaml)
        
        # Should fail to load dangerous content
        with pytest.raises(ConfigurationError) as exc_info:
            config_loader.load_single_file(dangerous_file)
        
        assert "Failed to parse YAML file" in str(exc_info.value)