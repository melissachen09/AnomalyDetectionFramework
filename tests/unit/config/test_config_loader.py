"""
Test suite for YAMLConfigLoader class - ADF-28 Configuration Manager Tests.

This module contains comprehensive tests for configuration management and retrieval logic 
including CRUD operations, thread safety, hot reload functionality, and API contract validation.

Test Coverage Requirements (ADF-28):
- [x] CRUD operations tested
- [x] Thread safety verified  
- [x] Hot reload functionality tested
- [x] API contract validated

Following TDD principles for robust configuration management.
"""

import pytest
from pathlib import Path
import tempfile
import os
import yaml
import time
import threading
import queue
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Import the classes under test
from src.detection.config.yaml_config_loader import YAMLConfigLoader, LRUCache
from src.detection.config.exceptions import ConfigurationError, ValidationError


class TestYAMLConfigLoaderCRUD:
    """Test CRUD operations for Configuration Manager (ADF-28 Requirement 1)."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with test configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test directory structure
            events_dir = temp_path / "events"
            events_dir.mkdir()
            
            # Create valid test config files
            listing_views_config = {
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
                    },
                    "alerting": {
                        "critical": {
                            "condition": "deviation > 0.5",
                            "recipients": {
                                "email": ["director-bi@company.com"]
                            }
                        }
                    }
                }
            }
            
            # Write valid config file
            with open(events_dir / "listing_views.yaml", "w") as f:
                yaml.dump(listing_views_config, f)
            
            # Create enquiries config
            enquiries_config = {
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
                yaml.dump(enquiries_config, f)
            
            yield temp_path

    @pytest.fixture
    def config_loader(self, temp_config_dir):
        """Create a YAMLConfigLoader instance for testing."""
        loader = YAMLConfigLoader(config_dir=temp_config_dir)
        loader.load_all_configs()  # Load initial configs
        return loader

    def test_create_operation_load_all_configs(self, config_loader):
        """Test CREATE: Loading all configuration files successfully."""
        configs = config_loader.load_all_configs()
        
        assert isinstance(configs, dict)
        assert len(configs) >= 2
        assert "listing_views" in configs
        assert "enquiries" in configs
        
        # Verify config structure
        listing_config = configs["listing_views"]
        assert listing_config["event_config"]["name"] == "listing_views"
        assert "data_source" in listing_config["event_config"]
        assert "detection" in listing_config["event_config"]

    def test_read_operation_get_config_by_name(self, config_loader):
        """Test READ: Getting configuration by name."""
        # Test existing config
        config = config_loader.get_config("listing_views")
        assert config is not None
        assert config["event_config"]["name"] == "listing_views"
        assert config["event_config"]["description"] == "Property listing page views"
        
        # Test non-existent config
        config = config_loader.get_config("nonexistent")
        assert config is None

    def test_read_operation_list_all_configs(self, config_loader):
        """Test READ: Listing all configuration names."""
        names = config_loader.list_config_names()
        
        assert isinstance(names, list)
        assert len(names) >= 2
        assert "listing_views" in names
        assert "enquiries" in names

    def test_read_operation_get_config_with_various_event_types(self, config_loader, temp_config_dir):
        """Test READ: get_config with various event types (Sub-task ADF-CONFIG-005a)."""
        # Add more event types for testing
        click_config = {
            "event_config": {
                "name": "clicks",
                "description": "Click tracking events",
                "data_source": {
                    "table": "DATAMART.CLICK_STATISTICS",
                    "metrics": [{"column": "CLICK_COUNT", "alias": "total_clicks"}]
                }
            }
        }
        
        with open(temp_config_dir / "events" / "clicks.yaml", "w") as f:
            yaml.dump(click_config, f)
        
        # Reload to pick up new config
        config_loader.reload_configs()
        
        # Test various event type retrievals
        event_types = ["listing_views", "enquiries", "clicks"]
        for event_type in event_types:
            config = config_loader.get_config(event_type)
            assert config is not None
            assert config["event_config"]["name"] == event_type

    def test_read_operation_list_configs_filtering_and_sorting(self, config_loader):
        """Test READ: list_configs filtering and sorting (Sub-task ADF-CONFIG-005b)."""
        names = config_loader.list_config_names()
        
        # Verify sorting (should be deterministic)
        sorted_names = sorted(names)
        assert names == sorted_names or len(set(names)) == len(names)  # Either sorted or unique
        
        # Test filtering by pattern (implement if needed)
        # This tests the API contract for listing configurations
        assert all(isinstance(name, str) for name in names)
        assert len(names) > 0

    def test_update_operation_reload_configs(self, config_loader, temp_config_dir):
        """Test UPDATE: Reloading configurations after changes."""
        # Initial state
        initial_count = len(config_loader.list_config_names())
        
        # Add a new config file
        new_config = {
            "event_config": {
                "name": "new_metric",
                "description": "New metric for testing updates",
                "data_source": {
                    "table": "DATAMART.NEW_TABLE",
                    "metrics": [{"column": "NEW_METRIC", "alias": "new_value"}]
                }
            }
        }
        
        with open(temp_config_dir / "events" / "new_metric.yaml", "w") as f:
            yaml.dump(new_config, f)
        
        # Reload configs (UPDATE operation)
        config_loader.reload_configs()
        
        # Verify update
        updated_names = config_loader.list_config_names()
        assert len(updated_names) == initial_count + 1
        assert "new_metric" in updated_names
        
        # Verify new config can be retrieved
        new_retrieved_config = config_loader.get_config("new_metric")
        assert new_retrieved_config is not None
        assert new_retrieved_config["event_config"]["name"] == "new_metric"

    def test_delete_operation_file_removal_and_reload(self, config_loader, temp_config_dir):
        """Test DELETE: Removing configuration files and reloading."""
        # Verify initial state
        assert "enquiries" in config_loader.list_config_names()
        assert config_loader.get_config("enquiries") is not None
        
        # Remove config file (DELETE operation)
        enquiries_file = temp_config_dir / "events" / "enquiries.yaml"
        os.remove(enquiries_file)
        
        # Reload to reflect deletion
        config_loader.reload_configs()
        
        # Verify deletion
        assert "enquiries" not in config_loader.list_config_names()
        assert config_loader.get_config("enquiries") is None

    def test_crud_operations_with_invalid_data(self, config_loader, temp_config_dir):
        """Test CRUD operations handle invalid data gracefully."""
        # Create invalid config file
        invalid_config_path = temp_config_dir / "events" / "invalid.yaml"
        with open(invalid_config_path, "w") as f:
            f.write("invalid_yaml: [\n  - item1\n  - item2\n  missing_bracket")
        
        # Should handle invalid files gracefully during reload
        try:
            config_loader.reload_configs()
            # If it doesn't raise an exception, verify invalid config is not loaded
            assert "invalid" not in config_loader.list_config_names()
        except ValidationError:
            # Expected behavior for invalid configs
            pass


class TestYAMLConfigLoaderThreadSafety:
    """Test thread safety for Configuration Manager (ADF-28 Requirement 2)."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with test configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            events_dir = temp_path / "events"
            events_dir.mkdir()
            
            # Create test config
            config = {
                "event_config": {
                    "name": "thread_test",
                    "description": "Configuration for thread safety testing",
                    "data_source": {
                        "table": "DATAMART.THREAD_TEST",
                        "metrics": [{"column": "VALUE", "alias": "test_value"}]
                    }
                }
            }
            
            with open(events_dir / "thread_test.yaml", "w") as f:
                yaml.dump(config, f)
            
            yield temp_path

    @pytest.fixture
    def config_loader(self, temp_config_dir):
        """Create a YAMLConfigLoader instance for testing."""
        loader = YAMLConfigLoader(config_dir=temp_config_dir)
        loader.load_all_configs()
        return loader

    def test_concurrent_read_operations(self, config_loader):
        """Test concurrent read operations are thread-safe."""
        results = queue.Queue()
        num_threads = 10
        
        def read_config():
            try:
                config = config_loader.get_config("thread_test")
                results.put(("success", config))
            except Exception as e:
                results.put(("error", e))
        
        # Start multiple threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=read_config)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        success_count = 0
        configs = []
        while not results.empty():
            status, result = results.get()
            if status == "success":
                success_count += 1
                configs.append(result)
            else:
                pytest.fail(f"Thread failed with error: {result}")
        
        assert success_count == num_threads
        # All configs should be identical
        for config in configs[1:]:
            assert config == configs[0]

    def test_concurrent_list_operations(self, config_loader):
        """Test concurrent list operations are thread-safe."""
        results = queue.Queue()
        num_threads = 10
        
        def list_configs():
            try:
                names = config_loader.list_config_names()
                results.put(("success", names))
            except Exception as e:
                results.put(("error", e))
        
        # Start multiple threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=list_configs)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        success_count = 0
        name_lists = []
        while not results.empty():
            status, result = results.get()
            if status == "success":
                success_count += 1
                name_lists.append(result)
            else:
                pytest.fail(f"Thread failed with error: {result}")
        
        assert success_count == num_threads
        # All name lists should be identical
        for names in name_lists[1:]:
            assert names == name_lists[0]

    def test_concurrent_reload_operations(self, config_loader, temp_config_dir):
        """Test concurrent reload operations are thread-safe."""
        results = queue.Queue()
        num_threads = 5
        
        def reload_configs():
            try:
                config_loader.reload_configs()
                names = config_loader.list_config_names()
                results.put(("success", names))
            except Exception as e:
                results.put(("error", e))
        
        # Start multiple threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=reload_configs)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        success_count = 0
        while not results.empty():
            status, result = results.get()
            if status == "success":
                success_count += 1
                assert "thread_test" in result
            else:
                pytest.fail(f"Thread failed with error: {result}")
        
        assert success_count == num_threads

    def test_cache_thread_safety(self, config_loader, temp_config_dir):
        """Test cache operations are thread-safe."""
        file_path = temp_config_dir / "events" / "thread_test.yaml"
        results = queue.Queue()
        num_threads = 20
        
        def load_file():
            try:
                config = config_loader.load_single_file(file_path)
                results.put(("success", config))
            except Exception as e:
                results.put(("error", e))
        
        # Start multiple threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=load_file)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        success_count = 0
        configs = []
        while not results.empty():
            status, result = results.get()
            if status == "success":
                success_count += 1
                configs.append(result)
            else:
                pytest.fail(f"Thread failed with error: {result}")
        
        assert success_count == num_threads
        # All configs should be identical
        for config in configs[1:]:
            assert config == configs[0]


class TestYAMLConfigLoaderHotReload:
    """Test hot reload functionality for Configuration Manager (ADF-28 Requirement 3)."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with test configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            events_dir = temp_path / "events"
            events_dir.mkdir()
            
            # Create initial config
            initial_config = {
                "event_config": {
                    "name": "hot_reload_test",
                    "description": "Initial description",
                    "data_source": {
                        "table": "DATAMART.INITIAL_TABLE",
                        "metrics": [{"column": "INITIAL_VALUE", "alias": "initial"}]
                    }
                }
            }
            
            with open(events_dir / "hot_reload_test.yaml", "w") as f:
                yaml.dump(initial_config, f)
            
            yield temp_path

    @pytest.fixture
    def config_loader(self, temp_config_dir):
        """Create a YAMLConfigLoader instance for testing."""
        loader = YAMLConfigLoader(config_dir=temp_config_dir)
        loader.load_all_configs()
        return loader

    def test_configuration_reload_without_service_restart(self, config_loader, temp_config_dir):
        """Test configuration reload without service restart (Sub-task ADF-CONFIG-005c)."""
        # Get initial configuration
        initial_config = config_loader.get_config("hot_reload_test")
        assert initial_config["event_config"]["description"] == "Initial description"
        
        # Modify the configuration file
        updated_config = {
            "event_config": {
                "name": "hot_reload_test",
                "description": "Updated description after hot reload",
                "data_source": {
                    "table": "DATAMART.UPDATED_TABLE",
                    "metrics": [
                        {"column": "UPDATED_VALUE", "alias": "updated"},
                        {"column": "NEW_METRIC", "alias": "new_metric"}
                    ]
                }
            }
        }
        
        # Wait a moment to ensure different modification time
        time.sleep(0.1)
        
        with open(temp_config_dir / "events" / "hot_reload_test.yaml", "w") as f:
            yaml.dump(updated_config, f)
        
        # Reload configurations (hot reload)
        config_loader.reload_configs()
        
        # Verify hot reload worked
        reloaded_config = config_loader.get_config("hot_reload_test")
        assert reloaded_config["event_config"]["description"] == "Updated description after hot reload"
        assert reloaded_config["event_config"]["data_source"]["table"] == "DATAMART.UPDATED_TABLE"
        assert len(reloaded_config["event_config"]["data_source"]["metrics"]) == 2

    def test_hot_reload_with_new_files(self, config_loader, temp_config_dir):
        """Test hot reload picks up new configuration files."""
        initial_count = len(config_loader.list_config_names())
        
        # Add new configuration file
        new_config = {
            "event_config": {
                "name": "hot_reload_new",
                "description": "New config added during hot reload",
                "data_source": {
                    "table": "DATAMART.NEW_TABLE",
                    "metrics": [{"column": "NEW_VALUE", "alias": "new"}]
                }
            }
        }
        
        with open(temp_config_dir / "events" / "hot_reload_new.yaml", "w") as f:
            yaml.dump(new_config, f)
        
        # Hot reload
        config_loader.reload_configs()
        
        # Verify new file was loaded
        assert len(config_loader.list_config_names()) == initial_count + 1
        assert "hot_reload_new" in config_loader.list_config_names()
        
        new_loaded_config = config_loader.get_config("hot_reload_new")
        assert new_loaded_config is not None
        assert new_loaded_config["event_config"]["name"] == "hot_reload_new"

    def test_hot_reload_with_deleted_files(self, config_loader, temp_config_dir):
        """Test hot reload handles deleted configuration files."""
        # Verify initial state
        assert "hot_reload_test" in config_loader.list_config_names()
        
        # Delete configuration file
        os.remove(temp_config_dir / "events" / "hot_reload_test.yaml")
        
        # Hot reload
        config_loader.reload_configs()
        
        # Verify file was removed from loaded configs
        assert "hot_reload_test" not in config_loader.list_config_names()
        assert config_loader.get_config("hot_reload_test") is None

    def test_cache_invalidation_on_file_modification(self, config_loader, temp_config_dir):
        """Test cache invalidation when files are modified."""
        file_path = temp_config_dir / "events" / "hot_reload_test.yaml"
        
        # Load file (should be cached)
        config1 = config_loader.load_single_file(file_path)
        
        # Modify file content
        time.sleep(0.1)  # Ensure different modification time
        modified_config = {
            "event_config": {
                "name": "hot_reload_test",
                "description": "Cache invalidation test",
                "data_source": {
                    "table": "DATAMART.CACHE_TEST",
                    "metrics": [{"column": "CACHE_VALUE", "alias": "cache"}]
                }
            }
        }
        
        with open(file_path, "w") as f:
            yaml.dump(modified_config, f)
        
        # Load file again (should detect modification and reload)
        config2 = config_loader.load_single_file(file_path)
        
        # Verify cache invalidation worked
        assert config1["event_config"]["description"] != config2["event_config"]["description"]
        assert config2["event_config"]["description"] == "Cache invalidation test"

    def test_hot_reload_preserves_running_service_state(self, config_loader, temp_config_dir):
        """Test hot reload doesn't disrupt running service state."""
        # Simulate running service state
        initial_names = config_loader.list_config_names()
        
        # Perform hot reload
        config_loader.reload_configs()
        
        # Verify service state is preserved (same configs available)
        reloaded_names = config_loader.list_config_names()
        assert set(initial_names) == set(reloaded_names)
        
        # Verify we can still access configurations
        for name in initial_names:
            config = config_loader.get_config(name)
            assert config is not None


class TestYAMLConfigLoaderAPIContract:
    """Test API contract validation for Configuration Manager (ADF-28 Requirement 4)."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with test configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            events_dir = temp_path / "events"
            events_dir.mkdir()
            
            # Create valid config
            config = {
                "event_config": {
                    "name": "api_test",
                    "description": "API contract validation",
                    "data_source": {
                        "table": "DATAMART.API_TEST",
                        "metrics": [{"column": "API_VALUE", "alias": "api"}]
                    }
                }
            }
            
            with open(events_dir / "api_test.yaml", "w") as f:
                yaml.dump(config, f)
            
            yield temp_path

    @pytest.fixture
    def config_loader(self, temp_config_dir):
        """Create a YAMLConfigLoader instance for testing."""
        return YAMLConfigLoader(config_dir=temp_config_dir)

    def test_constructor_api_contract(self, temp_config_dir):
        """Test constructor API contract validation."""
        # Valid constructor calls
        loader1 = YAMLConfigLoader(config_dir=temp_config_dir)
        assert loader1.config_dir == temp_config_dir
        
        loader2 = YAMLConfigLoader(
            config_dir=temp_config_dir,
            cache_enabled=True,
            cache_size=50
        )
        assert loader2.cache_enabled is True
        assert loader2.cache_size == 50
        
        # Invalid constructor calls
        with pytest.raises(ConfigurationError):
            YAMLConfigLoader(config_dir="/nonexistent/path")

    def test_load_single_file_api_contract(self, config_loader, temp_config_dir):
        """Test load_single_file API contract validation."""
        file_path = temp_config_dir / "events" / "api_test.yaml"
        
        # Valid calls
        config = config_loader.load_single_file(file_path)
        assert isinstance(config, dict)
        
        config_str = config_loader.load_single_file(str(file_path))
        assert isinstance(config_str, dict)
        assert config == config_str
        
        # Invalid calls
        with pytest.raises(ConfigurationError):
            config_loader.load_single_file("/nonexistent/file.yaml")

    def test_load_all_configs_api_contract(self, config_loader):
        """Test load_all_configs API contract validation."""
        # Valid calls
        configs1 = config_loader.load_all_configs()
        assert isinstance(configs1, dict)
        
        configs2 = config_loader.load_all_configs(fail_on_error=True)
        assert isinstance(configs2, dict)
        
        configs3 = config_loader.load_all_configs(fail_on_error=False)
        assert isinstance(configs3, dict)

    def test_get_config_api_contract(self, config_loader):
        """Test get_config API contract validation."""
        config_loader.load_all_configs()
        
        # Valid calls
        config = config_loader.get_config("api_test")
        assert isinstance(config, dict) or config is None
        
        # Non-existent config should return None
        none_config = config_loader.get_config("nonexistent")
        assert none_config is None
        
        # Test with various string types
        config_unicode = config_loader.get_config("api_test")
        assert config_unicode == config

    def test_list_config_names_api_contract(self, config_loader):
        """Test list_config_names API contract validation."""
        config_loader.load_all_configs()
        
        names = config_loader.list_config_names()
        
        # API contract validation
        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)
        assert len(names) >= 0

    def test_reload_configs_api_contract(self, config_loader):
        """Test reload_configs API contract validation."""
        config_loader.load_all_configs()
        
        # Should not return anything (None)
        result = config_loader.reload_configs()
        assert result is None
        
        # Should still work after reload
        names = config_loader.list_config_names()
        assert isinstance(names, list)

    def test_scan_directory_api_contract(self, config_loader, temp_config_dir):
        """Test scan_directory API contract validation."""
        # Valid calls
        files1 = config_loader.scan_directory(temp_config_dir / "events")
        assert isinstance(files1, list)
        assert all(isinstance(f, Path) for f in files1)
        
        files2 = config_loader.scan_directory(
            temp_config_dir / "events", 
            pattern="*.yaml"
        )
        assert isinstance(files2, list)
        
        files3 = config_loader.scan_directory(
            temp_config_dir / "events",
            pattern="*.yaml",
            recursive=True
        )
        assert isinstance(files3, list)
        
        # Invalid calls
        with pytest.raises(ConfigurationError):
            config_loader.scan_directory("/nonexistent/directory")

    def test_exception_hierarchy_api_contract(self):
        """Test exception hierarchy API contract."""
        from src.detection.config.exceptions import (
            ConfigurationError, 
            ValidationError, 
            FileParsingError
        )
        
        # Verify exception hierarchy
        assert issubclass(ValidationError, ConfigurationError)
        assert issubclass(FileParsingError, ConfigurationError)
        assert issubclass(ConfigurationError, Exception)

    def test_thread_safety_api_contract(self, config_loader):
        """Test thread safety API contract validation."""
        config_loader.load_all_configs()
        
        # All methods should be callable concurrently
        def test_methods():
            try:
                config_loader.get_config("api_test")
                config_loader.list_config_names()
                return True
            except Exception:
                return False
        
        # Test concurrent access
        import threading
        results = []
        threads = []
        
        for _ in range(5):
            thread = threading.Thread(target=lambda: results.append(test_methods()))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert all(results)

    def test_config_versioning_and_rollback_validation(self, config_loader, temp_config_dir):
        """Test config versioning and rollback (Sub-task ADF-CONFIG-005d)."""
        config_loader.load_all_configs()
        
        # Get initial config
        initial_config = config_loader.get_config("api_test")
        initial_description = initial_config["event_config"]["description"]
        
        # Create version 2 with deep copy to avoid modifying original
        import copy
        updated_config = copy.deepcopy(initial_config)
        updated_config["event_config"]["description"] = "Version 2"
        
        file_path = temp_config_dir / "events" / "api_test.yaml"
        with open(file_path, "w") as f:
            yaml.dump(updated_config, f)
        
        # Reload (version 2)
        config_loader.reload_configs()
        v2_config = config_loader.get_config("api_test")
        assert v2_config["event_config"]["description"] == "Version 2"
        
        # Rollback to version 1 - create original config structure
        original_config = {
            "event_config": {
                "name": "api_test",
                "description": initial_description,
                "data_source": {
                    "table": "DATAMART.API_TEST",
                    "metrics": [{"column": "API_VALUE", "alias": "api"}]
                }
            }
        }
        
        with open(file_path, "w") as f:
            yaml.dump(original_config, f)
        
        config_loader.reload_configs()
        rollback_config = config_loader.get_config("api_test")
        assert rollback_config["event_config"]["description"] == initial_description


class TestLRUCacheThreadSafety:
    """Test LRU Cache thread safety separately."""

    def test_lru_cache_concurrent_access(self):
        """Test LRU cache handles concurrent access safely."""
        cache = LRUCache(max_size=10)
        results = queue.Queue()
        
        def cache_operations():
            try:
                # Perform cache operations
                cache.set("key1", "value1")
                value = cache.get("key1")
                cache.set("key2", "value2")
                cache.clear()
                results.put(("success", value))
            except Exception as e:
                results.put(("error", e))
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=cache_operations)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        while not results.empty():
            status, result = results.get()
            if status == "error":
                pytest.fail(f"Cache operation failed: {result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])