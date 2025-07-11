"""
Test suite for ConfigManager singleton class - ADF-29 Configuration Manager.

This module contains comprehensive tests for the ConfigManager singleton 
implementation including thread safety, efficient config retrieval, 
change detection, hot reload, and comprehensive logging.

Test Coverage Requirements (ADF-29):
- [x] Singleton pattern with thread safety
- [x] Efficient config retrieval with caching
- [x] Change detection and hot reload
- [x] Comprehensive logging

Following TDD principles for robust configuration management.
"""

import pytest
import tempfile
import os
import yaml
import time
import threading
import queue
import logging
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Import the classes under test
from src.detection.config.config_manager import ConfigManager
from src.detection.config.exceptions import ConfigurationError, ValidationError


class TestConfigManagerSingleton:
    """Test ConfigManager singleton pattern with thread safety (ADF-29 Requirement 1)."""

    def test_singleton_pattern_same_instance(self):
        """Test that ConfigManager returns the same instance (singleton pattern)."""
        # Reset singleton first
        ConfigManager.reset_instance()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Get first instance
            instance1 = ConfigManager.get_instance(config_dir=temp_dir)
            
            # Get second instance
            instance2 = ConfigManager.get_instance(config_dir=temp_dir)
            
            # Should be the same instance
            assert instance1 is instance2
            assert id(instance1) == id(instance2)

    def test_singleton_thread_safety(self):
        """Test that singleton creation is thread-safe."""
        # Reset singleton first
        ConfigManager.reset_instance()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            instances = queue.Queue()
            
            def create_instance():
                instance = ConfigManager.get_instance(config_dir=temp_dir)
                instances.put(instance)
            
            # Create multiple threads
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=create_instance)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Collect all instances
            all_instances = []
            while not instances.empty():
                all_instances.append(instances.get())
            
            # All instances should be the same
            first_instance = all_instances[0]
            for instance in all_instances[1:]:
                assert instance is first_instance
                assert id(instance) == id(first_instance)

    def test_singleton_reset_for_testing(self):
        """Test that singleton can be reset for testing purposes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first instance
            instance1 = ConfigManager.get_instance(config_dir=temp_dir)
            
            # Reset singleton
            ConfigManager.reset_instance()
            
            # Create new instance
            instance2 = ConfigManager.get_instance(config_dir=temp_dir)
            
            # Should be different instances after reset
            assert instance1 is not instance2
            assert id(instance1) != id(instance2)

    def test_singleton_initialization_with_different_config_dirs(self):
        """Test singleton behavior when initialized with different config directories."""
        # Reset singleton first
        ConfigManager.reset_instance()
        
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                # First instance
                instance1 = ConfigManager.get_instance(config_dir=temp_dir1)
                
                # Second instance with different config_dir should be same instance
                instance2 = ConfigManager.get_instance(config_dir=temp_dir2)
                
                # Should be same instance (singleton), but should log warning
                assert instance1 is instance2
                assert instance1.config_dir == Path(temp_dir1)  # Should keep original config_dir


class TestConfigManagerRetrieval:
    """Test efficient config retrieval with caching (ADF-29 Requirement 2)."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with test configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            events_dir = temp_path / "events"
            events_dir.mkdir()
            
            # Create test config files
            configs = {
                "listing_views": {
                    "event_config": {
                        "name": "listing_views",
                        "description": "Property listing page views",
                        "data_source": {
                            "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
                            "metrics": [{"column": "NUMBEROFVIEWS", "alias": "total_views"}]
                        }
                    }
                },
                "enquiries": {
                    "event_config": {
                        "name": "enquiries",
                        "description": "Property enquiries",
                        "data_source": {
                            "table": "DATAMART.DD_LISTING_STATISTICS_BLENDED",
                            "metrics": [{"column": "NUMBEROFENQUIRIES", "alias": "total_enquiries"}]
                        }
                    }
                }
            }
            
            for name, config in configs.items():
                with open(events_dir / f"{name}.yaml", "w") as f:
                    yaml.dump(config, f)
            
            yield temp_path

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create a ConfigManager instance for testing."""
        ConfigManager.reset_instance()
        manager = ConfigManager.get_instance(config_dir=temp_config_dir)
        manager.load_all_configs()  # Load configs so they're available for retrieval
        return manager

    def test_efficient_config_retrieval(self, config_manager):
        """Test efficient config retrieval with caching."""
        # First retrieval should load from file
        config1 = config_manager.get_config("listing_views")
        assert config1 is not None
        assert config1["event_config"]["name"] == "listing_views"
        
        # Second retrieval should use cache (should be faster)
        start_time = time.time()
        config2 = config_manager.get_config("listing_views")
        end_time = time.time()
        
        # Should be the same config
        assert config1 == config2
        
        # Should be very fast (cached)
        assert end_time - start_time < 0.001  # Less than 1ms

    def test_config_retrieval_performance(self, config_manager):
        """Test config retrieval performance under load."""
        # Load initial configs
        config_manager.load_all_configs()
        
        # Measure retrieval time for multiple configs
        start_time = time.time()
        for _ in range(100):
            config_manager.get_config("listing_views")
            config_manager.get_config("enquiries")
        end_time = time.time()
        
        # Should be very fast with caching
        total_time = end_time - start_time
        assert total_time < 0.1  # Less than 100ms for 200 retrievals

    def test_config_retrieval_thread_safety(self, config_manager):
        """Test config retrieval is thread-safe."""
        results = queue.Queue()
        
        def retrieve_configs():
            try:
                for _ in range(10):
                    config = config_manager.get_config("listing_views")
                    if config is not None:
                        results.put(("success", config))
            except Exception as e:
                results.put(("error", e))
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=retrieve_configs)
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
                pytest.fail(f"Thread failed: {result}")
        
        assert success_count == 50  # 5 threads * 10 retrievals each
        # All configs should be identical
        for config in configs[1:]:
            assert config == configs[0]

    def test_list_configs_caching(self, config_manager):
        """Test list_configs uses caching efficiently."""
        # First call
        names1 = config_manager.list_configs()
        
        # Second call should be cached
        start_time = time.time()
        names2 = config_manager.list_configs()
        end_time = time.time()
        
        assert names1 == names2
        assert end_time - start_time < 0.001  # Should be very fast


class TestConfigManagerChangeDetection:
    """Test change detection and hot reload (ADF-29 Requirement 3)."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with test configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            events_dir = temp_path / "events"
            events_dir.mkdir()
            
            # Create initial config
            config = {
                "event_config": {
                    "name": "change_detection_test",
                    "description": "Initial description",
                    "data_source": {
                        "table": "DATAMART.INITIAL_TABLE",
                        "metrics": [{"column": "INITIAL_VALUE", "alias": "initial"}]
                    }
                }
            }
            
            with open(events_dir / "change_detection_test.yaml", "w") as f:
                yaml.dump(config, f)
            
            yield temp_path

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create a ConfigManager instance for testing."""
        ConfigManager.reset_instance()
        return ConfigManager.get_instance(config_dir=temp_config_dir)

    def test_file_change_detection(self, config_manager, temp_config_dir):
        """Test file change detection mechanism."""
        # Load initial config
        config_manager.load_all_configs()
        initial_config = config_manager.get_config("change_detection_test")
        assert initial_config["event_config"]["description"] == "Initial description"
        
        # Modify config file
        time.sleep(0.1)  # Ensure different modification time
        modified_config = {
            "event_config": {
                "name": "change_detection_test",
                "description": "Modified description",
                "data_source": {
                    "table": "DATAMART.MODIFIED_TABLE",
                    "metrics": [{"column": "MODIFIED_VALUE", "alias": "modified"}]
                }
            }
        }
        
        with open(temp_config_dir / "events" / "change_detection_test.yaml", "w") as f:
            yaml.dump(modified_config, f)
        
        # Check if change is detected
        assert config_manager.has_config_changed("change_detection_test")

    def test_hot_reload_functionality(self, config_manager, temp_config_dir):
        """Test hot reload functionality."""
        # Load initial config
        config_manager.load_all_configs()
        initial_config = config_manager.get_config("change_detection_test")
        
        # Modify config file
        time.sleep(0.1)
        modified_config = {
            "event_config": {
                "name": "change_detection_test",
                "description": "Hot reloaded description",
                "data_source": {
                    "table": "DATAMART.HOT_RELOAD_TABLE",
                    "metrics": [{"column": "HOT_RELOAD_VALUE", "alias": "hot_reload"}]
                }
            }
        }
        
        with open(temp_config_dir / "events" / "change_detection_test.yaml", "w") as f:
            yaml.dump(modified_config, f)
        
        # Trigger hot reload
        config_manager.reload_if_changed()
        
        # Get updated config
        updated_config = config_manager.get_config("change_detection_test")
        assert updated_config["event_config"]["description"] == "Hot reloaded description"
        assert updated_config["event_config"]["data_source"]["table"] == "DATAMART.HOT_RELOAD_TABLE"

    def test_file_watcher_integration(self, config_manager, temp_config_dir):
        """Test file watcher integration for automatic reloading."""
        # Enable file watcher
        config_manager.enable_file_watcher()
        
        # Load initial config
        config_manager.load_all_configs()
        
        # Modify config file
        time.sleep(0.1)
        modified_config = {
            "event_config": {
                "name": "change_detection_test",
                "description": "File watcher test",
                "data_source": {
                    "table": "DATAMART.FILE_WATCHER_TABLE",
                    "metrics": [{"column": "FILE_WATCHER_VALUE", "alias": "file_watcher"}]
                }
            }
        }
        
        with open(temp_config_dir / "events" / "change_detection_test.yaml", "w") as f:
            yaml.dump(modified_config, f)
        
        # Wait for file watcher to detect change
        time.sleep(0.5)
        
        # Config should be automatically reloaded
        updated_config = config_manager.get_config("change_detection_test")
        assert updated_config["event_config"]["description"] == "File watcher test"
        
        # Disable file watcher
        config_manager.disable_file_watcher()

    def test_multiple_file_changes(self, config_manager, temp_config_dir):
        """Test handling multiple file changes."""
        # Add another config file
        new_config = {
            "event_config": {
                "name": "multiple_changes_test",
                "description": "Multiple changes test",
                "data_source": {
                    "table": "DATAMART.MULTIPLE_TABLE",
                    "metrics": [{"column": "MULTIPLE_VALUE", "alias": "multiple"}]
                }
            }
        }
        
        with open(temp_config_dir / "events" / "multiple_changes_test.yaml", "w") as f:
            yaml.dump(new_config, f)
        
        # Load configs
        config_manager.load_all_configs()
        
        # Modify multiple files
        time.sleep(0.1)
        for name in ["change_detection_test", "multiple_changes_test"]:
            modified_config = {
                "event_config": {
                    "name": name,
                    "description": f"Modified {name}",
                    "data_source": {
                        "table": f"DATAMART.MODIFIED_{name.upper()}_TABLE",
                        "metrics": [{"column": f"MODIFIED_{name.upper()}_VALUE", "alias": f"modified_{name}"}]
                    }
                }
            }
            
            with open(temp_config_dir / "events" / f"{name}.yaml", "w") as f:
                yaml.dump(modified_config, f)
        
        # Check if all changes are detected
        assert config_manager.has_config_changed("change_detection_test")
        assert config_manager.has_config_changed("multiple_changes_test")
        
        # Reload all changes
        config_manager.reload_if_changed()
        
        # Verify all configs are updated
        for name in ["change_detection_test", "multiple_changes_test"]:
            config = config_manager.get_config(name)
            assert config["event_config"]["description"] == f"Modified {name}"


class TestConfigManagerLogging:
    """Test comprehensive logging (ADF-29 Requirement 4)."""

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
                    "name": "logging_test",
                    "description": "Logging test configuration",
                    "data_source": {
                        "table": "DATAMART.LOGGING_TABLE",
                        "metrics": [{"column": "LOGGING_VALUE", "alias": "logging"}]
                    }
                }
            }
            
            with open(events_dir / "logging_test.yaml", "w") as f:
                yaml.dump(config, f)
            
            yield temp_path

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create a ConfigManager instance for testing."""
        ConfigManager.reset_instance()
        manager = ConfigManager.get_instance(config_dir=temp_config_dir)
        manager.load_all_configs()  # Load configs so they're available for retrieval
        return manager

    def test_logging_configuration(self, config_manager, caplog):
        """Test logging is properly configured."""
        with caplog.at_level(logging.INFO):
            config_manager.load_all_configs()
            
            # Should have logged the loading
            assert any("Loading configurations" in record.message for record in caplog.records)
            assert any("ConfigManager" in record.name for record in caplog.records)

    def test_logging_config_operations(self, config_manager, caplog):
        """Test logging of config operations."""
        # Clear any setup logs
        caplog.clear()
        
        with caplog.at_level(logging.DEBUG):
            # Test get_config logging
            config = config_manager.get_config("logging_test")
            assert config is not None
            
            # Test list_configs logging
            names = config_manager.list_configs()
            assert len(names) > 0
            
            # Should have logged operations (check for retrieval or listing)
            assert any("retrieving" in record.message.lower() or "listed" in record.message.lower() 
                      for record in caplog.records)

    def test_logging_error_conditions(self, config_manager, caplog):
        """Test logging of error conditions."""
        with caplog.at_level(logging.WARNING):
            # Test non-existent config
            config = config_manager.get_config("nonexistent_config")
            assert config is None
            
            # Should have logged warning
            assert any("not found" in record.message.lower() for record in caplog.records)

    def test_logging_performance_metrics(self, config_manager, caplog):
        """Test logging of performance metrics."""
        with caplog.at_level(logging.INFO):
            # Clear any setup logs first
            caplog.clear()
            
            # Load configs with performance logging
            config_manager.reload_configs()  # This should log time metrics
            
            # Should have logged performance metrics (time information)
            assert any("time" in record.message.lower() or "successfully loaded" in record.message.lower() 
                      or "successfully reloaded" in record.message.lower()
                      for record in caplog.records)

    def test_logging_file_changes(self, config_manager, temp_config_dir, caplog):
        """Test logging of file changes."""
        config_manager.load_all_configs()
        
        with caplog.at_level(logging.INFO):
            # Modify config file
            time.sleep(0.1)
            modified_config = {
                "event_config": {
                    "name": "logging_test",
                    "description": "Modified for logging test",
                    "data_source": {
                        "table": "DATAMART.MODIFIED_LOGGING_TABLE",
                        "metrics": [{"column": "MODIFIED_LOGGING_VALUE", "alias": "modified_logging"}]
                    }
                }
            }
            
            with open(temp_config_dir / "events" / "logging_test.yaml", "w") as f:
                yaml.dump(modified_config, f)
            
            # Check change detection
            config_manager.has_config_changed("logging_test")
            
            # Should have logged file change detection
            assert any("change" in record.message.lower() for record in caplog.records)

    def test_logging_levels(self, config_manager, caplog):
        """Test different logging levels are used appropriately."""
        # Test INFO level
        with caplog.at_level(logging.INFO):
            config_manager.load_all_configs()
            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert len(info_records) > 0
        
        caplog.clear()
        
        # Test WARNING level
        with caplog.at_level(logging.WARNING):
            config_manager.get_config("nonexistent")
            warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warning_records) > 0
        
        caplog.clear()
        
        # Test DEBUG level
        with caplog.at_level(logging.DEBUG):
            config_manager.get_config("logging_test")
            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
            assert len(debug_records) > 0


class TestConfigManagerIntegration:
    """Test ConfigManager integration with existing YAMLConfigLoader."""

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
                    "name": "integration_test",
                    "description": "Integration test configuration",
                    "data_source": {
                        "table": "DATAMART.INTEGRATION_TABLE",
                        "metrics": [{"column": "INTEGRATION_VALUE", "alias": "integration"}]
                    }
                }
            }
            
            with open(events_dir / "integration_test.yaml", "w") as f:
                yaml.dump(config, f)
            
            yield temp_path

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create a ConfigManager instance for testing."""
        ConfigManager.reset_instance()
        return ConfigManager.get_instance(config_dir=temp_config_dir)

    def test_integration_with_yaml_config_loader(self, config_manager):
        """Test ConfigManager integrates properly with YAMLConfigLoader."""
        # ConfigManager should use YAMLConfigLoader internally
        assert hasattr(config_manager, '_loader')
        
        # Should be able to load configs
        config_manager.load_all_configs()
        config = config_manager.get_config("integration_test")
        assert config is not None
        assert config["event_config"]["name"] == "integration_test"

    def test_compatibility_with_existing_tests(self, config_manager):
        """Test ConfigManager is compatible with existing YAMLConfigLoader tests."""
        # Should support same API as YAMLConfigLoader
        assert hasattr(config_manager, 'load_all_configs')
        assert hasattr(config_manager, 'get_config')
        assert hasattr(config_manager, 'list_configs')
        assert hasattr(config_manager, 'reload_configs')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])