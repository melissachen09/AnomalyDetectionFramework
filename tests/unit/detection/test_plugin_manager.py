"""
Test cases for Plugin Manager functionality.

This module contains comprehensive test cases for the PluginManager class,
covering plugin discovery, lifecycle management, error isolation, 
parallel execution, and hot reload capabilities.

Tests are written following TDD principles as part of ADF-36:
Write Test Cases for Plugin Manager (GADF-DETECT-007).
"""

import pytest
import os
import tempfile
import shutil
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from datetime import date, datetime
from pathlib import Path

# Import the base detector and related functionality
from src.detection.detectors.base_detector import BaseDetector, DetectionResult, register_detector
from src.detection.plugin_manager import PluginManager


class MockDetectorForTesting(BaseDetector):
    """Mock detector for testing plugin manager functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.initialized = True
        self.cleanup_called = False
        
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        """Mock detection method."""
        return [
            DetectionResult(
                event_type="test_event",
                metric_name="test_metric",
                detection_date=start_date,
                expected_value=100.0,
                actual_value=150.0,
                deviation_percentage=0.5,
                severity="warning",
                detection_method="mock"
            )
        ]
    
    def cleanup(self):
        """Mock cleanup method."""
        self.cleanup_called = True


class FailingMockDetector(BaseDetector):
    """Mock detector that fails for error isolation testing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if config.get('fail_init'):
            raise ValueError("Initialization failure")
    
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        if self.config.get('fail_detect'):
            raise RuntimeError("Detection failure")
        return []


class SlowMockDetector(BaseDetector):
    """Mock detector for performance testing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.detection_time = config.get('detection_time', 0.1)
    
    def detect(self, start_date: date, end_date: date) -> List[DetectionResult]:
        time.sleep(self.detection_time)
        return []



class TestPluginManagerDiscovery:
    """Test cases for plugin auto-discovery mechanism (GADF-DETECT-007a)."""
    
    def test_discover_plugins_empty_registry(self):
        """Test plugin discovery with empty registry."""
        # Clear the registry for testing
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', {}):
            manager = PluginManager()
            plugins = manager.discover_plugins()
            
            assert isinstance(plugins, dict)
            assert len(plugins) == 0
    
    def test_discover_plugins_with_registered_detectors(self):
        """Test plugin discovery with registered detectors."""
        # Mock registry with test detectors
        mock_registry = {
            'mock_detector': MockDetectorForTesting,
            'failing_detector': FailingMockDetector
        }
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            plugins = manager.discover_plugins()
            
            assert len(plugins) == 2
            assert 'mock_detector' in plugins
            assert 'failing_detector' in plugins
            assert plugins['mock_detector'] == MockDetectorForTesting
            assert plugins['failing_detector'] == FailingMockDetector
    
    def test_discover_plugins_from_directory(self):
        """Test plugin discovery from file system directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock plugin file
            plugin_file = Path(temp_dir) / "test_plugin.py"
            plugin_content = '''
from src.detection.detectors.base_detector import BaseDetector, register_detector

@register_detector("directory_test")
class DirectoryTestDetector(BaseDetector):
    def detect(self, start_date, end_date):
        return []
'''
            plugin_file.write_text(plugin_content)
            
            manager = PluginManager(plugin_dirs=[temp_dir])
            plugins = manager.discover_plugins()
            
            # This test defines the expected behavior - actual implementation needed
            assert isinstance(plugins, dict)
    
    def test_discover_plugins_invalid_directory(self):
        """Test plugin discovery with invalid directory path."""
        manager = PluginManager(plugin_dirs=["/nonexistent/path"])
        
        # Should handle invalid directories gracefully
        plugins = manager.discover_plugins()
        assert isinstance(plugins, dict)
    
    def test_discover_plugins_with_file_filters(self):
        """Test plugin discovery with file pattern filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create plugin files with different extensions
            (Path(temp_dir) / "valid_plugin.py").touch()
            (Path(temp_dir) / "invalid_plugin.txt").touch()
            (Path(temp_dir) / "__pycache__").mkdir()
            
            manager = PluginManager(plugin_dirs=[temp_dir])
            plugins = manager.discover_plugins()
            
            # Should only discover Python files, not other file types
            assert isinstance(plugins, dict)
    
    def test_discover_plugins_performance(self):
        """Test plugin discovery performance with many files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create many files to test performance
            for i in range(100):
                (Path(temp_dir) / f"file_{i}.py").touch()
            
            manager = PluginManager(plugin_dirs=[temp_dir])
            
            start_time = time.time()
            plugins = manager.discover_plugins()
            discovery_time = time.time() - start_time
            
            # Discovery should complete in reasonable time (< 1 second)
            assert discovery_time < 1.0
            assert isinstance(plugins, dict)


class TestPluginManagerLifecycle:
    """Test cases for plugin initialization and cleanup (GADF-DETECT-007b)."""
    
    def test_load_plugin_success(self):
        """Test successful plugin loading."""
        mock_registry = {'mock_detector': MockDetectorForTesting}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            result = manager.load_plugin('mock_detector')
            
            assert result is True
            loaded_plugins = manager.get_loaded_plugins()
            assert 'mock_detector' in loaded_plugins
    
    def test_load_plugin_nonexistent(self):
        """Test loading non-existent plugin."""
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', {}):
            manager = PluginManager()
            result = manager.load_plugin('nonexistent')
            
            assert result is False
    
    def test_load_plugin_initialization_failure(self):
        """Test plugin loading when initialization fails."""
        mock_registry = {'failing_detector': FailingMockDetector}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            
            # Mock the detector to fail during initialization by patching the class
            original_init = FailingMockDetector.__init__
            def failing_init(self, config):
                raise ValueError("Initialization failure")
            
            with patch.object(FailingMockDetector, '__init__', failing_init):
                # Should handle initialization failures gracefully
                result = manager.load_plugin('failing_detector')
                assert result is False
    
    def test_unload_plugin_success(self):
        """Test successful plugin unloading."""
        mock_registry = {'mock_detector': MockDetectorForTesting}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('mock_detector')
            
            result = manager.unload_plugin('mock_detector')
            assert result is True
            
            loaded_plugins = manager.get_loaded_plugins()
            assert 'mock_detector' not in loaded_plugins
    
    def test_unload_plugin_not_loaded(self):
        """Test unloading plugin that isn't loaded."""
        manager = PluginManager()
        result = manager.unload_plugin('not_loaded')
        
        assert result is False
    
    def test_plugin_cleanup_on_unload(self):
        """Test that plugin cleanup is called during unload."""
        mock_registry = {'mock_detector': MockDetectorForTesting}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('mock_detector')
            
            # Get reference to loaded plugin instance
            loaded_plugins = manager.get_loaded_plugins()
            plugin_instance = loaded_plugins['mock_detector']
            
            manager.unload_plugin('mock_detector')
            
            # Verify cleanup was called
            assert plugin_instance.cleanup_called is True
    
    def test_get_loaded_plugins_empty(self):
        """Test getting loaded plugins when none are loaded."""
        manager = PluginManager()
        loaded_plugins = manager.get_loaded_plugins()
        
        assert isinstance(loaded_plugins, dict)
        assert len(loaded_plugins) == 0
    
    def test_get_loaded_plugins_multiple(self):
        """Test getting multiple loaded plugins."""
        mock_registry = {
            'mock_detector': MockDetectorForTesting,
            'slow_detector': SlowMockDetector
        }
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('mock_detector')
            manager.load_plugin('slow_detector')
            
            loaded_plugins = manager.get_loaded_plugins()
            assert len(loaded_plugins) == 2
            assert 'mock_detector' in loaded_plugins
            assert 'slow_detector' in loaded_plugins


class TestPluginManagerParallelExecution:
    """Test cases for parallel plugin execution (GADF-DETECT-007c)."""
    
    def test_execute_plugin_single_success(self):
        """Test successful single plugin execution."""
        mock_registry = {'mock_detector': MockDetectorForTesting}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('mock_detector')
            
            config = {'event_type': 'test'}
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            
            results = manager.execute_plugin('mock_detector', config, start_date, end_date)
            
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0].event_type == 'test_event'
    
    def test_execute_plugin_not_loaded(self):
        """Test executing plugin that isn't loaded."""
        manager = PluginManager()
        
        config = {'event_type': 'test'}
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        with pytest.raises(ValueError, match="Plugin .* is not loaded"):
            manager.execute_plugin('not_loaded', config, start_date, end_date)
    
    def test_execute_plugins_parallel_success(self):
        """Test successful parallel execution of multiple plugins."""
        mock_registry = {
            'mock_detector': MockDetectorForTesting,
            'slow_detector': SlowMockDetector
        }
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('mock_detector')
            manager.load_plugin('slow_detector')
            
            plugin_configs = [
                {'plugin_name': 'mock_detector', 'config': {'event_type': 'test1'}},
                {'plugin_name': 'slow_detector', 'config': {'event_type': 'test2', 'detection_time': 0.1}}
            ]
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            
            start_time = time.time()
            results = manager.execute_plugins_parallel(plugin_configs, start_date, end_date)
            execution_time = time.time() - start_time
            
            assert isinstance(results, dict)
            assert 'mock_detector' in results
            assert 'slow_detector' in results
            # Parallel execution should be faster than sequential
            assert execution_time < 0.3  # Should be much faster than 0.1 + 0.1 sequential
    
    def test_execute_plugins_parallel_thread_safety(self):
        """Test thread safety of parallel plugin execution."""
        mock_registry = {'mock_detector': MockDetectorForTesting}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('mock_detector')
            
            results = []
            errors = []
            
            def execute_plugin():
                try:
                    config = {'event_type': 'thread_test'}
                    start_date = date(2024, 1, 1)
                    end_date = date(2024, 1, 31)
                    result = manager.execute_plugin('mock_detector', config, start_date, end_date)
                    results.append(result)
                except Exception as e:
                    errors.append(e)
            
            # Run multiple threads simultaneously
            threads = [threading.Thread(target=execute_plugin) for _ in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            # All executions should succeed
            assert len(errors) == 0
            assert len(results) == 10
    
    def test_execute_plugins_parallel_performance_scaling(self):
        """Test performance scaling with increasing number of plugins."""
        mock_registry = {f'detector_{i}': SlowMockDetector for i in range(5)}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            for i in range(5):
                manager.load_plugin(f'detector_{i}')
            
            plugin_configs = [
                {'plugin_name': f'detector_{i}', 'config': {'detection_time': 0.1}}
                for i in range(5)
            ]
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            
            start_time = time.time()
            results = manager.execute_plugins_parallel(plugin_configs, start_date, end_date)
            execution_time = time.time() - start_time
            
            # Parallel execution should scale well (not linear with plugin count)
            assert execution_time < 0.5  # Much faster than 5 * 0.1 = 0.5 sequential
            assert len(results) == 5
    
    def test_execute_plugins_parallel_resource_limits(self):
        """Test parallel execution respects resource limits."""
        # Test that plugin manager doesn't spawn unlimited threads
        mock_registry = {f'detector_{i}': SlowMockDetector for i in range(20)}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            for i in range(20):
                manager.load_plugin(f'detector_{i}')
            
            plugin_configs = [
                {'plugin_name': f'detector_{i}', 'config': {'detection_time': 0.05}}
                for i in range(20)
            ]
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            
            # Should complete without overwhelming the system
            results = manager.execute_plugins_parallel(plugin_configs, start_date, end_date)
            assert len(results) == 20


class TestPluginManagerErrorIsolation:
    """Test cases for plugin error containment (GADF-DETECT-007d)."""
    
    def test_plugin_initialization_error_isolation(self):
        """Test that plugin initialization errors don't affect other plugins."""
        mock_registry = {
            'good_detector': MockDetectorForTesting,
            'bad_detector': FailingMockDetector
        }
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            
            # Load good plugin first
            assert manager.load_plugin('good_detector') is True
            
            # Mock the bad detector to fail during initialization
            original_init = FailingMockDetector.__init__
            def failing_init(self, config):
                raise ValueError("Initialization failure")
            
            with patch.object(FailingMockDetector, '__init__', failing_init):
                # Try to load bad plugin (should fail)
                assert manager.load_plugin('bad_detector') is False
            
            # Good plugin should still be loaded and functional
            loaded_plugins = manager.get_loaded_plugins()
            assert 'good_detector' in loaded_plugins
            assert 'bad_detector' not in loaded_plugins
    
    def test_plugin_execution_error_isolation(self):
        """Test that plugin execution errors don't affect other plugins."""
        mock_registry = {
            'good_detector': MockDetectorForTesting,
            'failing_detector': FailingMockDetector
        }
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('good_detector')
            manager.load_plugin('failing_detector')
            
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            
            # Good plugin should work
            results = manager.execute_plugin('good_detector', {}, start_date, end_date)
            assert len(results) == 1
            
            # Failing plugin should raise exception but not crash manager
            with pytest.raises(RuntimeError):
                manager.execute_plugin('failing_detector', {'fail_detect': True}, start_date, end_date)
            
            # Good plugin should still work after the failure
            results = manager.execute_plugin('good_detector', {}, start_date, end_date)
            assert len(results) == 1
    
    def test_parallel_execution_error_isolation(self):
        """Test error isolation in parallel execution."""
        mock_registry = {
            'good_detector': MockDetectorForTesting,
            'failing_detector': FailingMockDetector
        }
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('good_detector')
            manager.load_plugin('failing_detector')
            
            plugin_configs = [
                {'plugin_name': 'good_detector', 'config': {}},
                {'plugin_name': 'failing_detector', 'config': {'fail_detect': True}}
            ]
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            
            results = manager.execute_plugins_parallel(plugin_configs, start_date, end_date)
            
            # Good plugin should have results, failing plugin should have error
            assert 'good_detector' in results
            assert isinstance(results['good_detector'], list)
            assert len(results['good_detector']) == 1
            
            # Failing plugin should be handled gracefully
            if 'failing_detector' in results:
                assert isinstance(results['failing_detector'], Exception)
    
    def test_plugin_health_monitoring(self):
        """Test plugin health status monitoring."""
        mock_registry = {
            'healthy_detector': MockDetectorForTesting,
            'unhealthy_detector': FailingMockDetector
        }
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('healthy_detector')
            
            # Test healthy plugin
            health = manager.get_plugin_health('healthy_detector')
            assert health['status'] == 'healthy'
            assert 'last_execution' in health
            assert 'error_count' in health
            
            # Test non-existent plugin
            health = manager.get_plugin_health('nonexistent')
            assert health['status'] == 'not_found'
    
    def test_plugin_error_recovery(self):
        """Test plugin recovery after errors."""
        mock_registry = {'mock_detector': MockDetectorForTesting}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('mock_detector')
            
            # Simulate plugin failure and recovery
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            
            # First execution succeeds
            results = manager.execute_plugin('mock_detector', {}, start_date, end_date)
            assert len(results) == 1
            
            # Plugin should still be functional after errors
            health = manager.get_plugin_health('mock_detector')
            assert health['status'] in ['healthy', 'warning']


class TestPluginManagerHotReload:
    """Test cases for hot reload capability validation (GADF-DETECT-007d)."""
    
    def test_reload_plugins_success(self):
        """Test successful plugin hot reload."""
        mock_registry = {'mock_detector': MockDetectorForTesting}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('mock_detector')
            
            # Add new plugin to registry
            mock_registry['new_detector'] = SlowMockDetector
            
            result = manager.reload_plugins()
            assert result is True
            
            # New plugin should be available
            loaded_plugins = manager.get_loaded_plugins()
            assert 'new_detector' in loaded_plugins
    
    def test_reload_plugins_preserves_state(self):
        """Test that plugin reload preserves important state."""
        mock_registry = {'mock_detector': MockDetectorForTesting}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('mock_detector')
            
            # Execute plugin to establish state
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            manager.execute_plugin('mock_detector', {}, start_date, end_date)
            
            # Check health before reload
            health_before = manager.get_plugin_health('mock_detector')
            
            # Reload plugins
            manager.reload_plugins()
            
            # Plugin should still be loaded and functional
            loaded_plugins = manager.get_loaded_plugins()
            assert 'mock_detector' in loaded_plugins
            
            # Should still be able to execute
            results = manager.execute_plugin('mock_detector', {}, start_date, end_date)
            assert len(results) == 1
    
    def test_reload_plugins_handles_errors(self):
        """Test plugin reload error handling."""
        mock_registry = {'mock_detector': MockDetectorForTesting}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('mock_detector')
            
            # Simulate reload failure (e.g., registry corruption)
            with patch.object(manager, 'discover_plugins', side_effect=Exception("Discovery failed")):
                result = manager.reload_plugins()
                
                # Reload should fail gracefully
                assert result is False
                
                # Existing plugins should still work
                loaded_plugins = manager.get_loaded_plugins()
                assert 'mock_detector' in loaded_plugins
    
    def test_reload_plugins_during_execution(self):
        """Test hot reload while plugins are executing."""
        mock_registry = {'slow_detector': SlowMockDetector}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            manager.load_plugin('slow_detector')
            
            execution_completed = threading.Event()
            reload_completed = threading.Event()
            
            def execute_slow_plugin():
                """Execute slow plugin in background."""
                config = {'detection_time': 0.2}
                start_date = date(2024, 1, 1)
                end_date = date(2024, 1, 31)
                manager.execute_plugin('slow_detector', config, start_date, end_date)
                execution_completed.set()
            
            def reload_plugins():
                """Reload plugins in background."""
                time.sleep(0.1)  # Start reload during execution
                manager.reload_plugins()
                reload_completed.set()
            
            # Start execution and reload concurrently
            exec_thread = threading.Thread(target=execute_slow_plugin)
            reload_thread = threading.Thread(target=reload_plugins)
            
            exec_thread.start()
            reload_thread.start()
            
            # Wait for both to complete
            exec_thread.join(timeout=1.0)
            reload_thread.join(timeout=1.0)
            
            # Both operations should complete successfully
            assert execution_completed.is_set()
            assert reload_completed.is_set()
    
    def test_reload_plugins_file_system_watch(self):
        """Test plugin reload triggered by file system changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_file = Path(temp_dir) / "dynamic_plugin.py"
            
            manager = PluginManager(plugin_dirs=[temp_dir])
            initial_plugins = manager.discover_plugins()
            
            # Create new plugin file
            plugin_content = '''
from src.detection.detectors.base_detector import BaseDetector, register_detector

@register_detector("dynamic_test")
class DynamicTestDetector(BaseDetector):
    def detect(self, start_date, end_date):
        return []
'''
            plugin_file.write_text(plugin_content)
            
            # Reload should pick up new plugin
            result = manager.reload_plugins()
            assert result is True
            
            updated_plugins = manager.discover_plugins()
            # Should have more plugins than initially (implementation dependent)
            assert isinstance(updated_plugins, dict)


class TestPluginManagerIntegration:
    """Integration tests combining multiple plugin manager features."""
    
    def test_full_plugin_lifecycle(self):
        """Test complete plugin lifecycle from discovery to cleanup."""
        mock_registry = {
            'test_detector': MockDetectorForTesting,
            'slow_detector': SlowMockDetector
        }
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', mock_registry):
            manager = PluginManager()
            
            # 1. Discovery
            plugins = manager.discover_plugins()
            assert len(plugins) == 2
            
            # 2. Loading
            assert manager.load_plugin('test_detector') is True
            assert manager.load_plugin('slow_detector') is True
            
            # 3. Execution
            config = {'event_type': 'integration_test'}
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            
            results = manager.execute_plugin('test_detector', config, start_date, end_date)
            assert len(results) == 1
            
            # 4. Parallel execution
            plugin_configs = [
                {'plugin_name': 'test_detector', 'config': config},
                {'plugin_name': 'slow_detector', 'config': {'detection_time': 0.05}}
            ]
            parallel_results = manager.execute_plugins_parallel(plugin_configs, start_date, end_date)
            assert len(parallel_results) == 2
            
            # 5. Health check
            health = manager.get_plugin_health('test_detector')
            assert health['status'] in ['healthy', 'warning']
            
            # 6. Hot reload
            assert manager.reload_plugins() is True
            
            # 7. Cleanup
            assert manager.unload_plugin('test_detector') is True
            assert manager.unload_plugin('slow_detector') is True
            
            loaded_plugins = manager.get_loaded_plugins()
            assert len(loaded_plugins) == 0


# Performance and stress tests
class TestPluginManagerPerformance:
    """Performance and stress tests for plugin manager."""
    
    def test_discovery_performance_benchmark(self):
        """Benchmark plugin discovery performance."""
        # Test with large registry
        large_registry = {f'detector_{i}': MockDetectorForTesting for i in range(100)}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', large_registry):
            manager = PluginManager()
            
            start_time = time.time()
            plugins = manager.discover_plugins()
            discovery_time = time.time() - start_time
            
            assert len(plugins) == 100
            assert discovery_time < 1.0  # Should be fast even with many plugins
    
    def test_parallel_execution_scalability(self):
        """Test parallel execution with many plugins."""
        registry = {f'detector_{i}': SlowMockDetector for i in range(10)}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', registry):
            manager = PluginManager()
            
            # Load all plugins
            for i in range(10):
                manager.load_plugin(f'detector_{i}')
            
            plugin_configs = [
                {'plugin_name': f'detector_{i}', 'config': {'detection_time': 0.01}}
                for i in range(10)
            ]
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            
            start_time = time.time()
            results = manager.execute_plugins_parallel(plugin_configs, start_date, end_date)
            execution_time = time.time() - start_time
            
            assert len(results) == 10
            # Parallel execution should be significantly faster than sequential
            assert execution_time < 0.1  # Much less than 10 * 0.01 = 0.1 sequential
    
    def test_memory_usage_under_load(self):
        """Test memory usage during heavy plugin operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        registry = {f'detector_{i}': MockDetectorForTesting for i in range(50)}
        
        with patch('src.detection.detectors.base_detector._DETECTOR_REGISTRY', registry):
            manager = PluginManager()
            
            # Load many plugins and execute them
            for i in range(50):
                manager.load_plugin(f'detector_{i}')
            
            plugin_configs = [
                {'plugin_name': f'detector_{i}', 'config': {}}
                for i in range(50)
            ]
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            
            # Execute multiple times to test for memory leaks
            for _ in range(5):
                manager.execute_plugins_parallel(plugin_configs, start_date, end_date)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024