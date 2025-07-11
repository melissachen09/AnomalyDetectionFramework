"""
Plugin Manager for Anomaly Detection Framework.

This module implements dynamic plugin loading and execution system with automatic
plugin discovery, isolated execution environment, performance monitoring, and
graceful error handling as specified in ADF-37.

The PluginManager provides:
- Automatic plugin discovery from registry and file system
- Isolated execution environment for plugins
- Performance monitoring and health checks
- Graceful error handling and recovery
- Parallel plugin execution with thread safety
- Hot reload capability for dynamic plugin updates
"""

import importlib
import importlib.util
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable
import inspect

from .detectors.base_detector import (
    BaseDetector, 
    DetectionResult, 
    get_registered_detectors,
    create_detector
)


class PluginHealthStatus:
    """Health status tracking for plugins."""
    
    def __init__(self):
        self.status = "unknown"
        self.last_execution = None
        self.error_count = 0
        self.last_error = None
        self.execution_count = 0
        self.average_execution_time = 0.0
        self.lock = threading.Lock()
    
    def record_execution(self, execution_time: float, error: Optional[Exception] = None):
        """Record plugin execution metrics."""
        with self.lock:
            self.last_execution = datetime.now()
            self.execution_count += 1
            
            if error:
                self.error_count += 1
                self.last_error = str(error)
                self.status = "unhealthy" if self.error_count > 5 else "warning"
            else:
                # Update average execution time
                if self.execution_count == 1:
                    self.average_execution_time = execution_time
                else:
                    # Running average
                    alpha = 0.2  # Weight for new measurement
                    self.average_execution_time = (
                        alpha * execution_time + 
                        (1 - alpha) * self.average_execution_time
                    )
                
                # Update status based on error rate
                error_rate = self.error_count / self.execution_count
                if error_rate == 0:
                    self.status = "healthy"
                elif error_rate < 0.1:
                    self.status = "warning"
                else:
                    self.status = "unhealthy"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health status to dictionary."""
        with self.lock:
            return {
                "status": self.status,
                "last_execution": self.last_execution.isoformat() if self.last_execution else None,
                "error_count": self.error_count,
                "execution_count": self.execution_count,
                "average_execution_time": self.average_execution_time,
                "last_error": self.last_error
            }


class PluginManager:
    """
    Plugin manager for anomaly detection framework.
    
    Provides dynamic plugin loading and execution system with automatic plugin
    discovery, isolated execution environment, performance monitoring, and 
    graceful error handling.
    
    Features:
    - Automatic plugin discovery from registry and file system
    - Thread-safe plugin loading and execution
    - Parallel execution with configurable thread pool
    - Health monitoring and error tracking
    - Hot reload capability
    - Resource management and cleanup
    
    Example:
        manager = PluginManager()
        manager.discover_plugins()
        manager.load_plugin('threshold_detector')
        
        results = manager.execute_plugin(
            'threshold_detector',
            {'threshold': 100},
            date(2024, 1, 1),
            date(2024, 1, 31)
        )
    """
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None, max_workers: int = 4):
        """
        Initialize plugin manager.
        
        Args:
            plugin_dirs: Optional list of directories to scan for plugins
            max_workers: Maximum number of threads for parallel execution
        """
        self.plugin_dirs = plugin_dirs or []
        self.max_workers = max_workers
        
        # Plugin storage
        self._loaded_plugins: Dict[str, BaseDetector] = {}
        self._plugin_classes: Dict[str, Type[BaseDetector]] = {}
        self._plugin_health: Dict[str, PluginHealthStatus] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"PluginManager initialized with {max_workers} workers")
    
    def discover_plugins(self) -> Dict[str, Type[BaseDetector]]:
        """
        Discover and return available plugins.
        
        Discovers plugins from:
        1. Registered detector registry
        2. File system directories (if specified)
        
        Returns:
            Dictionary mapping plugin names to their classes
        """
        with self._lock:
            discovered_plugins = {}
            
            # Get plugins from registry
            try:
                registered_plugins = get_registered_detectors()
                discovered_plugins.update(registered_plugins)
                self.logger.debug(f"Found {len(registered_plugins)} registered plugins")
            except Exception as e:
                self.logger.error(f"Error discovering registered plugins: {e}")
            
            # Discover plugins from file system
            for plugin_dir in self.plugin_dirs:
                try:
                    file_plugins = self._discover_plugins_from_directory(plugin_dir)
                    discovered_plugins.update(file_plugins)
                    self.logger.debug(f"Found {len(file_plugins)} plugins in {plugin_dir}")
                except Exception as e:
                    self.logger.error(f"Error discovering plugins from {plugin_dir}: {e}")
            
            # Update internal cache
            self._plugin_classes.update(discovered_plugins)
            
            self.logger.info(f"Discovered {len(discovered_plugins)} total plugins")
            return discovered_plugins.copy()
    
    def _discover_plugins_from_directory(self, directory: str) -> Dict[str, Type[BaseDetector]]:
        """
        Discover plugins from a file system directory.
        
        Args:
            directory: Directory path to scan for Python files
            
        Returns:
            Dictionary of discovered plugin classes
        """
        plugins = {}
        directory_path = Path(directory)
        
        if not directory_path.exists() or not directory_path.is_dir():
            self.logger.warning(f"Plugin directory {directory} does not exist or is not a directory")
            return plugins
        
        # Find Python files
        python_files = list(directory_path.glob("*.py"))
        python_files.extend(directory_path.glob("**/*.py"))
        
        for python_file in python_files:
            # Skip __pycache__ and test files
            if "__pycache__" in str(python_file) or python_file.name.startswith("test_"):
                continue
                
            try:
                # Load module dynamically
                spec = importlib.util.spec_from_file_location(
                    python_file.stem, 
                    python_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find detector classes in module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BaseDetector) and 
                            obj is not BaseDetector and
                            hasattr(obj, '__module__')):
                            plugins[name.lower()] = obj
                            
            except Exception as e:
                self.logger.error(f"Error loading plugin file {python_file}: {e}")
        
        return plugins
    
    def load_plugin(self, name: str) -> bool:
        """
        Load a specific plugin by name.
        
        Args:
            name: Name of the plugin to load
            
        Returns:
            True if plugin loaded successfully, False otherwise
        """
        with self._lock:
            if name in self._loaded_plugins:
                self.logger.debug(f"Plugin {name} already loaded")
                return True
            
            # Find plugin class
            if name not in self._plugin_classes:
                # Try to discover plugins first
                self.discover_plugins()
                
            if name not in self._plugin_classes:
                self.logger.error(f"Plugin {name} not found in available plugins")
                return False
            
            try:
                # Create plugin instance with minimal config for loading
                plugin_class = self._plugin_classes[name]
                plugin_instance = plugin_class({})  # Minimal config for loading
                
                # Store plugin
                self._loaded_plugins[name] = plugin_instance
                self._plugin_health[name] = PluginHealthStatus()
                self._plugin_health[name].status = "healthy"  # Start as healthy
                
                self.logger.info(f"Successfully loaded plugin: {name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to load plugin {name}: {e}")
                self.logger.debug(f"Plugin loading error details: {traceback.format_exc()}")
                return False
    
    def unload_plugin(self, name: str) -> bool:
        """
        Unload a specific plugin by name.
        
        Args:
            name: Name of the plugin to unload
            
        Returns:
            True if plugin unloaded successfully, False otherwise
        """
        with self._lock:
            if name not in self._loaded_plugins:
                self.logger.debug(f"Plugin {name} is not loaded")
                return False
            
            try:
                plugin = self._loaded_plugins[name]
                
                # Call cleanup if available
                if hasattr(plugin, 'cleanup') and callable(getattr(plugin, 'cleanup')):
                    plugin.cleanup()
                
                # Remove from loaded plugins
                del self._loaded_plugins[name]
                if name in self._plugin_health:
                    del self._plugin_health[name]
                
                self.logger.info(f"Successfully unloaded plugin: {name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error unloading plugin {name}: {e}")
                return False
    
    def get_loaded_plugins(self) -> Dict[str, BaseDetector]:
        """
        Get all currently loaded plugins.
        
        Returns:
            Dictionary of loaded plugin instances
        """
        with self._lock:
            return self._loaded_plugins.copy()
    
    def execute_plugin(self, name: str, config: Dict[str, Any], 
                      start_date: date, end_date: date) -> List[DetectionResult]:
        """
        Execute a plugin with given configuration.
        
        Args:
            name: Name of the plugin to execute
            config: Configuration dictionary for the plugin
            start_date: Start date for detection
            end_date: End date for detection
            
        Returns:
            List of detection results
            
        Raises:
            ValueError: If plugin is not loaded
        """
        with self._lock:
            if name not in self._loaded_plugins:
                raise ValueError(f"Plugin '{name}' is not loaded")
            
            plugin = self._loaded_plugins[name]
        
        # Execute plugin outside of lock to allow concurrent execution
        start_time = time.time()
        error = None
        results = []
        
        try:
            # Create new instance with actual config for execution
            plugin_class = type(plugin)
            execution_plugin = plugin_class(config)
            
            # Execute detection
            results = execution_plugin.detect(start_date, end_date)
            
            self.logger.debug(f"Plugin {name} executed successfully, found {len(results)} results")
            
        except Exception as e:
            error = e
            self.logger.error(f"Plugin {name} execution failed: {e}")
            raise
            
        finally:
            execution_time = time.time() - start_time
            
            # Update health status
            with self._lock:
                if name in self._plugin_health:
                    self._plugin_health[name].record_execution(execution_time, error)
        
        return results
    
    def execute_plugins_parallel(self, plugin_configs: List[Dict[str, Any]], 
                                start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Execute multiple plugins in parallel.
        
        Args:
            plugin_configs: List of plugin configuration dictionaries.
                           Each should have 'plugin_name' and 'config' keys.
            start_date: Start date for detection
            end_date: End date for detection
            
        Returns:
            Dictionary mapping plugin names to their results or errors
        """
        results = {}
        
        # Submit all plugin executions to thread pool
        future_to_plugin = {}
        
        for plugin_config in plugin_configs:
            plugin_name = plugin_config.get('plugin_name')
            config = plugin_config.get('config', {})
            
            if not plugin_name:
                self.logger.error("Plugin config missing 'plugin_name' field")
                continue
                
            future = self._executor.submit(
                self._execute_plugin_safe,
                plugin_name, config, start_date, end_date
            )
            future_to_plugin[future] = plugin_name
        
        # Collect results as they complete
        for future in as_completed(future_to_plugin, timeout=300):  # 5 minute timeout
            plugin_name = future_to_plugin[future]
            try:
                plugin_results = future.result()
                results[plugin_name] = plugin_results
            except Exception as e:
                self.logger.error(f"Plugin {plugin_name} failed in parallel execution: {e}")
                results[plugin_name] = e
        
        return results
    
    def _execute_plugin_safe(self, name: str, config: Dict[str, Any],
                           start_date: date, end_date: date) -> Any:
        """
        Safe plugin execution wrapper for parallel execution.
        
        Args:
            name: Plugin name
            config: Plugin configuration
            start_date: Start date for detection
            end_date: End date for detection
            
        Returns:
            Detection results on success, Exception object on error
        """
        try:
            return self.execute_plugin(name, config, start_date, end_date)
        except Exception as e:
            self.logger.error(f"Safe execution failed for plugin {name}: {e}")
            # Record error in health status
            with self._lock:
                if name in self._plugin_health:
                    self._plugin_health[name].record_execution(0.0, e)
            return e  # Return the exception instead of empty list
    
    def reload_plugins(self) -> bool:
        """
        Reload all plugins (hot reload).
        
        Discovers new plugins and reloads existing loaded plugins.
        
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            with self._lock:
                # Store current loaded plugin names
                currently_loaded = list(self._loaded_plugins.keys())
                
                # Discover plugins again to find new ones
                newly_discovered = self.discover_plugins()
                
                # Load any new plugins that were discovered
                for plugin_name, plugin_class in newly_discovered.items():
                    if plugin_name not in self._loaded_plugins:
                        # Try to load new plugin
                        self.load_plugin(plugin_name)
                
                # Reload all previously loaded plugins
                for plugin_name in currently_loaded:
                    if plugin_name in self._plugin_classes:
                        # Unload and reload
                        self.unload_plugin(plugin_name)
                        if not self.load_plugin(plugin_name):
                            self.logger.error(f"Failed to reload plugin {plugin_name}")
                            return False
                
                self.logger.info("Plugin hot reload completed successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Plugin reload failed: {e}")
            return False
    
    def get_plugin_health(self, name: str) -> Dict[str, Any]:
        """
        Get health status of a plugin.
        
        Args:
            name: Name of the plugin
            
        Returns:
            Dictionary containing health information
        """
        with self._lock:
            if name not in self._plugin_health:
                return {"status": "not_found"}
            
            return self._plugin_health[name].to_dict()
    
    def shutdown(self):
        """Shutdown the plugin manager and clean up resources."""
        try:
            with self._lock:
                # Unload all plugins
                plugin_names = list(self._loaded_plugins.keys())
                for name in plugin_names:
                    self.unload_plugin(name)
                
                # Shutdown thread pool
                self._executor.shutdown(wait=True)
                
                self.logger.info("PluginManager shutdown completed")
                
        except Exception as e:
            self.logger.error(f"Error during PluginManager shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
    
    def __repr__(self) -> str:
        """String representation of the plugin manager."""
        with self._lock:
            loaded_count = len(self._loaded_plugins)
            available_count = len(self._plugin_classes)
            return (f"PluginManager(loaded={loaded_count}, "
                   f"available={available_count}, workers={self.max_workers})")