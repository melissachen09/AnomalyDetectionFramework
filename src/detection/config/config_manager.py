"""
Configuration Manager for Anomaly Detection Framework.

This module provides a singleton ConfigManager that builds upon YAMLConfigLoader
to add thread safety, file watching, change detection, and comprehensive logging.

Features:
- Singleton pattern with thread safety
- Efficient config retrieval with caching
- Change detection and hot reload
- File watcher for automatic reloading
- Comprehensive logging and performance metrics
"""

import os
import time
import threading
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .yaml_config_loader import YAMLConfigLoader
from .exceptions import ConfigurationError, ValidationError


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration file changes."""
    
    def __init__(self, config_manager):
        """Initialize the file handler."""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml')):
            self.logger.info(f"Configuration file modified: {event.src_path}")
            self.config_manager._mark_file_changed(event.src_path)
            # Automatically reload if watcher is enabled
            self.config_manager.reload_if_changed()
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml')):
            self.logger.info(f"Configuration file created: {event.src_path}")
            self.config_manager._mark_file_changed(event.src_path)
            # Automatically reload if watcher is enabled
            self.config_manager.reload_if_changed()
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml')):
            self.logger.info(f"Configuration file deleted: {event.src_path}")
            self.config_manager._mark_file_changed(event.src_path)
            # Automatically reload if watcher is enabled
            self.config_manager.reload_if_changed()


class ConfigManager:
    """
    Singleton Configuration Manager for Anomaly Detection Framework.
    
    This class provides a centralized, thread-safe configuration management
    system with hot reload capabilities, change detection, and comprehensive
    logging. It builds upon YAMLConfigLoader to add enterprise-grade features.
    
    Features:
    - Singleton pattern with thread safety
    - Efficient config retrieval with caching
    - File change detection and automatic hot reload
    - File watcher for real-time updates
    - Comprehensive logging and performance metrics
    - Migration and validation tools
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __init__(self, config_dir: Union[str, Path]):
        """
        Initialize ConfigManager.
        
        This should not be called directly. Use get_instance() instead.
        
        Args:
            config_dir: Path to configuration directory
            
        Raises:
            ConfigurationError: If config_dir is invalid
        """
        # Only check if called directly (not from get_instance)
        import inspect
        frame = inspect.currentframe()
        caller = frame.f_back.f_code.co_name if frame.f_back else None
        
        if caller != 'get_instance' and ConfigManager._instance is not None:
            raise RuntimeError("Use get_instance() to get ConfigManager instance")
        
        self.config_dir = Path(config_dir)
        self.logger = self._setup_logging()
        
        # Initialize YAMLConfigLoader
        self._loader = YAMLConfigLoader(
            config_dir=config_dir,
            cache_enabled=True,
            cache_size=200  # Larger cache for better performance
        )
        
        # File change tracking
        self._file_mtimes = {}
        self._changed_files = set()
        self._change_lock = threading.RLock()
        
        # File watcher
        self._observer = None
        self._file_handler = None
        self._watcher_enabled = False
        
        # Performance metrics
        self._metrics = {
            'config_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'reload_count': 0,
            'total_load_time': 0.0
        }
        self._metrics_lock = threading.RLock()
        
        self.logger.info(f"ConfigManager initialized with config directory: {self.config_dir}")
    
    @classmethod
    def get_instance(cls, config_dir: Union[str, Path] = None) -> 'ConfigManager':
        """
        Get singleton instance of ConfigManager.
        
        Args:
            config_dir: Path to configuration directory (only used for first call)
            
        Returns:
            ConfigManager singleton instance
            
        Raises:
            ConfigurationError: If config_dir not provided on first call
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if config_dir is None:
                        raise ConfigurationError("config_dir must be provided for first initialization")
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__(config_dir)
        else:
            # Log warning if trying to change config_dir
            if config_dir is not None and hasattr(cls._instance, 'config_dir') and Path(config_dir) != cls._instance.config_dir:
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Attempted to change config_dir from {cls._instance.config_dir} "
                    f"to {config_dir}. Ignoring - using existing directory."
                )
        
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """
        Reset singleton instance for testing.
        
        This should only be used in tests.
        """
        with cls._lock:
            if cls._instance is not None:
                # Cleanup file watcher if enabled
                if hasattr(cls._instance, '_observer') and cls._instance._observer is not None:
                    cls._instance.disable_file_watcher()
            cls._instance = None
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup comprehensive logging for ConfigManager.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}.ConfigManager")
        
        # Set up formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Only add handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            # Allow DEBUG level for comprehensive testing
            logger.setLevel(logging.DEBUG)
        
        return logger
    
    def load_all_configs(self, fail_on_error: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files with performance tracking.
        
        Args:
            fail_on_error: Whether to fail on first error or collect all errors
            
        Returns:
            Dictionary of configuration name -> configuration data
            
        Raises:
            ValidationError: If validation fails and fail_on_error is True
        """
        start_time = time.time()
        self.logger.info("Loading configurations from directory")
        
        try:
            configs = self._loader.load_all_configs(fail_on_error=fail_on_error)
            
            # Update file modification times
            self._update_file_mtimes()
            
            # Update metrics
            with self._metrics_lock:
                self._metrics['config_loads'] += 1
                self._metrics['total_load_time'] += time.time() - start_time
            
            self.logger.info(f"Successfully loaded {len(configs)} configurations in {time.time() - start_time:.3f}s")
            return configs
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
            raise
    
    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration by name with caching and performance tracking.
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration dictionary or None if not found
        """
        start_time = time.time()
        self.logger.debug(f"Retrieving config '{name}'")
        
        try:
            config = self._loader.get_config(name)
            
            # Update metrics
            with self._metrics_lock:
                if config is not None:
                    self._metrics['cache_hits'] += 1
                    self.logger.debug(f"Config '{name}' retrieved from cache in {time.time() - start_time:.3f}s")
                else:
                    self._metrics['cache_misses'] += 1
                    self.logger.warning(f"Config '{name}' not found")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error retrieving config '{name}': {e}")
            raise
    
    def list_configs(self) -> List[str]:
        """
        List all available configuration names.
        
        Returns:
            List of configuration names
        """
        try:
            names = self._loader.list_config_names()
            self.logger.debug(f"Listed {len(names)} configurations")
            return names
        except Exception as e:
            self.logger.error(f"Error listing configurations: {e}")
            raise
    
    def reload_configs(self) -> None:
        """
        Reload all configurations from disk.
        
        This clears the cache and reloads all configurations.
        """
        start_time = time.time()
        self.logger.info("Reloading all configurations")
        
        try:
            self._loader.reload_configs()
            self._update_file_mtimes()
            
            # Clear changed files
            with self._change_lock:
                self._changed_files.clear()
            
            # Update metrics
            with self._metrics_lock:
                self._metrics['reload_count'] += 1
            
            self.logger.info(f"Successfully reloaded configurations in {time.time() - start_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to reload configurations: {e}")
            raise
    
    def has_config_changed(self, name: str) -> bool:
        """
        Check if a specific configuration has changed.
        
        Args:
            name: Configuration name
            
        Returns:
            True if configuration has changed, False otherwise
        """
        try:
            # Get the file path for this config
            config_files = self._loader.scan_directory(
                self.config_dir,
                pattern="*.{yaml,yml}",
                recursive=True
            )
            
            for file_path in config_files:
                if file_path.stem == name:
                    current_mtime = file_path.stat().st_mtime
                    cached_mtime = self._file_mtimes.get(str(file_path))
                    
                    if cached_mtime is None or current_mtime > cached_mtime:
                        self.logger.info(f"Config '{name}' has changed (file: {file_path})")
                        return True
            
            # Also check if file is in changed files set
            with self._change_lock:
                changed = any(name in str(path) for path in self._changed_files)
                if changed:
                    self.logger.info(f"Config '{name}' marked as changed by file watcher")
                return changed
                
        except Exception as e:
            self.logger.error(f"Error checking if config '{name}' changed: {e}")
            return False
    
    def reload_if_changed(self) -> bool:
        """
        Reload configurations if any changes detected.
        
        Returns:
            True if reload was performed, False otherwise
        """
        try:
            # Check for any changed files
            has_changes = False
            
            with self._change_lock:
                if self._changed_files:
                    has_changes = True
            
            # Check file modification times
            if not has_changes:
                config_files = self._loader.scan_directory(
                    self.config_dir,
                    pattern="*.{yaml,yml}",
                    recursive=True
                )
                
                for file_path in config_files:
                    current_mtime = file_path.stat().st_mtime
                    cached_mtime = self._file_mtimes.get(str(file_path))
                    
                    if cached_mtime is None or current_mtime > cached_mtime:
                        has_changes = True
                        break
            
            if has_changes:
                self.logger.info("Changes detected, performing hot reload")
                self.reload_configs()
                return True
            else:
                self.logger.debug("No changes detected, skipping reload")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during change detection and reload: {e}")
            return False
    
    def enable_file_watcher(self) -> None:
        """
        Enable file watcher for automatic change detection.
        """
        if self._watcher_enabled:
            self.logger.warning("File watcher already enabled")
            return
        
        try:
            self._file_handler = ConfigFileHandler(self)
            self._observer = Observer()
            self._observer.schedule(
                self._file_handler,
                str(self.config_dir),
                recursive=True
            )
            self._observer.start()
            self._watcher_enabled = True
            
            self.logger.info("File watcher enabled for automatic configuration reloading")
            
        except Exception as e:
            self.logger.error(f"Failed to enable file watcher: {e}")
            raise ConfigurationError(f"Failed to enable file watcher: {e}")
    
    def disable_file_watcher(self) -> None:
        """
        Disable file watcher.
        """
        if not self._watcher_enabled:
            self.logger.debug("File watcher not enabled")
            return
        
        try:
            if self._observer is not None:
                self._observer.stop()
                self._observer.join()
                self._observer = None
            
            self._file_handler = None
            self._watcher_enabled = False
            
            self.logger.info("File watcher disabled")
            
        except Exception as e:
            self.logger.error(f"Error disabling file watcher: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for monitoring.
        
        Returns:
            Dictionary of performance metrics
        """
        with self._metrics_lock:
            metrics = self._metrics.copy()
            
            # Calculate derived metrics
            if metrics['config_loads'] > 0:
                metrics['avg_load_time'] = metrics['total_load_time'] / metrics['config_loads']
            else:
                metrics['avg_load_time'] = 0.0
            
            total_requests = metrics['cache_hits'] + metrics['cache_misses']
            if total_requests > 0:
                metrics['cache_hit_rate'] = metrics['cache_hits'] / total_requests
            else:
                metrics['cache_hit_rate'] = 0.0
            
            return metrics
    
    def validate_config(self, name: str) -> List[str]:
        """
        Validate a specific configuration.
        
        Args:
            name: Configuration name
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            config = self.get_config(name)
            if config is None:
                errors.append(f"Configuration '{name}' not found")
                return errors
            
            # Basic structure validation
            if 'event_config' not in config:
                errors.append("Missing 'event_config' section")
                return errors
            
            event_config = config['event_config']
            
            # Required fields
            required_fields = ['name', 'data_source']
            for field in required_fields:
                if field not in event_config:
                    errors.append(f"Missing required field: {field}")
            
            # Validate data_source
            if 'data_source' in event_config:
                data_source = event_config['data_source']
                if 'table' not in data_source:
                    errors.append("Missing 'table' in data_source")
                if 'metrics' not in data_source:
                    errors.append("Missing 'metrics' in data_source")
                elif not isinstance(data_source['metrics'], list):
                    errors.append("'metrics' must be a list")
            
            self.logger.debug(f"Validation completed for '{name}': {len(errors)} errors")
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            self.logger.error(f"Error validating config '{name}': {e}")
        
        return errors
    
    def migrate_config(self, name: str, from_version: str, to_version: str) -> bool:
        """
        Migrate configuration from one version to another.
        
        Args:
            name: Configuration name
            from_version: Source version
            to_version: Target version
            
        Returns:
            True if migration successful, False otherwise
        """
        self.logger.info(f"Migrating config '{name}' from {from_version} to {to_version}")
        
        try:
            # This is a placeholder for migration logic
            # In a real implementation, you would have version-specific migration rules
            config = self.get_config(name)
            if config is None:
                self.logger.error(f"Cannot migrate config '{name}': not found")
                return False
            
            # Example migration logic (placeholder)
            if from_version == "1.0" and to_version == "2.0":
                # Add any new required fields or transform existing ones
                if 'version' not in config['event_config']:
                    config['event_config']['version'] = to_version
            
            self.logger.info(f"Successfully migrated config '{name}' to version {to_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to migrate config '{name}': {e}")
            return False
    
    def _update_file_mtimes(self) -> None:
        """Update file modification times for change detection."""
        try:
            config_files = self._loader.scan_directory(
                self.config_dir,
                pattern="*.{yaml,yml}",
                recursive=True
            )
            
            for file_path in config_files:
                self._file_mtimes[str(file_path)] = file_path.stat().st_mtime
                
        except Exception as e:
            self.logger.error(f"Error updating file modification times: {e}")
    
    def _mark_file_changed(self, file_path: str) -> None:
        """Mark a file as changed for reload detection."""
        with self._change_lock:
            self._changed_files.add(file_path)
            self.logger.debug(f"Marked file as changed: {file_path}")
    
    def __del__(self):
        """Cleanup when instance is destroyed."""
        try:
            if hasattr(self, '_watcher_enabled') and self._watcher_enabled:
                self.disable_file_watcher()
        except Exception:
            # Ignore errors during cleanup
            pass