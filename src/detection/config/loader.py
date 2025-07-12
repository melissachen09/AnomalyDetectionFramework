"""
Configuration loader for anomaly detection framework.

This module provides functionality to load, validate, and cache configuration files
for the anomaly detection system. It handles YAML parsing, file system operations,
error handling, and performance optimization through caching.
"""

import os
import time
import glob
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from threading import Lock
import yaml

logger = logging.getLogger(__name__)


class ConfigLoaderError(Exception):
    """Base exception class for configuration loader errors."""
    pass


class ConfigValidationError(ConfigLoaderError):
    """Raised when configuration validation fails."""
    pass


class ConfigFileNotFoundError(ConfigLoaderError):
    """Raised when a configuration file is not found."""
    pass


class ConfigParsingError(ConfigLoaderError):
    """Raised when YAML parsing fails."""
    pass


@dataclass
class EventConfig:
    """Data class representing an event configuration."""
    name: str
    enabled: bool = True
    data_source: Dict[str, Any] = field(default_factory=dict)
    detection: Dict[str, Any] = field(default_factory=dict)
    alerting: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventConfig':
        """Create EventConfig from dictionary data."""
        event_config_data = data.get('event_config', {})
        
        return cls(
            name=event_config_data.get('name', ''),
            enabled=event_config_data.get('enabled', True),
            data_source=event_config_data.get('data_source', {}),
            detection=event_config_data.get('detection', {}),
            alerting=event_config_data.get('alerting', {}),
            metadata=event_config_data.get('metadata', {})
        )


@dataclass
class ConfigLoadResult:
    """Result of configuration loading operation."""
    event_config: EventConfig
    file_path: str
    load_time: float
    file_size: int
    
    @classmethod
    def from_data(cls, data: Dict[str, Any], file_path: str, 
                  load_time: float, file_size: int) -> 'ConfigLoadResult':
        """Create ConfigLoadResult from raw data."""
        return cls(
            event_config=EventConfig.from_dict(data),
            file_path=file_path,
            load_time=load_time,
            file_size=file_size
        )


class ConfigCache:
    """Thread-safe cache for configuration files."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, tuple] = {}  # file_path -> (result, mtime, cache_time)
        self._lock = Lock()
        self._stats = {"hits": 0, "misses": 0, "size": 0}
    
    def get(self, file_path: str) -> Optional[ConfigLoadResult]:
        """Get cached result if valid."""
        with self._lock:
            if file_path not in self._cache:
                self._stats["misses"] += 1
                return None
            
            result, cached_mtime, cache_time = self._cache[file_path]
            
            # Check if file has been modified
            try:
                current_mtime = os.path.getmtime(file_path)
                if current_mtime > cached_mtime:
                    # File modified, remove from cache
                    del self._cache[file_path]
                    self._stats["size"] = len(self._cache)
                    self._stats["misses"] += 1
                    return None
            except OSError:
                # File doesn't exist anymore, remove from cache
                del self._cache[file_path]
                self._stats["size"] = len(self._cache)
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            return result
    
    def put(self, file_path: str, result: ConfigLoadResult) -> None:
        """Cache the result."""
        with self._lock:
            try:
                mtime = os.path.getmtime(file_path)
                cache_time = time.time()
                
                # If cache is full, remove oldest entry
                if len(self._cache) >= self.max_size:
                    oldest_path = min(self._cache.keys(), 
                                    key=lambda k: self._cache[k][2])
                    del self._cache[oldest_path]
                
                self._cache[file_path] = (result, mtime, cache_time)
                self._stats["size"] = len(self._cache)
                
            except OSError:
                # Can't get mtime, don't cache
                pass
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._stats = {"hits": 0, "misses": 0, "size": 0}
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return self._stats.copy()


class ConfigLoader:
    """
    Configuration loader with caching, validation, and error handling.
    
    Features:
    - YAML file parsing with safety checks
    - File system operations (single/bulk loading)
    - Thread-safe caching with automatic invalidation
    - Comprehensive error handling
    - Performance monitoring
    """
    
    def __init__(self, enable_cache: bool = True, max_cache_size: int = 100, 
                 max_file_size_mb: int = 10):
        """
        Initialize ConfigLoader.
        
        Args:
            enable_cache: Whether to enable caching
            max_cache_size: Maximum number of configs to cache
            max_file_size_mb: Maximum file size in MB to process
        """
        self.enable_cache = enable_cache
        self.max_file_size_mb = max_file_size_mb
        self._cache = ConfigCache(max_cache_size) if enable_cache else None
        self._load_times: Dict[str, float] = {}
        
        # Configure YAML loader for security
        self._yaml_loader = yaml.SafeLoader
    
    def load_single_config(self, file_path: str) -> ConfigLoadResult:
        """
        Load a single configuration file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            ConfigLoadResult containing the loaded configuration
            
        Raises:
            ConfigFileNotFoundError: If file doesn't exist
            ConfigParsingError: If YAML parsing fails
            ConfigValidationError: If configuration is invalid
        """
        file_path = str(file_path)  # Ensure string
        
        # Check cache first
        if self._cache:
            cached_result = self._cache.get(file_path)
            if cached_result:
                return cached_result
        
        start_time = time.time()
        
        try:
            # Validate file exists and is readable
            if not os.path.isfile(file_path):
                if os.path.isdir(file_path):
                    raise ConfigLoaderError(f"Expected file, got directory: {file_path}")
                raise ConfigFileNotFoundError(f"Configuration file not found: {file_path}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            max_size_bytes = self.max_file_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                raise ConfigLoaderError(
                    f"File too large: {file_size / 1024 / 1024:.2f}MB "
                    f"(max: {self.max_file_size_mb}MB)"
                )
            
            # Check file permissions
            if not os.access(file_path, os.R_OK):
                raise ConfigLoaderError(f"Permission denied: {file_path}")
            
            # Load and parse YAML
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for empty file
                if not content.strip():
                    raise ConfigLoaderError(f"Empty configuration file: {file_path}")
                
                # Parse YAML safely
                data = yaml.load(content, Loader=self._yaml_loader)
                
                if data is None:
                    raise ConfigLoaderError(f"Empty configuration file: {file_path}")
                
            except yaml.YAMLError as e:
                raise ConfigParsingError(f"Invalid YAML syntax in {file_path}: {e}")
            except UnicodeDecodeError as e:
                raise ConfigLoaderError(f"Unable to decode file {file_path}: {e}")
            except OSError as e:
                raise ConfigLoaderError(f"Error reading file {file_path}: {e}")
            
            # Validate configuration structure
            self._validate_config(data, file_path)
            
            # Create result
            load_time = time.time() - start_time
            result = ConfigLoadResult.from_data(data, file_path, load_time, file_size)
            
            # Cache result
            if self._cache:
                self._cache.put(file_path, result)
            
            # Track load time
            self._load_times[file_path] = time.time()
            
            logger.debug(f"Loaded config {file_path} in {load_time:.3f}s")
            return result
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Failed to load config {file_path} after {load_time:.3f}s: {e}")
            raise
    
    def load_configs_from_directory(self, directory_path: str) -> List[ConfigLoadResult]:
        """
        Load all YAML configuration files from a directory.
        
        Args:
            directory_path: Path to directory containing config files
            
        Returns:
            List of ConfigLoadResult objects
        """
        directory_path = str(directory_path)
        
        if not os.path.isdir(directory_path):
            raise ConfigLoaderError(f"Directory not found: {directory_path}")
        
        pattern = os.path.join(directory_path, "*.yaml")
        config_files = glob.glob(pattern)
        
        if not config_files:
            logger.warning(f"No YAML files found in {directory_path}")
            return []
        
        results = []
        errors = []
        
        for config_file in sorted(config_files):
            try:
                result = self.load_single_config(config_file)
                results.append(result)
            except Exception as e:
                errors.append(f"{config_file}: {e}")
                logger.error(f"Failed to load {config_file}: {e}")
        
        if errors:
            logger.warning(f"Failed to load {len(errors)} config files: {errors}")
        
        logger.info(f"Loaded {len(results)} config files from {directory_path}")
        return results
    
    def load_configs_with_pattern(self, directory_path: str, pattern: str) -> List[ConfigLoadResult]:
        """
        Load configuration files matching a specific pattern.
        
        Args:
            directory_path: Directory to search in
            pattern: Glob pattern to match files
            
        Returns:
            List of ConfigLoadResult objects
        """
        directory_path = str(directory_path)
        full_pattern = os.path.join(directory_path, pattern)
        config_files = glob.glob(full_pattern)
        
        results = []
        for config_file in sorted(config_files):
            try:
                result = self.load_single_config(config_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to load {config_file}: {e}")
        
        return results
    
    def load_configs_recursive(self, directory_path: str) -> List[ConfigLoadResult]:
        """
        Recursively load all YAML files from directory and subdirectories.
        
        Args:
            directory_path: Root directory to search
            
        Returns:
            List of ConfigLoadResult objects
        """
        directory_path = str(directory_path)
        pattern = os.path.join(directory_path, "**", "*.yaml")
        config_files = glob.glob(pattern, recursive=True)
        
        results = []
        for config_file in sorted(config_files):
            try:
                result = self.load_single_config(config_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to load {config_file}: {e}")
        
        return results
    
    def refresh_if_modified(self, file_path: str) -> bool:
        """
        Refresh config if file has been modified since last load.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            True if file was reloaded, False if not modified
        """
        file_path = str(file_path)
        
        if file_path not in self._load_times:
            return False
        
        try:
            current_mtime = os.path.getmtime(file_path)
            last_load_time = self._load_times[file_path]
            
            if current_mtime > last_load_time:
                # File modified, reload
                self.load_single_config(file_path)
                return True
            
        except OSError:
            # File doesn't exist anymore
            if file_path in self._load_times:
                del self._load_times[file_path]
        
        return False
    
    def get_file_load_time(self, file_path: str) -> Optional[float]:
        """Get the timestamp when file was last loaded."""
        return self._load_times.get(str(file_path))
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.get_stats()
        return {"hits": 0, "misses": 0, "size": 0}
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        if self._cache:
            self._cache.clear()
    
    def _validate_config(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Validate configuration structure.
        
        Args:
            data: Parsed YAML data
            file_path: Path to file being validated
            
        Raises:
            ConfigValidationError: If validation fails
        """
        if not isinstance(data, dict):
            raise ConfigValidationError(
                f"Configuration must be a dictionary in {file_path}"
            )
        
        if 'event_config' not in data:
            raise ConfigValidationError(
                f"Missing required 'event_config' section in {file_path}"
            )
        
        event_config = data['event_config']
        if not isinstance(event_config, dict):
            raise ConfigValidationError(
                f"'event_config' must be a dictionary in {file_path}"
            )
        
        # Check required fields
        required_fields = ['name']
        for field in required_fields:
            if field not in event_config:
                raise ConfigValidationError(
                    f"Missing required field 'event_config.{field}' in {file_path}"
                )
        
        # Validate name is string
        if not isinstance(event_config['name'], str) or not event_config['name'].strip():
            raise ConfigValidationError(
                f"'event_config.name' must be a non-empty string in {file_path}"
            )
        
        # Validate data_source if present
        if 'data_source' in event_config:
            data_source = event_config['data_source']
            if not isinstance(data_source, dict):
                raise ConfigValidationError(
                    f"'event_config.data_source' must be a dictionary in {file_path}"
                )