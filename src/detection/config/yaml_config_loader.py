"""
YAML Configuration Loader for Anomaly Detection Framework.

This module provides robust YAML configuration loading with:
- Recursive directory scanning
- YAML safe loading 
- LRU caching for performance
- Clear error reporting
- Thread-safe operations
"""

import os
import yaml
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from functools import lru_cache
from collections import OrderedDict

from .exceptions import ConfigurationError, ValidationError, FileParsingError


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.access_times = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing item
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Add new item
                if len(self.cache) >= self.max_size:
                    # Remove least recently used item
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.access_times[oldest_key]
                
                self.cache[key] = value
            
            self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def __len__(self) -> int:
        """Return cache size."""
        return len(self.cache)


class YAMLConfigLoader:
    """
    Robust YAML configuration loader with caching and error handling.
    
    Features:
    - Recursive directory scanning with glob pattern support
    - YAML safe loading to prevent code execution
    - LRU caching for performance optimization
    - Thread-safe operations
    - Comprehensive error reporting
    - File modification time tracking for cache invalidation
    """
    
    def __init__(
        self,
        config_dir: Union[str, Path],
        cache_enabled: bool = True,
        cache_size: int = 100
    ):
        """
        Initialize YAMLConfigLoader.
        
        Args:
            config_dir: Path to configuration directory
            cache_enabled: Whether to enable caching
            cache_size: Maximum number of cached configurations
            
        Raises:
            ConfigurationError: If config_dir is invalid
        """
        self.config_dir = Path(config_dir)
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        
        # Validate config directory
        if not self.config_dir.exists():
            raise ConfigurationError(f"Configuration directory does not exist: {config_dir}")
        
        if not self.config_dir.is_dir():
            raise ConfigurationError(f"Configuration directory path is not a directory: {config_dir}")
        
        # Initialize cache
        self._cache = LRUCache(cache_size) if cache_enabled else None
        self._file_mtimes = {}  # Track file modification times
        self._lock = threading.RLock()
        
        # Loaded configurations
        self._configs = {}
    
    def load_single_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load a single YAML configuration file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Parsed configuration dictionary or None if empty
            
        Raises:
            ConfigurationError: If file doesn't exist or parsing fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file does not exist: {file_path}")
        
        # Generate cache key
        cache_key = str(file_path.absolute())
        
        # Check cache if enabled
        if self.cache_enabled and self._cache:
            with self._lock:
                # Check if file has been modified
                current_mtime = file_path.stat().st_mtime
                cached_mtime = self._file_mtimes.get(cache_key)
                
                if cached_mtime and current_mtime == cached_mtime:
                    cached_config = self._cache.get(cache_key)
                    if cached_config is not None:
                        return cached_config
        
        # Load and parse file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Handle empty files
                if not content:
                    return None
                
                # Parse YAML safely
                config = yaml.safe_load(content)
                
                # Cache the result if caching is enabled
                if self.cache_enabled and self._cache:
                    with self._lock:
                        self._cache.set(cache_key, config)
                        self._file_mtimes[cache_key] = file_path.stat().st_mtime
                
                return config
                
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML file {file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading file {file_path}: {e}")
    
    def scan_directory(
        self, 
        directory: Union[str, Path], 
        pattern: str = "*.yaml",
        recursive: bool = False
    ) -> List[Path]:
        """
        Scan directory for configuration files.
        
        Args:
            directory: Directory to scan
            pattern: Glob pattern for files to include
            recursive: Whether to scan recursively
            
        Returns:
            List of matching file paths
            
        Raises:
            ConfigurationError: If directory doesn't exist
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise ConfigurationError(f"Directory does not exist: {directory}")
        
        if not directory.is_dir():
            raise ConfigurationError(f"Path is not a directory: {directory}")
        
        files = []
        
        # Handle multiple patterns
        patterns = []
        if pattern == "*.{yaml,yml}":
            patterns = ["*.yaml", "*.yml"]
        else:
            patterns = [pattern]
        
        for pat in patterns:
            if recursive:
                files.extend(directory.rglob(pat))
            else:
                files.extend(directory.glob(pat))
        
        # Remove duplicates and sort
        unique_files = list(set(files))
        unique_files.sort()
        
        return unique_files
    
    def load_all_configs(self, fail_on_error: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files from the config directory.
        
        Args:
            fail_on_error: Whether to fail on first error or collect all errors
            
        Returns:
            Dictionary of configuration name -> configuration data
            
        Raises:
            ValidationError: If validation fails and fail_on_error is True
        """
        configs = {}
        errors = []
        
        # Scan for configuration files
        config_files = self.scan_directory(self.config_dir, pattern="*.{yaml,yml}", recursive=True)
        
        for file_path in config_files:
            try:
                config = self.load_single_file(file_path)
                
                if config is None:
                    continue
                
                # Extract configuration name
                config_name = self._extract_config_name(config, file_path)
                
                if config_name:
                    configs[config_name] = config
                else:
                    error_msg = f"Could not extract configuration name from {file_path}"
                    if fail_on_error:
                        raise ValidationError(error_msg)
                    errors.append(error_msg)
                    
            except Exception as e:
                error_msg = f"Error loading {file_path}: {e}"
                if fail_on_error:
                    if errors:
                        # Aggregate all errors
                        all_errors = errors + [error_msg]
                        raise ValidationError(f"Validation failed for multiple files: {'; '.join(all_errors)}")
                    else:
                        raise ValidationError(f"Validation failed: {error_msg}")
                errors.append(error_msg)
        
        if errors and not configs:
            raise ValidationError(f"No valid configurations found. Errors: {'; '.join(errors)}")
        
        # Update internal configs
        with self._lock:
            self._configs.update(configs)
        
        return configs
    
    def _extract_config_name(self, config: Dict[str, Any], file_path: Path) -> Optional[str]:
        """
        Extract configuration name from config data.
        
        Args:
            config: Configuration dictionary
            file_path: Path to configuration file
            
        Returns:
            Configuration name or None if not found
        """
        # Try to get name from event_config.name
        if isinstance(config, dict) and 'event_config' in config:
            event_config = config['event_config']
            if isinstance(event_config, dict) and 'name' in event_config:
                name = event_config['name']
                if name and isinstance(name, str) and name.strip():
                    return name.strip()
        
        # Fallback to filename without extension
        return file_path.stem
    
    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration by name.
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration dictionary or None if not found
        """
        with self._lock:
            return self._configs.get(name)
    
    def list_config_names(self) -> List[str]:
        """
        List all available configuration names.
        
        Returns:
            List of configuration names
        """
        with self._lock:
            return list(self._configs.keys())
    
    def reload_configs(self) -> None:
        """
        Reload all configurations from disk.
        
        This clears the cache and reloads all configurations.
        """
        with self._lock:
            # Clear cache
            if self._cache:
                self._cache.clear()
            self._file_mtimes.clear()
            self._configs.clear()
            
            # Reload configurations
            self.load_all_configs(fail_on_error=False)