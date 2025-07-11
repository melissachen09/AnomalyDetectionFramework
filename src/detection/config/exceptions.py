"""Configuration-related custom exceptions."""


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass


class ValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""
    pass


class FileParsingError(ConfigurationError):
    """Exception raised when file parsing fails."""
    pass


class CacheError(ConfigurationError):
    """Exception raised when cache operations fail."""
    pass