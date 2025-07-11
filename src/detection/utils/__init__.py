"""
Detection utilities module.

This module contains utility classes and functions for the detection system,
including database connectors, configuration parsers, and common helpers.
"""

from .snowflake_connector import (
    SnowflakeConnector,
    SnowflakeConnectionPool,
    SnowflakeDetectorBase,
    SnowflakeConnectionError,
    SnowflakeAuthenticationError,
    SnowflakeTimeoutError
)

__all__ = [
    'SnowflakeConnector',
    'SnowflakeConnectionPool', 
    'SnowflakeDetectorBase',
    'SnowflakeConnectionError',
    'SnowflakeAuthenticationError',
    'SnowflakeTimeoutError'
]