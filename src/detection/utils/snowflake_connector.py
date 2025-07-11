"""
Snowflake database connector with connection pooling and error handling.

This module provides a robust Snowflake database connector that supports:
- Connection lifecycle management
- Connection pooling for performance
- Comprehensive error handling
- Thread-safe operations
- Retry mechanisms for transient failures

Part of Epic ADF-4: Snowflake Integration Layer
"""

import logging
import threading
import time
from contextlib import contextmanager
from queue import Queue, Empty
from typing import Dict, Any, Optional, List
import snowflake.connector
from snowflake.connector import ProgrammingError, DatabaseError, OperationalError


logger = logging.getLogger(__name__)


class SnowflakeConnectionError(Exception):
    """Base exception for Snowflake connection errors."""
    pass


class SnowflakeAuthenticationError(SnowflakeConnectionError):
    """Exception raised for authentication failures."""
    pass


class SnowflakeTimeoutError(SnowflakeConnectionError):
    """Exception raised for timeout scenarios."""
    pass


class SnowflakeConnector:
    """
    Snowflake database connector with retry logic and error handling.
    
    Provides a reliable connection to Snowflake with automatic retry
    on transient failures and proper error classification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Snowflake connector with configuration.
        
        Args:
            config: Configuration dictionary containing connection parameters
                   Required: account, user, password
                   Optional: warehouse, database, schema, role, max_retries, retry_delay
        
        Raises:
            ValueError: If required configuration is missing
        """
        self.config = self._validate_config(config)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        
        logger.info("Snowflake connector initialized for account: %s", 
                   self.config['account'])
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize configuration parameters.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        required_fields = ['account', 'user', 'password']
        
        for field in required_fields:
            if not config.get(field):
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Create a copy to avoid modifying the original
        validated_config = config.copy()
        
        # Set default values for optional fields
        optional_defaults = {
            'warehouse': None,
            'database': None,
            'schema': None,
            'role': None,
            'timeout': 60,
            'login_timeout': 60
        }
        
        for field, default_value in optional_defaults.items():
            if field not in validated_config:
                validated_config[field] = default_value
        
        return validated_config
    
    def get_connection(self):
        """
        Get a Snowflake connection.
        
        Returns:
            Snowflake connection object
            
        Raises:
            SnowflakeConnectionError: For connection failures
            SnowflakeAuthenticationError: For authentication failures
        """
        return self._create_connection()
    
    @contextmanager
    def get_connection_context(self):
        """
        Get a Snowflake connection with automatic cleanup.
        
        Yields:
            Snowflake connection object
            
        Raises:
            SnowflakeConnectionError: For connection failures
            SnowflakeAuthenticationError: For authentication failures
        """
        connection = None
        try:
            connection = self._create_connection()
            yield connection
        finally:
            if connection:
                try:
                    connection.close()
                except Exception as e:
                    logger.warning("Error closing connection: %s", e)
    
    def _create_connection(self):
        """
        Create a new Snowflake connection with retry logic.
        
        Returns:
            Snowflake connection object
            
        Raises:
            SnowflakeConnectionError: For connection failures
            SnowflakeAuthenticationError: For authentication failures
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug("Attempting to connect to Snowflake (attempt %d/%d)", 
                           attempt + 1, self.max_retries + 1)
                
                # Filter out None values from config
                connection_params = {
                    k: v for k, v in self.config.items() 
                    if v is not None and k not in ['max_retries', 'retry_delay']
                }
                
                connection = snowflake.connector.connect(**connection_params)
                logger.info("Successfully connected to Snowflake")
                return connection
                
            except ProgrammingError as e:
                last_exception = e
                if e.errno in [250001, 250002, 2043]:  # Auth-related errors
                    logger.error("Authentication failed: %s", e.msg)
                    raise SnowflakeAuthenticationError(f"Authentication failed: {e.msg}")
                elif e.errno in [2003, 2043]:  # Object doesn't exist or insufficient privileges
                    logger.error("Configuration error: %s", e.msg)
                    raise SnowflakeConnectionError(f"Configuration error: {e.msg}")
                else:
                    logger.error("Programming error: %s", e.msg)
                    raise SnowflakeConnectionError(f"Programming error: {e.msg}")
                    
            except OperationalError as e:
                last_exception = e
                if e.errno == 604:  # Query timeout
                    logger.error("Query timeout: %s", e.msg)
                    raise SnowflakeTimeoutError(f"Query timeout: {e.msg}")
                elif e.errno in [250003, 250004]:  # Network-related errors
                    logger.warning("Network error (attempt %d/%d): %s", 
                                 attempt + 1, self.max_retries + 1, e.msg)
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        raise SnowflakeConnectionError(f"Network error after {self.max_retries} retries: {e.msg}")
                else:
                    logger.error("Operational error: %s", e.msg)
                    raise SnowflakeConnectionError(f"Operational error: {e.msg}")
                    
            except Exception as e:
                last_exception = e
                logger.error("Unexpected error: %s", e)
                raise SnowflakeConnectionError(f"Unexpected error: {e}")
        
        # If we get here, all retries were exhausted
        raise SnowflakeConnectionError(f"Connection failed after {self.max_retries} retries: {last_exception}")


class SnowflakeConnectionPool:
    """
    Thread-safe connection pool for Snowflake connections.
    
    Manages a pool of reusable connections to improve performance
    and handle concurrent access safely.
    """
    
    def __init__(self, config: Dict[str, Any], pool_size: int = 5, 
                 max_connections: int = 20, timeout: float = 30.0):
        """
        Initialize connection pool.
        
        Args:
            config: Snowflake connection configuration
            pool_size: Initial number of connections to create
            max_connections: Maximum number of connections allowed
            timeout: Maximum time to wait for connection acquisition
        """
        self.config = config
        self.pool_size = pool_size
        self.max_connections = max_connections
        self.timeout = timeout
        
        self._available_connections = Queue(maxsize=max_connections)
        self._all_connections = []
        self._active_connections = set()
        self._lock = threading.RLock()
        
        # Create initial pool
        self._create_initial_pool()
        
        logger.info("Connection pool initialized with %d connections", pool_size)
    
    def _create_initial_pool(self):
        """Create initial pool of connections."""
        connector = SnowflakeConnector(self.config)
        
        for i in range(self.pool_size):
            try:
                with connector.get_connection() as conn:
                    # Create a new connection for the pool
                    connection = connector._create_connection()
                    self._available_connections.put(connection)
                    self._all_connections.append(connection)
                    logger.debug("Created initial connection %d/%d", i + 1, self.pool_size)
            except Exception as e:
                logger.error("Failed to create initial connection %d: %s", i + 1, e)
                # Continue with fewer connections if some fail
                break
    
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Returns:
            Snowflake connection object
            
        Raises:
            SnowflakeTimeoutError: If no connection available within timeout
            SnowflakeConnectionError: If connection creation fails
        """
        try:
            # Try to get an available connection
            connection = self._available_connections.get(timeout=self.timeout)
            
            # Validate connection before returning
            if self._is_connection_valid(connection):
                with self._lock:
                    self._active_connections.add(connection)
                logger.debug("Retrieved connection from pool")
                return connection
            else:
                # Connection is invalid, create a new one
                logger.debug("Retrieved invalid connection, creating new one")
                return self._create_new_connection()
                
        except Empty:
            # No connections available, try to create new one if under limit
            logger.debug("No connections available in pool")
            return self._create_new_connection()
    
    def _create_new_connection(self):
        """
        Create a new connection if under the maximum limit.
        
        Returns:
            Snowflake connection object
            
        Raises:
            SnowflakeTimeoutError: If maximum connections reached
            SnowflakeConnectionError: If connection creation fails
        """
        with self._lock:
            if len(self._all_connections) >= self.max_connections:
                raise SnowflakeTimeoutError(
                    f"Maximum connections ({self.max_connections}) reached"
                )
            
            # Create new connection
            connector = SnowflakeConnector(self.config)
            connection = connector._create_connection()
            
            self._all_connections.append(connection)
            self._active_connections.add(connection)
            
            logger.debug("Created new connection (total: %d)", len(self._all_connections))
            return connection
    
    def return_connection(self, connection):
        """
        Return a connection to the pool.
        
        Args:
            connection: Snowflake connection to return
        """
        if not connection:
            return
        
        with self._lock:
            if connection in self._active_connections:
                self._active_connections.remove(connection)
            
            if self._is_connection_valid(connection):
                try:
                    self._available_connections.put_nowait(connection)
                    logger.debug("Returned connection to pool")
                except:
                    # Queue is full, close the connection
                    self._close_connection(connection)
            else:
                # Connection is invalid, remove it
                self._close_connection(connection)
                if connection in self._all_connections:
                    self._all_connections.remove(connection)
    
    def _is_connection_valid(self, connection) -> bool:
        """
        Check if a connection is still valid.
        
        Args:
            connection: Snowflake connection to validate
            
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            return connection and not connection.is_closed()
        except Exception:
            return False
    
    def _close_connection(self, connection):
        """
        Safely close a connection.
        
        Args:
            connection: Snowflake connection to close
        """
        try:
            if connection:
                connection.close()
                logger.debug("Closed connection")
        except Exception as e:
            logger.warning("Error closing connection: %s", e)
    
    def close(self):
        """Close all connections in the pool."""
        with self._lock:
            logger.info("Closing connection pool")
            
            # Close all connections
            for connection in self._all_connections:
                self._close_connection(connection)
            
            # Clear all tracking structures
            self._all_connections.clear()
            self._active_connections.clear()
            
            # Clear the queue
            while not self._available_connections.empty():
                try:
                    self._available_connections.get_nowait()
                except Empty:
                    break
            
            logger.info("Connection pool closed")
    
    def get_pool_stats(self) -> Dict[str, int]:
        """
        Get current pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        with self._lock:
            return {
                'total_connections': len(self._all_connections),
                'active_connections': len(self._active_connections),
                'available_connections': self._available_connections.qsize(),
                'max_connections': self.max_connections
            }


class SnowflakeDetectorBase:
    """
    Base class for Snowflake-based detectors.
    
    Provides common functionality for detectors that need to query Snowflake.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base detector with Snowflake configuration.
        
        Args:
            config: Configuration dictionary containing detector and connection config
        """
        self.config = config
        self.name = config.get('name', 'unnamed_detector')
        
        # Extract Snowflake connection config
        snowflake_config = config.get('snowflake', {})
        if not snowflake_config:
            raise ValueError("Snowflake configuration required")
        
        # Initialize connection pool
        pool_size = config.get('pool_size', 3)
        max_connections = config.get('max_connections', 10)
        
        self.connection_pool = SnowflakeConnectionPool(
            config=snowflake_config,
            pool_size=pool_size,
            max_connections=max_connections
        )
        
        logger.info("Initialized Snowflake detector: %s", self.name)
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool with automatic return.
        
        Yields:
            Snowflake connection object
        """
        connection = None
        try:
            connection = self.connection_pool.get_connection()
            yield connection
        finally:
            if connection:
                self.connection_pool.return_connection(connection)
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """
        Execute a query and return results.
        
        Args:
            query: SQL query to execute
            params: Optional query parameters
            
        Returns:
            List of tuples containing query results
            
        Raises:
            SnowflakeConnectionError: For query execution errors
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                results = cursor.fetchall()
                logger.debug("Query executed successfully, %d rows returned", len(results))
                return results
                
        except ProgrammingError as e:
            logger.error("SQL error in query: %s", e.msg)
            raise SnowflakeConnectionError(f"SQL error: {e.msg}")
        except OperationalError as e:
            if e.errno == 604:
                logger.error("Query timeout: %s", e.msg)
                raise SnowflakeTimeoutError(f"Query timeout: {e.msg}")
            else:
                logger.error("Operational error: %s", e.msg)
                raise SnowflakeConnectionError(f"Operational error: {e.msg}")
        except Exception as e:
            logger.error("Unexpected error executing query: %s", e)
            raise SnowflakeConnectionError(f"Unexpected error: {e}")
    
    def close(self):
        """Close the connection pool."""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.close()
            logger.info("Closed connection pool for detector: %s", self.name)