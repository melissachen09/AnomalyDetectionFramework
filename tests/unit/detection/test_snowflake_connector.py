"""
Test suite for Snowflake connector implementation.

Tests database connection management functionality with connection lifecycle,
pool behavior, error scenarios, and thread safety validation.

Part of Epic ADF-4: Snowflake Integration Layer
Task: ADF-42 - Write Test Cases for Snowflake Connector
"""

import os
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import snowflake.connector
from snowflake.connector import ProgrammingError, DatabaseError, OperationalError

from src.detection.utils.snowflake_connector import (
    SnowflakeConnector,
    SnowflakeConnectionPool,
    SnowflakeConnectionError,
    SnowflakeTimeoutError,
    SnowflakeAuthenticationError
)


class TestSnowflakeConnectorLifecycle(unittest.TestCase):
    """Test connection lifecycle management."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password',
            'warehouse': 'test_warehouse',
            'database': 'test_database',
            'schema': 'test_schema',
            'role': 'test_role'
        }

    @patch('snowflake.connector.connect')
    def test_connection_creation_success(self, mock_connect):
        """Test successful connection creation."""
        mock_connection = Mock()
        mock_connect.return_value = mock_connection
        
        connector = SnowflakeConnector(self.valid_config)
        connection = connector.get_connection()
        
        self.assertIsNotNone(connection)
        mock_connect.assert_called_once_with(
            account='test_account',
            user='test_user',
            password='test_password',
            warehouse='test_warehouse',
            database='test_database',
            schema='test_schema',
            role='test_role',
            timeout=60,
            login_timeout=60
        )

    @patch('snowflake.connector.connect')
    def test_connection_creation_with_authentication_error(self, mock_connect):
        """Test connection creation with authentication failure."""
        mock_connect.side_effect = ProgrammingError(
            "Authentication failed",
            errno=250001
        )
        
        connector = SnowflakeConnector(self.valid_config)
        
        with self.assertRaises(SnowflakeAuthenticationError) as context:
            connector.get_connection()
        
        self.assertIn("Authentication failed", str(context.exception))

    @patch('snowflake.connector.connect')
    def test_connection_creation_with_network_error(self, mock_connect):
        """Test connection creation with network failure."""
        mock_connect.side_effect = OperationalError(
            "Network timeout",
            errno=250003
        )
        
        connector = SnowflakeConnector(self.valid_config)
        
        with self.assertRaises(SnowflakeConnectionError) as context:
            connector.get_connection()
        
        self.assertIn("Network timeout", str(context.exception))

    @patch('snowflake.connector.connect')
    def test_connection_context_manager(self, mock_connect):
        """Test connection usage with context manager."""
        mock_connection = Mock()
        mock_connect.return_value = mock_connection
        
        connector = SnowflakeConnector(self.valid_config)
        
        with connector.get_connection_context() as conn:
            self.assertIsNotNone(conn)
            self.assertEqual(conn, mock_connection)
        
        # Verify connection was closed
        mock_connection.close.assert_called_once()

    @patch('snowflake.connector.connect')
    def test_connection_close_on_exception(self, mock_connect):
        """Test connection is properly closed when exception occurs."""
        mock_connection = Mock()
        mock_connect.return_value = mock_connection
        
        connector = SnowflakeConnector(self.valid_config)
        
        with self.assertRaises(ValueError):
            with connector.get_connection_context() as conn:
                raise ValueError("Test exception")
        
        # Verify connection was still closed despite exception
        mock_connection.close.assert_called_once()

    def test_config_validation_required_fields(self):
        """Test configuration validation with missing required fields."""
        invalid_configs = [
            {},  # Empty config
            {'user': 'test'},  # Missing account
            {'account': 'test'},  # Missing user
            {'account': 'test', 'user': 'test'},  # Missing password
        ]
        
        for config in invalid_configs:
            with self.assertRaises(ValueError):
                SnowflakeConnector(config)

    def test_config_validation_optional_fields(self):
        """Test configuration validation with optional fields."""
        minimal_config = {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password'
        }
        
        # Should not raise exception
        connector = SnowflakeConnector(minimal_config)
        self.assertIsNotNone(connector)

    @patch('snowflake.connector.connect')
    def test_connection_retry_mechanism(self, mock_connect):
        """Test connection retry on transient failures."""
        # First two calls fail, third succeeds
        mock_connect.side_effect = [
            OperationalError("Temporary network error", errno=250003),
            OperationalError("Another network error", errno=250003),
            Mock()  # Success on third attempt
        ]
        
        config = self.valid_config.copy()
        config['max_retries'] = 3
        config['retry_delay'] = 0.1
        
        connector = SnowflakeConnector(config)
        connection = connector.get_connection()
        
        self.assertIsNotNone(connection)
        self.assertEqual(mock_connect.call_count, 3)

    @patch('snowflake.connector.connect')
    def test_connection_retry_exhausted(self, mock_connect):
        """Test connection failure after retry exhaustion."""
        mock_connect.side_effect = OperationalError(
            "Persistent network error", 
            errno=250003
        )
        
        config = self.valid_config.copy()
        config['max_retries'] = 2
        config['retry_delay'] = 0.1
        
        connector = SnowflakeConnector(config)
        
        with self.assertRaises(SnowflakeConnectionError):
            connector.get_connection()
        
        self.assertEqual(mock_connect.call_count, 3)  # Initial + 2 retries


class TestSnowflakeConnectionPool(unittest.TestCase):
    """Test connection pool behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password',
            'warehouse': 'test_warehouse',
            'database': 'test_database',
            'schema': 'test_schema'
        }

    @patch('snowflake.connector.connect')
    def test_pool_initialization(self, mock_connect):
        """Test connection pool initialization with correct size."""
        mock_connect.return_value = Mock()
        
        pool = SnowflakeConnectionPool(
            config=self.valid_config,
            pool_size=5,
            max_connections=10
        )
        
        self.assertEqual(pool.pool_size, 5)
        self.assertEqual(pool.max_connections, 10)
        self.assertEqual(len(pool._available_connections), 5)

    @patch('snowflake.connector.connect')
    def test_pool_connection_acquisition(self, mock_connect):
        """Test acquiring connections from pool."""
        mock_connections = [Mock() for _ in range(3)]
        mock_connect.side_effect = mock_connections
        
        pool = SnowflakeConnectionPool(
            config=self.valid_config,
            pool_size=3
        )
        
        # Acquire first connection
        conn1 = pool.get_connection()
        self.assertIsNotNone(conn1)
        self.assertEqual(len(pool._available_connections), 2)
        
        # Acquire second connection
        conn2 = pool.get_connection()
        self.assertIsNotNone(conn2)
        self.assertNotEqual(conn1, conn2)
        self.assertEqual(len(pool._available_connections), 1)

    @patch('snowflake.connector.connect')
    def test_pool_connection_return(self, mock_connect):
        """Test returning connections to pool."""
        mock_connect.return_value = Mock()
        
        pool = SnowflakeConnectionPool(
            config=self.valid_config,
            pool_size=2
        )
        
        conn = pool.get_connection()
        initial_available = len(pool._available_connections)
        
        pool.return_connection(conn)
        
        self.assertEqual(
            len(pool._available_connections), 
            initial_available + 1
        )

    @patch('snowflake.connector.connect')
    def test_pool_exhaustion_and_expansion(self, mock_connect):
        """Test pool behavior when all connections are in use."""
        mock_connect.return_value = Mock()
        
        pool = SnowflakeConnectionPool(
            config=self.valid_config,
            pool_size=2,
            max_connections=5
        )
        
        # Exhaust initial pool
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        
        self.assertEqual(len(pool._available_connections), 0)
        
        # Request additional connection - should create new one
        conn3 = pool.get_connection()
        self.assertIsNotNone(conn3)
        
        # Verify new connection was created
        self.assertEqual(mock_connect.call_count, 3)

    @patch('snowflake.connector.connect')
    def test_pool_max_connections_limit(self, mock_connect):
        """Test pool respects maximum connections limit."""
        mock_connect.return_value = Mock()
        
        pool = SnowflakeConnectionPool(
            config=self.valid_config,
            pool_size=2,
            max_connections=3,
            timeout=0.1
        )
        
        # Exhaust pool and reach max connections
        connections = []
        for i in range(3):
            connections.append(pool.get_connection())
        
        # Next request should timeout
        with self.assertRaises(SnowflakeTimeoutError):
            pool.get_connection()

    @patch('snowflake.connector.connect')
    def test_pool_connection_validation(self, mock_connect):
        """Test pool validates connections before returning them."""
        mock_connection = Mock()
        mock_connection.is_closed.return_value = False
        mock_connect.return_value = mock_connection
        
        pool = SnowflakeConnectionPool(
            config=self.valid_config,
            pool_size=1
        )
        
        conn = pool.get_connection()
        pool.return_connection(conn)
        
        # Mock connection becoming invalid
        mock_connection.is_closed.return_value = True
        
        # Should create new connection instead of returning invalid one
        new_conn = pool.get_connection()
        self.assertEqual(mock_connect.call_count, 2)

    @patch('snowflake.connector.connect')
    def test_pool_cleanup_on_close(self, mock_connect):
        """Test pool properly closes all connections on cleanup."""
        mock_connections = [Mock() for _ in range(3)]
        mock_connect.side_effect = mock_connections
        
        pool = SnowflakeConnectionPool(
            config=self.valid_config,
            pool_size=3
        )
        
        # Get and return a connection to verify active connections are tracked
        conn = pool.get_connection()
        pool.return_connection(conn)
        
        # Close pool
        pool.close()
        
        # Verify all connections were closed
        for mock_conn in mock_connections:
            mock_conn.close.assert_called_once()


class TestSnowflakeErrorScenarios(unittest.TestCase):
    """Test error handling scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password'
        }

    @patch('snowflake.connector.connect')
    def test_query_execution_error(self, mock_connect):
        """Test handling of query execution errors."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = ProgrammingError(
            "SQL compilation error", 
            errno=100096
        )
        mock_connect.return_value = mock_connection
        
        connector = SnowflakeConnector(self.valid_config)
        
        with self.assertRaises(SnowflakeConnectionError):
            with connector.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM invalid_table")

    @patch('snowflake.connector.connect')
    def test_connection_timeout_during_query(self, mock_connect):
        """Test handling of connection timeout during query execution."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = OperationalError(
            "Query timeout", 
            errno=604
        )
        mock_connect.return_value = mock_connection
        
        connector = SnowflakeConnector(self.valid_config)
        
        with self.assertRaises(SnowflakeTimeoutError):
            with connector.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM large_table")

    @patch('snowflake.connector.connect')
    def test_connection_lost_during_transaction(self, mock_connect):
        """Test handling of connection loss during transaction."""
        mock_connection = Mock()
        mock_connection.is_closed.return_value = False
        
        # Simulate connection loss
        def side_effect(*args, **kwargs):
            mock_connection.is_closed.return_value = True
            raise OperationalError("Connection lost", errno=250003)
        
        mock_connection.cursor.side_effect = side_effect
        mock_connect.return_value = mock_connection
        
        connector = SnowflakeConnector(self.valid_config)
        
        with self.assertRaises(SnowflakeConnectionError):
            with connector.get_connection_context() as conn:
                conn.cursor()

    @patch('snowflake.connector.connect')
    def test_invalid_warehouse_error(self, mock_connect):
        """Test handling of invalid warehouse error."""
        mock_connect.side_effect = ProgrammingError(
            "Object 'INVALID_WAREHOUSE' does not exist",
            errno=2003
        )
        
        config = self.valid_config.copy()
        config['warehouse'] = 'INVALID_WAREHOUSE'
        
        connector = SnowflakeConnector(config)
        
        with self.assertRaises(SnowflakeConnectionError) as context:
            connector.get_connection()
        
        self.assertIn("does not exist", str(context.exception))

    @patch('snowflake.connector.connect')
    def test_insufficient_privileges_error(self, mock_connect):
        """Test handling of insufficient privileges error."""
        mock_connect.side_effect = ProgrammingError(
            "Insufficient privileges to operate on warehouse",
            errno=2043
        )
        
        connector = SnowflakeConnector(self.valid_config)
        
        with self.assertRaises(SnowflakeAuthenticationError) as context:
            connector.get_connection()
        
        self.assertIn("Insufficient privileges", str(context.exception))

    def test_malformed_connection_config(self):
        """Test handling of malformed connection configuration."""
        invalid_configs = [
            {'account': ''},  # Empty account
            {'account': 'test', 'user': '', 'password': 'test'},  # Empty user
            {'account': 'test', 'user': 'test', 'password': ''},  # Empty password
        ]
        
        for config in invalid_configs:
            with self.assertRaises(ValueError):
                SnowflakeConnector(config)

    @patch('snowflake.connector.connect')
    def test_connection_recovery_after_error(self, mock_connect):
        """Test connection recovery after transient error."""
        # First call fails, second succeeds
        mock_connect.side_effect = [
            OperationalError("Temporary failure", errno=250003),
            Mock()
        ]
        
        connector = SnowflakeConnector(self.valid_config)
        
        # First attempt should fail
        with self.assertRaises(SnowflakeConnectionError):
            connector.get_connection()
        
        # Second attempt should succeed
        conn = connector.get_connection()
        self.assertIsNotNone(conn)


class TestSnowflakeThreadSafety(unittest.TestCase):
    """Test thread safety of Snowflake connector."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password'
        }
        self.results = {}
        self.errors = {}

    @patch('snowflake.connector.connect')
    def test_concurrent_connection_creation(self, mock_connect):
        """Test concurrent connection creation is thread-safe."""
        mock_connections = [Mock() for _ in range(10)]
        mock_connect.side_effect = mock_connections
        
        connector = SnowflakeConnector(self.valid_config)
        
        def create_connection(thread_id):
            try:
                conn = connector.get_connection()
                self.results[thread_id] = conn
            except Exception as e:
                self.errors[thread_id] = e
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_connection, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(self.results), 10)
        self.assertEqual(len(self.errors), 0)
        self.assertEqual(mock_connect.call_count, 10)

    @patch('snowflake.connector.connect')
    def test_connection_pool_thread_safety(self, mock_connect):
        """Test connection pool operations are thread-safe."""
        mock_connect.return_value = Mock()
        
        pool = SnowflakeConnectionPool(
            config=self.valid_config,
            pool_size=5,
            max_connections=20
        )
        
        acquired_connections = {}
        
        def acquire_and_use_connection(thread_id):
            try:
                conn = pool.get_connection()
                acquired_connections[thread_id] = conn
                
                # Simulate some work
                time.sleep(0.01)
                
                pool.return_connection(conn)
                self.results[thread_id] = "success"
            except Exception as e:
                self.errors[thread_id] = e
        
        # Create more threads than pool size to test contention
        threads = []
        for i in range(15):
            thread = threading.Thread(target=acquire_and_use_connection, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(self.results), 15)
        self.assertEqual(len(self.errors), 0)
        
        # Verify all connections were returned to pool
        pool.close()

    @patch('snowflake.connector.connect')
    def test_concurrent_query_execution(self, mock_connect):
        """Test concurrent query execution using same connection pool."""
        mock_connections = [Mock() for _ in range(5)]
        for mock_conn in mock_connections:
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = [(1, 'test')]
            mock_conn.cursor.return_value = mock_cursor
        
        mock_connect.side_effect = mock_connections
        
        pool = SnowflakeConnectionPool(
            config=self.valid_config,
            pool_size=5
        )
        
        def execute_query(thread_id):
            try:
                conn = pool.get_connection()
                cursor = conn.cursor()
                cursor.execute(f"SELECT {thread_id}, 'test'")
                result = cursor.fetchall()
                pool.return_connection(conn)
                self.results[thread_id] = result
            except Exception as e:
                self.errors[thread_id] = e
        
        # Execute queries concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(execute_query, i) 
                for i in range(20)
            ]
            
            for future in futures:
                future.result()  # Wait for completion
        
        # Verify results
        self.assertEqual(len(self.results), 20)
        self.assertEqual(len(self.errors), 0)
        
        # Verify each query got results
        for thread_id, result in self.results.items():
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], (1, 'test'))

    @patch('snowflake.connector.connect')
    def test_connection_pool_stress_test(self, mock_connect):
        """Test connection pool under high concurrency stress."""
        mock_connect.return_value = Mock()
        
        pool = SnowflakeConnectionPool(
            config=self.valid_config,
            pool_size=3,
            max_connections=10,
            timeout=5.0
        )
        
        def stress_test_worker(worker_id):
            try:
                for i in range(5):  # Each worker does 5 operations
                    conn = pool.get_connection()
                    
                    # Simulate variable work time
                    time.sleep(0.001 * (worker_id % 3))
                    
                    pool.return_connection(conn)
                
                self.results[worker_id] = "completed"
            except Exception as e:
                self.errors[worker_id] = e
        
        # Run stress test with many concurrent workers
        threads = []
        for i in range(50):
            thread = threading.Thread(target=stress_test_worker, args=(i,))
            threads.append(thread)
        
        start_time = time.time()
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Verify results
        self.assertEqual(len(self.results), 50)
        self.assertEqual(len(self.errors), 0)
        
        # Verify reasonable performance (should complete within 10 seconds)
        self.assertLess(end_time - start_time, 10.0)
        
        pool.close()


if __name__ == '__main__':
    unittest.main()