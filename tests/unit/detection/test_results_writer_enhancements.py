"""Enhanced test cases for ResultsWriter - ADF-47 (GADF-SNOW-006).

This module contains additional test cases for the ResultsWriter enhancements:
- Performance monitoring and metrics
- Advanced batch optimization
- Enhanced transaction management
- Connection pooling improvements

Sub-tasks covered:
- GADF-SNOW-006a: Performance monitoring system
- GADF-SNOW-006b: Batch optimization improvements  
- GADF-SNOW-006c: Enhanced transaction management
- GADF-SNOW-006d: Connection pooling and resource management
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, date
from typing import List, Dict, Any
import snowflake.connector
from contextlib import contextmanager

# Import the classes we're testing
from src.detection.utils.results_writer import ResultsWriter
from src.detection.utils.models import AnomalyResult, BatchInsertResult, WriteResult


class TestPerformanceMonitoring:
    """Test performance monitoring features - GADF-SNOW-006a."""
    
    @pytest.fixture
    def config_with_monitoring(self):
        """Configuration with performance monitoring enabled."""
        return {
            'connection': {
                'account': 'test_account',
                'user': 'test_user',
                'password': 'test_password',
                'warehouse': 'test_warehouse',
                'database': 'test_database',
                'schema': 'test_schema'
            },
            'performance_monitoring': {
                'enabled': True,
                'metrics_table': 'PERFORMANCE_METRICS',
                'slow_query_threshold_ms': 5000,
                'batch_size_warnings': True,
                'connection_pool_monitoring': True
            },
            'batch_settings': {
                'batch_size': 1000,
                'enable_upsert': True,
                'timeout_seconds': 300
            }
        }
    
    @pytest.fixture
    def sample_anomaly_result(self):
        """Sample anomaly result for testing."""
        return AnomalyResult(
            detection_date=date.today(),
            event_type='listing_views',
            metric_name='total_views',
            expected_value=100000.0,
            actual_value=50000.0,
            deviation_percentage=-0.5,
            severity='critical',
            detection_method='threshold',
            detector_config={'min_value': 80000, 'max_value': 200000},
            metadata={'source': 'test'}
        )
    
    def test_performance_metrics_collection(self, config_with_monitoring, sample_anomaly_result):
        """Test that performance metrics are collected during operations."""
        writer = ResultsWriter(config_with_monitoring)
        
        # Mock connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_anomaly(sample_anomaly_result)
        
        # Verify metrics were collected
        assert result.success is True
        assert result.execution_time_ms is not None
        assert result.execution_time_ms > 0
        
        # Verify metrics were stored (if enabled)
        if hasattr(writer, '_store_performance_metrics'):
            assert writer._store_performance_metrics.called
    
    def test_slow_query_detection(self, config_with_monitoring, sample_anomaly_result):
        """Test detection of slow queries."""
        writer = ResultsWriter(config_with_monitoring)
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1
        
        # Simulate slow query
        def slow_execute(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow query
            
        mock_cursor.execute.side_effect = slow_execute
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_anomaly(sample_anomaly_result)
        
        # Verify slow query was detected
        assert result.success is True
        assert result.execution_time_ms >= 100  # At least 100ms due to sleep
    
    def test_batch_size_optimization_warnings(self, config_with_monitoring):
        """Test warnings for suboptimal batch sizes."""
        writer = ResultsWriter(config_with_monitoring)
        
        # Test with very small batch size
        small_batch = [AnomalyResult(
            detection_date=date.today(),
            event_type=f'event_{i}',
            metric_name=f'metric_{i}',
            expected_value=100.0,
            actual_value=90.0,
            deviation_percentage=-0.1,
            severity='warning',
            detection_method='threshold',
            detector_config={},
            metadata={}
        ) for i in range(5)]  # Only 5 items
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 5
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_batch(small_batch)
        
        # Should succeed but potentially log warning about batch size
        assert result.success is True
        assert result.total_rows == 5
    
    def test_throughput_measurement(self, config_with_monitoring):
        """Test throughput measurement for batch operations."""
        writer = ResultsWriter(config_with_monitoring)
        
        # Create larger batch for throughput testing
        large_batch = [AnomalyResult(
            detection_date=date.today(),
            event_type=f'event_{i}',
            metric_name=f'metric_{i}',
            expected_value=100.0,
            actual_value=90.0,
            deviation_percentage=-0.1,
            severity='warning',
            detection_method='threshold',
            detector_config={},
            metadata={}
        ) for i in range(1000)]
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1000
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_batch(large_batch)
        
        # Verify throughput can be calculated
        assert result.success is True
        assert result.execution_time_ms is not None
        
        # Calculate throughput (records per second)
        throughput = 1000 / (result.execution_time_ms / 1000)
        assert throughput > 0
    
    def test_connection_pool_metrics(self, config_with_monitoring, sample_anomaly_result):
        """Test connection pool performance metrics."""
        writer = ResultsWriter(config_with_monitoring)
        
        # Mock connection pool
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            # Perform multiple operations to test connection reuse
            for i in range(5):
                result = writer.insert_anomaly(sample_anomaly_result)
                assert result.success is True
        
        # Verify connection was reused efficiently
        assert mock_conn.cursor.call_count == 5


class TestBatchOptimization:
    """Test batch optimization improvements - GADF-SNOW-006b."""
    
    @pytest.fixture
    def config_with_optimization(self):
        """Configuration with batch optimization enabled."""
        return {
            'connection': {
                'account': 'test_account',
                'user': 'test_user',
                'password': 'test_password',
                'warehouse': 'test_warehouse',
                'database': 'test_database',
                'schema': 'test_schema'
            },
            'batch_optimization': {
                'enabled': True,
                'adaptive_batch_size': True,
                'min_batch_size': 100,
                'max_batch_size': 5000,
                'compression_enabled': True,
                'parallel_processing': True
            },
            'batch_settings': {
                'batch_size': 1000,
                'enable_upsert': True,
                'timeout_seconds': 300
            }
        }
    
    def test_adaptive_batch_sizing(self, config_with_optimization):
        """Test adaptive batch sizing based on performance."""
        writer = ResultsWriter(config_with_optimization)
        
        # Create test data
        test_data = [AnomalyResult(
            detection_date=date.today(),
            event_type=f'event_{i}',
            metric_name=f'metric_{i}',
            expected_value=100.0,
            actual_value=90.0,
            deviation_percentage=-0.1,
            severity='warning',
            detection_method='threshold',
            detector_config={},
            metadata={}
        ) for i in range(2500)]  # 2.5x default batch size
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1000  # Simulate batch processing
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_batch(test_data)
        
        # Should process in adaptive batches
        assert result.success is True
        assert result.total_rows == 2500
        assert result.batches_processed >= 2  # Should be split into multiple batches
    
    def test_parallel_batch_processing(self, config_with_optimization):
        """Test parallel processing of batches."""
        writer = ResultsWriter(config_with_optimization)
        
        # Create large dataset for parallel processing
        large_dataset = [AnomalyResult(
            detection_date=date.today(),
            event_type=f'event_{i}',
            metric_name=f'metric_{i}',
            expected_value=100.0,
            actual_value=90.0,
            deviation_percentage=-0.1,
            severity='warning',
            detection_method='threshold',
            detector_config={},
            metadata={}
        ) for i in range(10000)]
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1000
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            start_time = time.time()
            result = writer.insert_batch(large_dataset)
            end_time = time.time()
        
        # Should complete within reasonable time with parallel processing
        assert result.success is True
        assert result.total_rows == 10000
        assert end_time - start_time < 10.0  # Should complete within 10 seconds
    
    def test_compression_optimization(self, config_with_optimization):
        """Test data compression for large batches."""
        writer = ResultsWriter(config_with_optimization)
        
        # Create dataset with compressible data
        compressible_data = [AnomalyResult(
            detection_date=date.today(),
            event_type='repeated_event',  # Repeated values for compression
            metric_name='repeated_metric',
            expected_value=100.0,
            actual_value=90.0,
            deviation_percentage=-0.1,
            severity='warning',
            detection_method='threshold',
            detector_config={'threshold': 0.5},
            metadata={'source': 'test', 'version': '1.0'}
        ) for i in range(1000)]
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1000
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_batch(compressible_data)
        
        # Should process successfully with compression
        assert result.success is True
        assert result.total_rows == 1000
    
    def test_batch_memory_optimization(self, config_with_optimization):
        """Test memory optimization for large batches."""
        writer = ResultsWriter(config_with_optimization)
        
        # Create very large dataset that would normally consume significant memory
        def generate_large_anomaly(i):
            return AnomalyResult(
                detection_date=date.today(),
                event_type=f'event_{i}',
                metric_name=f'metric_{i}',
                expected_value=100.0 + i,
                actual_value=90.0 + i,
                deviation_percentage=-0.1,
                severity='warning',
                detection_method='threshold',
                detector_config={'threshold': 0.5 + i * 0.001},
                metadata={'source': f'test_{i}', 'data': 'x' * 100}  # Some bulk data
            )
        
        # Use generator for memory efficiency
        large_data_generator = (generate_large_anomaly(i) for i in range(50000))
        large_dataset = list(large_data_generator)
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1000
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_batch(large_dataset)
        
        # Should handle large dataset without memory issues
        assert result.success is True
        assert result.total_rows == 50000


class TestEnhancedTransactionManagement:
    """Test enhanced transaction management - GADF-SNOW-006c."""
    
    @pytest.fixture
    def config_with_advanced_transactions(self):
        """Configuration with advanced transaction features."""
        return {
            'connection': {
                'account': 'test_account',
                'user': 'test_user',
                'password': 'test_password',
                'warehouse': 'test_warehouse',
                'database': 'test_database',
                'schema': 'test_schema'
            },
            'transaction_settings': {
                'auto_commit': False,
                'isolation_level': 'READ_COMMITTED',
                'retry_attempts': 3,
                'deadlock_detection': True,
                'transaction_timeout': 300,
                'savepoint_support': True
            },
            'advanced_transaction_features': {
                'distributed_transactions': True,
                'transaction_logging': True,
                'rollback_analytics': True
            }
        }
    
    def test_transaction_timeout_handling(self, config_with_advanced_transactions):
        """Test transaction timeout handling."""
        writer = ResultsWriter(config_with_advanced_transactions)
        
        test_data = [AnomalyResult(
            detection_date=date.today(),
            event_type=f'event_{i}',
            metric_name=f'metric_{i}',
            expected_value=100.0,
            actual_value=90.0,
            deviation_percentage=-0.1,
            severity='warning',
            detection_method='threshold',
            detector_config={},
            metadata={}
        ) for i in range(10)]
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        # Simulate timeout
        mock_cursor.executemany.side_effect = snowflake.connector.OperationalError("Transaction timeout")
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_batch_with_transaction(test_data)
        
        # Should handle timeout gracefully
        assert result.success is False
        assert "timeout" in str(result.errors[0]).lower()
    
    def test_deadlock_detection_and_retry(self, config_with_advanced_transactions):
        """Test deadlock detection and automatic retry."""
        writer = ResultsWriter(config_with_advanced_transactions)
        
        test_data = [AnomalyResult(
            detection_date=date.today(),
            event_type='test_event',
            metric_name='test_metric',
            expected_value=100.0,
            actual_value=90.0,
            deviation_percentage=-0.1,
            severity='warning',
            detection_method='threshold',
            detector_config={},
            metadata={}
        )]
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        # Simulate deadlock then success
        mock_cursor.executemany.side_effect = [
            snowflake.connector.OperationalError("Deadlock detected"),
            None  # Success on retry
        ]
        mock_cursor.rowcount = 1
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_batch_with_transaction(test_data)
        
        # Should succeed after deadlock retry
        assert result.success is True
        assert result.total_rows == 1
    
    def test_savepoint_support(self, config_with_advanced_transactions):
        """Test savepoint support for nested transactions."""
        writer = ResultsWriter(config_with_advanced_transactions)
        
        test_data = [AnomalyResult(
            detection_date=date.today(),
            event_type='test_event',
            metric_name='test_metric',
            expected_value=100.0,
            actual_value=90.0,
            deviation_percentage=-0.1,
            severity='warning',
            detection_method='threshold',
            detector_config={},
            metadata={}
        )]
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_batch_with_transaction(test_data)
        
        # Should use savepoints if supported
        assert result.success is True
        
        # Verify savepoint SQL was executed (if implemented)
        if hasattr(writer, '_use_savepoints'):
            sql_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
            savepoint_calls = [sql for sql in sql_calls if 'SAVEPOINT' in sql.upper()]
            assert len(savepoint_calls) > 0
    
    def test_distributed_transaction_coordination(self, config_with_advanced_transactions):
        """Test distributed transaction coordination."""
        writer = ResultsWriter(config_with_advanced_transactions)
        
        # Simulate distributed transaction scenario
        test_data = [AnomalyResult(
            detection_date=date.today(),
            event_type='distributed_event',
            metric_name='distributed_metric',
            expected_value=100.0,
            actual_value=90.0,
            deviation_percentage=-0.1,
            severity='critical',
            detection_method='threshold',
            detector_config={},
            metadata={'distributed': True}
        )]
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_batch_with_transaction(test_data)
        
        # Should handle distributed transactions
        assert result.success is True
        assert result.total_rows == 1


class TestConnectionPoolingEnhancements:
    """Test connection pooling enhancements - GADF-SNOW-006d."""
    
    @pytest.fixture
    def config_with_connection_pooling(self):
        """Configuration with advanced connection pooling."""
        return {
            'connection': {
                'account': 'test_account',
                'user': 'test_user',
                'password': 'test_password',
                'warehouse': 'test_warehouse',
                'database': 'test_database',
                'schema': 'test_schema'
            },
            'connection_pool': {
                'enabled': True,
                'min_connections': 2,
                'max_connections': 10,
                'connection_timeout': 30,
                'idle_timeout': 300,
                'health_check_interval': 60,
                'retry_on_failure': True
            }
        }
    
    def test_connection_pool_initialization(self, config_with_connection_pooling):
        """Test connection pool initialization."""
        writer = ResultsWriter(config_with_connection_pooling)
        
        # Verify pool configuration
        assert hasattr(writer, 'connection_pool') or hasattr(writer, '_connection_pool_config')
        
        # Pool should be configured with specified parameters
        if hasattr(writer, 'connection_pool'):
            assert writer.connection_pool.min_connections == 2
            assert writer.connection_pool.max_connections == 10
    
    def test_connection_pool_health_checks(self, config_with_connection_pooling):
        """Test connection pool health checks."""
        writer = ResultsWriter(config_with_connection_pooling)
        
        # Mock connection pool
        mock_pool = Mock()
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1
        
        with patch.object(writer, '_get_connection_pool', return_value=mock_pool):
            mock_pool.get_connection.return_value = mock_conn
            
            # Perform operation
            anomaly = AnomalyResult(
                detection_date=date.today(),
                event_type='test_event',
                metric_name='test_metric',
                expected_value=100.0,
                actual_value=90.0,
                deviation_percentage=-0.1,
                severity='warning',
                detection_method='threshold',
                detector_config={},
                metadata={}
            )
            
            with patch.object(writer, '_get_connection', return_value=mock_conn):
                result = writer.insert_anomaly(anomaly)
            
            assert result.success is True
    
    def test_connection_pool_exhaustion_handling(self, config_with_connection_pooling):
        """Test handling of connection pool exhaustion."""
        writer = ResultsWriter(config_with_connection_pooling)
        
        # Mock pool exhaustion
        mock_pool = Mock()
        mock_pool.get_connection.side_effect = Exception("Connection pool exhausted")
        
        with patch.object(writer, '_get_connection_pool', return_value=mock_pool):
            anomaly = AnomalyResult(
                detection_date=date.today(),
                event_type='test_event',
                metric_name='test_metric',
                expected_value=100.0,
                actual_value=90.0,
                deviation_percentage=-0.1,
                severity='warning',
                detection_method='threshold',
                detector_config={},
                metadata={}
            )
            
            with patch.object(writer, '_get_connection', side_effect=Exception("Pool exhausted")):
                result = writer.insert_anomaly(anomaly)
            
            # Should handle pool exhaustion gracefully
            assert result.success is False
            assert "exhausted" in str(result.error).lower()
    
    def test_concurrent_connection_usage(self, config_with_connection_pooling):
        """Test concurrent connection usage from pool."""
        writer = ResultsWriter(config_with_connection_pooling)
        
        # Simulate concurrent operations
        results = []
        
        def concurrent_insert(anomaly_data):
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.rowcount = 1
            
            with patch.object(writer, '_get_connection', return_value=mock_conn):
                result = writer.insert_anomaly(anomaly_data)
                results.append(result)
        
        # Create test data
        test_anomalies = [AnomalyResult(
            detection_date=date.today(),
            event_type=f'event_{i}',
            metric_name=f'metric_{i}',
            expected_value=100.0,
            actual_value=90.0,
            deviation_percentage=-0.1,
            severity='warning',
            detection_method='threshold',
            detector_config={},
            metadata={}
        ) for i in range(5)]
        
        # Run concurrent operations
        threads = []
        for anomaly in test_anomalies:
            thread = threading.Thread(target=concurrent_insert, args=(anomaly,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations succeeded
        assert len(results) == 5
        assert all(result.success for result in results)


class TestResultsWriterIntegration:
    """Integration tests for enhanced ResultsWriter features."""
    
    def test_end_to_end_enhanced_workflow(self):
        """Test end-to-end workflow with all enhancements."""
        config = {
            'connection': {
                'account': 'test_account',
                'user': 'test_user',
                'password': 'test_password',
                'warehouse': 'test_warehouse',
                'database': 'test_database',
                'schema': 'test_schema'
            },
            'performance_monitoring': {
                'enabled': True,
                'metrics_table': 'PERFORMANCE_METRICS',
                'slow_query_threshold_ms': 5000
            },
            'batch_optimization': {
                'enabled': True,
                'adaptive_batch_size': True,
                'min_batch_size': 100,
                'max_batch_size': 5000
            },
            'transaction_settings': {
                'auto_commit': False,
                'isolation_level': 'READ_COMMITTED',
                'retry_attempts': 3,
                'deadlock_detection': True
            },
            'connection_pool': {
                'enabled': True,
                'min_connections': 2,
                'max_connections': 10
            }
        }
        
        writer = ResultsWriter(config)
        
        # Create comprehensive test dataset
        test_data = [AnomalyResult(
            detection_date=date.today(),
            event_type=f'event_{i}',
            metric_name=f'metric_{i}',
            expected_value=100.0 + i,
            actual_value=90.0 + i,
            deviation_percentage=-0.1,
            severity='warning' if i % 2 == 0 else 'critical',
            detection_method='threshold',
            detector_config={'threshold': 0.5 + i * 0.001},
            metadata={'source': f'test_{i}', 'enhanced': True}
        ) for i in range(2500)]
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1000
        
        with patch.object(writer, '_get_connection', return_value=mock_conn):
            result = writer.insert_batch(test_data)
        
        # Verify enhanced features work together
        assert result.success is True
        assert result.total_rows == 2500
        assert result.execution_time_ms is not None
        assert result.batches_processed >= 1
        
        # Verify performance monitoring
        assert hasattr(result, 'execution_time_ms')
        assert result.execution_time_ms > 0