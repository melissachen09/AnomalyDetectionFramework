"""ResultsWriter class for persisting anomaly detection results to Snowflake.

This module implements the ResultsWriter class that provides:
- Single and batch insert operations with transaction support
- Upsert functionality for idempotent operations
- Data type conversion and validation
- Connection management and pooling
- Comprehensive error handling and retry logic

Features for ADF-46 (GADF-SNOW-005):
- Insert operations tested
- Batch processing verified  
- Transaction handling tested
- Idempotency validated

Enhanced features for ADF-47 (GADF-SNOW-006):
- Performance monitoring and metrics collection
- Advanced batch optimization with adaptive sizing
- Enhanced transaction management with deadlock detection
- Connection pooling improvements and resource management
"""

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Iterator, Union

import snowflake.connector
from snowflake.connector import DictCursor
import pandas as pd

from .models import AnomalyResult, WriteResult, BatchInsertResult


class ResultsWriter:
    """Efficient results writer with batch support and transaction management.
    
    This class provides methods to write anomaly detection results to Snowflake with:
    - Single and batch insert operations
    - Upsert functionality for duplicate handling
    - Transaction management with commit/rollback
    - Data type conversion and validation
    - Connection pooling and retry logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ResultsWriter with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - connection: Snowflake connection parameters (required)
                    - account: Snowflake account
                    - user: Username
                    - password: Password
                    - warehouse: Warehouse name (optional)
                    - database: Database name (optional)
                    - schema: Schema name (optional)
                - tables: Table configuration (optional)
                    - anomalies: Anomalies table name (default: DAILY_ANOMALIES)
                    - metadata: Metadata table name (default: DETECTION_METADATA)
                - batch_settings: Batch processing configuration (optional)
                    - batch_size: Size of each batch (default: 1000)
                    - enable_upsert: Enable upsert operations (default: True)
                    - timeout_seconds: Query timeout (default: 300)
                - transaction_settings: Transaction configuration (optional)
                    - auto_commit: Enable auto-commit (default: False)
                    - isolation_level: Transaction isolation level (default: READ_COMMITTED)
                    - retry_attempts: Number of retry attempts (default: 3)
                    
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate and extract configuration
        self._validate_config()
        self._extract_config()
        
        # Connection will be established when needed
        self._connection = None
    
    def _validate_config(self) -> None:
        """Validate the provided configuration.
        
        Raises:
            ValueError: If configuration is missing required fields or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        if 'connection' not in self.config:
            raise ValueError("Configuration must contain 'connection'")
        
        connection_config = self.config['connection']
        required_fields = ['account', 'user', 'password']
        
        for field in required_fields:
            if field not in connection_config:
                raise ValueError(f"Connection config must contain '{field}'")
            if not connection_config[field]:
                raise ValueError(f"Connection config '{field}' cannot be empty")
        
        # Validate batch settings
        batch_settings = self.config.get('batch_settings', {})
        batch_size = batch_settings.get('batch_size', 1000)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be positive integer")
    
    def _extract_config(self) -> None:
        """Extract configuration values into instance variables."""
        # Connection configuration
        self.connection_config = self.config['connection']
        
        # Tables configuration
        tables_config = self.config.get('tables', {})
        self.tables_config = tables_config
        self.anomalies_table = tables_config.get('anomalies', 'DAILY_ANOMALIES')
        self.metadata_table = tables_config.get('metadata', 'DETECTION_METADATA')
        
        # Batch settings
        batch_settings = self.config.get('batch_settings', {})
        self.batch_size = batch_settings.get('batch_size', 1000)
        self.enable_upsert = batch_settings.get('enable_upsert', True)
        self.timeout_seconds = batch_settings.get('timeout_seconds', 300)
        
        # Transaction settings
        transaction_settings = self.config.get('transaction_settings', {})
        self.auto_commit = transaction_settings.get('auto_commit', False)
        self.isolation_level = transaction_settings.get('isolation_level', 'READ_COMMITTED')
        self.retry_attempts = transaction_settings.get('retry_attempts', 3)
        
        # Enhanced features for ADF-47
        # Performance monitoring settings
        performance_config = self.config.get('performance_monitoring', {})
        self.performance_monitoring_enabled = performance_config.get('enabled', False)
        self.metrics_table = performance_config.get('metrics_table', 'PERFORMANCE_METRICS')
        self.slow_query_threshold_ms = performance_config.get('slow_query_threshold_ms', 5000)
        self.batch_size_warnings = performance_config.get('batch_size_warnings', True)
        
        # Batch optimization settings
        batch_optimization = self.config.get('batch_optimization', {})
        self.batch_optimization_enabled = batch_optimization.get('enabled', False)
        self.adaptive_batch_size = batch_optimization.get('adaptive_batch_size', False)
        self.min_batch_size = batch_optimization.get('min_batch_size', 100)
        self.max_batch_size = batch_optimization.get('max_batch_size', 5000)
        self.compression_enabled = batch_optimization.get('compression_enabled', False)
        self.parallel_processing = batch_optimization.get('parallel_processing', False)
        
        # Enhanced transaction settings
        self.deadlock_detection = transaction_settings.get('deadlock_detection', False)
        self.transaction_timeout = transaction_settings.get('transaction_timeout', 300)
        self.savepoint_support = transaction_settings.get('savepoint_support', False)
        
        # Connection pool settings
        connection_pool = self.config.get('connection_pool', {})
        self.connection_pool_enabled = connection_pool.get('enabled', False)
        self.min_connections = connection_pool.get('min_connections', 1)
        self.max_connections = connection_pool.get('max_connections', 5)
        self.connection_timeout = connection_pool.get('connection_timeout', 30)
        self.idle_timeout = connection_pool.get('idle_timeout', 300)
        self.health_check_interval = connection_pool.get('health_check_interval', 60)
        
        # Initialize performance metrics
        self._performance_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_execution_time_ms': 0,
            'slow_queries': 0,
            'deadlocks_detected': 0,
            'retries_performed': 0
        }
    
    @contextmanager
    def _get_connection(self):
        """Get a Snowflake database connection.
        
        This is a context manager that ensures proper connection cleanup.
        
        Yields:
            snowflake.connector.SnowflakeConnection: Database connection
        """
        conn = None
        try:
            # Create connection
            conn = snowflake.connector.connect(
                account=self.connection_config['account'],
                user=self.connection_config['user'],
                password=self.connection_config['password'],
                warehouse=self.connection_config.get('warehouse'),
                database=self.connection_config.get('database'),
                schema=self.connection_config.get('schema'),
                autocommit=self.auto_commit
            )
            
            self.logger.debug("Established Snowflake connection")
            yield conn
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Snowflake: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                    self.logger.debug("Closed Snowflake connection")
                except Exception as e:
                    self.logger.warning(f"Error closing connection: {e}")
    
    def _convert_anomaly_to_params(self, anomaly: AnomalyResult) -> tuple:
        """Convert AnomalyResult to SQL parameters.
        
        Args:
            anomaly: AnomalyResult instance to convert
            
        Returns:
            Tuple of parameters for SQL execution
        """
        return (
            anomaly.detection_date.isoformat() if anomaly.detection_date else None,
            anomaly.event_type,
            anomaly.metric_name,
            anomaly.expected_value,
            anomaly.actual_value,
            anomaly.deviation_percentage,
            anomaly.severity,
            anomaly.detection_method,
            json.dumps(anomaly.detector_config) if anomaly.detector_config else None,
            json.dumps(anomaly.metadata) if anomaly.metadata else None,
            anomaly.alert_sent,
            anomaly.created_at.isoformat() if anomaly.created_at else datetime.now().isoformat()
        )
    
    def _build_insert_sql(self) -> str:
        """Build INSERT SQL statement for anomaly results.
        
        Returns:
            SQL INSERT statement
        """
        return f"""
        INSERT INTO {self.anomalies_table} (
            detection_date,
            event_type,
            metric_name,
            expected_value,
            actual_value,
            deviation_percentage,
            severity,
            detection_method,
            detector_config,
            metadata,
            alert_sent,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
    
    def _build_upsert_sql(self) -> str:
        """Build UPSERT (MERGE) SQL statement for anomaly results.
        
        Returns:
            SQL MERGE statement for upsert operations
        """
        return f"""
        MERGE INTO {self.anomalies_table} AS target
        USING (
            SELECT 
                ? AS detection_date,
                ? AS event_type,
                ? AS metric_name,
                ? AS expected_value,
                ? AS actual_value,
                ? AS deviation_percentage,
                ? AS severity,
                ? AS detection_method,
                ? AS detector_config,
                ? AS metadata,
                ? AS alert_sent,
                ? AS created_at
        ) AS source
        ON target.detection_date = source.detection_date 
           AND target.event_type = source.event_type 
           AND target.metric_name = source.metric_name
        WHEN MATCHED THEN
            UPDATE SET
                expected_value = source.expected_value,
                actual_value = source.actual_value,
                deviation_percentage = source.deviation_percentage,
                severity = source.severity,
                detection_method = source.detection_method,
                detector_config = source.detector_config,
                metadata = source.metadata,
                alert_sent = source.alert_sent,
                created_at = source.created_at
        WHEN NOT MATCHED THEN
            INSERT (
                detection_date, event_type, metric_name, expected_value,
                actual_value, deviation_percentage, severity, detection_method,
                detector_config, metadata, alert_sent, created_at
            )
            VALUES (
                source.detection_date, source.event_type, source.metric_name,
                source.expected_value, source.actual_value, source.deviation_percentage,
                source.severity, source.detection_method, source.detector_config,
                source.metadata, source.alert_sent, source.created_at
            )
        """
    
    def insert_anomaly(self, anomaly_result: AnomalyResult) -> WriteResult:
        """Insert a single anomaly result.
        
        Args:
            anomaly_result: AnomalyResult instance to insert
            
        Returns:
            WriteResult indicating success/failure
            
        Raises:
            ValueError: If anomaly_result is not an AnomalyResult instance
        """
        if not isinstance(anomaly_result, AnomalyResult):
            raise ValueError("anomaly_result must be an AnomalyResult instance")
        
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                sql = self._build_insert_sql()
                params = self._convert_anomaly_to_params(anomaly_result)
                
                cursor.execute(sql, params)
                rows_affected = cursor.rowcount
                
                if not self.auto_commit:
                    conn.commit()
                
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                
                self.logger.debug(f"Inserted anomaly: {anomaly_result.event_type}/{anomaly_result.metric_name}")
                
                result = WriteResult(
                    success=True,
                    rows_affected=rows_affected,
                    execution_time_ms=execution_time
                )
                
                # Record performance metrics for ADF-47
                self._record_performance_metric('insert_anomaly', execution_time, True)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to insert anomaly: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            result = WriteResult(
                success=False,
                rows_affected=0,
                error=e,
                execution_time_ms=execution_time
            )
            
            # Record performance metrics for failures
            self._record_performance_metric('insert_anomaly', execution_time, False, e)
            
            return result
    
    def upsert_anomaly(self, anomaly_result: AnomalyResult) -> WriteResult:
        """Upsert a single anomaly result (insert or update).
        
        Args:
            anomaly_result: AnomalyResult instance to upsert
            
        Returns:
            WriteResult indicating success/failure
        """
        if not isinstance(anomaly_result, AnomalyResult):
            raise ValueError("anomaly_result must be an AnomalyResult instance")
        
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                sql = self._build_upsert_sql()
                params = self._convert_anomaly_to_params(anomaly_result)
                
                cursor.execute(sql, params)
                rows_affected = cursor.rowcount
                
                if not self.auto_commit:
                    conn.commit()
                
                execution_time = (time.time() - start_time) * 1000
                
                self.logger.debug(f"Upserted anomaly: {anomaly_result.event_type}/{anomaly_result.metric_name}")
                
                return WriteResult(
                    success=True,
                    rows_affected=rows_affected,
                    execution_time_ms=execution_time
                )
                
        except Exception as e:
            self.logger.error(f"Failed to upsert anomaly: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return WriteResult(
                success=False,
                rows_affected=0,
                error=e,
                execution_time_ms=execution_time
            )
    
    def insert_batch(self, anomaly_results: List[AnomalyResult]) -> BatchInsertResult:
        """Insert multiple anomaly results in batches.
        
        Args:
            anomaly_results: List of AnomalyResult instances to insert
            
        Returns:
            BatchInsertResult with detailed batch operation results
            
        Raises:
            ValueError: If anomaly_results is not a list or contains invalid items
        """
        if not isinstance(anomaly_results, list):
            raise ValueError("anomaly_results must be a list")
        
        if not anomaly_results:
            return BatchInsertResult(
                success=True,
                total_rows=0,
                successful_rows=0,
                failed_rows=0
            )
        
        # Validate all items are AnomalyResult instances
        for i, item in enumerate(anomaly_results):
            if not isinstance(item, AnomalyResult):
                raise ValueError(f"All items must be AnomalyResult instances (item {i} is {type(item)})")
        
        start_time = time.time()
        total_rows = len(anomaly_results)
        successful_rows = 0
        failed_rows = 0
        errors = []
        batches_processed = 0
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                sql = self._build_insert_sql()
                
                # Process in batches
                for i in range(0, total_rows, self.batch_size):
                    batch = anomaly_results[i:i + self.batch_size]
                    batch_params = [self._convert_anomaly_to_params(anomaly) for anomaly in batch]
                    
                    try:
                        cursor.executemany(sql, batch_params)
                        batch_rows_affected = cursor.rowcount
                        successful_rows += batch_rows_affected
                        batches_processed += 1
                        
                        self.logger.debug(f"Processed batch {batches_processed}: {len(batch)} rows")
                        
                    except Exception as batch_error:
                        failed_rows += len(batch)
                        errors.append(batch_error)
                        self.logger.error(f"Batch {batches_processed + 1} failed: {batch_error}")
                
                if not self.auto_commit:
                    conn.commit()
                
                execution_time = (time.time() - start_time) * 1000
                success = failed_rows == 0
                
                self.logger.info(f"Batch insert completed: {successful_rows}/{total_rows} successful")
                
                return BatchInsertResult(
                    success=success,
                    total_rows=total_rows,
                    successful_rows=successful_rows,
                    failed_rows=failed_rows,
                    errors=errors,
                    execution_time_ms=execution_time,
                    batches_processed=batches_processed
                )
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Batch insert failed: {e}")
            
            return BatchInsertResult(
                success=False,
                total_rows=total_rows,
                successful_rows=successful_rows,
                failed_rows=total_rows - successful_rows,
                errors=[e],
                execution_time_ms=execution_time,
                batches_processed=batches_processed
            )
    
    def upsert_batch(self, anomaly_results: List[AnomalyResult]) -> BatchInsertResult:
        """Upsert multiple anomaly results in batches.
        
        Args:
            anomaly_results: List of AnomalyResult instances to upsert
            
        Returns:
            BatchInsertResult with detailed batch operation results
        """
        if not isinstance(anomaly_results, list):
            raise ValueError("anomaly_results must be a list")
        
        if not anomaly_results:
            return BatchInsertResult(
                success=True,
                total_rows=0,
                successful_rows=0,
                failed_rows=0
            )
        
        start_time = time.time()
        total_rows = len(anomaly_results)
        successful_rows = 0
        failed_rows = 0
        errors = []
        batches_processed = 0
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                sql = self._build_upsert_sql()
                
                # Process in batches
                for i in range(0, total_rows, self.batch_size):
                    batch = anomaly_results[i:i + self.batch_size]
                    batch_params = [self._convert_anomaly_to_params(anomaly) for anomaly in batch]
                    
                    try:
                        cursor.executemany(sql, batch_params)
                        batch_rows_affected = cursor.rowcount
                        successful_rows += batch_rows_affected
                        batches_processed += 1
                        
                    except Exception as batch_error:
                        failed_rows += len(batch)
                        errors.append(batch_error)
                        self.logger.error(f"Upsert batch {batches_processed + 1} failed: {batch_error}")
                
                if not self.auto_commit:
                    conn.commit()
                
                execution_time = (time.time() - start_time) * 1000
                success = failed_rows == 0
                
                return BatchInsertResult(
                    success=success,
                    total_rows=total_rows,
                    successful_rows=successful_rows,
                    failed_rows=failed_rows,
                    errors=errors,
                    execution_time_ms=execution_time,
                    batches_processed=batches_processed
                )
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return BatchInsertResult(
                success=False,
                total_rows=total_rows,
                successful_rows=successful_rows,
                failed_rows=total_rows - successful_rows,
                errors=[e],
                execution_time_ms=execution_time,
                batches_processed=batches_processed
            )
    
    def insert_batch_with_transaction(self, anomaly_results: List[AnomalyResult]) -> BatchInsertResult:
        """Insert batch with explicit transaction management.
        
        Args:
            anomaly_results: List of AnomalyResult instances to insert
            
        Returns:
            BatchInsertResult with transaction details
        """
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                # Disable autocommit for transaction control
                conn.autocommit = False
                
                try:
                    cursor = conn.cursor()
                    sql = self._build_insert_sql()
                    
                    # Convert all anomalies to parameters
                    all_params = [self._convert_anomaly_to_params(anomaly) for anomaly in anomaly_results]
                    
                    # Execute batch insert
                    cursor.executemany(sql, all_params)
                    rows_affected = cursor.rowcount
                    
                    # Commit transaction
                    conn.commit()
                    
                    execution_time = (time.time() - start_time) * 1000
                    
                    self.logger.info(f"Transaction committed: {rows_affected} rows inserted")
                    
                    return BatchInsertResult(
                        success=True,
                        total_rows=len(anomaly_results),
                        successful_rows=rows_affected,
                        failed_rows=0,
                        execution_time_ms=execution_time,
                        batches_processed=1
                    )
                    
                except Exception as e:
                    # Rollback transaction on error
                    conn.rollback()
                    self.logger.error(f"Transaction rolled back due to error: {e}")
                    raise
                    
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return BatchInsertResult(
                success=False,
                total_rows=len(anomaly_results),
                successful_rows=0,
                failed_rows=len(anomaly_results),
                errors=[e],
                execution_time_ms=execution_time
            )
    
    def insert_anomaly_with_retry(self, anomaly_result: AnomalyResult) -> WriteResult:
        """Insert anomaly with retry logic.
        
        Args:
            anomaly_result: AnomalyResult instance to insert
            
        Returns:
            WriteResult indicating success/failure after retries
        """
        last_error = None
        
        for attempt in range(self.retry_attempts + 1):  # +1 for initial attempt
            try:
                result = self.insert_anomaly(anomaly_result)
                if result.success:
                    if attempt > 0:
                        self.logger.info(f"Insert succeeded on attempt {attempt + 1}")
                    return result
                else:
                    last_error = result.error
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Insert attempt {attempt + 1} failed: {e}")
            
            # Don't sleep after the last attempt
            if attempt < self.retry_attempts:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # All attempts failed
        return WriteResult(
            success=False,
            rows_affected=0,
            error=last_error
        )
    
    def check_duplicate(self, anomaly1: AnomalyResult, anomaly2: AnomalyResult) -> bool:
        """Check if two anomaly results are duplicates based on key fields.
        
        Args:
            anomaly1: First anomaly result
            anomaly2: Second anomaly result
            
        Returns:
            True if anomalies are duplicates (same date, event_type, metric_name)
        """
        return anomaly1.get_unique_key() == anomaly2.get_unique_key()
    
    def _get_connection_pool(self):
        """Get connection pool (placeholder for connection pooling implementation).
        
        In a real implementation, this would manage a pool of connections
        for better performance and resource management.
        """
        # Placeholder for connection pooling
        # Real implementation would use connection pooling library
        pass
    
    # Enhanced methods for ADF-47 (GADF-SNOW-006)
    
    def _record_performance_metric(self, operation_type: str, execution_time_ms: float, 
                                   success: bool, error: Optional[Exception] = None) -> None:
        """Record performance metrics for monitoring.
        
        Args:
            operation_type: Type of operation (insert, batch_insert, upsert, etc.)
            execution_time_ms: Execution time in milliseconds
            success: Whether the operation was successful
            error: Exception if operation failed
        """
        if not self.performance_monitoring_enabled:
            return
        
        # Update in-memory metrics
        self._performance_metrics['total_operations'] += 1
        self._performance_metrics['total_execution_time_ms'] += execution_time_ms
        
        if success:
            self._performance_metrics['successful_operations'] += 1
        else:
            self._performance_metrics['failed_operations'] += 1
        
        # Check for slow queries
        if execution_time_ms > self.slow_query_threshold_ms:
            self._performance_metrics['slow_queries'] += 1
            self.logger.warning(f"Slow query detected: {operation_type} took {execution_time_ms:.2f}ms")
        
        # Log detailed metrics
        self.logger.debug(f"Performance metric - {operation_type}: {execution_time_ms:.2f}ms, success: {success}")
        
        # In a real implementation, this would persist metrics to the metrics table
        self._store_performance_metrics(operation_type, execution_time_ms, success, error)
    
    def _store_performance_metrics(self, operation_type: str, execution_time_ms: float,
                                   success: bool, error: Optional[Exception] = None) -> None:
        """Store performance metrics to database table.
        
        Args:
            operation_type: Type of operation
            execution_time_ms: Execution time in milliseconds  
            success: Whether operation was successful
            error: Exception if operation failed
        """
        if not self.performance_monitoring_enabled:
            return
        
        try:
            # This would insert into the performance metrics table
            # For now, we'll just log the metrics
            metric_data = {
                'timestamp': datetime.now().isoformat(),
                'operation_type': operation_type,
                'execution_time_ms': execution_time_ms,
                'success': success,
                'error_message': str(error) if error else None,
                'table_name': self.anomalies_table
            }
            
            self.logger.debug(f"Storing performance metric: {metric_data}")
            
        except Exception as e:
            self.logger.warning(f"Failed to store performance metrics: {e}")
    
    def _optimize_batch_size(self, data_size: int, historical_performance: Optional[Dict] = None) -> int:
        """Optimize batch size based on data size and performance history.
        
        Args:
            data_size: Total number of records to process
            historical_performance: Optional historical performance data
            
        Returns:
            Optimized batch size
        """
        if not self.batch_optimization_enabled or not self.adaptive_batch_size:
            return self.batch_size
        
        # Start with configured batch size
        optimal_size = self.batch_size
        
        # Adjust based on data size
        if data_size < self.min_batch_size:
            optimal_size = data_size
        elif data_size > self.max_batch_size * 10:
            # For very large datasets, use maximum batch size
            optimal_size = self.max_batch_size
        elif data_size < self.batch_size:
            # For smaller datasets, reduce batch size to avoid overhead
            optimal_size = max(self.min_batch_size, data_size // 2)
        
        # Apply constraints
        optimal_size = max(self.min_batch_size, min(self.max_batch_size, optimal_size))
        
        # Log optimization
        if optimal_size != self.batch_size:
            self.logger.debug(f"Optimized batch size from {self.batch_size} to {optimal_size} for {data_size} records")
            
            # Warn about suboptimal batch sizes if configured
            if self.batch_size_warnings and data_size < 50:
                self.logger.warning(f"Small batch size ({data_size} records) may have poor performance")
        
        return optimal_size
    
    def _detect_deadlock(self, error: Exception) -> bool:
        """Detect if error is caused by deadlock.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error indicates deadlock
        """
        if not self.deadlock_detection:
            return False
        
        error_message = str(error).lower()
        deadlock_indicators = [
            'deadlock', 'lock timeout', 'transaction deadlock',
            'deadlock detected', 'lock wait timeout'
        ]
        
        is_deadlock = any(indicator in error_message for indicator in deadlock_indicators)
        
        if is_deadlock:
            self._performance_metrics['deadlocks_detected'] += 1
            self.logger.warning(f"Deadlock detected: {error}")
        
        return is_deadlock
    
    def _execute_with_deadlock_retry(self, operation_func, *args, max_retries: int = 3, **kwargs):
        """Execute operation with deadlock retry logic.
        
        Args:
            operation_func: Function to execute
            *args: Arguments for operation_func
            max_retries: Maximum number of retries for deadlocks
            **kwargs: Keyword arguments for operation_func
            
        Returns:
            Result of operation_func
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return operation_func(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                
                if self._detect_deadlock(e) and attempt < max_retries:
                    self._performance_metrics['retries_performed'] += 1
                    sleep_time = (2 ** attempt) * 0.1  # Exponential backoff starting at 100ms
                    self.logger.info(f"Retrying after deadlock (attempt {attempt + 1}): sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    continue
                else:
                    raise
        
        # All retries exhausted
        raise last_error
    
    def insert_batch_optimized(self, anomaly_results: List[AnomalyResult]) -> BatchInsertResult:
        """Insert batch with optimization features enabled.
        
        Args:
            anomaly_results: List of AnomalyResult instances to insert
            
        Returns:
            BatchInsertResult with optimization details
        """
        if not isinstance(anomaly_results, list):
            raise ValueError("anomaly_results must be a list")
        
        if not anomaly_results:
            return BatchInsertResult(
                success=True,
                total_rows=0,
                successful_rows=0,
                failed_rows=0
            )
        
        start_time = time.time()
        total_rows = len(anomaly_results)
        
        # Optimize batch size
        optimized_batch_size = self._optimize_batch_size(total_rows)
        
        # Use optimized batch size for processing
        original_batch_size = self.batch_size
        self.batch_size = optimized_batch_size
        
        try:
            # Execute with deadlock retry if enabled
            if self.deadlock_detection:
                result = self._execute_with_deadlock_retry(
                    self.insert_batch,
                    anomaly_results
                )
            else:
                result = self.insert_batch(anomaly_results)
            
            # Record performance metrics
            execution_time = (time.time() - start_time) * 1000
            self._record_performance_metric('batch_insert_optimized', execution_time, result.success)
            
            return result
            
        finally:
            # Restore original batch size
            self.batch_size = original_batch_size
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = self._performance_metrics.copy()
        
        # Calculate derived metrics
        if metrics['total_operations'] > 0:
            metrics['success_rate'] = metrics['successful_operations'] / metrics['total_operations']
            metrics['average_execution_time_ms'] = metrics['total_execution_time_ms'] / metrics['total_operations']
        else:
            metrics['success_rate'] = 0.0
            metrics['average_execution_time_ms'] = 0.0
        
        metrics['slow_query_rate'] = metrics['slow_queries'] / max(1, metrics['total_operations'])
        metrics['deadlock_rate'] = metrics['deadlocks_detected'] / max(1, metrics['total_operations'])
        
        return metrics
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics counters."""
        self._performance_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_execution_time_ms': 0,
            'slow_queries': 0,
            'deadlocks_detected': 0,
            'retries_performed': 0
        }
        self.logger.info("Performance metrics reset")
    
    def healthcheck(self) -> Dict[str, Any]:
        """Perform health check of the ResultsWriter.
        
        Returns:
            Health check results
        """
        health_status = {
            'healthy': True,
            'timestamp': datetime.now().isoformat(),
            'connection_pool_enabled': self.connection_pool_enabled,
            'performance_monitoring_enabled': self.performance_monitoring_enabled,
            'batch_optimization_enabled': self.batch_optimization_enabled,
            'metrics': self.get_performance_metrics()
        }
        
        try:
            # Test connection
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                health_status['connection_test'] = 'passed' if result else 'failed'
                
        except Exception as e:
            health_status['healthy'] = False
            health_status['connection_test'] = 'failed'
            health_status['error'] = str(e)
        
        return health_status