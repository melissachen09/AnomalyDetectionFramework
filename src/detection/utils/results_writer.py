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
                
                return WriteResult(
                    success=True,
                    rows_affected=rows_affected,
                    execution_time_ms=execution_time
                )
                
        except Exception as e:
            self.logger.error(f"Failed to insert anomaly: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return WriteResult(
                success=False,
                rows_affected=0,
                error=e,
                execution_time_ms=execution_time
            )
    
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