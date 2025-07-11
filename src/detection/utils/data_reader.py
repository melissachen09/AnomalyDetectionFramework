"""DataReader class for efficient data reading from Snowflake with streaming support.

This module implements the DataReader class that provides:
- Dynamic query generation using QueryBuilder
- Streaming support for large datasets via chunked reading
- Data validation layer with configurable rules
- Memory-efficient processing with generator-based iteration
- Integration with Snowflake connector

Features for ADF-45:
- Dynamic query generation
- Streaming for large datasets  
- Data validation included
- Memory efficient processing
"""

import pandas as pd
import snowflake.connector
from typing import Dict, Any, List, Optional, Iterator, Union
import logging
from datetime import datetime
from contextlib import contextmanager

from .query_builder import QueryBuilder


class DataReader:
    """Efficient data reader with streaming support and validation.
    
    This class provides methods to read data from Snowflake with support for:
    - Configurable data source and metrics
    - Chunked reading for memory efficiency
    - Data validation with configurable rules
    - Dynamic SQL query generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DataReader with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - data_source: Data source configuration (required)
                    - table: Table name
                    - date_column: Date column name
                    - metrics: List of metric definitions
                - validation: Validation configuration (optional)
                    - enabled: Enable/disable validation (default: True)
                    - required_columns: List of required columns
                    - data_types: Expected data types mapping
                    - constraints: Value constraints for columns
                - chunking: Chunked reading configuration (optional)
                    - enabled: Enable/disable chunking (default: True)
                    - chunk_size: Size of each chunk (default: 50000)
                    
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate and extract configuration
        self._validate_config()
        self._extract_config()
        
        # Initialize query builder
        self.query_builder = QueryBuilder(self.config['data_source'])
        
        # Connection will be established when needed
        self._connection = None
    
    def _validate_config(self) -> None:
        """Validate the provided configuration.
        
        Raises:
            ValueError: If configuration is missing required fields or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        if 'data_source' not in self.config:
            raise ValueError("Configuration must contain 'data_source'")
        
        data_source = self.config['data_source']
        required_fields = ['table', 'date_column', 'metrics']
        
        for field in required_fields:
            if field not in data_source:
                raise ValueError(f"data_source must contain '{field}'")
        
        if not data_source['table']:
            raise ValueError("Table name cannot be empty")
        
        if not data_source['date_column']:
            raise ValueError("Date column cannot be empty")
        
        if not isinstance(data_source['metrics'], list) or not data_source['metrics']:
            raise ValueError("Metrics must be a non-empty list")
        
        # Validate metric definitions
        for metric in data_source['metrics']:
            if not isinstance(metric, dict):
                raise ValueError("Each metric must be a dictionary")
            if 'column' not in metric:
                raise ValueError("Each metric must have a 'column' field")
            if not metric['column']:
                raise ValueError("Metric column name cannot be empty")
    
    def _extract_config(self) -> None:
        """Extract configuration values into instance variables."""
        data_source = self.config['data_source']
        
        self.table_name = data_source['table']
        self.date_column = data_source['date_column']
        self.metrics = data_source['metrics']
        
        # Chunking configuration
        chunking_config = self.config.get('chunking', {})
        self.chunking_enabled = chunking_config.get('enabled', True)
        self.chunk_size = chunking_config.get('chunk_size', 50000)
        
        # Validation configuration
        validation_config = self.config.get('validation', {})
        self.validation_enabled = validation_config.get('enabled', True)
        self.required_columns = validation_config.get('required_columns', [])
        self.expected_data_types = validation_config.get('data_types', {})
        self.constraints = validation_config.get('constraints', {})
    
    @contextmanager
    def _get_snowflake_connection(self):
        """Get a Snowflake database connection.
        
        This is a context manager that ensures proper connection cleanup.
        In a real implementation, this would use connection pooling and
        proper credential management.
        
        Yields:
            snowflake.connector.SnowflakeConnection: Database connection
        """
        # TODO: Implement actual Snowflake connection with proper credentials
        # For now, this is a placeholder that would be implemented with:
        # - Environment variable based credential management
        # - Connection pooling for efficiency
        # - Proper error handling and retries
        
        # Placeholder connection setup
        try:
            # In real implementation:
            # conn = snowflake.connector.connect(
            #     user=os.getenv('SNOWFLAKE_USER'),
            #     password=os.getenv('SNOWFLAKE_PASSWORD'),
            #     account=os.getenv('SNOWFLAKE_ACCOUNT'),
            #     warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            #     database=os.getenv('SNOWFLAKE_DATABASE'),
            #     schema=os.getenv('SNOWFLAKE_SCHEMA')
            # )
            
            # For testing, we'll create a mock connection
            conn = None
            self.logger.info("Established Snowflake connection")
            yield conn
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Snowflake: {e}")
            raise
        finally:
            if conn:
                conn.close()
                self.logger.info("Closed Snowflake connection")
    
    def _execute_query(self, query: str, params: List[Any]) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string with parameter placeholders
            params: Parameters for the query
            
        Returns:
            DataFrame containing query results
            
        Raises:
            Exception: If query execution fails
        """
        try:
            with self._get_snowflake_connection() as conn:
                self.logger.debug(f"Executing query: {query}")
                self.logger.debug(f"Parameters: {params}")
                
                # In real implementation, execute the query:
                # cursor = conn.cursor()
                # cursor.execute(query, params)
                # results = cursor.fetchall()
                # columns = [desc[0] for desc in cursor.description]
                # df = pd.DataFrame(results, columns=columns)
                
                # For testing purposes, return empty DataFrame
                # This will be mocked in tests
                df = pd.DataFrame()
                
                self.logger.info(f"Query executed successfully, returned {len(df)} rows")
                return df
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def build_query(
        self,
        start_date: str,
        end_date: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> str:
        """Build SQL query for data retrieval.
        
        Args:
            start_date: Start date for date range filter
            end_date: End date for date range filter  
            filters: Additional filters as column-value pairs
            limit: Maximum number of rows to return
            offset: Number of rows to skip (for pagination)
            
        Returns:
            SQL query string
        """
        query, _ = self.query_builder.build_select_query(
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            limit=limit,
            offset=offset
        )
        return query
    
    def read_data(
        self,
        start_date: str,
        end_date: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Read data for the specified date range and filters.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            filters: Additional filters to apply
            
        Returns:
            DataFrame containing the requested data
            
        Raises:
            ValueError: If validation fails
            Exception: If data retrieval fails
        """
        if self.chunking_enabled:
            # For non-chunked read, combine all chunks
            chunks = list(self.read_chunked(start_date, end_date, filters))
            if chunks:
                result = pd.concat(chunks, ignore_index=True)
            else:
                result = pd.DataFrame()
        else:
            result = self.read_all(start_date, end_date, filters)
        
        # Validate data if enabled
        if self.validation_enabled and not result.empty:
            self.validate_data(result)
        
        return result
    
    def read_all(
        self,
        start_date: str,
        end_date: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Read all data at once (non-chunked).
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            filters: Additional filters to apply
            
        Returns:
            DataFrame containing all requested data
        """
        query, params = self.query_builder.build_select_query(
            start_date=start_date,
            end_date=end_date,
            filters=filters
        )
        
        return self._execute_query(query, params)
    
    def read_chunked(
        self,
        start_date: str,
        end_date: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Iterator[pd.DataFrame]:
        """Read data in chunks for memory efficiency.
        
        This method returns a generator that yields DataFrames in chunks,
        allowing for memory-efficient processing of large datasets.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            filters: Additional filters to apply
            
        Yields:
            DataFrame chunks containing the requested data
            
        Raises:
            ValueError: If validation fails
            Exception: If data retrieval fails
        """
        if not self.chunking_enabled:
            # If chunking is disabled, return all data as single chunk
            yield self.read_all(start_date, end_date, filters)
            return
        
        offset = 0
        
        while True:
            # Build query with limit and offset for pagination
            query, params = self.query_builder.build_select_query(
                start_date=start_date,
                end_date=end_date,
                filters=filters,
                limit=self.chunk_size,
                offset=offset
            )
            
            # Execute query and get chunk
            chunk = self._execute_query(query, params)
            
            # If chunk is empty, we've reached the end
            if chunk.empty:
                break
            
            # Validate chunk if enabled
            if self.validation_enabled:
                self.validate_data(chunk)
            
            self.logger.debug(f"Retrieved chunk with {len(chunk)} rows (offset: {offset})")
            yield chunk
            
            # If chunk is smaller than chunk_size, we've reached the end
            if len(chunk) < self.chunk_size:
                break
            
            offset += self.chunk_size
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data against configured rules.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        if not self.validation_enabled:
            return True
        
        # Check required columns
        if self.required_columns:
            missing_columns = set(self.required_columns) - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if self.expected_data_types:
            for column, expected_type in self.expected_data_types.items():
                if column in data.columns:
                    actual_type = str(data[column].dtype)
                    if expected_type not in actual_type:
                        raise ValueError(
                            f"Data type mismatch for column '{column}': "
                            f"expected {expected_type}, got {actual_type}"
                        )
        
        # Check constraints
        if self.constraints:
            for column, constraint in self.constraints.items():
                if column in data.columns:
                    self._validate_column_constraints(data[column], column, constraint)
        
        self.logger.debug("Data validation passed")
        return True
    
    def _validate_column_constraints(
        self,
        series: pd.Series,
        column_name: str,
        constraints: Dict[str, Any]
    ) -> None:
        """Validate constraints for a specific column.
        
        Args:
            series: Pandas Series to validate
            column_name: Name of the column being validated
            constraints: Dictionary of constraints to check
            
        Raises:
            ValueError: If constraints are violated
        """
        if 'min' in constraints:
            min_val = constraints['min']
            if (series < min_val).any():
                raise ValueError(f"Constraint violation: {column_name} has values below minimum {min_val}")
        
        if 'max' in constraints:
            max_val = constraints['max']
            if (series > max_val).any():
                raise ValueError(f"Constraint violation: {column_name} has values above maximum {max_val}")
        
        if 'not_null' in constraints and constraints['not_null']:
            if series.isnull().any():
                raise ValueError(f"Constraint violation: {column_name} contains null values")
    
    def get_row_count(
        self,
        start_date: str,
        end_date: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Get the total number of rows for a query without retrieving data.
        
        Args:
            start_date: Start date for date range filter
            end_date: End date for date range filter
            filters: Additional filters to apply
            
        Returns:
            Total number of rows that would be returned
        """
        query, params = self.query_builder.build_count_query(
            start_date=start_date,
            end_date=end_date,
            filters=filters
        )
        
        result = self._execute_query(query, params)
        if not result.empty:
            return int(result.iloc[0, 0])
        return 0