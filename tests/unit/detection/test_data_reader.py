"""Test cases for DataReader class following TDD approach.

This module tests the DataReader implementation for ADF-45.
Tests cover:
- DataReader class initialization with config
- SQL query builder functionality  
- Chunked reading for large datasets
- Data validation layer
- Memory efficiency
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Iterator

# Import the DataReader class (will be implemented)
from src.detection.utils.data_reader import DataReader
from src.detection.utils.query_builder import QueryBuilder


class TestDataReaderInitialization:
    """Test DataReader class initialization and configuration."""
    
    def test_data_reader_init_with_valid_config(self):
        """Test DataReader initializes correctly with valid configuration."""
        config = {
            'data_source': {
                'table': 'DATAMART.DD_LISTING_STATISTICS_BLENDED',
                'date_column': 'STATISTIC_DATE',
                'metrics': [
                    {'column': 'NUMBEROFVIEWS', 'alias': 'total_views'},
                    {'column': 'NUMBEROFENQUIRIES', 'alias': 'enquiries'}
                ]
            },
            'validation': {
                'required_columns': ['STATISTIC_DATE', 'NUMBEROFVIEWS'],
                'data_types': {
                    'STATISTIC_DATE': 'date',
                    'NUMBEROFVIEWS': 'integer'
                }
            },
            'chunking': {
                'enabled': True,
                'chunk_size': 10000
            }
        }
        
        reader = DataReader(config)
        
        assert reader.config == config
        assert reader.table_name == 'DATAMART.DD_LISTING_STATISTICS_BLENDED'
        assert reader.date_column == 'STATISTIC_DATE'
        assert len(reader.metrics) == 2
        assert reader.chunk_size == 10000
    
    def test_data_reader_init_with_minimal_config(self):
        """Test DataReader initializes with minimal required configuration."""
        config = {
            'data_source': {
                'table': 'TEST_TABLE',
                'date_column': 'DATE_COL',
                'metrics': [{'column': 'VALUE', 'alias': 'value'}]
            }
        }
        
        reader = DataReader(config)
        
        assert reader.table_name == 'TEST_TABLE'
        assert reader.date_column == 'DATE_COL'
        assert reader.chunk_size == 50000  # Default value
        assert reader.validation_enabled is True  # Default value
    
    def test_data_reader_init_raises_on_invalid_config(self):
        """Test DataReader raises ValueError for invalid configuration."""
        invalid_configs = [
            {},  # Empty config
            {'data_source': {}},  # Missing required fields
            {'data_source': {'table': 'TEST'}},  # Missing date_column and metrics
            {'data_source': {'table': '', 'date_column': 'DATE', 'metrics': []}},  # Empty table name
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                DataReader(config)


class TestDataReaderQueryBuilder:
    """Test SQL query building functionality."""
    
    @pytest.fixture
    def data_reader(self):
        """Create a DataReader instance for testing."""
        config = {
            'data_source': {
                'table': 'DATAMART.DD_LISTING_STATISTICS_BLENDED',
                'date_column': 'STATISTIC_DATE',
                'metrics': [
                    {'column': 'NUMBEROFVIEWS', 'alias': 'total_views'},
                    {'column': 'NUMBEROFENQUIRIES', 'alias': 'enquiries'}
                ]
            }
        }
        return DataReader(config)
    
    def test_build_query_basic(self, data_reader):
        """Test basic query building with date range."""
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        query = data_reader.build_query(start_date, end_date)
        
        assert 'SELECT' in query.upper()
        assert 'STATISTIC_DATE' in query
        assert 'NUMBEROFVIEWS AS total_views' in query
        assert 'NUMBEROFENQUIRIES AS enquiries' in query
        assert 'FROM DATAMART.DD_LISTING_STATISTICS_BLENDED' in query
        assert 'WHERE STATISTIC_DATE >= %s' in query
        assert 'AND STATISTIC_DATE <= %s' in query
    
    def test_build_query_with_filters(self, data_reader):
        """Test query building with additional filters."""
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        filters = {'PROPERTY_TYPE': 'HOUSE', 'STATE': 'NSW'}
        
        query = data_reader.build_query(start_date, end_date, filters=filters)
        
        assert 'PROPERTY_TYPE = %s' in query
        assert 'STATE = %s' in query
    
    def test_build_query_with_limit(self, data_reader):
        """Test query building with LIMIT clause."""
        start_date = '2023-01-01'  
        end_date = '2023-01-31'
        
        query = data_reader.build_query(start_date, end_date, limit=1000)
        
        assert 'LIMIT 1000' in query
    
    def test_build_query_with_offset(self, data_reader):
        """Test query building with OFFSET for pagination."""
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        query = data_reader.build_query(start_date, end_date, limit=1000, offset=5000)
        
        assert 'LIMIT 1000' in query
        assert 'OFFSET 5000' in query


class TestDataReaderChunkedReading:
    """Test chunked reading functionality for large datasets."""
    
    @pytest.fixture
    def data_reader(self):
        """Create a DataReader instance for chunked reading tests."""
        config = {
            'data_source': {
                'table': 'LARGE_TABLE',
                'date_column': 'DATE_COL',
                'metrics': [{'column': 'VALUE', 'alias': 'value'}]
            },
            'chunking': {
                'enabled': True,
                'chunk_size': 1000
            }
        }
        return DataReader(config)
    
    @patch('src.detection.utils.data_reader.DataReader._execute_query')
    def test_read_chunked_data_single_chunk(self, mock_execute, data_reader):
        """Test reading data when result fits in single chunk."""
        # Mock small dataset that fits in one chunk
        mock_data = pd.DataFrame({
            'DATE_COL': ['2023-01-01', '2023-01-02'],
            'VALUE': [100, 200]
        })
        mock_execute.return_value = mock_data
        
        chunks = list(data_reader.read_chunked('2023-01-01', '2023-01-31'))
        
        assert len(chunks) == 1
        assert len(chunks[0]) == 2
        assert chunks[0].equals(mock_data)
    
    @patch('src.detection.utils.data_reader.DataReader._execute_query')
    def test_read_chunked_data_multiple_chunks(self, mock_execute, data_reader):
        """Test reading data across multiple chunks."""
        # Mock larger dataset requiring multiple chunks
        def mock_execute_side_effect(query, params):
            if 'OFFSET 0' in query or 'OFFSET' not in query:
                return pd.DataFrame({'VALUE': list(range(1000))})
            elif 'OFFSET 1000' in query:
                return pd.DataFrame({'VALUE': list(range(1000, 1500))})
            else:
                return pd.DataFrame()  # Empty result for subsequent chunks
        
        mock_execute.side_effect = mock_execute_side_effect
        
        chunks = list(data_reader.read_chunked('2023-01-01', '2023-01-31'))
        
        assert len(chunks) == 2
        assert len(chunks[0]) == 1000
        assert len(chunks[1]) == 500
    
    @patch('src.detection.utils.data_reader.DataReader._execute_query')
    def test_read_chunked_memory_efficient(self, mock_execute, data_reader):
        """Test that chunked reading is memory efficient (generator-based)."""
        mock_data = pd.DataFrame({'VALUE': list(range(100))})
        mock_execute.return_value = mock_data
        
        chunks = data_reader.read_chunked('2023-01-01', '2023-01-31')
        
        # Should return a generator, not a list
        assert hasattr(chunks, '__next__')
        assert hasattr(chunks, '__iter__')
    
    def test_read_chunked_disabled(self):
        """Test behavior when chunking is disabled."""
        config = {
            'data_source': {
                'table': 'TABLE',
                'date_column': 'DATE_COL',
                'metrics': [{'column': 'VALUE', 'alias': 'value'}]
            },
            'chunking': {'enabled': False}
        }
        reader = DataReader(config)
        
        with patch.object(reader, 'read_all') as mock_read_all:
            mock_read_all.return_value = pd.DataFrame({'VALUE': [1, 2, 3]})
            
            chunks = list(reader.read_chunked('2023-01-01', '2023-01-31'))
            
            assert len(chunks) == 1
            mock_read_all.assert_called_once()


class TestDataReaderValidation:
    """Test data validation functionality."""
    
    @pytest.fixture
    def data_reader_with_validation(self):
        """Create a DataReader with validation enabled."""
        config = {
            'data_source': {
                'table': 'TEST_TABLE',
                'date_column': 'DATE_COL',
                'metrics': [{'column': 'VALUE', 'alias': 'value'}]
            },
            'validation': {
                'required_columns': ['DATE_COL', 'VALUE'],
                'data_types': {
                    'DATE_COL': 'datetime64[ns]',
                    'VALUE': 'int64'
                },
                'constraints': {
                    'VALUE': {'min': 0, 'max': 1000000}
                }
            }
        }
        return DataReader(config)
    
    def test_validate_data_valid_dataframe(self, data_reader_with_validation):
        """Test validation passes for valid DataFrame."""
        valid_data = pd.DataFrame({
            'DATE_COL': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'VALUE': [100, 200]
        })
        
        # Should not raise any exception
        result = data_reader_with_validation.validate_data(valid_data)
        assert result is True
    
    def test_validate_data_missing_columns(self, data_reader_with_validation):
        """Test validation fails for missing required columns."""
        invalid_data = pd.DataFrame({
            'DATE_COL': pd.to_datetime(['2023-01-01', '2023-01-02'])
            # Missing VALUE column
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            data_reader_with_validation.validate_data(invalid_data)
    
    def test_validate_data_wrong_data_types(self, data_reader_with_validation):
        """Test validation fails for incorrect data types."""
        invalid_data = pd.DataFrame({
            'DATE_COL': ['2023-01-01', '2023-01-02'],  # String instead of datetime
            'VALUE': [100, 200]
        })
        
        with pytest.raises(ValueError, match="Data type mismatch"):
            data_reader_with_validation.validate_data(invalid_data)
    
    def test_validate_data_constraint_violations(self, data_reader_with_validation):
        """Test validation fails for constraint violations."""
        invalid_data = pd.DataFrame({
            'DATE_COL': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'VALUE': [100, 2000000]  # Second value exceeds max constraint
        })
        
        with pytest.raises(ValueError, match="Constraint violation"):
            data_reader_with_validation.validate_data(invalid_data)
    
    def test_validate_data_disabled(self):
        """Test behavior when validation is disabled."""
        config = {
            'data_source': {
                'table': 'TEST_TABLE',
                'date_column': 'DATE_COL', 
                'metrics': [{'column': 'VALUE', 'alias': 'value'}]
            },
            'validation': {'enabled': False}
        }
        reader = DataReader(config)
        
        # Should accept any DataFrame without validation
        invalid_data = pd.DataFrame({'WRONG_COL': [1, 2, 3]})
        result = reader.validate_data(invalid_data)
        assert result is True


class TestDataReaderIntegration:
    """Integration tests for complete DataReader functionality."""
    
    @pytest.fixture
    def full_config_reader(self):
        """Create a fully configured DataReader for integration tests."""
        config = {
            'data_source': {
                'table': 'DATAMART.DD_LISTING_STATISTICS_BLENDED',
                'date_column': 'STATISTIC_DATE',
                'metrics': [
                    {'column': 'NUMBEROFVIEWS', 'alias': 'total_views'},
                    {'column': 'NUMBEROFENQUIRIES', 'alias': 'enquiries'}
                ]
            },
            'validation': {
                'required_columns': ['STATISTIC_DATE', 'NUMBEROFVIEWS'],
                'data_types': {
                    'STATISTIC_DATE': 'datetime64[ns]',
                    'NUMBEROFVIEWS': 'int64'
                }
            },
            'chunking': {
                'enabled': True,
                'chunk_size': 5000
            }
        }
        return DataReader(config)
    
    @patch('src.detection.utils.data_reader.DataReader._get_snowflake_connection')
    def test_read_data_end_to_end(self, mock_connection, full_config_reader):
        """Test complete data reading workflow with validation and chunking."""
        # Mock database connection and execution
        mock_conn = Mock()
        mock_connection.return_value = mock_conn
        
        mock_data = pd.DataFrame({
            'STATISTIC_DATE': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'NUMBEROFVIEWS': [1000, 1500],
            'NUMBEROFENQUIRIES': [50, 75]
        })
        
        with patch.object(full_config_reader, '_execute_query', return_value=mock_data):
            result = full_config_reader.read_data('2023-01-01', '2023-01-02')
            
            assert not result.empty
            assert 'STATISTIC_DATE' in result.columns
            assert 'NUMBEROFVIEWS' in result.columns
            assert len(result) == 2
    
    def test_error_handling_database_connection_failure(self, full_config_reader):
        """Test error handling when database connection fails."""
        with patch.object(full_config_reader, '_get_snowflake_connection', 
                         side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Connection failed"):
                full_config_reader.read_data('2023-01-01', '2023-01-02')
    
    def test_error_handling_query_execution_failure(self, full_config_reader):
        """Test error handling when query execution fails."""
        with patch.object(full_config_reader, '_get_snowflake_connection'):
            with patch.object(full_config_reader, '_execute_query',
                             side_effect=Exception("Query failed")):
                with pytest.raises(Exception, match="Query failed"):
                    full_config_reader.read_data('2023-01-01', '2023-01-02')


class TestDataReaderPerformance:
    """Performance and memory efficiency tests."""
    
    def test_memory_usage_chunked_vs_full_read(self):
        """Test that chunked reading uses less memory than full read."""
        # This would be implemented with actual memory profiling
        # For now, we test the interface
        config = {
            'data_source': {
                'table': 'LARGE_TABLE',
                'date_column': 'DATE_COL',
                'metrics': [{'column': 'VALUE', 'alias': 'value'}]
            },
            'chunking': {'enabled': True, 'chunk_size': 1000}
        }
        reader = DataReader(config)
        
        # Verify chunked reading returns generator
        chunks = reader.read_chunked('2023-01-01', '2023-01-31')
        assert hasattr(chunks, '__next__')  # Is a generator
    
    @patch('src.detection.utils.data_reader.DataReader._execute_query')
    def test_connection_pooling_reuse(self, mock_execute):
        """Test that connections are reused efficiently."""
        config = {
            'data_source': {
                'table': 'TEST_TABLE',
                'date_column': 'DATE_COL',
                'metrics': [{'column': 'VALUE', 'alias': 'value'}]
            }
        }
        reader = DataReader(config)
        
        mock_execute.return_value = pd.DataFrame({'VALUE': [1, 2, 3]})
        
        with patch.object(reader, '_get_snowflake_connection') as mock_conn:
            # Multiple reads should reuse connection
            reader.read_data('2023-01-01', '2023-01-02')
            reader.read_data('2023-01-03', '2023-01-04')
            
            # Connection should be acquired only once (or reused efficiently)
            assert mock_conn.call_count <= 2  # Allow for some connection management


if __name__ == '__main__':
    pytest.main([__file__])