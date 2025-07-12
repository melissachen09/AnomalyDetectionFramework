"""Test cases for QueryBuilder utility class.

This module tests the SQL query building functionality for ADF-45.
Tests cover:
- Dynamic SQL query generation
- Parameter binding and SQL injection prevention
- Query optimization and structure
- Complex query scenarios
"""

import pytest
from typing import Dict, Any, List, Optional

# Import the QueryBuilder class (will be implemented)
from src.detection.utils.query_builder import QueryBuilder


class TestQueryBuilderInitialization:
    """Test QueryBuilder initialization and basic setup."""
    
    def test_query_builder_init(self):
        """Test QueryBuilder initializes correctly."""
        builder = QueryBuilder()
        assert builder is not None
        assert hasattr(builder, 'build_select_query')
        assert hasattr(builder, 'build_count_query')
    
    def test_query_builder_with_table_config(self):
        """Test QueryBuilder with table configuration."""
        table_config = {
            'table': 'DATAMART.DD_LISTING_STATISTICS_BLENDED',
            'date_column': 'STATISTIC_DATE',
            'metrics': [
                {'column': 'NUMBEROFVIEWS', 'alias': 'total_views'},
                {'column': 'NUMBEROFENQUIRIES', 'alias': 'enquiries'}
            ]
        }
        
        builder = QueryBuilder(table_config)
        assert builder.table_name == 'DATAMART.DD_LISTING_STATISTICS_BLENDED'
        assert builder.date_column == 'STATISTIC_DATE'
        assert len(builder.metrics) == 2


class TestQueryBuilderSelectQueries:
    """Test SELECT query building functionality."""
    
    @pytest.fixture
    def query_builder(self):
        """Create a QueryBuilder instance for testing."""
        table_config = {
            'table': 'TEST_TABLE',
            'date_column': 'DATE_COL',
            'metrics': [
                {'column': 'VALUE1', 'alias': 'value_one'},
                {'column': 'VALUE2', 'alias': 'value_two'}
            ]
        }
        return QueryBuilder(table_config)
    
    def test_build_basic_select_query(self, query_builder):
        """Test building basic SELECT query with date range."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        expected_structure = [
            'SELECT',
            'DATE_COL',
            'VALUE1 AS value_one',
            'VALUE2 AS value_two',
            'FROM TEST_TABLE',
            'WHERE DATE_COL >= %s',
            'AND DATE_COL <= %s',
            'ORDER BY DATE_COL'
        ]
        
        for element in expected_structure:
            assert element in query
        
        assert params == ['2023-01-01', '2023-01-31']
    
    def test_build_select_query_with_filters(self, query_builder):
        """Test building SELECT query with additional filters."""
        filters = {
            'REGION': 'NSW',
            'PROPERTY_TYPE': 'HOUSE'
        }
        
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters=filters
        )
        
        assert 'REGION = %s' in query
        assert 'PROPERTY_TYPE = %s' in query
        assert params == ['2023-01-01', '2023-01-31', 'NSW', 'HOUSE']
    
    def test_build_select_query_with_limit(self, query_builder):
        """Test building SELECT query with LIMIT clause."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            limit=1000
        )
        
        assert 'LIMIT 1000' in query
    
    def test_build_select_query_with_offset(self, query_builder):
        """Test building SELECT query with OFFSET for pagination."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            limit=1000,
            offset=5000
        )
        
        assert 'LIMIT 1000' in query
        assert 'OFFSET 5000' in query
    
    def test_build_select_query_custom_columns(self, query_builder):
        """Test building SELECT query with custom column selection."""
        custom_columns = ['DATE_COL', 'VALUE1', 'CUSTOM_FIELD']
        
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            columns=custom_columns
        )
        
        assert 'SELECT DATE_COL, VALUE1, CUSTOM_FIELD' in query
    
    def test_build_select_query_with_aggregation(self, query_builder):
        """Test building SELECT query with aggregation functions."""
        aggregations = {
            'VALUE1': 'SUM',
            'VALUE2': 'AVG'
        }
        
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            aggregations=aggregations,
            group_by=['DATE_COL']
        )
        
        assert 'SUM(VALUE1)' in query
        assert 'AVG(VALUE2)' in query
        assert 'GROUP BY DATE_COL' in query


class TestQueryBuilderCountQueries:
    """Test COUNT query building functionality."""
    
    @pytest.fixture  
    def query_builder(self):
        """Create a QueryBuilder instance for testing."""
        table_config = {
            'table': 'TEST_TABLE',
            'date_column': 'DATE_COL',
            'metrics': [{'column': 'VALUE', 'alias': 'value'}]
        }
        return QueryBuilder(table_config)
    
    def test_build_count_query_basic(self, query_builder):
        """Test building basic COUNT query."""
        query, params = query_builder.build_count_query(
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        assert 'SELECT COUNT(*)' in query
        assert 'FROM TEST_TABLE' in query
        assert 'WHERE DATE_COL >= %s' in query
        assert 'AND DATE_COL <= %s' in query
        assert params == ['2023-01-01', '2023-01-31']
    
    def test_build_count_query_with_filters(self, query_builder):
        """Test building COUNT query with filters."""
        filters = {'STATUS': 'ACTIVE'}
        
        query, params = query_builder.build_count_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters=filters
        )
        
        assert 'STATUS = %s' in query
        assert params == ['2023-01-01', '2023-01-31', 'ACTIVE']


class TestQueryBuilderSQLInjectionPrevention:
    """Test SQL injection prevention and security."""
    
    @pytest.fixture
    def query_builder(self):
        """Create a QueryBuilder instance for security testing."""
        table_config = {
            'table': 'SECURE_TABLE',
            'date_column': 'DATE_COL',
            'metrics': [{'column': 'VALUE', 'alias': 'value'}]
        }
        return QueryBuilder(table_config)
    
    def test_parameterized_queries_prevent_injection(self, query_builder):
        """Test that all user inputs are properly parameterized."""
        malicious_input = "'; DROP TABLE users; --"
        
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date=malicious_input
        )
        
        # Query should use parameterized binding
        assert 'DROP TABLE' not in query
        assert '%s' in query  # Parameter placeholder should be present
        assert malicious_input in params  # Malicious input should be in params, not query
    
    def test_filter_values_are_parameterized(self, query_builder):
        """Test that filter values are properly parameterized."""
        malicious_filters = {
            'COLUMN1': "'; DELETE FROM table; --",
            'COLUMN2': "UNION SELECT * FROM sensitive_table"
        }
        
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters=malicious_filters
        )
        
        # SQL injection attempts should not appear in the query structure
        assert 'DELETE FROM' not in query
        assert 'UNION SELECT' not in query
        assert 'sensitive_table' not in query
        
        # Values should be parameterized
        assert malicious_filters['COLUMN1'] in params
        assert malicious_filters['COLUMN2'] in params
    
    def test_table_names_are_validated(self):
        """Test that table names are validated and cannot be injected."""
        # Should raise an error for invalid table names
        invalid_table_configs = [
            {'table': 'TABLE; DROP TABLE users;', 'date_column': 'DATE', 'metrics': []},
            {'table': '', 'date_column': 'DATE', 'metrics': []},
            {'table': None, 'date_column': 'DATE', 'metrics': []}
        ]
        
        for config in invalid_table_configs:
            with pytest.raises(ValueError):
                QueryBuilder(config)


class TestQueryBuilderValidation:
    """Test query validation and error handling."""
    
    def test_invalid_date_range_raises_error(self):
        """Test that invalid date ranges raise appropriate errors."""
        table_config = {
            'table': 'TEST_TABLE',
            'date_column': 'DATE_COL',
            'metrics': [{'column': 'VALUE', 'alias': 'value'}]
        }
        builder = QueryBuilder(table_config)
        
        # End date before start date should raise error
        with pytest.raises(ValueError, match="End date must be after start date"):
            builder.build_select_query(
                start_date='2023-01-31',
                end_date='2023-01-01'
            )
    
    def test_missing_required_parameters_raises_error(self):
        """Test that missing required parameters raise errors."""
        table_config = {
            'table': 'TEST_TABLE',
            'date_column': 'DATE_COL',
            'metrics': [{'column': 'VALUE', 'alias': 'value'}]
        }
        builder = QueryBuilder(table_config)
        
        # Missing start_date should raise error
        with pytest.raises(ValueError, match="start_date is required"):
            builder.build_select_query(end_date='2023-01-31')
        
        # Missing end_date should raise error  
        with pytest.raises(ValueError, match="end_date is required"):
            builder.build_select_query(start_date='2023-01-01')
    
    def test_invalid_limit_offset_raises_error(self):
        """Test that invalid limit/offset values raise errors."""
        table_config = {
            'table': 'TEST_TABLE',
            'date_column': 'DATE_COL',
            'metrics': [{'column': 'VALUE', 'alias': 'value'}]
        }
        builder = QueryBuilder(table_config)
        
        # Negative limit should raise error
        with pytest.raises(ValueError, match="Limit must be positive"):
            builder.build_select_query(
                start_date='2023-01-01',
                end_date='2023-01-31',
                limit=-1
            )
        
        # Negative offset should raise error
        with pytest.raises(ValueError, match="Offset must be non-negative"):
            builder.build_select_query(
                start_date='2023-01-01',
                end_date='2023-01-31',
                offset=-1
            )


class TestQueryBuilderOptimization:
    """Test query optimization features."""
    
    @pytest.fixture
    def query_builder(self):
        """Create a QueryBuilder instance for optimization testing."""
        table_config = {
            'table': 'LARGE_TABLE',
            'date_column': 'DATE_COL',
            'metrics': [
                {'column': 'VALUE1', 'alias': 'value_one'},
                {'column': 'VALUE2', 'alias': 'value_two'}
            ]
        }
        return QueryBuilder(table_config)
    
    def test_query_includes_performance_hints(self, query_builder):
        """Test that queries include performance optimization hints."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        # Should include proper indexing hints for date range queries
        assert 'ORDER BY DATE_COL' in query  # Helps with index usage
    
    def test_efficient_column_selection(self, query_builder):
        """Test that only necessary columns are selected."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        # Should not use SELECT * for efficiency
        assert 'SELECT *' not in query
        assert 'VALUE1 AS value_one' in query
        assert 'VALUE2 AS value_two' in query
    
    def test_pagination_optimization(self, query_builder):
        """Test that pagination is implemented efficiently."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            limit=1000,
            offset=5000
        )
        
        # Should use LIMIT and OFFSET for efficient pagination
        assert 'LIMIT 1000' in query
        assert 'OFFSET 5000' in query
        # ORDER BY should come before LIMIT/OFFSET for consistent results
        order_by_pos = query.find('ORDER BY')
        limit_pos = query.find('LIMIT')
        assert order_by_pos < limit_pos


class TestQueryBuilderComplexScenarios:
    """Test complex query building scenarios."""
    
    @pytest.fixture
    def complex_query_builder(self):
        """Create a QueryBuilder for complex scenario testing."""
        table_config = {
            'table': 'DATAMART.DD_LISTING_STATISTICS_BLENDED',
            'date_column': 'STATISTIC_DATE',
            'metrics': [
                {'column': 'NUMBEROFVIEWS', 'alias': 'total_views'},
                {'column': 'NUMBEROFENQUIRIES', 'alias': 'enquiries'},
                {'column': 'NUMBEROFSHORTLISTS', 'alias': 'shortlists'}
            ]
        }
        return QueryBuilder(table_config)
    
    def test_complex_filter_combinations(self, complex_query_builder):
        """Test complex filter combinations."""
        complex_filters = {
            'STATE': 'NSW',
            'PROPERTY_TYPE': 'HOUSE',
            'PRICE_RANGE': 'HIGH',
            'BEDROOMS': 4
        }
        
        query, params = complex_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-12-31',
            filters=complex_filters,
            limit=10000
        )
        
        # All filters should be properly included
        for filter_name in complex_filters.keys():
            assert f'{filter_name} = %s' in query
        
        # Parameters should be in correct order
        expected_params = ['2023-01-01', '2023-12-31'] + list(complex_filters.values())
        assert params == expected_params
    
    def test_aggregation_with_grouping(self, complex_query_builder):
        """Test aggregation queries with grouping."""
        query, params = complex_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            aggregations={
                'NUMBEROFVIEWS': 'SUM',
                'NUMBEROFENQUIRIES': 'AVG'
            },
            group_by=['STATISTIC_DATE', 'STATE']
        )
        
        assert 'SUM(NUMBEROFVIEWS)' in query
        assert 'AVG(NUMBEROFENQUIRIES)' in query
        assert 'GROUP BY STATISTIC_DATE, STATE' in query
    
    def test_query_string_formatting(self, complex_query_builder):
        """Test that generated queries are properly formatted and readable."""
        query, params = complex_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters={'STATE': 'NSW'},
            limit=1000
        )
        
        # Query should be well-formatted (not all on one line)
        lines = query.split('\n')
        assert len(lines) > 1  # Multi-line formatting
        
        # Should not have excessive whitespace
        assert '  ' not in query.replace('\n', ' ')  # No double spaces


if __name__ == '__main__':
    pytest.main([__file__])