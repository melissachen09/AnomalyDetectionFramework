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
    
    def test_column_names_are_validated_in_filters(self, query_builder):
        """Test that column names in filters are validated."""
        malicious_filters = {
            'COLUMN1; DROP TABLE users;': 'value1',
            'COLUMN2 OR 1=1': 'value2',
        }
        
        # Each malicious column name should raise an error
        for malicious_column, value in malicious_filters.items():
            with pytest.raises(ValueError, match="Invalid column name"):
                query_builder.build_select_query(
                    start_date='2023-01-01',
                    end_date='2023-01-31',
                    filters={malicious_column: value}
                )
    
    def test_column_names_are_validated_in_group_by(self, query_builder):
        """Test that column names in GROUP BY are validated."""
        malicious_group_by = [
            'COLUMN1; DROP TABLE users;',
            'COLUMN2 OR 1=1',
        ]
        
        for malicious_column in malicious_group_by:
            with pytest.raises(ValueError, match="Invalid column name in GROUP BY"):
                query_builder.build_select_query(
                    start_date='2023-01-01',
                    end_date='2023-01-31',
                    group_by=[malicious_column]
                )
    
    def test_complex_sql_injection_attempts(self, query_builder):
        """Test protection against complex SQL injection patterns."""
        complex_injection_attempts = [
            "' UNION ALL SELECT user, password FROM users WHERE 'a'='a",
            "'; INSERT INTO logs VALUES ('hacked'); --",
            "' OR EXISTS(SELECT * FROM sensitive_data) --",
            "'; EXEC xp_cmdshell('format c:'); --",
            "' AND (SELECT COUNT(*) FROM information_schema.tables) > 0 --"
        ]
        
        for injection_attempt in complex_injection_attempts:
            # Should be safely parameterized in date values
            query, params = query_builder.build_select_query(
                start_date='2023-01-01',
                end_date=injection_attempt
            )
            
            # Injection patterns should not appear in the query structure
            assert 'UNION' not in query.upper() or query.upper().count('UNION') == 0
            assert 'INSERT' not in query.upper()
            assert 'EXEC' not in query.upper()
            assert 'xp_cmdshell' not in query.lower()
            assert injection_attempt in params
    
    def test_unicode_and_special_characters_in_parameters(self, query_builder):
        """Test handling of unicode and special characters in parameters."""
        special_values = [
            "Müller",  # Unicode characters
            "O'Brien",  # Apostrophe
            "data\nwith\nnewlines",  # Newlines
            "data\twith\ttabs",  # Tabs
            "data with ⚡ emojis 🔒",  # Emojis
            "data with null \x00 bytes",  # Null bytes
        ]
        
        for special_value in special_values:
            query, params = query_builder.build_select_query(
                start_date='2023-01-01',
                end_date='2023-01-31',
                filters={'SPECIAL_COLUMN': special_value}
            )
            
            # Special characters should be in parameters, not in query structure
            assert special_value in params
            assert 'SPECIAL_COLUMN = %s' in query


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


class TestQueryBuilderEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def query_builder(self):
        """Create a QueryBuilder instance for edge case testing."""
        table_config = {
            'table': 'EDGE_CASE_TABLE',
            'date_column': 'DATE_COL',
            'metrics': [
                {'column': 'VALUE1', 'alias': 'value_one'},
                {'column': 'VALUE2', 'alias': 'value_two'}
            ]
        }
        return QueryBuilder(table_config)
    
    def test_empty_filters_dictionary(self, query_builder):
        """Test handling of empty filters dictionary."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters={}
        )
        
        # Should work without filters
        assert 'WHERE DATE_COL >= %s' in query
        assert 'WHERE DATE_COL >= %s AND DATE_COL <= %s' in query
        assert params == ['2023-01-01', '2023-01-31']
    
    def test_none_filters_parameter(self, query_builder):
        """Test handling of None filters parameter."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters=None
        )
        
        # Should work without filters
        assert 'WHERE DATE_COL >= %s' in query
        assert params == ['2023-01-01', '2023-01-31']
    
    def test_empty_group_by_list(self, query_builder):
        """Test handling of empty group by list."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            group_by=[]
        )
        
        # Should not include GROUP BY clause
        assert 'GROUP BY' not in query
        assert 'ORDER BY DATE_COL' in query  # Should still order by date
    
    def test_none_group_by_parameter(self, query_builder):
        """Test handling of None group by parameter."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            group_by=None
        )
        
        # Should not include GROUP BY clause
        assert 'GROUP BY' not in query
        assert 'ORDER BY DATE_COL' in query
    
    def test_empty_columns_list(self, query_builder):
        """Test handling of empty columns list."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            columns=[]
        )
        
        # Should use empty SELECT clause (edge case)
        assert 'SELECT ' in query
    
    def test_empty_aggregations_dictionary(self, query_builder):
        """Test handling of empty aggregations dictionary."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            aggregations={}
        )
        
        # Should fall back to default metrics
        assert 'VALUE1 AS value_one' in query
        assert 'VALUE2 AS value_two' in query
    
    def test_zero_limit(self, query_builder):
        """Test handling of zero limit."""
        with pytest.raises(ValueError, match="Limit must be positive"):
            query_builder.build_select_query(
                start_date='2023-01-01',
                end_date='2023-01-31',
                limit=0
            )
    
    def test_zero_offset(self, query_builder):
        """Test handling of zero offset."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            limit=100,
            offset=0
        )
        
        # Zero offset should be valid
        assert 'LIMIT 100' in query
        assert 'OFFSET 0' in query
    
    def test_very_large_limit(self, query_builder):
        """Test handling of very large limit values."""
        large_limit = 999999999
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            limit=large_limit
        )
        
        assert f'LIMIT {large_limit}' in query
    
    def test_very_large_offset(self, query_builder):
        """Test handling of very large offset values."""
        large_offset = 999999999
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            limit=100,
            offset=large_offset
        )
        
        assert f'OFFSET {large_offset}' in query
    
    def test_single_character_dates(self, query_builder):
        """Test handling of single character date inputs."""
        query, params = query_builder.build_select_query(
            start_date='1',
            end_date='2'
        )
        
        # Should accept any string as date (validation happens at DB level)
        assert params == ['1', '2']
    
    def test_very_long_date_strings(self, query_builder):
        """Test handling of very long date strings."""
        long_date = '2023-01-01T00:00:00.000000+00:00' + 'Z' * 1000
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date=long_date
        )
        
        # Should accept long strings
        assert long_date in params
    
    def test_date_format_variations(self, query_builder):
        """Test various date format inputs."""
        # Test ISO format dates that should work
        iso_dates = [
            '2023-01-01',
            '2023-01-01T00:00:00',
            '2023-01-01 00:00:00',
        ]
        
        for date_format in iso_dates:
            query, params = query_builder.build_select_query(
                start_date=date_format,
                end_date='2023-01-31'
            )
            # Should accept ISO formats
            assert date_format in params
        
        # Test non-ISO formats that may fail validation but should still be parameterized
        non_iso_dates = [
            '2023/01/01',
            '01-01-2023',
            'Jan 1, 2023',
            '1/1/2023'
        ]
        
        for date_format in non_iso_dates:
            try:
                query, params = query_builder.build_select_query(
                    start_date=date_format,
                    end_date='2023-01-31'
                )
                # If it doesn't raise an error, should be parameterized
                assert date_format in params
            except ValueError:
                # Some formats might fail validation, which is acceptable
                pass
    
    def test_filter_with_none_values(self, query_builder):
        """Test handling of None values in filters."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters={'COLUMN1': None, 'COLUMN2': 'value'}
        )
        
        # None values should be included as parameters
        assert None in params
        assert 'value' in params
        assert 'COLUMN1 = %s' in query
        assert 'COLUMN2 = %s' in query
    
    def test_filter_with_empty_string_values(self, query_builder):
        """Test handling of empty string values in filters."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters={'COLUMN1': '', 'COLUMN2': 'value'}
        )
        
        # Empty strings should be valid filter values
        assert '' in params
        assert 'value' in params
        assert 'COLUMN1 = %s' in query
        assert 'COLUMN2 = %s' in query
    
    def test_filter_with_numeric_values(self, query_builder):
        """Test handling of various numeric types in filters."""
        numeric_filters = {
            'INT_COL': 42,
            'FLOAT_COL': 3.14159,
            'NEGATIVE_COL': -100,
            'ZERO_COL': 0,
            'LARGE_COL': 999999999999
        }
        
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters=numeric_filters
        )
        
        # All numeric values should be in parameters
        for value in numeric_filters.values():
            assert value in params
    
    def test_filter_with_boolean_values(self, query_builder):
        """Test handling of boolean values in filters."""
        query, params = query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters={'BOOL_COL': True, 'BOOL_COL2': False}
        )
        
        # Boolean values should be included
        assert True in params
        assert False in params
    
    def test_very_long_table_name(self):
        """Test handling of very long table names."""
        long_table_name = 'SCHEMA.' + 'A' * 1000
        table_config = {
            'table': long_table_name,
            'date_column': 'DATE_COL',
            'metrics': [{'column': 'VALUE', 'alias': 'value'}]
        }
        
        # Should accept long table names (database will enforce limits)
        builder = QueryBuilder(table_config)
        assert builder.table_name == long_table_name
    
    def test_query_builder_without_table_config(self):
        """Test QueryBuilder initialization without table config."""
        builder = QueryBuilder()
        
        # Should initialize with None values
        assert builder.table_name is None
        assert builder.date_column is None
        assert builder.metrics == []
    
    def test_table_config_with_minimal_metrics(self):
        """Test table config with minimal metric configuration."""
        table_config = {
            'table': 'TEST_TABLE',
            'date_column': 'DATE_COL',
            'metrics': [{'column': 'VALUE'}]  # No alias
        }
        
        builder = QueryBuilder(table_config)
        query, params = builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        # Should use column name when no alias provided
        assert 'VALUE' in query
        # Should not include AS clause when no alias
        assert 'VALUE AS' not in query
    
    def test_metric_with_same_column_and_alias(self):
        """Test metric where column and alias are the same."""
        table_config = {
            'table': 'TEST_TABLE',
            'date_column': 'DATE_COL',
            'metrics': [{'column': 'VALUE', 'alias': 'VALUE'}]  # Same name
        }
        
        builder = QueryBuilder(table_config)
        query, params = builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        # Should not include redundant AS clause
        assert 'VALUE AS VALUE' not in query
        assert 'VALUE' in query


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


class TestQueryBuilderPerformanceOptimization:
    """Test advanced performance optimization features."""
    
    @pytest.fixture
    def perf_query_builder(self):
        """Create a QueryBuilder for performance testing."""
        table_config = {
            'table': 'LARGE_PRODUCTION_TABLE',
            'date_column': 'CREATED_DATE',
            'metrics': [
                {'column': 'METRIC_A', 'alias': 'metric_a'},
                {'column': 'METRIC_B', 'alias': 'metric_b'},
                {'column': 'METRIC_C', 'alias': 'metric_c'}
            ]
        }
        return QueryBuilder(table_config)
    
    def test_query_structure_is_database_optimized(self, perf_query_builder):
        """Test that query structure follows database optimization patterns."""
        query, params = perf_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters={'REGION': 'NSW', 'STATUS': 'ACTIVE'},
            limit=1000
        )
        
        query_lines = query.split('\n')
        
        # Verify optimal query structure order
        # 1. SELECT should be first
        assert query_lines[0].startswith('SELECT')
        
        # 2. FROM should come after SELECT
        from_line_index = next(i for i, line in enumerate(query_lines) if 'FROM' in line)
        assert from_line_index > 0
        
        # 3. WHERE should come after FROM
        where_line_index = next(i for i, line in enumerate(query_lines) if 'WHERE' in line)
        assert where_line_index > from_line_index
        
        # 4. ORDER BY should come before LIMIT for index optimization
        order_by_index = next(i for i, line in enumerate(query_lines) if 'ORDER BY' in line)
        limit_index = next(i for i, line in enumerate(query_lines) if 'LIMIT' in line)
        assert order_by_index < limit_index
    
    def test_date_filter_ordering_for_index_usage(self, perf_query_builder):
        """Test that date filters are ordered optimally for index usage."""
        query, params = perf_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        # Date range filters should be structured for index scans
        assert 'CREATED_DATE >= %s' in query
        assert 'CREATED_DATE <= %s' in query
        
        # Start date should come before end date in WHERE clause
        start_pos = query.find('CREATED_DATE >= %s')
        end_pos = query.find('CREATED_DATE <= %s')
        assert start_pos < end_pos
    
    def test_selective_column_projection_optimization(self, perf_query_builder):
        """Test that only required columns are selected for performance."""
        # Test with specific columns
        query, params = perf_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            columns=['CREATED_DATE', 'METRIC_A']
        )
        
        # Should only select specified columns
        assert 'METRIC_A' in query
        assert 'METRIC_B' not in query
        assert 'METRIC_C' not in query
        
        # Should not use SELECT *
        assert 'SELECT *' not in query
    
    def test_limit_prevents_runaway_queries(self, perf_query_builder):
        """Test that LIMIT clause is properly applied to prevent runaway queries."""
        query, params = perf_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-12-31',  # Large date range
            limit=1000
        )
        
        # Should include LIMIT to prevent full table scans
        assert 'LIMIT 1000' in query
        
        # LIMIT should be at the end of the query
        assert query.strip().endswith('LIMIT 1000')
    
    def test_pagination_optimization_with_consistent_ordering(self, perf_query_builder):
        """Test that pagination uses consistent ordering for repeatable results."""
        query, params = perf_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            limit=100,
            offset=500
        )
        
        # Should include ORDER BY for consistent pagination
        assert 'ORDER BY' in query
        
        # ORDER BY should come before LIMIT/OFFSET
        order_by_pos = query.find('ORDER BY')
        limit_pos = query.find('LIMIT')
        offset_pos = query.find('OFFSET')
        
        assert order_by_pos < limit_pos
        assert limit_pos < offset_pos
    
    def test_aggregation_with_appropriate_grouping(self, perf_query_builder):
        """Test that aggregations use proper grouping for performance."""
        query, params = perf_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            aggregations={'METRIC_A': 'SUM', 'METRIC_B': 'AVG'},
            group_by=['CREATED_DATE', 'REGION']
        )
        
        # Should include GROUP BY for aggregations
        assert 'GROUP BY CREATED_DATE, REGION' in query
        
        # Aggregation functions should be properly formatted
        assert 'SUM(METRIC_A)' in query
        assert 'AVG(METRIC_B)' in query
        
        # GROUP BY should come before ORDER BY
        group_by_pos = query.find('GROUP BY')
        order_by_pos = query.find('ORDER BY')
        assert group_by_pos < order_by_pos


class TestQueryBuilderComplexJoinQueries:
    """Test complex query construction capabilities."""
    
    @pytest.fixture
    def join_query_builder(self):
        """Create a QueryBuilder for complex join testing."""
        table_config = {
            'table': 'MAIN_TABLE',
            'date_column': 'DATE_COL',
            'metrics': [
                {'column': 'VALUE1', 'alias': 'value_one'},
                {'column': 'VALUE2', 'alias': 'value_two'}
            ]
        }
        return QueryBuilder(table_config)
    
    def test_complex_multi_column_filters(self, join_query_builder):
        """Test complex filtering scenarios with multiple columns."""
        complex_filters = {
            'REGION': 'NSW',
            'PROPERTY_TYPE': 'HOUSE',
            'BEDROOMS': 4,
            'PRICE_RANGE': 'HIGH',
            'STATUS': 'ACTIVE',
            'LISTING_TYPE': 'SALE',
            'AGENT_ID': 12345,
            'CREATED_BY': 'system',
            'IS_FEATURED': True,
            'HAS_IMAGES': True
        }
        
        query, params = join_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters=complex_filters
        )
        
        # All filters should be present in WHERE clause
        for column_name in complex_filters.keys():
            assert f'{column_name} = %s' in query
        
        # Parameters should be in correct order
        # Date parameters first, then filter parameters
        expected_params = ['2023-01-01', '2023-01-31'] + list(complex_filters.values())
        assert params == expected_params
        
        # Should use AND to combine all conditions
        and_count = query.count(' AND ')
        assert and_count >= len(complex_filters)  # At least one AND per filter + date conditions
    
    def test_complex_aggregation_scenarios(self, join_query_builder):
        """Test complex aggregation and grouping scenarios."""
        query, params = join_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-12-31',
            aggregations={
                'VALUE1': 'SUM',
                'VALUE2': 'AVG'
            },
            group_by=['DATE_COL', 'REGION', 'PROPERTY_TYPE'],
            filters={'STATUS': 'ACTIVE'},
            limit=5000
        )
        
        # Should include all aggregation functions
        assert 'SUM(VALUE1)' in query
        assert 'AVG(VALUE2)' in query
        
        # Should group by all specified columns
        assert 'GROUP BY DATE_COL, REGION, PROPERTY_TYPE' in query
        
        # Should order by the grouped columns for consistency
        assert 'ORDER BY DATE_COL, REGION, PROPERTY_TYPE' in query
        
        # Should include filter
        assert 'STATUS = %s' in query
        assert 'ACTIVE' in params
        
        # Should include limit
        assert 'LIMIT 5000' in query
    
    def test_complex_custom_column_selection(self, join_query_builder):
        """Test complex custom column selection scenarios."""
        custom_columns = [
            'DATE_COL',
            'REGION',
            'PROPERTY_TYPE',
            'VALUE1',
            'VALUE2',
            'CALCULATED_FIELD',
            'DERIVED_METRIC'
        ]
        
        query, params = join_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            columns=custom_columns,
            filters={'REGION': 'NSW'},
            limit=10000
        )
        
        # Should select all custom columns
        for column in custom_columns:
            assert column in query
        
        # Should not include the default metric aliases
        assert 'value_one' not in query
        assert 'value_two' not in query
        
        # Should be formatted as comma-separated list
        select_clause = query.split('\n')[0]
        for i, column in enumerate(custom_columns):
            if i < len(custom_columns) - 1:
                assert f'{column},' in select_clause or f'{column} ' in select_clause
    
    def test_query_with_all_optional_parameters(self, join_query_builder):
        """Test query building with all optional parameters specified."""
        query, params = join_query_builder.build_select_query(
            start_date='2023-01-01',
            end_date='2023-01-31',
            filters={
                'REGION': 'NSW',
                'TYPE': 'PREMIUM'
            },
            columns=['DATE_COL', 'REGION', 'VALUE1'],
            aggregations=None,  # Explicitly None
            group_by=['DATE_COL', 'REGION'],
            limit=1000,
            offset=5000
        )
        
        # Should include custom columns (aggregations=None should be ignored)
        assert 'DATE_COL, REGION, VALUE1' in query
        
        # Should include grouping
        assert 'GROUP BY DATE_COL, REGION' in query
        
        # Should include filters
        assert 'REGION = %s' in query
        assert 'TYPE = %s' in query
        
        # Should include pagination
        assert 'LIMIT 1000' in query
        assert 'OFFSET 5000' in query
        
        # Parameters should be correctly ordered
        expected_params = ['2023-01-01', '2023-01-31', 'NSW', 'PREMIUM']
        assert params == expected_params


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