"""QueryBuilder utility for dynamic SQL query generation.

This module provides SQL query building functionality for the DataReader class.
Implements safe, parameterized query construction with SQL injection prevention.

Features:
- Dynamic SELECT query building
- Parameterized query binding
- SQL injection prevention
- Query optimization hints
- Flexible filtering and pagination
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import re
from datetime import datetime


class QueryBuilder:
    """Utility class for building safe, parameterized SQL queries.
    
    This class provides methods to construct SQL queries dynamically while
    preventing SQL injection through proper parameterization.
    """
    
    def __init__(self, table_config: Optional[Dict[str, Any]] = None):
        """Initialize QueryBuilder with optional table configuration.
        
        Args:
            table_config: Dictionary containing table configuration:
                - table: Table name (required)
                - date_column: Date column name (required)
                - metrics: List of metric column definitions (required)
        
        Raises:
            ValueError: If table_config is provided but invalid
        """
        self.table_name = None
        self.date_column = None
        self.metrics = []
        
        if table_config:
            self._validate_table_config(table_config)
            self.table_name = table_config['table']
            self.date_column = table_config['date_column']
            self.metrics = table_config['metrics']
    
    def _validate_table_config(self, config: Dict[str, Any]) -> None:
        """Validate table configuration.
        
        Args:
            config: Table configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['table', 'date_column', 'metrics']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in table config: {field}")
        
        if not config['table'] or not isinstance(config['table'], str):
            raise ValueError("Table name must be a non-empty string")
        
        if not config['date_column'] or not isinstance(config['date_column'], str):
            raise ValueError("Date column must be a non-empty string")
        
        if not isinstance(config['metrics'], list):
            raise ValueError("Metrics must be a list")
        
        # Validate table name format (basic SQL injection prevention)
        if not self._is_valid_identifier(config['table']):
            raise ValueError(f"Invalid table name format: {config['table']}")
        
        if not self._is_valid_identifier(config['date_column']):
            raise ValueError(f"Invalid date column format: {config['date_column']}")
    
    def _is_valid_identifier(self, identifier: str) -> bool:
        """Check if an identifier is safe for SQL queries.
        
        Args:
            identifier: SQL identifier to validate
            
        Returns:
            True if identifier is safe, False otherwise
        """
        # Allow alphanumeric, underscores, dots (for schema.table), and hyphens
        pattern = r'^[A-Za-z0-9_.-]+$'
        return bool(re.match(pattern, identifier)) and len(identifier) > 0
    
    def build_select_query(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        aggregations: Optional[Dict[str, str]] = None,
        group_by: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> Tuple[str, List[Any]]:
        """Build a SELECT query with the specified parameters.
        
        Args:
            start_date: Start date for date range filter
            end_date: End date for date range filter
            filters: Additional filters as column-value pairs
            columns: Custom columns to select (overrides default metrics)
            aggregations: Aggregation functions for columns
            group_by: Columns to group by
            limit: Maximum number of rows to return
            offset: Number of rows to skip (for pagination)
            
        Returns:
            Tuple of (query_string, parameters_list)
            
        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_query_parameters(start_date, end_date, limit, offset)
        
        params = []
        
        # Build SELECT clause
        select_clause = self._build_select_clause(columns, aggregations)
        
        # Build FROM clause
        from_clause = f"FROM {self.table_name}"
        
        # Build WHERE clause
        where_clause, where_params = self._build_where_clause(
            start_date, end_date, filters
        )
        params.extend(where_params)
        
        # Build GROUP BY clause
        group_by_clause = self._build_group_by_clause(group_by)
        
        # Build ORDER BY clause
        order_by_clause = self._build_order_by_clause(group_by)
        
        # Build LIMIT/OFFSET clause
        limit_clause = self._build_limit_clause(limit, offset)
        
        # Combine all clauses
        query_parts = [
            select_clause,
            from_clause,
            where_clause,
            group_by_clause,
            order_by_clause,
            limit_clause
        ]
        
        query = "\n".join(part for part in query_parts if part)
        
        return query, params
    
    def build_count_query(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[Any]]:
        """Build a COUNT query to get row count.
        
        Args:
            start_date: Start date for date range filter
            end_date: End date for date range filter
            filters: Additional filters as column-value pairs
            
        Returns:
            Tuple of (query_string, parameters_list)
            
        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_query_parameters(start_date, end_date)
        
        params = []
        
        # Build COUNT query
        select_clause = "SELECT COUNT(*)"
        from_clause = f"FROM {self.table_name}"
        where_clause, where_params = self._build_where_clause(
            start_date, end_date, filters
        )
        params.extend(where_params)
        
        query_parts = [select_clause, from_clause, where_clause]
        query = "\n".join(part for part in query_parts if part)
        
        return query, params
    
    def _validate_query_parameters(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> None:
        """Validate query parameters.
        
        Args:
            start_date: Start date string
            end_date: End date string
            limit: Row limit
            offset: Row offset
            
        Raises:
            ValueError: If parameters are invalid
        """
        if start_date is None:
            raise ValueError("start_date is required")
        
        if end_date is None:
            raise ValueError("end_date is required")
        
        # Validate date order (basic check)
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            if end_dt <= start_dt:
                raise ValueError("End date must be after start date")
        except ValueError as e:
            if "End date must be after start date" in str(e):
                raise
            # If date parsing fails, let the database handle the validation
            pass
        
        if limit is not None and limit <= 0:
            raise ValueError("Limit must be positive")
        
        if offset is not None and offset < 0:
            raise ValueError("Offset must be non-negative")
    
    def _build_select_clause(
        self,
        columns: Optional[List[str]] = None,
        aggregations: Optional[Dict[str, str]] = None
    ) -> str:
        """Build the SELECT clause of the query.
        
        Args:
            columns: Custom columns to select
            aggregations: Aggregation functions for columns
            
        Returns:
            SELECT clause string
        """
        if columns:
            # Use custom columns
            select_items = columns
        elif aggregations:
            # Use aggregations
            select_items = []
            for column, func in aggregations.items():
                select_items.append(f"{func}({column})")
        else:
            # Use default metrics configuration
            select_items = [self.date_column] if self.date_column else []
            for metric in self.metrics:
                column = metric['column']
                alias = metric.get('alias', column)
                if alias != column:
                    select_items.append(f"{column} AS {alias}")
                else:
                    select_items.append(column)
        
        return f"SELECT {', '.join(select_items)}"
    
    def _build_where_clause(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        filters: Optional[Dict[str, Any]]
    ) -> Tuple[str, List[Any]]:
        """Build the WHERE clause of the query.
        
        Args:
            start_date: Start date for range filter
            end_date: End date for range filter
            filters: Additional filters
            
        Returns:
            Tuple of (WHERE clause string, parameters list)
        """
        conditions = []
        params = []
        
        # Add date range conditions
        if start_date and self.date_column:
            conditions.append(f"{self.date_column} >= %s")
            params.append(start_date)
        
        if end_date and self.date_column:
            conditions.append(f"{self.date_column} <= %s")
            params.append(end_date)
        
        # Add filter conditions
        if filters:
            for column, value in filters.items():
                # Validate column name to prevent injection
                if not self._is_valid_identifier(column):
                    raise ValueError(f"Invalid column name: {column}")
                conditions.append(f"{column} = %s")
                params.append(value)
        
        if conditions:
            return f"WHERE {' AND '.join(conditions)}", params
        else:
            return "", params
    
    def _build_group_by_clause(self, group_by: Optional[List[str]]) -> str:
        """Build the GROUP BY clause.
        
        Args:
            group_by: Columns to group by
            
        Returns:
            GROUP BY clause string or empty string
        """
        if group_by:
            # Validate column names
            for column in group_by:
                if not self._is_valid_identifier(column):
                    raise ValueError(f"Invalid column name in GROUP BY: {column}")
            return f"GROUP BY {', '.join(group_by)}"
        return ""
    
    def _build_order_by_clause(self, group_by: Optional[List[str]]) -> str:
        """Build the ORDER BY clause.
        
        Args:
            group_by: Columns being grouped by (affects ordering)
            
        Returns:
            ORDER BY clause string
        """
        if group_by:
            # When grouping, order by the group columns
            return f"ORDER BY {', '.join(group_by)}"
        elif self.date_column:
            # Default to ordering by date column for consistent results
            return f"ORDER BY {self.date_column}"
        else:
            return ""
    
    def _build_limit_clause(self, limit: Optional[int], offset: Optional[int]) -> str:
        """Build the LIMIT/OFFSET clause.
        
        Args:
            limit: Maximum number of rows
            offset: Number of rows to skip
            
        Returns:
            LIMIT/OFFSET clause string or empty string
        """
        if limit is not None:
            clause = f"LIMIT {limit}"
            if offset is not None:
                clause += f" OFFSET {offset}"
            return clause
        return ""