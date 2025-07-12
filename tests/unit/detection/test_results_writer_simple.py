"""Simple test cases for ResultsWriter to verify basic functionality.

This is a simplified version focusing on core acceptance criteria:
- Insert operations tested
- Batch processing verified
- Transaction handling tested
- Idempotency validated
"""

import pytest
from unittest.mock import Mock, patch
from datetime import date, datetime

# Import the classes we're testing
from src.detection.utils.results_writer import ResultsWriter
from src.detection.utils.models import AnomalyResult, BatchInsertResult, WriteResult


def test_results_writer_init():
    """Test ResultsWriter initialization."""
    config = {
        'connection': {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password'
        }
    }
    
    writer = ResultsWriter(config)
    assert writer.connection_config == config['connection']
    assert writer.batch_size == 1000  # Default


def test_anomaly_result_creation():
    """Test AnomalyResult creation and validation."""
    anomaly = AnomalyResult(
        detection_date=date.today(),
        event_type='test_event',
        metric_name='test_metric',
        expected_value=100.0,
        actual_value=90.0,
        deviation_percentage=-0.1,
        severity='warning',
        detection_method='threshold',
        detector_config={'threshold': 95.0},
        metadata={'source': 'test'}
    )
    
    assert anomaly.event_type == 'test_event'
    assert anomaly.severity == 'warning'
    assert anomaly.get_unique_key() == (date.today(), 'test_event', 'test_metric')


def test_anomaly_result_validation():
    """Test AnomalyResult validation."""
    # Test invalid severity
    with pytest.raises(ValueError, match="Invalid severity"):
        AnomalyResult(
            detection_date=date.today(),
            event_type='test_event',
            metric_name='test_metric',
            expected_value=100.0,
            actual_value=90.0,
            deviation_percentage=-0.1,
            severity='invalid_severity',
            detection_method='threshold',
            detector_config={},
            metadata={}
        )


def test_config_validation():
    """Test configuration validation."""
    # Missing connection config
    with pytest.raises(ValueError, match="Configuration must contain 'connection'"):
        ResultsWriter({})
    
    # Missing required connection fields
    with pytest.raises(ValueError, match="Connection config must contain 'user'"):
        ResultsWriter({
            'connection': {
                'account': 'test_account',
                'password': 'test_password'
                # Missing 'user'
            }
        })


@patch('snowflake.connector.connect')
def test_insert_single_anomaly(mock_connect):
    """Test single anomaly insert."""
    # Setup mocks
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.rowcount = 1
    mock_connect.return_value = mock_conn
    
    # Setup writer
    config = {
        'connection': {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password'
        }
    }
    writer = ResultsWriter(config)
    
    # Create test anomaly
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
    
    # Execute insert
    result = writer.insert_anomaly(anomaly)
    
    # Verify results
    assert result.success is True
    assert result.rows_affected == 1
    assert result.error is None
    
    # Verify SQL execution
    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()


@patch('snowflake.connector.connect')
def test_batch_insert(mock_connect):
    """Test batch insert functionality."""
    # Setup mocks
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.rowcount = 3
    mock_connect.return_value = mock_conn
    
    # Setup writer
    config = {
        'connection': {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password'
        }
    }
    writer = ResultsWriter(config)
    
    # Create test anomalies
    anomalies = []
    for i in range(3):
        anomalies.append(AnomalyResult(
            detection_date=date.today(),
            event_type=f'event_{i}',
            metric_name=f'metric_{i}',
            expected_value=100.0 + i,
            actual_value=90.0 + i,
            deviation_percentage=-0.1,
            severity='warning',
            detection_method='threshold',
            detector_config={},
            metadata={}
        ))
    
    # Execute batch insert
    result = writer.insert_batch(anomalies)
    
    # Verify results
    assert result.success is True
    assert result.total_rows == 3
    assert result.successful_rows == 3
    assert result.failed_rows == 0
    
    # Verify SQL execution
    mock_cursor.executemany.assert_called_once()
    mock_conn.commit.assert_called_once()


@patch('snowflake.connector.connect')
def test_transaction_handling(mock_connect):
    """Test transaction commit and rollback."""
    # Setup mocks
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.rowcount = 1
    mock_connect.return_value = mock_conn
    
    # Setup writer
    config = {
        'connection': {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password'
        }
    }
    writer = ResultsWriter(config)
    
    # Create test anomaly
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
    
    # Test successful transaction
    result = writer.insert_batch_with_transaction([anomaly])
    
    # Verify transaction handling
    assert result.success is True
    mock_conn.commit.assert_called_once()
    mock_conn.rollback.assert_not_called()


@patch('snowflake.connector.connect')
def test_upsert_functionality(mock_connect):
    """Test upsert (insert or update) functionality."""
    # Setup mocks
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.rowcount = 1
    mock_connect.return_value = mock_conn
    
    # Setup writer
    config = {
        'connection': {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password'
        }
    }
    writer = ResultsWriter(config)
    
    # Create test anomaly
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
    
    # Execute upsert
    result = writer.upsert_anomaly(anomaly)
    
    # Verify results
    assert result.success is True
    assert result.rows_affected == 1
    
    # Verify MERGE SQL is used
    mock_cursor.execute.assert_called_once()
    sql = mock_cursor.execute.call_args[0][0]
    assert 'MERGE' in sql.upper()


def test_duplicate_detection():
    """Test duplicate detection logic."""
    config = {
        'connection': {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password'
        }
    }
    writer = ResultsWriter(config)
    
    # Create identical anomalies (same key fields)
    anomaly1 = AnomalyResult(
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
    
    anomaly2 = AnomalyResult(
        detection_date=date.today(),  # Same date
        event_type='test_event',      # Same event
        metric_name='test_metric',    # Same metric
        expected_value=110.0,         # Different value (not part of key)
        actual_value=95.0,
        deviation_percentage=-0.05,
        severity='high',
        detection_method='statistical',
        detector_config={},
        metadata={}
    )
    
    # Should be detected as duplicates
    assert writer.check_duplicate(anomaly1, anomaly2) is True
    
    # Create different anomaly (different key fields)
    anomaly3 = AnomalyResult(
        detection_date=date.today(),
        event_type='different_event',  # Different event type
        metric_name='test_metric',
        expected_value=100.0,
        actual_value=90.0,
        deviation_percentage=-0.1,
        severity='warning',
        detection_method='threshold',
        detector_config={},
        metadata={}
    )
    
    # Should not be duplicates
    assert writer.check_duplicate(anomaly1, anomaly3) is False


def test_data_type_conversion():
    """Test data type conversion for SQL parameters."""
    config = {
        'connection': {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password'
        }
    }
    writer = ResultsWriter(config)
    
    # Create anomaly with various data types
    anomaly = AnomalyResult(
        detection_date=date(2024, 1, 15),
        event_type='test_event',
        metric_name='test_metric',
        expected_value=123.456,
        actual_value=None,  # None value
        deviation_percentage=-0.1,
        severity='warning',
        detection_method='threshold',
        detector_config={'key': 'value'},  # Dict to be JSON serialized
        metadata=None  # None value
    )
    
    # Convert to SQL parameters
    params = writer._convert_anomaly_to_params(anomaly)
    
    # Verify conversions
    assert '2024-01-15' in params[0]  # Date converted to ISO string
    assert params[1] == 'test_event'  # String unchanged
    assert params[3] == 123.456       # Float unchanged
    assert params[4] is None          # None preserved
    assert '"key": "value"' in params[8]  # Dict converted to JSON
    assert params[9] is None          # None preserved


if __name__ == '__main__':
    pytest.main([__file__, '-v'])