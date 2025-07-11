"""Unit tests for the main detection module."""

import pytest
from unittest.mock import patch, MagicMock
from detection.main import main


class TestMain:
    """Test cases for the main detection function."""
    
    def test_main_success(self):
        """Test successful execution of main function."""
        result = main("test_event")
        assert result == 0
    
    def test_main_with_none_event_type(self):
        """Test main function with None event type."""
        result = main(None)
        assert result == 0
    
    @patch('detection.main.logger')
    def test_main_logs_info(self, mock_logger):
        """Test that main function logs appropriate information."""
        main("test_event")
        mock_logger.info.assert_called()
    
    @patch('detection.main.logger')
    def test_main_handles_exception(self, mock_logger):
        """Test main function handles exceptions gracefully."""
        # Mock logger to raise exception
        mock_logger.info.side_effect = Exception("Test exception")
        
        result = main("test_event")
        assert result == 1
        mock_logger.error.assert_called()