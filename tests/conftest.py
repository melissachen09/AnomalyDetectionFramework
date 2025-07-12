"""
Pytest configuration for detection module tests.
"""

import pytest
import sys
import os

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # Reduce log noise during tests