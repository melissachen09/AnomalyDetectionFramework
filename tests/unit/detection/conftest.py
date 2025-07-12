"""
Pytest configuration for detection unit tests.
"""

import pytest
import sys
import os

# Add src directory to Python path for imports
src_path = os.path.join(os.path.dirname(__file__), '../../../src')
sys.path.insert(0, os.path.abspath(src_path))

print(f"DEBUG: Added to Python path: {os.path.abspath(src_path)}")
print(f"DEBUG: Python path: {sys.path[:3]}")

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # Reduce log noise during tests