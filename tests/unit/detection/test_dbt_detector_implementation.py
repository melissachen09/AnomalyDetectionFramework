"""
Test suite specifically for DbtTestDetector implementation.

This module tests the actual DbtTestDetector class implementation
to ensure it correctly implements the expected interface and behavior.

Part of Epic ADF-3: Detection Plugin Architecture
Task: ADF-41 - Implement dbt Test Detector
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess
import pytest

from src.detection.detectors import DbtTestDetector, DetectionResult


class TestDbtTestDetectorImplementation(unittest.TestCase):
    """Test the actual DbtTestDetector implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dbt_project_dir = Path(self.temp_dir) / "dbt_project"
        self.dbt_project_dir.mkdir(parents=True)
        
        # Create mock dbt_project.yml
        dbt_project_yml = self.dbt_project_dir / "dbt_project.yml"
        dbt_project_yml.write_text("""
name: 'test_project'
version: '1.0.0'
profile: 'test_profile'
model-paths: ["models"]
test-paths: ["tests"]
""")
        
        # Create target directory
        target_dir = self.dbt_project_dir / "target"
        target_dir.mkdir(parents=True)
        
        self.valid_config = {
            'name': 'test_dbt_detector',
            'project_dir': str(self.dbt_project_dir),
            'timeout': 300
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_detector_initialization(self):
        """Test DbtTestDetector initialization with valid config."""
        detector = DbtTestDetector(self.valid_config)
        
        self.assertEqual(detector.name, 'test_dbt_detector')
        self.assertEqual(detector.project_dir, Path(self.dbt_project_dir))
        self.assertEqual(detector.timeout, 300)
        self.assertTrue(detector.validate_config())

    def test_detector_initialization_missing_project_dir(self):
        """Test detector initialization fails with missing project directory."""
        invalid_config = {
            'name': 'test_detector',
            'project_dir': '/nonexistent/path'
        }
        
        with self.assertRaises(ValueError) as context:
            DbtTestDetector(invalid_config)
        
        self.assertIn("dbt project directory does not exist", str(context.exception))

    def test_config_validation_success(self):
        """Test configuration validation with valid config."""
        detector = DbtTestDetector(self.valid_config)
        self.assertTrue(detector.validate_config())

    def test_config_validation_missing_project_dir(self):
        """Test configuration validation fails without project_dir."""
        invalid_config = {'name': 'test_detector'}
        
        with self.assertRaises(ValueError):
            DbtTestDetector(invalid_config)

    def test_get_detector_info(self):
        """Test detector info retrieval."""
        detector = DbtTestDetector(self.valid_config)
        info = detector.get_detector_info()
        
        self.assertEqual(info['name'], 'test_dbt_detector')
        self.assertEqual(info['type'], 'DbtTestDetector')
        self.assertEqual(info['project_dir'], str(self.dbt_project_dir))
        self.assertIn('timeout', info)

    @patch('subprocess.run')
    def test_successful_detection_flow(self, mock_run):
        """Test complete detection flow with successful dbt execution."""
        # Setup mock dbt command execution
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Running dbt tests...\nCompleted successfully",
            stderr=""
        )
        
        # Create mock results file
        results_data = {
            "metadata": {
                "dbt_version": "1.6.0",
                "generated_at": "2024-01-01T12:00:00.000000Z"
            },
            "results": [
                {
                    "unique_id": "test.project.test_not_null",
                    "status": "pass",
                    "execution_time": 1.5,
                    "failures": 0,
                    "message": "Test passed successfully",
                    "timing": []
                }
            ]
        }
        
        results_file = self.dbt_project_dir / "target" / "run_results.json"
        results_file.write_text(json.dumps(results_data))
        
        # Execute detection
        detector = DbtTestDetector(self.valid_config)
        results = detector.detect()
        
        # Verify results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.status, 'pass')
        self.assertEqual(result.test_id, 'test.project.test_not_null')
        self.assertEqual(result.execution_time, 1.5)
        self.assertEqual(result.failures, 0)
        
        # Verify dbt command was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "dbt")
        self.assertEqual(call_args[1], "test")
        self.assertIn("--project-dir", call_args)

    @patch('subprocess.run')
    def test_detection_with_failures(self, mock_run):
        """Test detection with failing dbt tests."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Running dbt tests...\nSome tests failed",
            stderr=""
        )
        
        # Create mock results with failures
        results_data = {
            "metadata": {"dbt_version": "1.6.0"},
            "results": [
                {
                    "unique_id": "test.project.test_unique_id",
                    "status": "fail",
                    "execution_time": 2.0,
                    "failures": 3,
                    "message": "Got 3 results, expected 0",
                    "timing": []
                }
            ]
        }
        
        results_file = self.dbt_project_dir / "target" / "run_results.json"
        results_file.write_text(json.dumps(results_data))
        
        detector = DbtTestDetector(self.valid_config)
        results = detector.detect()
        
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.status, 'fail')
        self.assertEqual(result.failures, 3)
        self.assertIn("Got 3 results", result.message)

    @patch('subprocess.run')
    def test_detection_with_test_selection(self, mock_run):
        """Test detection with test selection criteria."""
        config_with_selection = self.valid_config.copy()
        config_with_selection['selection'] = 'tag:data_quality'
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Running selected tests",
            stderr=""
        )
        
        # Create empty results file
        results_file = self.dbt_project_dir / "target" / "run_results.json"
        results_file.write_text(json.dumps({"metadata": {}, "results": []}))
        
        detector = DbtTestDetector(config_with_selection)
        results = detector.detect()
        
        # Verify --select parameter was used
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertIn("--select", call_args)
        self.assertIn("tag:data_quality", call_args)

    @patch('subprocess.run')
    def test_detection_timeout_handling(self, mock_run):
        """Test detection handles subprocess timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["dbt", "test"],
            timeout=300
        )
        
        detector = DbtTestDetector(self.valid_config)
        results = detector.detect()
        
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.test_id, 'timeout_error')
        self.assertIn("timed out", result.message)

    @patch('subprocess.run')
    def test_detection_command_failure(self, mock_run):
        """Test detection handles subprocess command failure."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["dbt", "test"],
            stderr="Compilation error: Model not found"
        )
        
        detector = DbtTestDetector(self.valid_config)
        results = detector.detect()
        
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.test_id, 'execution_error')
        self.assertIn("Compilation error", result.message)

    def test_detection_missing_results_file(self):
        """Test detection handles missing results file."""
        detector = DbtTestDetector(self.valid_config)
        
        # Mock successful command execution but no results file
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            
            results = detector.detect()
            
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.status, 'error')
            self.assertIn("unexpected", result.test_id)

    def test_detection_invalid_json_results(self):
        """Test detection handles invalid JSON in results file."""
        # Create invalid JSON results file
        results_file = self.dbt_project_dir / "target" / "run_results.json"
        results_file.write_text('{"invalid": json}')
        
        detector = DbtTestDetector(self.valid_config)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            
            results = detector.detect()
            
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.status, 'error')

    def test_status_mapping(self):
        """Test dbt status mapping to standard statuses."""
        detector = DbtTestDetector(self.valid_config)
        
        self.assertEqual(detector._map_dbt_status('pass'), 'pass')
        self.assertEqual(detector._map_dbt_status('fail'), 'fail')
        self.assertEqual(detector._map_dbt_status('error'), 'error')
        self.assertEqual(detector._map_dbt_status('skip'), 'skip')
        self.assertEqual(detector._map_dbt_status('warn'), 'warning')
        self.assertEqual(detector._map_dbt_status('unknown_status'), 'unknown')


if __name__ == '__main__':
    unittest.main()