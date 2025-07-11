"""
dbt Test Detector for the Anomaly Detection Framework.

This detector integrates with existing dbt tests to provide data quality
monitoring by executing dbt test commands and parsing their results.

Part of Epic ADF-3: Detection Plugin Architecture
Task: ADF-41 - Implement dbt Test Detector (GADF-DETECT-012)
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class DbtTestDetector(BaseDetector):
    """
    Detector that executes dbt tests and parses results for anomaly detection.
    
    This detector integrates with existing dbt test infrastructure to:
    - Execute dbt test commands via subprocess
    - Parse dbt run_results.json artifacts
    - Handle errors, timeouts, and various execution scenarios
    - Support test selection via dbt's --select parameter
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dbt test detector.
        
        Args:
            config: Configuration dictionary containing:
                - project_dir: Path to dbt project directory
                - profiles_dir: Path to dbt profiles directory (optional)
                - target_path: Path to dbt target directory (optional)
                - log_level: dbt log level (default: info)
                - selection: dbt test selection criteria (optional)
                - timeout: Command timeout in seconds (default: 300)
        """
        super().__init__(config)
        
        # Required configuration
        if 'project_dir' not in config:
            raise ValueError("Missing required configuration: project_dir")
            
        self.project_dir = Path(config['project_dir'])
        if not self.project_dir.exists():
            raise ValueError(f"dbt project directory does not exist: {self.project_dir}")
        
        # Optional configuration with defaults
        self.profiles_dir = config.get('profiles_dir', self.project_dir / "profiles")
        self.target_path = config.get('target_path', self.project_dir / "target")
        self.log_level = config.get('log_level', 'info')
        self.selection = config.get('selection')
        
        # Ensure target directory exists
        Path(self.target_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized dbt detector for project: {self.project_dir}")
    
    def detect(self, **kwargs) -> List[DetectionResult]:
        """
        Execute dbt tests and return detection results.
        
        Returns:
            List of DetectionResult objects, one per dbt test executed
        """
        try:
            # Execute dbt test command
            self.logger.info("Starting dbt test execution")
            self._execute_dbt_test()
            
            # Parse results from run_results.json
            results = self._parse_dbt_results()
            
            self.logger.info(f"dbt test execution completed with {len(results)} results")
            return results
            
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"dbt test execution timed out after {e.timeout} seconds")
            return [DetectionResult(
                detector_name=self.name,
                test_id="timeout_error",
                status="error",
                message=f"dbt test execution timed out after {e.timeout} seconds",
                metadata={"error_type": "timeout", "command": e.cmd}
            )]
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"dbt test execution failed: {e.stderr}")
            return [DetectionResult(
                detector_name=self.name,
                test_id="execution_error",
                status="error",
                message=f"dbt test execution failed: {e.stderr}",
                metadata={"error_type": "command_failure", "returncode": e.returncode}
            )]
            
        except Exception as e:
            self.logger.error(f"Unexpected error during dbt test execution: {str(e)}")
            return [DetectionResult(
                detector_name=self.name,
                test_id="unexpected_error",
                status="error",
                message=f"Unexpected error: {str(e)}",
                metadata={"error_type": "unexpected"}
            )]
    
    def _execute_dbt_test(self) -> None:
        """
        Execute dbt test command using subprocess.
        
        Raises:
            subprocess.TimeoutExpired: If command times out
            subprocess.CalledProcessError: If command fails
            FileNotFoundError: If dbt command is not found
        """
        # Build dbt command
        command = [
            "dbt", "test",
            "--project-dir", str(self.project_dir),
            "--profiles-dir", str(self.profiles_dir),
            "--target-path", str(self.target_path),
            "--log-level", self.log_level
        ]
        
        # Add selection criteria if specified
        if self.selection:
            command.extend(["--select", self.selection])
        
        self.logger.debug(f"Executing command: {' '.join(command)}")
        
        # Execute command with timeout
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=str(self.project_dir)
        )
        
        # Log output for debugging
        if result.stdout:
            self.logger.debug(f"dbt stdout: {result.stdout}")
        if result.stderr:
            self.logger.debug(f"dbt stderr: {result.stderr}")
        
        # Check for command failure
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                command,
                output=result.stdout,
                stderr=result.stderr
            )
    
    def _parse_dbt_results(self) -> List[DetectionResult]:
        """
        Parse dbt run_results.json file and convert to DetectionResult objects.
        
        Returns:
            List of DetectionResult objects
            
        Raises:
            FileNotFoundError: If run_results.json file is missing
            json.JSONDecodeError: If results file is invalid JSON
        """
        results_file = Path(self.target_path) / "run_results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"dbt results file not found: {results_file}")
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                dbt_results = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in dbt results file: {e.msg}",
                doc=e.doc,
                pos=e.pos
            )
        
        # Convert dbt results to DetectionResult objects
        detection_results = []
        
        for test_result in dbt_results.get('results', []):
            detection_result = DetectionResult(
                detector_name=self.name,
                test_id=test_result.get('unique_id', 'unknown'),
                status=self._map_dbt_status(test_result.get('status', 'unknown')),
                message=test_result.get('message', ''),
                execution_time=test_result.get('execution_time', 0.0),
                failures=test_result.get('failures', 0),
                metadata={
                    'dbt_metadata': dbt_results.get('metadata', {}),
                    'timing': test_result.get('timing', []),
                    'thread_id': test_result.get('thread_id'),
                    'adapter_response': test_result.get('adapter_response', {})
                }
            )
            detection_results.append(detection_result)
        
        return detection_results
    
    def _map_dbt_status(self, dbt_status: str) -> str:
        """
        Map dbt test status to standard detection result status.
        
        Args:
            dbt_status: Status from dbt test result
            
        Returns:
            Standardized status string
        """
        status_mapping = {
            'pass': 'pass',
            'fail': 'fail', 
            'error': 'error',
            'skip': 'skip',
            'warn': 'warning'
        }
        
        return status_mapping.get(dbt_status.lower(), 'unknown')
    
    def validate_config(self) -> bool:
        """
        Validate dbt detector configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required fields
            if 'project_dir' not in self.config:
                self.logger.error("Missing required configuration: project_dir")
                return False
            
            # Check project directory exists
            if not self.project_dir.exists():
                self.logger.error(f"dbt project directory does not exist: {self.project_dir}")
                return False
            
            # Check for dbt_project.yml
            dbt_project_file = self.project_dir / "dbt_project.yml"
            if not dbt_project_file.exists():
                self.logger.error(f"dbt_project.yml not found in: {self.project_dir}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {str(e)}")
            return False
    
    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about this dbt detector.
        
        Returns:
            Dictionary with detector metadata
        """
        info = super().get_detector_info()
        info.update({
            'project_dir': str(self.project_dir),
            'profiles_dir': str(self.profiles_dir),
            'target_path': str(self.target_path),
            'log_level': self.log_level,
            'selection': self.selection
        })
        return info