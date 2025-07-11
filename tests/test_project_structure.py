"""
Test suite for project directory structure validation.

This test verifies that the required directory structure exists
as specified in ADF-12.
"""

import os
import unittest
from pathlib import Path


class TestProjectStructure(unittest.TestCase):
    """Test cases for project directory structure."""

    def setUp(self):
        """Set up test with project root path."""
        self.project_root = Path(__file__).parent.parent

    def test_src_directory_exists(self):
        """Test that /src directory exists."""
        src_dir = self.project_root / "src"
        self.assertTrue(
            src_dir.exists(),
            f"src directory should exist at {src_dir}"
        )
        self.assertTrue(
            src_dir.is_dir(),
            f"src should be a directory at {src_dir}"
        )

    def test_tests_directory_exists(self):
        """Test that /tests directory exists."""
        tests_dir = self.project_root / "tests"
        self.assertTrue(
            tests_dir.exists(),
            f"tests directory should exist at {tests_dir}"
        )
        self.assertTrue(
            tests_dir.is_dir(),
            f"tests should be a directory at {tests_dir}"
        )

    def test_configs_directory_exists(self):
        """Test that /configs directory exists."""
        configs_dir = self.project_root / "configs"
        self.assertTrue(
            configs_dir.exists(),
            f"configs directory should exist at {configs_dir}"
        )
        self.assertTrue(
            configs_dir.is_dir(),
            f"configs should be a directory at {configs_dir}"
        )

    def test_docs_directory_exists(self):
        """Test that /docs directory exists."""
        docs_dir = self.project_root / "docs"
        self.assertTrue(
            docs_dir.exists(),
            f"docs directory should exist at {docs_dir}"
        )
        self.assertTrue(
            docs_dir.is_dir(),
            f"docs should be a directory at {docs_dir}"
        )

    def test_docker_directory_exists(self):
        """Test that /docker directory exists."""
        docker_dir = self.project_root / "docker"
        self.assertTrue(
            docker_dir.exists(),
            f"docker directory should exist at {docker_dir}"
        )
        self.assertTrue(
            docker_dir.is_dir(),
            f"docker should be a directory at {docker_dir}"
        )

    def test_git_repository_initialized(self):
        """Test that Git repository is properly initialized."""
        git_dir = self.project_root / ".git"
        self.assertTrue(
            git_dir.exists(),
            f".git directory should exist at {git_dir}"
        )
        self.assertTrue(
            git_dir.is_dir(),
            f".git should be a directory at {git_dir}"
        )

    def test_git_config_exists(self):
        """Test that Git configuration is set up."""
        git_config = self.project_root / ".git" / "config"
        self.assertTrue(
            git_config.exists(),
            f"Git config should exist at {git_config}"
        )
        self.assertTrue(
            git_config.is_file(),
            f"Git config should be a file at {git_config}"
        )

    def test_all_required_directories_present(self):
        """Test that all required directories are present."""
        required_dirs = ['src', 'tests', 'configs', 'docs', 'docker']
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            with self.subTest(directory=dir_name):
                self.assertTrue(
                    dir_path.exists(),
                    f"{dir_name} directory should exist"
                )
                self.assertTrue(
                    dir_path.is_dir(),
                    f"{dir_name} should be a directory"
                )


if __name__ == '__main__':
    unittest.main()