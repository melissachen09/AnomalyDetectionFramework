"""
Test suite for Docker Development Environment setup [ADF-19]
Tests for multi-stage Dockerfile, docker-compose, environment variables, and Makefile
"""

import os
import pytest
import yaml
import subprocess
from pathlib import Path


class TestDockerSetup:
    """Test Docker development environment configuration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path(__file__).parent.parent.parent
        self.docker_dir = self.project_root / "docker"
        self.detector_dir = self.docker_dir / "detector"
        
    def test_detector_dockerfile_exists(self):
        """Test that detector Dockerfile exists"""
        dockerfile_path = self.detector_dir / "Dockerfile"
        assert dockerfile_path.exists(), "Detector Dockerfile should exist"
        
    def test_dockerfile_multi_stage(self):
        """Test that Dockerfile uses multi-stage build"""
        dockerfile_path = self.detector_dir / "Dockerfile"
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            assert "FROM python:3.9" in content, "Should use Python 3.9 base image"
            assert "AS builder" in content or "AS base" in content, "Should use multi-stage build"
            assert "COPY --from=" in content, "Should copy from previous stage"
            
    def test_dockerfile_optimization(self):
        """Test Dockerfile optimization practices"""
        dockerfile_path = self.detector_dir / "Dockerfile"
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            assert "requirements.txt" in content, "Should copy requirements.txt separately"
            assert "RUN pip install" in content, "Should install dependencies"
            assert "WORKDIR" in content, "Should set working directory"
            assert "USER" in content, "Should set non-root user"
            
    def test_docker_compose_detector_service(self):
        """Test docker-compose includes detector service"""
        compose_path = self.project_root / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml should exist"
        
        with open(compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
            
        assert "services" in compose_config, "Should have services section"
        assert "anomaly-detector" in compose_config["services"], "Should have anomaly-detector service"
        
    def test_docker_compose_environment_variables(self):
        """Test docker-compose has proper environment variable setup"""
        compose_path = self.project_root / "docker-compose.yml"
        
        with open(compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
            
        detector_service = compose_config["services"]["anomaly-detector"]
        assert "environment" in detector_service, "Should have environment section"
        
        # Check for required environment variables
        env_vars = detector_service["environment"]
        required_vars = [
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_USER", 
            "SNOWFLAKE_PASSWORD",
            "SNOWFLAKE_DATABASE",
            "SNOWFLAKE_WAREHOUSE"
        ]
        
        for var in required_vars:
            assert any(var in str(env_var) for env_var in env_vars), f"Should have {var} environment variable"
            
    def test_docker_compose_volume_mounts(self):
        """Test docker-compose has proper volume mounts"""
        compose_path = self.project_root / "docker-compose.yml"
        
        with open(compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
            
        detector_service = compose_config["services"]["anomaly-detector"]
        assert "volumes" in detector_service, "Should have volumes section"
        
        # Check for config volume mount
        volumes = detector_service["volumes"]
        assert any("./configs:" in vol for vol in volumes), "Should mount configs directory"
        
    def test_env_example_file_exists(self):
        """Test that .env.example file exists"""
        env_example_path = self.project_root / ".env.example"
        assert env_example_path.exists(), ".env.example should exist"
        
    def test_env_example_has_required_variables(self):
        """Test .env.example has all required environment variables"""
        env_example_path = self.project_root / ".env.example"
        
        if env_example_path.exists():
            content = env_example_path.read_text()
            required_vars = [
                "SNOWFLAKE_ACCOUNT",
                "SNOWFLAKE_USER",
                "SNOWFLAKE_PASSWORD", 
                "SNOWFLAKE_DATABASE",
                "SNOWFLAKE_WAREHOUSE",
                "SLACK_WEBHOOK_URL"
            ]
            
            for var in required_vars:
                assert var in content, f"Should have {var} in .env.example"
                
    def test_makefile_exists(self):
        """Test that Makefile exists"""
        makefile_path = self.project_root / "Makefile"
        assert makefile_path.exists(), "Makefile should exist"
        
    def test_makefile_has_required_targets(self):
        """Test Makefile has required build targets"""
        makefile_path = self.project_root / "Makefile"
        
        if makefile_path.exists():
            content = makefile_path.read_text()
            required_targets = ["build", "test", "run", "clean", "dev"]
            
            for target in required_targets:
                assert f"{target}:" in content, f"Should have {target} target"
                
    def test_makefile_docker_commands(self):
        """Test Makefile contains proper Docker commands"""
        makefile_path = self.project_root / "Makefile"
        
        if makefile_path.exists():
            content = makefile_path.read_text()
            assert "docker build" in content, "Should have docker build command"
            assert "docker-compose" in content, "Should have docker-compose commands"
            assert "docker run" in content, "Should have docker run command"
            
    def test_detector_requirements_file(self):
        """Test that detector has requirements.txt"""
        requirements_path = self.detector_dir / "requirements.txt"
        assert requirements_path.exists(), "requirements.txt should exist"
        
    def test_detector_requirements_content(self):
        """Test requirements.txt has necessary packages"""
        requirements_path = self.detector_dir / "requirements.txt"
        
        if requirements_path.exists():
            content = requirements_path.read_text()
            required_packages = [
                "snowflake-connector-python",
                "pandas",
                "pyyaml",
                "pytest"
            ]
            
            for package in required_packages:
                assert package in content, f"Should have {package} in requirements.txt"
                
    def test_gitignore_docker_entries(self):
        """Test .gitignore has Docker-related entries"""
        gitignore_path = self.project_root / ".gitignore"
        
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            docker_entries = [".env", "*.log", "__pycache__/"]
            
            for entry in docker_entries:
                assert entry in content, f"Should have {entry} in .gitignore"


class TestDockerIntegration:
    """Integration tests for Docker setup"""
    
    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path(__file__).parent.parent.parent
        
    @pytest.mark.slow
    def test_docker_build_success(self):
        """Test that Docker image builds successfully"""
        # This is a slow test that actually builds the Docker image
        result = subprocess.run(
            ["docker", "build", "-t", "anomaly-detector:test", "./docker/detector"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"
        
    @pytest.mark.slow  
    def test_docker_compose_validation(self):
        """Test that docker-compose file is valid"""
        result = subprocess.run(
            ["docker-compose", "config"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"docker-compose config invalid: {result.stderr}"
        
    @pytest.mark.slow
    def test_makefile_build_target(self):
        """Test that Makefile build target works"""
        result = subprocess.run(
            ["make", "build"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Make build failed: {result.stderr}"