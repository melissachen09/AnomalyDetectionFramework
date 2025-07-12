"""
Test cases for ADF-16: Set Up Local Airflow Environment

This module tests the Airflow environment setup including:
- Docker-compose configuration
- Snowflake connections
- Volume mounts for DAGs and plugins
"""

import unittest
import os
import yaml
from pathlib import Path


class TestAirflowEnvironment(unittest.TestCase):
    """Test Airflow local environment setup"""

    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent
        self.docker_compose_path = self.project_root / "docker-compose.yml"
        self.airflow_dockerfile_path = self.project_root / "docker" / "airflow" / "Dockerfile"
        self.dags_path = self.project_root / "airflow" / "dags"
        self.plugins_path = self.project_root / "airflow" / "plugins"

    def test_docker_compose_file_exists(self):
        """Test that docker-compose.yml exists in project root"""
        self.assertTrue(
            self.docker_compose_path.exists(),
            "docker-compose.yml file should exist in project root"
        )

    def test_docker_compose_has_airflow_services(self):
        """Test that docker-compose.yml contains required Airflow services"""
        self.assertTrue(self.docker_compose_path.exists(), "docker-compose.yml must exist")
        
        with open(self.docker_compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config.get('services', {})
        
        # Check for required Airflow services
        required_services = ['airflow-webserver', 'airflow-scheduler', 'postgres']
        for service in required_services:
            self.assertIn(
                service, services,
                f"Service '{service}' should be defined in docker-compose.yml"
            )

    def test_airflow_webserver_configuration(self):
        """Test Airflow webserver service configuration"""
        with open(self.docker_compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        webserver = compose_config['services']['airflow-webserver']
        
        # Check port mapping
        self.assertIn('ports', webserver, "Webserver should have port mapping")
        self.assertIn('8080:8080', webserver['ports'], "Webserver should map port 8080")
        
        # Check depends_on
        self.assertIn('depends_on', webserver, "Webserver should depend on other services")
        self.assertIn('postgres', webserver['depends_on'], "Webserver should depend on postgres")

    def test_postgres_database_configuration(self):
        """Test PostgreSQL database service configuration"""
        with open(self.docker_compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        postgres = compose_config['services']['postgres']
        
        # Check environment variables
        self.assertIn('environment', postgres, "Postgres should have environment variables")
        env = postgres['environment']
        self.assertIn('POSTGRES_USER', env, "Postgres should have POSTGRES_USER")
        self.assertIn('POSTGRES_PASSWORD', env, "Postgres should have POSTGRES_PASSWORD")
        self.assertIn('POSTGRES_DB', env, "Postgres should have POSTGRES_DB")

    def test_snowflake_connection_environment_variables(self):
        """Test that Snowflake connection variables are configured"""
        with open(self.docker_compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check that services have Snowflake environment variables
        airflow_services = ['airflow-webserver', 'airflow-scheduler']
        
        for service_name in airflow_services:
            service = compose_config['services'][service_name]
            self.assertIn('environment', service, f"{service_name} should have environment variables")
            
            env = service['environment']
            snowflake_vars = [
                'SNOWFLAKE_ACCOUNT',
                'SNOWFLAKE_USER', 
                'SNOWFLAKE_PASSWORD',
                'SNOWFLAKE_DATABASE',
                'SNOWFLAKE_SCHEMA',
                'SNOWFLAKE_WAREHOUSE'
            ]
            
            for var in snowflake_vars:
                self.assertIn(var, env, f"{service_name} should have {var} environment variable")

    def test_volume_mounts_configuration(self):
        """Test that DAGs and plugins directories are mounted correctly"""
        with open(self.docker_compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        airflow_services = ['airflow-webserver', 'airflow-scheduler']
        
        for service_name in airflow_services:
            service = compose_config['services'][service_name]
            self.assertIn('volumes', service, f"{service_name} should have volume mounts")
            
            volumes = service['volumes']
            
            # Check for DAGs mount
            dags_mount_found = any('./airflow/dags:/opt/airflow/dags' in volume for volume in volumes)
            self.assertTrue(dags_mount_found, f"{service_name} should mount DAGs directory")
            
            # Check for plugins mount
            plugins_mount_found = any('./airflow/plugins:/opt/airflow/plugins' in volume for volume in volumes)
            self.assertTrue(plugins_mount_found, f"{service_name} should mount plugins directory")

    def test_airflow_directories_exist(self):
        """Test that required Airflow directories exist"""
        required_dirs = [
            self.dags_path,
            self.plugins_path
        ]
        
        for dir_path in required_dirs:
            self.assertTrue(
                dir_path.exists(),
                f"Directory {dir_path} should exist"
            )

    def test_airflow_dockerfile_exists(self):
        """Test that custom Airflow Dockerfile exists"""
        self.assertTrue(
            self.airflow_dockerfile_path.exists(),
            "Custom Airflow Dockerfile should exist in docker/airflow/"
        )

    def test_airflow_dockerfile_has_requirements(self):
        """Test that Airflow Dockerfile installs required packages"""
        self.assertTrue(self.airflow_dockerfile_path.exists(), "Dockerfile must exist")
        
        with open(self.airflow_dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Check for required packages
        required_packages = [
            'snowflake-connector-python',
            'pandas',
            'pyyaml'
        ]
        
        for package in required_packages:
            self.assertIn(
                package, dockerfile_content,
                f"Dockerfile should install {package}"
            )

    def test_docker_compose_uses_custom_airflow_image(self):
        """Test that docker-compose uses the custom Airflow image"""
        with open(self.docker_compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        airflow_services = ['airflow-webserver', 'airflow-scheduler']
        
        for service_name in airflow_services:
            service = compose_config['services'][service_name]
            
            # Check that service either builds from dockerfile or uses custom image
            has_build = 'build' in service
            has_custom_image = 'image' in service and 'anomaly-detection-airflow' in service['image']
            
            self.assertTrue(
                has_build or has_custom_image,
                f"{service_name} should use custom Airflow image or build configuration"
            )


if __name__ == '__main__':
    unittest.main()