.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check security clean build docker-build docker-run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt -r requirements-dev.txt

test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/ -m "not slow"

test-integration: ## Run integration tests only
	pytest tests/integration/

test-e2e: ## Run end-to-end tests only
	pytest tests/e2e/

lint: ## Run linting checks
	flake8 .
	black --check .

format: ## Format code with black
	black .

type-check: ## Run type checking with mypy
	mypy --ignore-missing-imports .

security: ## Run security checks
	bandit -r . -f txt
	safety check

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build the project
	python -m build

docker-build: ## Build Docker image
	docker build -t anomaly-detector:latest .

docker-run: ## Run Docker container
	docker run --rm -it anomaly-detector:latest

docker-test: ## Test Docker image build
	docker build --target builder -t anomaly-detector:test .

check: lint type-check security test ## Run all quality checks

ci: check ## Run CI pipeline locally
	@echo "All checks passed!"