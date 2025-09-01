.PHONY: help install test lint format clean build run dev check all

# Default target
help:
	@echo "Available targets:"
	@echo "  install   - Install dependencies"
	@echo "  test      - Run tests with coverage"
	@echo "  lint      - Run pylint"
	@echo "  format    - Format code with black"
	@echo "  clean     - Remove build artifacts and cache"
	@echo "  build     - Run lint, format check, and tests"
	@echo "  run       - Run the quantum comic generator"
	@echo "  dev       - Install in development mode"
	@echo "  check     - Check code formatting without changing files"
	@echo "  all       - Run format, lint, and test"

# Install dependencies
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

# Install in development mode
dev: install
	pip install -e .

# Run tests with coverage
test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run linter
lint:
	python -m pylint src/ tests/ --rcfile=.pylintrc || true
	python -m mypy src/ --ignore-missing-imports

# Format code
format:
	python -m black src/ tests/

# Check formatting without changing files
check:
	python -m black src/ tests/ --check

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true

# Build target (for CI/CD)
build: check lint test
	@echo "Build successful!"

# Run the main application
run:
	python -m src.main

# Run the main application with a small number of panels
run_small:
	python -m src.main --panels 2

# Run all checks
all: format lint test
	@echo "All checks passed!"

# Watch for changes and run tests (requires pytest-watch)
watch:
	python -m pytest_watch tests/ -v

# Generate documentation
docs:
	@echo "Documentation generation not yet configured"

# Type checking
typecheck:
	python -m mypy src/ --ignore-missing-imports --strict
