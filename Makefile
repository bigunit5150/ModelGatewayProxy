.DEFAULT_GOAL := help
.PHONY: help dev test lint format type-check clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

dev: ## Run the development server with hot-reload
	uvicorn src.llmgateway.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run the test suite with coverage
	pytest --cov=src/llmgateway --cov-report=term-missing --cov-report=html:htmlcov

lint: ## Lint source and tests with Ruff
	ruff check src/ tests/

format: ## Auto-format source and tests with Ruff
	ruff format src/ tests/

type-check: ## Run mypy static type checks
	mypy src/

clean: ## Remove all generated artefacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete
	find . -type f -name "*\$$py.class" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
