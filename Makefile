.PHONY: help install install-dev test lint clean release pkgcheck

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install TLabel (pip)
	pip install -e .

install-dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

install-all:  ## Install with all optional dependencies
	pip install -e ".[all,dev]"

test:  ## Run test suite
	pytest tests/ -v --tb=short

test-quick:  ## Run core tests only
	pytest tests/test_tlabel.py -v

clean:  ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

pkgcheck:  ## 发版前包内容审查：内部文档不得进 sdist/wheel
	python scripts/check_package_contents.py

release: clean test pkgcheck  ## Build and publish to PyPI
	python -m build
	twine upload dist/*

demo:  ## Run TLabel demo
	python -c "import tlabel; tlabel.demo()"

panel:  ## Launch standalone HTML panel
	@echo "Open tlabel_panel_standalone.html in your browser"

docs-serve:  ## Serve docs locally (requires Python HTTP server)
	cd docs && python -m http.server 8080
