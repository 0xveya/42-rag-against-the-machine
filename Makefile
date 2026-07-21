PYTHON ?= python3
UV ?= uv
PIP ?= $(PYTHON) -m pip
VENV ?= .venv
VENV_PYTHON = $(VENV)/bin/python
MAIN ?= -m rag_against_the_machine.main
ARGS ?= 

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
FLAKE8_FLAGS = --max-line-length=100 --extend-ignore=E203 --exclude=.venv,.git,data,__pycache__,.mypy_cache,.ruff_cache,.pytest_cache,.ty,dist,build,test_env,test_venv,test_install

FLAKE8 = uv run flake8
MYPY = uv run mypy
RUFF = uv run ruff
TY = uv run ty
PYTEST = uv run pytest

.PHONY: install install-pip build-package build run clean lint lint-strict format check-modern typecheck typecheck-ty test

build: build-package

build-package:
	rm -rf dist build rag_against_the_machine-*.whl rag_against_the_machine-*.tar.gz
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) build; \
	else \
		$(PYTHON) -m pip install --upgrade build; \
		$(PYTHON) -m build; \
	fi

install: build-package
	uv sync --dev

install-pip: build-package
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"

run:
	uv run python $(MAIN) $(ARGS)


clean:
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".ruff_cache" \
		-o -name ".pytest_cache" -o -name ".ty" \) -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	rm -rf dist build rag_against_the_machine.egg-info rag_against_the_machine-*.whl rag_against_the_machine-*.tar.gz

lint:
	$(FLAKE8) $(FLAKE8_FLAGS) .
	$(MYPY) rag_against_the_machine $(MYPY_FLAGS)

lint-strict:
	$(FLAKE8) $(FLAKE8_FLAGS) .
	$(MYPY) rag_against_the_machine

format:
	$(RUFF) format .
	$(RUFF) check --fix .

check-modern:
	$(RUFF) check .
	$(TY) check rag_against_the_machine

typecheck:
	$(MYPY) rag_against_the_machine $(MYPY_FLAGS)
	$(TY) check rag_against_the_machine

typecheck-ty:
	$(TY) check rag_against_the_machine

test:
	$(PYTEST)
