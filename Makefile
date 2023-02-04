SHELL := /usr/bin/env bash

$(shell if [ ! -f .env ] && [ -f .example.env ]; then cp .example.env .env; fi)
-include .env
export

# make help: @ List available tasks on this project
help:
	@grep -h -E '[a-zA-Z0-9\.\-]+:.*?@ .*$$' $(MAKEFILE_LIST) | tr -d '#' | awk 'BEGIN {FS = ":.*?@ "}; {printf "\033[36m%-40s\033[0m %s\n", $$1, $$2}'

.PHONY: help install clean format
.SILENT: help install clean format .venv

define install
	if ! type poetry >/dev/null; then curl -sSL https://install.python-poetry.org | python3 -; fi

	poetry config virtualenvs.in-project true
	poetry config virtualenvs.prefer-active-python true
	poetry env use $(shell python3 --version | sed 's/[^[0-9\.]]*//g')

	if ! poetry lock --check 2>/dev/null; then poetry lock --no-update; fi

	poetry install

	poetry run pre-commit install
	poetry run pre-commit autoupdate
endef

.venv:
	$(call install)

# make install: @ Install dependencies
install:
	$(call install)

# poetry add [--group dev|...] ...@latest: @ Add / update to latest [dev|... group] dependencies

# poetry remove [--group dev|...] ...: @ Remove [dev|... group] dependencies

# make clean: @ Remove cache, checkpoints, coverage, etc.
clean:
	find . -type f -name *.DS_Store -ls -delete
	find . | grep -E '(__pycache__|\.pyc|\.pyo)' | xargs rm -rf
	find . | grep -E .mypy_cache | xargs rm -rf
	find . | grep -E .pytest_cache | xargs rm -rf
	find . | grep -E .ipynb_checkpoints | xargs rm -rf
	find . | grep -E .trash | xargs rm -rf
	rm -f .coverage

# make format: @ Format code
format: .venv
	poetry run isort .
	poetry run black .
