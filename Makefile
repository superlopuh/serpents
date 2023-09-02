MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# allow overriding the name of the venv directory
VENV_DIR ?= venv

# use a default prefix for coverage data files
COVERAGE_FILE ?= .coverage

# use different coverage data file per coverage run, otherwise combine complains
TESTS_COVERAGE_FILE = ${COVERAGE_FILE}.tests

# make tasks run all commands in a single shell
.ONESHELL:

# these targets don't produce files:
.PHONY: clean pytest tests precommit-install precommit black pyright
.PHONY: coverage coverage-tests coverage-report-html coverage-report-md

# remove all caches and the venv
clean:
	rm -rf ${VENV_DIR} .pytest_cache *.egg-info .coverage.*

# run pytest tests
pytest:
	pytest -W error -v

# set up all precommit hooks
precommit-install:
	pre-commit install

# run all precommit hooks and apply them
precommit:
	pre-commit run --all

# run pyright on all files in the current git commit
pyright:
	pyright $(shell git diff --staged --name-only  -- '*.py')

# run black on all files currently staged
black:
	staged_files="$(shell git diff --staged --name-only)"
	# run black on all of xdsl if no staged files exist
	black $${staged_files:-xdsl}

# run coverage over all tests and combine data files
coverage: coverage-tests
	coverage combine --append

# run coverage over tests
coverage-tests:
	COVERAGE_FILE=${TESTS_COVERAGE_FILE} pytest -W error --cov --cov-config=.coveragerc

# generate html coverage report
coverage-report-html:
	coverage html

# generate markdown coverage report
coverage-report-md:
	coverage report --format=markdown
