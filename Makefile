.PHONY: lint format

lint:
	ruff check --fix --show-fixes .

format:
	ruff check --select I --fix --show-fixes .
	ruff format .
