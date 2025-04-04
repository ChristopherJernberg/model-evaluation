.PHONY: lint format

lint:
	ruff check .

format:
	ruff check --select I --fix --show-fixes .
	ruff format .
