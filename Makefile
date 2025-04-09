.PHONY: run lint format clean deep-clean

run:
	python3 main.py

lint:
	ruff check .

format:
	ruff check --select I --fix --show-fixes .
	ruff format .

clean:
	rm -rf results/*

deep-clean: clean
	find . -path "*/weights/*" -delete
	rm -rf __pycache__
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf .pytest_cache
