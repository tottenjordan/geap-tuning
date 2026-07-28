.PHONY: dev lint format test build

dev:
	uv sync --all-groups

lint:
	uv run ruff format --check . && uv run ruff check . && uv run ty check src/

format:
	uv run ruff format .

test:
	uv run pytest
