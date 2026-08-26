.PHONY: install api demo test lint
install:
	uv sync --extra dev
api:
	uv run uvicorn fl_async.main:app --host 127.0.0.1 --port 8000 --reload
demo:
	uv run fl_async-demo
test:
	uv run pytest
lint:
	uv run ruff check .
