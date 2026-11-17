.PHONY: install test lint run train up

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check . && ruff format --check . && mypy src

run:
	uvicorn antifraude.api:app --reload

train:
	antifraude-train --rows 50000

up:
	docker compose up --build
