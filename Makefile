.RECIPEPREFIX := >
.PHONY: dev test docker-build docker-up docker-down format lint

dev:
> python -m venv .venv && . .venv/bin/activate && python -m dotenv run -- uvicorn app.main:app --reload --port 5000

test:
> pytest -q

docker-build:
> docker compose build

docker-up:
> docker compose up -d

docker-down:
> docker compose down

format:
> pre-commit run --files Dockerfile docker-compose.yml Makefile app/config.py .editorconfig .env.example .pre-commit-config.yaml mypy.ini pyproject.toml --hook-stage manual

lint:
> pre-commit run --all-files
