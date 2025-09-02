.PHONY: dev test docker-build docker-up docker-down format

dev:
	python -m venv .venv && . .venv/bin/activate && python -m dotenv run -- uvicorn app.main:app --reload --port 5000

test:
	pytest -q

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

format:
	trufflehog -v >/dev/null 2>&1 || true
