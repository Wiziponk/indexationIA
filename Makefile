.PHONY: dev test docker-build docker-up docker-down format

dev:
	python -m venv .venv && source .venv/bin/activate
	python -m dotenv run -- uvicorn app.main:app --reload --port 5000

test:
	tpytest -q

docker-build:
	tdocker compose build

docker-up:
	tdocker compose up -d

docker-down:
	tdocker compose down

format:
	trufflehog -v >/dev/null 2>&1 || true
