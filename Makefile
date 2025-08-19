setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

dev:
	flask --app app run --debug --port 5000

test:
	pytest -q || true

fmt:
	black . && isort .

lint:
	ruff check app.py
