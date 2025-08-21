# Multi-arch friendly, small image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (build + optional for numpy/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Try to pre-fetch NLTK punkt (won't fail the build if offline)
RUN python - <<'PY'\ntry:\n import nltk; nltk.download('punkt', quiet=True)\nexcept Exception: pass\nPY

COPY . /app

EXPOSE 5000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
