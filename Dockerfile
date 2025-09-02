# Builder image
FROM node:20-alpine AS web-builder

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ .
RUN npm run build

FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt
# Try to pre-fetch NLTK punkt (won't fail if offline)
RUN python - <<'PY'
try:
    import nltk
    nltk.download('punkt', quiet=True)
except Exception:
    pass
PY

# Runtime image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=builder /root/.local /usr/local
COPY . /app
COPY --from=web-builder /web/dist /app/web/dist

EXPOSE 5000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
