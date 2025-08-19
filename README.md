# Embeddings & Clustering (FastAPI + React)

A production-ready scaffold that preserves the public contract of the previous app while improving UX and robustness.

## Quick start (dev)

1. Copy `backend/.env.example` to `.env` and fill `OPENAI_API_KEY`. Optionally set `API_BASE`.
2. Run:
   ```bash
   docker-compose up --build
   ```
3. Open the UI: http://localhost:5173

## Public endpoints preserved

- `GET /` – app info
- `GET /healthz` – readiness
- `GET /api/fields` – enumerate source fields
- `GET /api/sample` – sample rows
- `POST /preview-excel` – Excel/CSV preview → `{ token, columns, head }`
- `GET /download/<name>` – download artifacts
- `POST /generate/preview` – preview how text will be built
- `POST /generate` – synchronous (waits for job and returns)
- `POST /cluster` – synchronous (waits for job and returns)

## Background jobs (new, used by the new UI)

- `POST /_jobs/generate` → `{ job_id }`
- `POST /_jobs/cluster` → `{ job_id }`
- `GET /_jobs/{id}` → `{ status, progress, message, result? }`

## Artifacts

- `raw_<uid>.parquet`
- `emb_<uid>.npy`
- `result_<uid>.parquet`

Stored in the Docker volume `data:`.

## Notes

- If `API_BASE` is empty, the app uses a small built-in sample dataset so you can test the flow.
- Frontend uses polling for job status to keep it simple and robust.
- For production, restrict CORS and move secrets out of compose.
