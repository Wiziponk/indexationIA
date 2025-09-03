# Indexation v2 (API-first + Single-page UI)

A clean-room rewrite with the same functional inputs/outputs as your original app:
- **Generates** `data/raw_<uid>.parquet` (rows + `_text_for_embedding`) and `data/emb_<uid>.npy` (embeddings)
- **Clusters** (K-Means auto-k or DBSCAN), **projects** to 2D (PCA or t-SNE), **names clusters**, and lets you **download** results

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure your API and OpenAI
export OPENAI_API_KEY=sk-...
export EDUC_API_BASE="https://educ.arte.tv/api/list/programs"
# optional: export EDUC_API_TOKEN="..."

# Run
python -m dotenv run -- uvicorn app.main:app --reload --port 5000
```
"docker compose --profile dev up --build"

Open http://localhost:5000 and use the UI.

## Notes
- Parquet requires `pyarrow` (already in requirements).
- For large datasets, embedding is batched with retry/backoff.

## Status API contract

The `/api/segment/status/{uid}` endpoint reports long-running batch jobs.
It returns one of four states:

```json
{"status": "running", "progress": 3, "total": 10}
{"status": "done", "result": {"uid": "abcd", "count": 10, "master_zip": "/api/download/zips/abcd.zip", "zips": [{"programme_id": "p1", "path": "/api/download/zips/abcd/p1.zip"}]}}
{"status": "error", "message": "something went wrong"}
{"status": "not_found", "message": "No job with this uid"}
```

Clients should handle each state accordingly and avoid assuming `result` exists
unless `status` is `"done"`.
