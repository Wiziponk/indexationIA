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
uvicorn app.main:app --reload --port 5000
```

Open http://localhost:5000 and use the UI.

## Notes
- Parquet requires `pyarrow` (already in requirements).
- For large datasets, embedding is batched with retry/backoff.
