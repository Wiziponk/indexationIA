# Indexation v3 — Patch Bundle

This bundle fixes critical issues, adds robustness, and includes a production setup (Docker, compose, tests).

## What’s fixed

1) **Duplicate `emb_clips.npy` in ZIPs → invalid reads**  
   The original `make_zip_for_program` wrote `emb_clips.npy` twice (first incorrectly to an in-memory buffer, then again). Python's `zipfile` will return the *first* entry by default, so downstream `/api/cluster/*` could fail when reading clip embeddings.  
   **Fix:** write each `.npy` once, correctly, with `np.save` to a `BytesIO` buffer before `writestr`. (See `app/services/clipmaker.py`).

2) **Can't download ZIPs from subfolders**  
   `/api/download/{name}` only served files from `data/`, but `segment/batch` stored artifacts under `data/zips/{uid}/...`. Links like `/api/download/{uid}.zip` 404'd.  
   **Fix:** new route `@router.get("/download/{path:path}")` with traversal protection so `/api/download/zips/{uid}.zip` and `/api/download/zips/{uid}/{file}.zip` work. Also updated `segment/batch` to emit the correct URLs.

3) **`n_init="auto"` breaks on older scikit-learn**  
   Some environments ship sklearn where `n_init="auto"` isn't available.  
   **Fix:** use `n_init=10` everywhere (`app/services/clustering.py`, `app/routers/cluster.py`).

4) **NLTK punkt availability**  
   If the server cannot download NLTK data, sentence splitting crashes.  
   **Fix:** robust fallback regex splitter is used automatically when punkt is missing (`_split_sentences`).

## What’s added

- **Dockerfile + docker-compose.yml** for a one-command run.
- **`.env.example`**: copy to `.env` and fill `OPENAI_API_KEY` and `EDUC_API_BASE`.
- **Basic tests** (`pytest`) for segmentation and clustering plumbing.
- **Makefile** targets for dev and Docker workflows.

## How to apply

1. Unzip this archive at the root of your repo (same folder that contains `app/`, `static/`, `requirements.txt`).  
   Allow it to **overwrite** existing files.

2. Create your environment:
   ```bash
   cp .env.example .env
   # Set OPENAI_API_KEY and EDUC_API_BASE in .env
   ```

3. Run locally:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload --port 5000
   # then open http://localhost:5000
   ```

   Or with Docker:
   ```bash
   docker compose up --build -d
   ```

4. In the UI:
   - Click **Load fields from API** (configure `EDUC_API_BASE` first).
   - Choose a **Primary key** and **Fields**.
   - Upload transcripts (`.docx` – filenames must contain the numeric ID).
   - **Prepare dataset** → **Preview clip maker** → **Produce ZIPs**.
   - Use **Cluster from ZIPs** to visualize by clips or by emissions.

## Notes

- For private/self-signed APIs, set `EDUC_VERIFY_SSL=0` in `.env`.
- Batch embedding is retried on rate-limits; tune `EMBED_BATCH_SIZE` via env.
- The UI constructs download links that now work for nested ZIPs.

Enjoy!
