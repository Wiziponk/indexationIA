# agents.md

## 📌 Project Intent

This repository provides a **dataset & clip segmentation tool** with two main workflows:

1. **Dataset & Clips (Wizard)**
   - Connect to an EDUC API or load IDs from Excel/CSV.
   - Upload programme transcripts (`.docx`).
   - Auto-segment into clips using embeddings + scoring.
   - Optionally summarise & title clips with OpenAI.
   - Export per-programme ZIPs and a master ZIP (with `manifest.csv`).

2. **Explore & Refine**
   - Upload ZIPs to explore clusters (by clips or programmes).
   - Visualize clusters in 2D projections.
   - Save, rename, and export cluster results.
   - Library to browse projects/programmes, edit clips, and re-run segmentation.

The tool is designed for **education/media cataloguing**, where long programmes need to be broken into semantically coherent clips with searchable metadata.

---

## ⚙️ Architecture Overview

- **Backend**
  - Framework: [FastAPI](https://fastapi.tiangolo.com/).
  - Entrypoint: `app/main.py`.
  - Routers: `app/routers/*` (datasets, segment, cluster, db).
  - Async batch jobs with in-memory job map (`/api/segment/status/{uid}`).
  - SQLite DB for projects/programmes/clips.
  - Exposes REST API with typed schemas (Pydantic).

- **Frontend**
  - Current: static HTML/Plotly under `static/`.
  - Target: React+TS SPA (via Vite, Tailwind, shadcn/ui).
  - Served by FastAPI static mount at `/`.

- **Data pipeline**
  - Input: EDUC API JSON or Excel/CSV list of IDs + `.docx` transcripts.
  - Processing: sentence chunking → clip boundary detection → embedding → scoring → OpenAI titles/summaries (optional).
  - Output: `raw_*.parquet`, `emb_*.npy`, clustered datasets, per-programme ZIPs, master ZIP with manifest.

- **Batching**
  - Runs long jobs in background threads.
  - Job UIDs tracked with a `/status` endpoint.
  - Result artifacts downloadable via `/api/download/*`.

---

## 📂 Repository Structure

- `app/`
  - `main.py` → FastAPI setup, CORS, static mount.
  - `routers/` → API modules:
    - `segment.py` → preview, batch, status, downloads.
    - `datasets.py` → dataset CRUD & rerun.
    - `cluster.py` → clustering endpoints.
    - `db.py` → library: projects, programmes, clips, reruns.
  - `schemas.py` (planned) → shared Pydantic models.
- `static/` → legacy HTML/Plotly UI.
- `tests/` → pytest API smoke tests.
- `Dockerfile`, `docker-compose.yml`, `Makefile`.

---

## 🔑 Key API Endpoints

- **Health & Fields**
  - `GET /api/health` → API alive + git sha.
  - `GET /api/fields` → list available fields from EDUC API.

- **Segmentation**
  - `POST /api/segment/prepare` → transcripts summary (included/excluded).
  - `POST /api/segment/preview` → preview clips for one programme.
  - `POST /api/segment/batch` → launch batch job.
  - `GET /api/segment/status/{uid}` → job status.
  - `GET /api/download/{filename}` → download ZIPs.

- **Datasets**
  - `GET/POST /api/datasets` → list/create.
  - `PUT /api/datasets/{uid}` → update label.
  - `DELETE /api/datasets/{uid}` → delete.
  - `POST /api/datasets/{uid}/rerun` → rerun dataset.

- **Clustering**
  - `POST /api/cluster/clips` → cluster uploaded ZIPs (clip embeddings).
  - `POST /api/cluster/emissions` → cluster by programme embeddings.

- **Library / DB**
  - `GET /api/db/projects` → list projects.
  - `GET /api/db/programs/{id}` → programme details.
  - `PATCH /api/db/clips/{clip_id}` → edit clip.
  - `POST /api/db/programs/{id}/rerun` → rerun segmentation.

---

## 🔒 Environment

Defined via `.env` (see `.env.example`):

- `OPENAI_API_KEY` → required if using titles/summaries.
- `EDUC_API_BASE`, `EDUC_API_TOKEN` → connection to EDUC API.
- `HTTP_TIMEOUT`, `EDUC_VERIFY_SSL`.

---

## 🚀 Development & Deployment

- Run locally:
  ```bash
  docker compose up --build
