# agents.md

## 🔑 Agent Plan (read me first)

This section is the **single source of truth** for what to do next. Keep it short; update as PRs land.

### Coding standards
- **Python**: Ruff + Black, type hints, fastapi/pydantic; prefer dependency injection over globals.
- **Frontend**: React + TypeScript (Vite), Tailwind + shadcn/ui, React Router, Plotly for charts.
- **Testing**: pytest for API, Playwright for UI smoke; add tests in same PR when feasible.
- **Commits/PRs**: small, atomic; follow Conventional Commits (`feat:`, `fix:`, `chore:`, `test:`).

### Guardrails
- Don’t alter `.docx` parsing/segmentation heuristics unless a PR explicitly says so.
- Don’t remove existing endpoints; extend with typed schemas and docs.
- Long jobs must remain asynchronous; report status via `/api/segment/status/{uid}`.

---

## ✅ Roadmap (first 10 PRs)

**PR1 — Tooling & repo hygiene**
- **Scope**: Add `.pre-commit-config.yaml`, Ruff/Black config in `pyproject.toml`, `mypy.ini`, `.editorconfig`, `.env.example`; improve Docker (multi-stage).
- **Acceptance**: `make format && make lint` passes; `docker compose up` runs API on :5000.
- **Commit**: `chore: add pre-commit, ruff/black, mypy and improve Docker`

**PR2 — Typed API contracts**
- **Scope**: Introduce `app/schemas.py`; make routers return Pydantic models; unify error envelope.
- **Acceptance**: `/docs` shows models for all routes; clients receive stable shapes.
- **Commit**: `feat(api): add response models and unified errors`

**PR3 — Frontend scaffold (React+TS SPA)**
- **Scope**: New `/web` (Vite, Tailwind, shadcn/ui, Router); mount `/web/dist` via FastAPI at `/`.
- **Acceptance**: SPA shell renders with sidebar; API still served under `/api`.
- **Commit**: `feat(web): bootstrap React+TS app and mount via FastAPI`

**PR4 — Settings page (health & fields)**
- **Scope**: `/web/src/pages/Settings.tsx`; ping `/api/health` and `/api/fields`; show base URL, git sha, field stats.
- **Acceptance**: Green “OK” state shows sha; fields list loads or shows actionable error.
- **Commit**: `feat(web): add Settings with health and fields discovery`

**PR5 — Dataset Wizard (steps 1–3)**
- **Scope**: Stepper UI: (1) pick `primary_key` + `embed_fields`, (2) scope via API vs CSV/XLSX upload, (3) upload transcripts + call `/api/segment/prepare`.
- **Acceptance**: Validation gates “Next”; prepare shows counts + sample IDs.
- **Commit**: `feat(web): wizard steps 1–3 (fields, scope, transcripts)`

**PR6 — Dataset Wizard (steps 4–6: preview & batch)**
- **Scope**: Preview one programme via `/api/segment/preview`; launch batch via `/api/segment/batch`; poll `/api/segment/status/{uid}`; show master/per-programme ZIP links.
- **Acceptance**: Progress states visible; master ZIP downloads; preview table shows top clips.
- **Commit**: `feat(web): preview & batch with status polling`

**PR7 — Datasets screen (CRUD + rerun)**
- **Scope**: List datasets; inline rename; delete; `POST /api/datasets/{uid}/rerun`; optional file links.
- **Acceptance**: Edit persists; rerun creates new dataset and toasts new uid.
- **Commit**: `feat(web): datasets CRUD and rerun`

**PR8 — Clustering UI (clips/programmes)**
- **Scope**: Drag-and-drop ZIPs; options (algo, k, projection); call `/api/cluster/clips` and `/api/cluster/emissions`; Plotly scatter; CSV export.
- **Acceptance**: Error on invalid ZIPs is friendly; clusters render; summary counts shown.
- **Commit**: `feat(web): clustering UI with dnd and Plotly`

**PR9 — Library (Projects → Programmes → Clips editor)**
- **Scope**: 3-pane editor; load projects/programmes; edit clip title/summary/score (`PATCH /api/db/clips/{id}`); rerun programme.
- **Acceptance**: Edits persist on reload; rerun updates counts and exposes ZIP link.
- **Commit**: `feat(web): library with clip editing and programme rerun`

**PR10 — Jobs & downloads**
- **Scope**: Reusable `useJobPolling(uid)`; jobs drawer; unified downloads panel for last batch artifacts.
- **Acceptance**: Jobs survive route changes; completed jobs auto-clear; downloads open reliably.
- **Commit**: `feat(web): jobs drawer and downloads center`

---

## 🧪 How to run (for the agent)
- **Dev**: `docker compose up --build` → backend on `http://localhost:5000/api`, SPA at `/`.
- **Tests**: `pytest -q`
