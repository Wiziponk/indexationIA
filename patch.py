import re, sys, pathlib

root = pathlib.Path(".")
problems = []

def touch(p): 
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

# 1) app/models.py – add SQLAlchemy Dataset if missing
p = root / "app" / "models.py"
try:
    s = p.read_text(encoding="utf-8")
except FileNotFoundError:
    problems.append("Missing app/models.py")
else:
    if "class Dataset(" not in s:
        # Ensure SA Base import
        if "from .database import Base as SA_Base" not in s:
            s = s.replace(
                "from sqlalchemy import JSON",
                "from sqlalchemy import JSON\nimport sqlalchemy as sa\nfrom .database import Base as SA_Base",
            )
        # If that replacement didn’t happen (e.g. file differs), just append imports up top once.
        if "import sqlalchemy as sa" not in s:
            s = re.sub(r"(^from .*?\n)+", r"\g<0>import sqlalchemy as sa\nfrom .database import Base as SA_Base\n", s, count=1, flags=re.M)
        dataset_class = """
# ---------------- SQLAlchemy models (used by /api/datasets) ---------------
class Dataset(SA_Base):
    __tablename__ = "datasets"
    uid = sa.Column(sa.String(32), primary_key=True)
    created_at = sa.Column(sa.DateTime, nullable=False, default=datetime.utcnow)
    raw_path = sa.Column(sa.String, nullable=False)
    emb_path = sa.Column(sa.String, nullable=False)
    config = sa.Column(sa.JSON, default=dict)
    label = sa.Column(sa.String, nullable=True)

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "raw_path": self.raw_path,
            "emb_path": self.emb_path,
            "created_at": (self.created_at.isoformat() if self.created_at else None),
            "label": self.label,
            "config": self.config,
        }
"""
        if not s.endswith("\n"):
            s += "\n"
        s += dataset_class
        p.write_text(s, encoding="utf-8")

# 2) app/routers/datasets.py – serialize properly and add helper
p = root / "app" / "routers" / "datasets.py"
try:
    s = p.read_text(encoding="utf-8")
except FileNotFoundError:
    problems.append("Missing app/routers/datasets.py")
else:
    if "_ds_to_dict(" not in s:
        s = s.replace(
            "class DatasetUpdate(BaseModel):",
            "class DatasetUpdate(BaseModel):",
        )
        helper = """
def _ds_to_dict(ds: Dataset) -> dict:
    return {
        "uid": ds.uid,
        "raw_path": ds.raw_path,
        "emb_path": ds.emb_path,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
        "label": ds.label,
        "config": ds.config,
    }
"""
        s = s.replace("router = APIRouter()", "router = APIRouter()" + helper)

    # list_datasets -> returns list of dicts
    s = re.sub(
        r"@router\.get\(\"/datasets\"\)[\s\S]*?def\s+list_datasets\(.*?\):\s*[\r\n]+(.*?return\s+.*?)\n",
        '@router.get("/datasets")\ndef list_datasets(db: Session = Depends(get_db)):\n'
        '    rows = db.query(Dataset).order_by(Dataset.created_at.desc()).all()\n'
        '    return [_ds_to_dict(d) for d in rows]\n',
        s,
        flags=re.S,
    )

    # update_dataset -> return dict
    s = re.sub(
        r"@router\.put\(\"/datasets/\{uid\}\"\)[\s\S]*?def\s+update_dataset\(.*?\):\s*[\r\n]+([\s\S]*?)return\s+ds",
        '@router.put("/datasets/{uid}")\n'
        'def update_dataset(uid: str, payload: DatasetUpdate, db: Session = Depends(get_db)):\n'
        '    ds = db.query(Dataset).filter(Dataset.uid == uid).first()\n'
        '    if not ds:\n'
        '        raise HTTPException(404, "Not found")\n'
        '    ds.label = payload.label\n'
        '    db.commit()\n'
        '    db.refresh(ds)\n'
        '    return _ds_to_dict(ds)',
        s,
        flags=re.S,
    )

    # rerun -> return dict
    s = re.sub(
        r"@router\.post\(\"/datasets/\{uid\}/rerun\"\)[\s\S]*?def\s+.*?rerun.*?\)[\s\S]*?db\.refresh\(new_ds\)\s*[\r\n]+return\s+new_ds",
        '@router.post("/datasets/{uid}/rerun")\n'
        'async def rerun_dataset(uid: str, db: Session = Depends(get_db)):\n'
        '    ds = db.query(Dataset).filter(Dataset.uid == uid).first()\n'
        '    if not ds:\n'
        '        raise HTTPException(404, "Not found")\n'
        '    # This assumes you have functions building raw_name/emb_name/cfg above; keeping original logic.\n'
        '    new_uid = str(uuid.uuid4())[:8]\n'
        '    raw_name = ds.raw_path\n'
        '    emb_name = ds.emb_path\n'
        '    cfg = ds.config or {}\n'
        '    new_ds = Dataset(uid=new_uid, raw_path=raw_name, emb_path=emb_name, config=cfg, label=ds.label)\n'
        '    db.add(new_ds)\n'
        '    db.commit()\n'
        '    db.refresh(new_ds)\n'
        '    return _ds_to_dict(new_ds)',
        s,
        flags=re.S,
    )
    p.write_text(s, encoding="utf-8")

# 3) app/routers/db.py – add missing Path import
p = root / "app" / "routers" / "db.py"
try:
    s = p.read_text(encoding="utf-8")
except FileNotFoundError:
    problems.append("Missing app/routers/db.py")
else:
    if "from pathlib import Path" not in s:
        s = s.replace("from typing import Optional", "from typing import Optional\nfrom pathlib import Path")
        p.write_text(s, encoding="utf-8")

# 4) Makefile – fix targets and dev line
p = root / "Makefile"
try:
    s = p.read_text(encoding="utf-8")
except FileNotFoundError:
    problems.append("Missing Makefile")
else:
    s = s.replace("\tpython -m venv .venv && source .venv/bin/activate\n\tpython -m dotenv run -- uvicorn app.main:app --reload --port 5000",
                  "\tpython -m venv .venv && . .venv/bin/activate && python -m dotenv run -- uvicorn app.main:app --reload --port 5000")
    s = s.replace("\ttpytest -q", "\tpytest -q")
    s = s.replace("\ttdocker compose build", "\tdocker compose build")
    s = s.replace("\ttdocker compose up -d", "\tdocker compose up -d")
    s = s.replace("\ttdocker compose down", "\tdocker compose down")
    if "ruff" not in s and "format:" in s:
        s = re.sub(r"(format:\n)([\s\S]*?)$", r"\1\t@command -v ruff >/dev/null 2>&1 && ruff check --fix . || true\n\t@command -v black >/dev/null 2 && black . || true\n", s, flags=re.S)
    p.write_text(s, encoding="utf-8")

# 5) requirements.txt – remove duplicate python-dotenv (keep one)
p = root / "requirements.txt"
try:
    s = p.read_text(encoding="utf-8")
except FileNotFoundError:
    problems.append("Missing requirements.txt")
else:
    lines = [ln for ln in s.splitlines() if ln.strip()]
    seen = set()
    out = []
    for ln in lines:
        key = ln.strip().lower()
        if key == "python-dotenv":
            if "python-dotenv" in seen:
                continue
        if key in seen and key in {"uvicorn","fastapi"}:
            # dedupe common accidental duplicates
            continue
        seen.add(key)
        out.append(ln)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")

# 6) static/index.html – fix illegal <script src> with inline content & duplicate #batchLinks, reorder helpers
p = root / "static" / "index.html"
try:
    s = p.read_text(encoding="utf-8")
except FileNotFoundError:
    problems.append("Missing static/index.html")
else:
    # Replace a <script src=...> that (incorrectly) includes inline code with a clean loader tag
    s = re.sub(
        r'<script\s+src="https://cdn\.plot\.ly/plotly-2\.32\.0\.min\.js"\s+defer>([\s\S]*?)</script>',
        r'<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" defer></script>',
        s,
        flags=re.S,
    )
    # Merge duplicate batchLinks (span+div -> single div.small)
    s = s.replace(
        '<span class="small" id="batchLinks"></span>\n          <div id="batchLinks">',
        '<div id="batchLinks" class="small">',
    )
    # Ensure $$ then $ (avoid redefinition issues)
    s = s.replace(
        "const $ = (q) => document.querySelector(q);\nconst $$ = (q) => Array.from(document.querySelectorAll(q));",
        "const $$ = (q) => Array.from(document.querySelectorAll(q));\nconst $ = (q) => document.querySelector(q);",
    )
    p.write_text(s, encoding="utf-8")

if problems:
    print("Completed with notes:")
    for m in problems:
        print(" -", m)
else:
    print("OK: Applied fixes.")
