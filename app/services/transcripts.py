from __future__ import annotations

import re
from typing import Dict, Iterable, List
from docx import Document
from fastapi import UploadFile

async def load_transcripts(files: List[UploadFile], pattern: str = r"(\d+)") -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    for fs in files:
        name = fs.filename or ""
        m = re.search(pattern, name)
        if not m:
            continue
        key = m.group(1)
        try:
            content = await fs.read()
            from io import BytesIO
            doc = Document(BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
            transcripts[key] = text
        except Exception:
            continue
    return transcripts
