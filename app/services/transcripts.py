from __future__ import annotations

import re
from typing import Dict, List, Tuple
from docx import Document
from fastapi import UploadFile


async def load_transcripts(
    files: List[UploadFile], pattern: str = r"(\d+)"
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return two dicts:
      - transcripts: {id -> transcript text}
      - names: {id -> source filename}
    IDs are extracted from digits in the filename.
    """
    transcripts: Dict[str, str] = {}
    names: Dict[str, str] = {}
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
            names[key] = name
        except Exception:
            continue
    return transcripts, names
