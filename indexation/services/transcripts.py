from __future__ import annotations

"""Transcript loading utilities."""

import re
from pathlib import Path
from typing import Dict, Iterable

from docx import Document


def load_transcripts(files: Iterable, pattern: str = r"(\d+)") -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    for fs in files:
        name = Path(fs.filename or "").name
        m = re.search(pattern, name)
        if not m:
            continue
        key = m.group(1)
        try:
            fs.stream.seek(0)
            doc = Document(fs.stream)
            text = "\n".join(p.text for p in doc.paragraphs)
            transcripts[key] = text
        except Exception:
            continue
    return transcripts
