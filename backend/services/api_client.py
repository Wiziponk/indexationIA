from typing import List, Dict, Any, Optional
import httpx
from pydantic import BaseModel
from ..config import settings

# Fallback sample data if API_BASE isn't set
SAMPLE = [
    {"id": "p1", "title": "Climate Change in Europe", "description": "Explores policies and impacts.", "theme":"Climate"},
    {"id": "p2", "title": "Art in Motion", "description": "A journey through contemporary dance.", "theme":"Culture"},
    {"id": "p3", "title": "Coding for Kids", "description": "An intro series to programming.", "theme":"Education"},
    {"id": "p4", "title": "History of Jazz", "description": "From New Orleans to the world.", "theme":"Music"},
]

async def fetch_programs(base: Optional[str], max_items: int = 500) -> List[Dict[str, Any]]:
    """Fetch records from the source API. Supports a simple page=? scheme.
    If no base is provided, returns SAMPLE data."""
    if not base:
        return SAMPLE

    records: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=20) as client:
        page = 1
        while len(records) < max_items:
            url = base
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}page={page}&page_size=100"
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "results" in data:
                batch = data["results"]
            elif isinstance(data, list):
                batch = data
            else:
                batch = []
            if not batch:
                break
            records.extend(batch)
            page += 1
    return records[:max_items]

def fields_from_records(records: List[Dict[str, Any]]) -> List[str]:
    if not records: 
        return []
    keys = set()
    for r in records[:10]:
        keys.update(r.keys())
    return sorted(keys)

def sample_rows(records: List[Dict[str, Any]], fields: List[str], n: int = 5) -> List[Dict[str, Any]]:
    out = []
    for r in records[:n]:
        row = {k: r.get(k) for k in fields}
        out.append(row)
    return out
