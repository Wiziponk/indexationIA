from __future__ import annotations

from typing import List, Dict, Any
import requests

from ..config import API_BASE, API_TOKEN, TIMEOUT


def auth_headers() -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    return headers


def _http_get(url: str) -> requests.Response:
    """One place to apply headers & timeout and raise meaningful errors."""
    r = requests.get(url, headers=auth_headers(), timeout=TIMEOUT)
    r.raise_for_status()
    return r


def _extract_list(payload: Any) -> List[Dict[str, Any]]:
    """
    Your API returns a single big JSON list.

    Still, we keep a tiny bit of tolerance:
    - If it's already a list, return it.
    - If it's a dict with a top-level list under a common key (results/data/items/programs),
      return that list.
    - Otherwise, return [].
    """
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("results", "data", "items", "programs"):
            val = payload.get(key)
            if isinstance(val, list):
                return val

    return []


def fetch_all_programs() -> List[Dict[str, Any]]:
    """
    Fetch once, assume the API returns ALL items as a single JSON list.
    No pagination, no "next" following, no page guessing.
    """
    if not API_BASE or not str(API_BASE).strip():
        raise RuntimeError("EDUC_API_BASE not set on the server.")

    r = _http_get(API_BASE)
    data = r.json()
    items = _extract_list(data)
    return items


def discover_fields() -> List[str]:
    """
    Return a flat list of field names to present in the UI.
    - Includes top-level keys
    - Includes one-level dotted keys for nested dicts (k.sub)
    Uses ONLY the single response body (no pagination).
    """
    try:
        if not API_BASE or not str(API_BASE).strip():
            return []

        r = _http_get(API_BASE)
        data = r.json()
        items = _extract_list(data)
        sample = items[0] if items else (data if isinstance(data, dict) else {})

        if not isinstance(sample, dict):
            return []

        fields = set()
        for k, v in sample.items():
            fields.add(k)
            if isinstance(v, dict):
                for sk in v.keys():
                    fields.add(f"{k}.{sk}")
        return sorted(fields)
    except Exception:
        return []
