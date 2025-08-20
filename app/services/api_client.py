from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Set
import requests

from ..config import API_BASE, API_TOKEN, TIMEOUT, VERIFY_SSL


def auth_headers() -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def _http_get(url: str) -> requests.Response:
    r = requests.get(url, headers=auth_headers(), timeout=TIMEOUT, verify=VERIFY_SSL)
    r.raise_for_status()
    return r


def _extract_list(payload: Any) -> List[Dict[str, Any]]:
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
    Your API is a single big JSON list (or a dict wrapping it).
    No pagination: fetch once.
    """
    if not API_BASE or not API_BASE.strip():
        raise RuntimeError("EDUC_API_BASE not set.")
    data = _http_get(API_BASE).json()
    return _extract_list(data)


def _union_keys(items: List[Dict[str, Any]], max_items: int = 50) -> Tuple[List[str], int]:
    """
    Union top-level keys + one-level dotted keys from up to max_items elements.
    """
    fields: Set[str] = set()
    sample = items[: max(1, min(max_items, len(items)))]
    for row in sample:
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            fields.add(k)
            if isinstance(v, dict):
                for sk in v.keys():
                    fields.add(f"{k}.{sk}")
    ordered = sorted(fields)
    return ordered, len(sample)


def discover_fields() -> Tuple[List[str], str, int, Optional[str]]:
    """
    Best-effort field discovery with diagnostics.
    Returns: (fields, source, sample_size, error_message)
      - source: 'direct' if taken from API_BASE payload,
                'fallback' if taken from fetch_all_programs(),
                'empty' if nothing found
    """
    if not API_BASE or not API_BASE.strip():
        return [], "empty", 0, "EDUC_API_BASE is not configured."

    # Try direct
    try:
        data = _http_get(API_BASE).json()
        items = _extract_list(data)
        if not items and isinstance(data, dict) and data:
            # payload is a single object — treat it as one item
            items = [data]
        if items:
            fields, n = _union_keys(items)
            if fields:
                return fields, "direct", n, None
    except Exception as e:
        # fall through to fallback with reason
        last_err = f"Direct call failed: {e}"
    else:
        last_err = None

    # Fallback: use fetch_all_programs()
    try:
        items = fetch_all_programs()
        if items:
            fields, n = _union_keys(items)
            if fields:
                return fields, "fallback", n, last_err
    except Exception as e2:
        err = (last_err + " | " if last_err else "") + f"Fallback failed: {e2}"
        return [], "empty", 0, err

    return [], "empty", 0, last_err
