from __future__ import annotations

from typing import List, Dict, Any
import requests

from ..config import API_BASE, API_TOKEN, TIMEOUT

def auth_headers() -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    return headers

def fetch_all_programs(max_pages: int = 50, page_param: str = "page", start_page: int = 1) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    url = API_BASE
    seen = set()
    for _ in range(max_pages):
        if not url or url in seen:
            break
        seen.add(url)
        r = requests.get(url, headers=auth_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            if isinstance(data.get("results"), list):
                items.extend(data["results"])
            elif isinstance(data.get("data"), list):
                items.extend(data["data"])
            elif isinstance(data.get("items"), list):
                items.extend(data["items"])
            elif isinstance(data.get("programs"), list):
                items.extend(data["programs"])
            next_url = data.get("next") or data.get("links", {}).get("next")
            if next_url:
                url = next_url
                continue
            # fallback: naive page increment
            import urllib.parse as up
            parsed = up.urlparse(url)
            qs = dict(up.parse_qsl(parsed.query))
            try:
                cur = int(qs.get(page_param, start_page))
                qs[page_param] = str(cur + 1)
                url = parsed._replace(query=up.urlencode(qs)).geturl()
            except Exception:
                url = None
        elif isinstance(data, list):
            items.extend(data)
            break
        else:
            break
    return items

def discover_fields() -> List[str]:
    try:
        r = requests.get(API_BASE, headers=auth_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            sample = None
            for key in ["results", "data", "items", "programs"]:
                if isinstance(data.get(key), list) and data[key]:
                    sample = data[key][0]
                    break
            if sample is None and data:
                sample = data
        elif isinstance(data, list) and data:
            sample = data[0]
        else:
            sample = {}
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
