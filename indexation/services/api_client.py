from __future__ import annotations

"""API client helper functions."""

from typing import List, Dict, Any
import requests

from indexation.config import API_BASE, API_TOKEN, TIMEOUT


def auth_headers() -> Dict[str, str]:
    """Return authorization headers for the external API."""
    headers = {"Accept": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    return headers


def fetch_all_programs(max_pages: int = 50, page_param: str = "page", start_page: int = 1) -> List[Dict[str, Any]]:
    """Fetch programs from the API with minimal assumptions about its shape."""
    items: List[Dict[str, Any]] = []
    url = API_BASE
    seen_urls = set()
    for _ in range(max_pages):
        if not url or url in seen_urls:
            break
        seen_urls.add(url)
        r = requests.get(url, headers=auth_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                items.extend(data["results"])
            elif "data" in data and isinstance(data["data"], list):
                items.extend(data["data"])
            elif "items" in data and isinstance(data["items"], list):
                items.extend(data["items"])
            elif isinstance(data.get("programs"), list):
                items.extend(data["programs"])

            next_url = data.get("next") or data.get("links", {}).get("next")
            if next_url:
                url = next_url
                continue

            if page_param in (url.split("?")[-1] if "?" in url else ""):
                import urllib.parse as up

                parsed = up.urlparse(url)
                qs = dict(up.parse_qsl(parsed.query))
                try:
                    cur = int(qs.get(page_param, start_page))
                    qs[page_param] = str(cur + 1)
                    url = parsed._replace(query=up.urlencode(qs)).geturl()
                    continue
                except Exception:
                    pass

            if "?" in url:
                url = url + f"&{page_param}={start_page+1}"
            else:
                url = url + f"?{page_param}={start_page+1}"
            start_page += 1
            continue

        elif isinstance(data, list):
            items.extend(data)
            break
        else:
            break

    return items


def discover_fields() -> List[str]:
    """Attempt to discover available fields by sampling the API."""
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
