import json
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .config import API_BASE, API_TOKEN, TIMEOUT, EMBED_MODEL, client


def auth_headers():
    headers = {"Accept": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    return headers


def fetch_all_programs(max_pages=50, page_param="page", start_page=1):
    """Fetch programs from the API with minimal assumptions about its shape."""
    items = []
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


def discover_fields():
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
        return sorted(sample.keys())
    except Exception:
        return []


def ensure_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    return [val]


def build_text_from_fields(row, fields):
    parts = []
    for f in fields:
        v = row.get(f, "")
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        parts.append(str(v))
    return "\n".join([p for p in parts if p and p.strip()])


def batch_embed(texts, model=EMBED_MODEL):
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def auto_kmeans(X, k_min=4, k_max=10):
    best_k, best_score, best_labels = None, -1, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_k is None:
        km = KMeans(n_clusters=min(4, len(X)), n_init="auto", random_state=42)
        best_labels = km.fit_predict(X)
        best_k = len(set(best_labels))
        best_score = -1
    return best_k, best_score, best_labels


def pca_2d(X):
    p = PCA(n_components=2, random_state=42)
    return p.fit_transform(X)
